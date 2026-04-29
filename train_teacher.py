import os

# 必须在 import torch 之前
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import random
import argparse
import numpy as np

import torch
import torch.backends.cudnn as cudnn

from scipy.ndimage import gaussian_filter, label as cc_label
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, auc

from dataset import get_data_transforms
from dataset import MVTecTrainPseudoDomainDataset

from resnet import wide_resnet50_2
from de_resnet import de_wide_resnet50_2

from cond_modules import (
    DomainFiLM, ConditionalTeacher, ClusterDomainRouter,
    infer_z_channels,
    build_merged_domain_test_loader, get_domain_name_to_id,
)


def setup_seed(seed: int, deterministic: bool):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    if deterministic:
        cudnn.deterministic = True
        cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)

        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
    else:
        cudnn.deterministic = False
        cudnn.benchmark = True
        torch.use_deterministic_algorithms(False)

        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_loader(
    dataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
    deterministic: bool,
    pin_memory: bool = True,
    drop_last: bool = False,
):
    effective_num_workers = num_workers

    g = torch.Generator()
    g.manual_seed(seed)

    kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=effective_num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        worker_init_fn=seed_worker,
        generator=g,
    )

    if effective_num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    else:
        kwargs["persistent_workers"] = False

    return torch.utils.data.DataLoader(**kwargs)


def domain_id_to_name_map(domain_name_to_id):
    return {int(v): str(k) for k, v in dict(domain_name_to_id).items()}


def feature_l2_sample_losses(ref_feats, pred_feats):
    """
    Return per-sample multi-scale squared L2 reconstruction losses.

    This implements the sample-level loss used by the updated method:
      L(x_i) = sum_l ||pred_i^l - ref_i^l||_2^2 normalized over the feature map.

    The normalization is over all non-batch dimensions to keep the scale stable
    across different ResNet stages while preserving sample-wise/domain-wise hardness.
    """
    if len(ref_feats) != len(pred_feats):
        raise ValueError(f"Feature list length mismatch: {len(ref_feats)} vs {len(pred_feats)}")

    losses = []
    for ref, pred in zip(ref_feats, pred_feats):
        if ref.shape != pred.shape:
            raise ValueError(f"Feature shape mismatch: ref={tuple(ref.shape)}, pred={tuple(pred.shape)}")
        diff = pred.float() - ref.float()
        losses.append(diff.pow(2).flatten(1).mean(dim=1))

    return torch.stack(losses, dim=0).sum(dim=0)


def enforce_domain_q_bounds(domain_q, q_min_floor: float = 0.0, q_max_cap: float = 1.0):
    """
    Project pseudo-domain weights onto a bounded probability simplex:
      sum_k q_k = 1,
      q_min_floor <= q_k <= q_max_cap.

    This keeps GroupDRO aggressive enough to emphasize hard pseudo-domains, while
    preventing winner-take-all collapse such as q_max -> 0.99. The default
    recommended setting for the current texture experiment is:
      dro_eta=0.08, q_min_floor=0.005, q_max_cap=0.90.
    """
    with torch.no_grad():
        q = domain_q.detach().float().clamp_min(1e-12)
        n = int(q.numel())
        if n <= 0:
            return q.detach()

        q = q / q.sum().clamp_min(1e-12)

        floor = max(0.0, float(q_min_floor))
        cap = float(q_max_cap)
        if cap <= 0:
            cap = 1.0

        # Make the bounds feasible for n domains.
        floor = min(floor, 1.0 / float(n))
        cap = max(cap, 1.0 / float(n))
        cap = min(cap, 1.0 - (n - 1) * floor)
        cap = max(cap, floor)

        if floor <= 0.0 and cap >= 1.0:
            return q.detach()

        # Bounded simplex projection. Find tau such that
        # sum(clamp(q - tau, floor, cap)) = 1.
        lo = float((q - cap).min().item()) - 1.0
        hi = float((q - floor).max().item()) + 1.0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            projected = torch.clamp(q - mid, min=floor, max=cap)
            if float(projected.sum().item()) > 1.0:
                lo = mid
            else:
                hi = mid

        q = torch.clamp(q - 0.5 * (lo + hi), min=floor, max=cap)
        q = q / q.sum().clamp_min(1e-12)
        return q.detach()


def update_groupdro_weights(
    domain_q,
    sample_losses,
    domain_id,
    eta: float,
    q_min_floor: float = 0.0,
    q_max_cap: float = 1.0,
):
    """
    Stable GroupDRO update:
        q_k <- q_k exp(eta * mean_loss_k) / sum_j q_j exp(eta * mean_loss_j)

    Only pseudo-domains present in the current mini-batch receive a loss update.
    Absent pseudo-domains keep their previous log weight before global renormalization.
    After the update, q is projected to a bounded simplex so hard domains can be
    emphasized without letting one pseudo-domain absorb nearly all probability mass.
    """
    if eta <= 0:
        return enforce_domain_q_bounds(domain_q.detach(), q_min_floor, q_max_cap)

    with torch.no_grad():
        log_q = torch.log(domain_q.detach().clamp_min(1e-12))
        unique_domain_ids = torch.unique(domain_id.detach())

        for k_tensor in unique_domain_ids:
            k = int(k_tensor.item())
            if k < 0 or k >= domain_q.numel():
                continue
            mask = (domain_id == k)
            if bool(mask.any()):
                mean_loss_k = sample_losses.detach()[mask].mean()
                log_q[k] = log_q[k] + float(eta) * mean_loss_k

        log_q = log_q - torch.logsumexp(log_q, dim=0)
        q = torch.exp(log_q).detach()
        return enforce_domain_q_bounds(q, q_min_floor, q_max_cap)

def groupdro_batch_loss(sample_losses, domain_id, domain_q):
    """Compute q-weighted average over pseudo-domain mean losses in the batch."""
    weighted_terms = []
    active_weights = []
    active_losses = []

    for k_tensor in torch.unique(domain_id.detach()):
        k = int(k_tensor.item())
        if k < 0 or k >= domain_q.numel():
            continue
        mask = (domain_id == k)
        if not bool(mask.any()):
            continue
        mean_loss_k = sample_losses[mask].mean()
        weight_k = domain_q[k].detach()
        weighted_terms.append(weight_k * mean_loss_k)
        active_weights.append(weight_k)
        active_losses.append(mean_loss_k.detach())

    if len(weighted_terms) == 0:
        return sample_losses.mean(), sample_losses.detach().mean()

    loss = torch.stack(weighted_terms).sum() / torch.stack(active_weights).sum().clamp_min(1e-12)
    active_mean = torch.stack(active_losses).mean()
    return loss, active_mean



def compute_groupdro_sample_coefficients(sample_losses, domain_id, domain_q):
    """
    Compute exact per-sample coefficients for the existing GroupDRO batch loss:
        loss = sum_k q_k * mean(loss_i | d_i=k) / sum_{k active} q_k

    This helper is used only by the micro-batch memory path. It preserves the
    original objective while allowing backward() to be executed on smaller
    chunks of a logical batch.
    """
    weighted_terms = []
    active_weights = []
    active_losses = []
    coeff = torch.zeros_like(sample_losses, dtype=sample_losses.dtype, device=sample_losses.device)

    active = []
    for k_tensor in torch.unique(domain_id.detach()):
        k = int(k_tensor.item())
        if k < 0 or k >= domain_q.numel():
            continue
        mask = (domain_id == k)
        if not bool(mask.any()):
            continue
        active.append((k, mask, int(mask.sum().item())))
        mean_loss_k = sample_losses.detach()[mask].mean()
        weight_k = domain_q[k].detach()
        weighted_terms.append(weight_k * mean_loss_k)
        active_weights.append(weight_k)
        active_losses.append(mean_loss_k.detach())

    if len(active) == 0:
        coeff.fill_(1.0 / max(1, int(sample_losses.numel())))
        return coeff.detach(), sample_losses.detach().mean()

    denom = torch.stack(active_weights).sum().clamp_min(1e-12)
    for k, mask, count_k in active:
        coeff[mask] = domain_q[k].detach() / denom / float(max(1, count_k))

    active_mean = torch.stack(active_losses).mean()
    return coeff.detach(), active_mean.detach()


def train_teacher_batch_with_optional_microbatch(
    *,
    encoder,
    teacher,
    img_cpu,
    domain_id_cpu,
    device,
    opt,
    scaler,
    domain_q,
    args,
):
    """
    One logical teacher training step.

    Fast path keeps the original full-batch implementation. Memory-safe path
    splits a logical batch into micro-batches for forward/backward while keeping
    the GroupDRO batch objective unchanged. It does not alter the model, loss,
    labels, router, or optimizer schedule.
    """
    micro_bs = int(getattr(args, "micro_batch_size", 0) or 0)
    batch_size = int(img_cpu.shape[0])

    if micro_bs <= 0 or micro_bs >= batch_size:
        img = img_cpu.to(device, non_blocking=True)
        domain_id = domain_id_cpu.to(device, non_blocking=True).long()

        with torch.no_grad():
            feats = encoder(img)
        del img

        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(args.amp and str(device).startswith("cuda"))):
            outs, _ = teacher(feats, domain_id)
            sample_losses = feature_l2_sample_losses(feats, outs)

        domain_q = update_groupdro_weights(
            domain_q=domain_q,
            sample_losses=sample_losses,
            domain_id=domain_id,
            eta=args.dro_eta,
            q_min_floor=args.q_min_floor,
            q_max_cap=args.q_max_cap,
        )
        loss, raw_domain_loss = groupdro_batch_loss(
            sample_losses=sample_losses,
            domain_id=domain_id,
            domain_q=domain_q,
        )
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        return domain_q, float(loss.item()), float(raw_domain_loss.item())

    # Pass 1: no-grad losses for the existing GroupDRO q update.
    det_losses = []
    domain_id = domain_id_cpu.to(device, non_blocking=True).long()
    with torch.no_grad():
        for st in range(0, batch_size, micro_bs):
            ed = min(batch_size, st + micro_bs)
            img_mb = img_cpu[st:ed].to(device, non_blocking=True)
            dom_mb = domain_id[st:ed]
            feats_mb = encoder(img_mb)
            outs_mb, _ = teacher(feats_mb, dom_mb)
            loss_mb = feature_l2_sample_losses(feats_mb, outs_mb)
            det_losses.append(loss_mb.detach())
            del img_mb, feats_mb, outs_mb, loss_mb

    sample_losses_det = torch.cat(det_losses, dim=0)
    domain_q = update_groupdro_weights(
        domain_q=domain_q,
        sample_losses=sample_losses_det,
        domain_id=domain_id,
        eta=args.dro_eta,
        q_min_floor=args.q_min_floor,
        q_max_cap=args.q_max_cap,
    )
    coeff, raw_domain_loss = compute_groupdro_sample_coefficients(sample_losses_det, domain_id, domain_q)
    loss_value = float((sample_losses_det * coeff).sum().detach().cpu().item())
    raw_loss_value = float(raw_domain_loss.detach().cpu().item())

    # Pass 2: gradient pass in micro-batches using exact full-batch coefficients.
    opt.zero_grad(set_to_none=True)
    for st in range(0, batch_size, micro_bs):
        ed = min(batch_size, st + micro_bs)
        img_mb = img_cpu[st:ed].to(device, non_blocking=True)
        dom_mb = domain_id[st:ed]
        coeff_mb = coeff[st:ed]

        with torch.no_grad():
            feats_mb = encoder(img_mb)
        del img_mb

        with torch.cuda.amp.autocast(enabled=(args.amp and str(device).startswith("cuda"))):
            outs_mb, _ = teacher(feats_mb, dom_mb)
            loss_mb_vec = feature_l2_sample_losses(feats_mb, outs_mb)
            loss_mb = (loss_mb_vec * coeff_mb).sum()

        scaler.scale(loss_mb).backward()
        del feats_mb, outs_mb, loss_mb_vec, loss_mb

    scaler.step(opt)
    scaler.update()
    return domain_q, loss_value, raw_loss_value



FULL_METRIC_KEYS = [
    "mAUROC_sp_max",
    "mAP_sp_max",
    "mF1_max_sp_max",
    "mAUPRO_px",
    "mAUROC_px",
    "mAP_px",
    "mF1_max_px",
    "mF1_px_0.2_0.8_0.1",
    "mAcc_px_0.2_0.8_0.1",
    "mIoU_px_0.2_0.8_0.1",
    "mIoU_max_px",
]


def _fmt5(v) -> str:
    if v is None:
        return "nan"
    try:
        v = float(v)
    except Exception:
        return "nan"
    return "nan" if np.isnan(v) else f"{v:.5f}"


def _is_valid_metric(v) -> bool:
    try:
        return v is not None and not np.isnan(float(v))
    except Exception:
        return False


def _nanmean_metric(xs) -> float:
    xs = np.asarray(xs, dtype=np.float64)
    return float(np.nanmean(xs)) if np.any(~np.isnan(xs)) else float("nan")


def _safe_auroc(y_true, y_score) -> float:
    y_true = np.asarray(y_true).astype(np.uint8)
    y_score = np.asarray(y_score, dtype=np.float64)
    if y_true.size == 0 or np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def _safe_ap(y_true, y_score) -> float:
    y_true = np.asarray(y_true).astype(np.uint8)
    y_score = np.asarray(y_score, dtype=np.float64)
    if y_true.size == 0 or np.unique(y_true).size < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def _safe_f1_max(y_true, y_score) -> float:
    y_true = np.asarray(y_true).astype(np.uint8)
    y_score = np.asarray(y_score, dtype=np.float64)
    if y_true.size == 0 or np.unique(y_true).size < 2:
        return float("nan")
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    f1 = 2.0 * precision * recall / np.clip(precision + recall, 1e-12, None)
    return float(np.nanmax(f1))


def _safe_iou_max(y_true, y_score) -> float:
    y_true = np.asarray(y_true).astype(np.uint8)
    y_score = np.asarray(y_score, dtype=np.float64)
    if y_true.size == 0 or np.unique(y_true).size < 2:
        return float("nan")
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    iou = precision * recall / np.clip(precision + recall - precision * recall, 1e-12, None)
    return float(np.nanmax(iou))


def _normalize_amaps_01(amaps: np.ndarray) -> np.ndarray:
    amaps = np.asarray(amaps, dtype=np.float32)
    mn = float(np.min(amaps))
    mx = float(np.max(amaps))
    if np.isclose(mx, mn):
        return np.zeros_like(amaps, dtype=np.float32)
    return ((amaps - mn) / (mx - mn)).astype(np.float32)


def _fixed_threshold_pixel_metrics(masks: np.ndarray, amaps: np.ndarray) -> dict:
    masks = (np.asarray(masks) > 0).astype(np.uint8)
    scores = _normalize_amaps_01(amaps)
    y_true = masks.reshape(-1).astype(bool)
    y_score = scores.reshape(-1)

    if y_true.size == 0:
        return {
            "mF1_px_0.2_0.8_0.1": float("nan"),
            "mAcc_px_0.2_0.8_0.1": float("nan"),
            "mIoU_px_0.2_0.8_0.1": float("nan"),
        }

    f1s, accs, ious = [], [], []
    thresholds = np.arange(0.2, 0.8000001, 0.1, dtype=np.float32)
    for th in thresholds:
        y_pred = y_score >= float(th)
        tp = float(np.logical_and(y_pred, y_true).sum())
        fp = float(np.logical_and(y_pred, ~y_true).sum())
        fn = float(np.logical_and(~y_pred, y_true).sum())
        tn = float(np.logical_and(~y_pred, ~y_true).sum())

        f1s.append((2.0 * tp) / max(2.0 * tp + fp + fn, 1e-12))
        accs.append((tp + tn) / max(tp + fp + fn + tn, 1e-12))
        ious.append(tp / max(tp + fp + fn, 1e-12))

    return {
        "mF1_px_0.2_0.8_0.1": float(np.mean(f1s)),
        "mAcc_px_0.2_0.8_0.1": float(np.mean(accs)),
        "mIoU_px_0.2_0.8_0.1": float(np.mean(ious)),
    }


def _compute_aupro(masks: np.ndarray, amaps: np.ndarray, num_th: int = 200, fpr_limit: float = 0.3) -> float:
    masks = np.asarray(masks).astype(np.uint8)
    amaps = np.asarray(amaps, dtype=np.float32)

    if masks.ndim != 3 or amaps.ndim != 3 or masks.shape != amaps.shape:
        return float("nan")

    masks = (masks > 0).astype(np.uint8)
    if masks.max() == 0:
        return float("nan")

    region_sets = []
    for img_idx, mask in enumerate(masks):
        labeled, num = cc_label(mask)
        if num > 0:
            # Keep the original image index. This is required when good images
            # are interleaved with anomalous images; otherwise PRO regions can
            # be evaluated against the wrong anomaly map.
            region_sets.append((img_idx, labeled, num))
    if len(region_sets) == 0:
        return float("nan")

    bg_pixels = np.logical_not(masks.astype(bool)).sum()
    if bg_pixels == 0:
        return float("nan")

    min_th = float(amaps.min())
    max_th = float(amaps.max())
    if np.isclose(min_th, max_th):
        return float("nan")

    thresholds = np.linspace(max_th, min_th, num=num_th, endpoint=True)
    fprs = [0.0]
    pros = [0.0]

    for th in thresholds:
        bin_amaps = amaps >= th
        fp = np.logical_and(bin_amaps, masks == 0).sum()
        fpr = float(fp) / float(bg_pixels)

        region_overlaps = []
        for img_idx, labeled, num in region_sets:
            pred = bin_amaps[img_idx]
            for rid in range(1, num + 1):
                region = labeled == rid
                region_overlaps.append(pred[region].mean())

        if len(region_overlaps) == 0:
            continue
        pro = float(np.mean(region_overlaps))
        if not np.isnan(pro):
            fprs.append(fpr)
            pros.append(pro)

    fprs = np.asarray(fprs, dtype=np.float64)
    pros = np.asarray(pros, dtype=np.float64)

    order = np.argsort(fprs)
    fprs = fprs[order]
    pros = pros[order]

    uniq_fprs = []
    uniq_pros = []
    for fpr in np.unique(fprs):
        uniq_fprs.append(fpr)
        uniq_pros.append(np.max(pros[fprs == fpr]))
    uniq_fprs = np.asarray(uniq_fprs, dtype=np.float64)
    uniq_pros = np.asarray(uniq_pros, dtype=np.float64)

    valid = uniq_fprs <= fpr_limit
    fprs = uniq_fprs[valid]
    pros = uniq_pros[valid]
    if fprs.size == 0:
        return float("nan")

    if fprs[-1] < fpr_limit:
        pro_at_limit = float(np.interp(fpr_limit, uniq_fprs, uniq_pros))
        fprs = np.append(fprs, fpr_limit)
        pros = np.append(pros, pro_at_limit)

    return float(auc(fprs / fpr_limit, pros))


def _init_full_metric_bucket():
    # Store per-image arrays, not flattened Python lists. Flattened pixel lists are
    # extremely memory-expensive for multi-class MVTec evaluation and caused
    # MemoryError during teacher eval. Metric definitions are unchanged: pixel
    # metrics are computed from the same masks/amaps at finalize.
    return {
        "gt_list_sp": [],
        "pr_list_sp": [],
        "masks": [],
        "amaps": [],
    }


def _add_metric_sample(bucket, gt_mask: np.ndarray, anomaly_map: np.ndarray, label_i: int):
    gt_mask = (np.asarray(gt_mask) > 0).astype(np.uint8, copy=False)
    anomaly_map = np.asarray(anomaly_map, dtype=np.float32)

    bucket["gt_list_sp"].append(int(label_i))
    bucket["pr_list_sp"].append(float(np.max(anomaly_map)))
    bucket["masks"].append(gt_mask)
    bucket["amaps"].append(anomaly_map)


def _finalize_full_metric_bucket(bucket) -> dict:
    n = len(bucket["gt_list_sp"])
    if n == 0:
        out = {k: float("nan") for k in FULL_METRIC_KEYS}
        out["n"] = 0
        return out

    masks = np.stack(bucket["masks"], axis=0).astype(np.uint8, copy=False)
    amaps = np.stack(bucket["amaps"], axis=0).astype(np.float32, copy=False)
    gt_px = masks.reshape(-1)
    pr_px = amaps.reshape(-1)

    fixed = _fixed_threshold_pixel_metrics(masks, amaps)
    out = {
        "mAUROC_sp_max": _safe_auroc(bucket["gt_list_sp"], bucket["pr_list_sp"]),
        "mAP_sp_max": _safe_ap(bucket["gt_list_sp"], bucket["pr_list_sp"]),
        "mF1_max_sp_max": _safe_f1_max(bucket["gt_list_sp"], bucket["pr_list_sp"]),
        "mAUPRO_px": _compute_aupro(masks, amaps, num_th=200, fpr_limit=0.3),
        "mAUROC_px": _safe_auroc(gt_px, pr_px),
        "mAP_px": _safe_ap(gt_px, pr_px),
        "mF1_max_px": _safe_f1_max(gt_px, pr_px),
        "mIoU_max_px": _safe_iou_max(gt_px, pr_px),
        "n": int(n),
    }
    out.update(fixed)
    bucket["masks"].clear()
    bucket["amaps"].clear()
    return out

def _update_metric_max(metric_max: dict, current: dict) -> dict:
    for k in FULL_METRIC_KEYS:
        v = current.get(k, float("nan"))
        old = metric_max.get(k, float("nan"))
        if _is_valid_metric(v) and (not _is_valid_metric(old) or float(v) > float(old)):
            metric_max[k] = float(v)
    return metric_max


def _print_eval_metric_table(name: str, current: dict, metric_max: dict):
    columns = ["Name"]
    for k in FULL_METRIC_KEYS:
        columns.append(k)
        columns.append(f"{k} (Max)")

    row = [str(name)]
    for k in FULL_METRIC_KEYS:
        row.append(_fmt5(current.get(k, float("nan"))))
        row.append(_fmt5(metric_max.get(k, float("nan"))))

    widths = [max(len(columns[i]), len(row[i])) for i in range(len(columns))]
    header = "| " + " | ".join(columns[i].center(widths[i]) for i in range(len(columns))) + " |"
    sep = "|" + "|".join("-" * (widths[i] + 2) for i in range(len(columns))) + "|"
    body = "| " + " | ".join(row[i].center(widths[i]) for i in range(len(row))) + " |"
    print(header)
    print(sep)
    print(body)


def _metric_value(metrics: dict, key: str) -> float:
    v = metrics.get(key, float("nan")) if isinstance(metrics, dict) else float("nan")
    try:
        return float(v)
    except Exception:
        return float("nan")


def _teacher_selection_score(metrics: dict) -> float:
    """
    Secondary tie-break score for teacher selection.
    The primary checkpoint criterion remains mAUROC_sp_max.
    """
    vals = [
        0.4 * _metric_value(metrics, "mAUROC_sp_max"),
        0.3 * _metric_value(metrics, "mAUROC_px"),
        0.3 * _metric_value(metrics, "mAUPRO_px"),
    ]
    if any(np.isnan(v) for v in vals):
        return float("nan")
    return float(sum(vals))


def _is_better_teacher(metrics: dict, best_info: dict, primary_key: str, eps: float = 1e-12) -> bool:
    """
    Best checkpoint rule:
      1) maximize the primary metric, default mAUROC_sp_max;
      2) only when primary metric is numerically tied, use the composite teacher score.
    """
    cur_primary = _metric_value(metrics, primary_key)
    if np.isnan(cur_primary):
        return False

    if best_info is None:
        return True

    best_primary = float(best_info.get("best_primary", float("nan")))
    if np.isnan(best_primary):
        return True
    if cur_primary > best_primary + eps:
        return True
    if abs(cur_primary - best_primary) <= eps:
        cur_score = _teacher_selection_score(metrics)
        best_score = float(best_info.get("best_score", float("nan")))
        if _is_valid_metric(cur_score) and (not _is_valid_metric(best_score) or cur_score > best_score + eps):
            return True
    return False


def build_teacher_payload(
    teacher,
    domain_q,
    args,
    num_domains: int,
    z_channels: int,
    domain_name_to_id: dict,
    epoch: int,
    latest_metrics: dict = None,
    metric_max: dict = None,
    best_info: dict = None,
    checkpoint_type: str = "latest",
    cluster_router: ClusterDomainRouter = None,
):
    meta = {
        "class_name": args.class_name,
        "mvtec_root": args.mvtec_root,
        "domains_dir": args.domains_dir,
        "pseudo_domain_json": args.pseudo_domain_json,
        "domain_source": getattr(args, "domain_source", "json" if args.pseudo_domain_json else "txt"),
        "num_domains": num_domains,
        "z_channels": z_channels,
        "domain_name_to_id": domain_name_to_id,
        "film_emb_dim": args.film_emb_dim,
        "film_hidden": args.film_hidden,
        "film_dropout": args.film_dropout,
        "router_type": "cluster_model_router",
        "cluster_dir": str(getattr(args, "cluster_dir", "")),
        "pseudo_script": str(getattr(args, "pseudo_script", "pseudo-domain_discover_json.py")),
        "cluster_infer_method": str(getattr(args, "cluster_infer_method", "hybrid")),
        "cluster_router_device": str(getattr(args, "cluster_router_device", "cpu")),
        "cluster_hdbscan_min_conf": float(getattr(args, "cluster_hdbscan_min_conf", 0.0)),
        "cluster_router_cache_enabled": not bool(getattr(args, "no_cluster_router_cache", False)),
        "cluster_router_cache_dir": str(getattr(args, "cluster_router_cache_dir", "")),
        "test_domain_assignment": "cluster_model_online",
        "dro_eta": args.dro_eta,
        "q_min_floor": args.q_min_floor,
        "q_max_cap": args.q_max_cap,
        "best_metric": args.best_metric,
        "checkpoint_type": checkpoint_type,
        "epoch": int(epoch),
        "final_domain_q": domain_q.detach().cpu().tolist(),
        "seed": args.seed,
        "deterministic": bool(args.deterministic),
        "amp": bool(args.amp),
        "eval_num_workers": args.eval_num_workers,
        "eval_batch_size": args.eval_batch_size,
    }
    if latest_metrics is not None:
        meta["latest_metrics"] = {k: float(v) for k, v in latest_metrics.items()}
    if metric_max is not None:
        meta["metric_max"] = {k: float(v) for k, v in metric_max.items()}
    if best_info is not None:
        meta["best_info"] = dict(best_info)

    payload = {
        "bn": teacher.bn.state_dict(),
        "decoder": teacher.decoder.state_dict(),
        "film": teacher.film.state_dict(),
        "domain_q": domain_q.detach().cpu(),
        "meta": meta,
    }
    return payload



@torch.no_grad()
def evaluate_full_metrics_over_merged_loader(
    encoder: torch.nn.Module,
    model: torch.nn.Module,
    domain_loader: torch.utils.data.DataLoader,
    device: str,
    model_kind: str,
    print_per_domain: bool = False,
    cluster_router: ClusterDomainRouter = None,
    id_to_domain_name: dict = None,
):
    from test import cal_anomaly_map

    encoder.eval()
    model.eval()
    if hasattr(model, "bn"):
        model.bn.eval()
    if hasattr(model, "decoder"):
        model.decoder.eval()
    if hasattr(model, "film"):
        model.film.eval()

    dataset = domain_loader.dataset
    all_domain_names = list(getattr(dataset, "all_domain_names", []))
    valid_domain_names = list(getattr(dataset, "valid_domain_names", []))
    skipped_domains = dict(getattr(dataset, "skipped_domains", {}))
    ordered_names = all_domain_names if len(all_domain_names) > 0 else valid_domain_names

    buckets = {name: _init_full_metric_bucket() for name in valid_domain_names}

    if id_to_domain_name is None:
        id_to_domain_name = {}

    for batch in domain_loader:
        if len(batch) >= 7:
            img, gt, label, _, domain_name, domain_id, img_paths = batch
        else:
            img, gt, label, _, domain_name, domain_id = batch
            img_paths = None
        img = img.to(device, non_blocking=True)
        domain_id = domain_id.to(device, non_blocking=True).long()

        feats = encoder(img)
        if cluster_router is not None:
            routed_domain_id = cluster_router.predict_paths(img_paths, output_device=device)
        else:
            routed_domain_id = domain_id

        if model_kind == "teacher":
            outs, _ = model(feats, routed_domain_id)
        elif model_kind == "student":
            outs, _ = model(feats)
        else:
            raise ValueError(f"Unsupported model_kind: {model_kind}")

        bs = img.shape[0]
        for i in range(bs):
            if cluster_router is not None:
                rid = int(routed_domain_id[i].detach().cpu().item())
                name_i = id_to_domain_name.get(rid, f"domain_{rid}")
            else:
                name_i = str(domain_name[i])
            if name_i not in buckets:
                continue

            feats_i = [x[i:i + 1] for x in feats]
            outs_i = [x[i:i + 1] for x in outs]
            anomaly_map, _ = cal_anomaly_map(feats_i, outs_i, img.shape[-1], amap_mode='a')
            anomaly_map = gaussian_filter(anomaly_map, sigma=4).astype(np.float32)

            gt_i = gt[i].detach().cpu().numpy()
            gt_mask = (np.squeeze(gt_i) > 0.5).astype(np.uint8)
            label_i = int(label[i].item()) if torch.is_tensor(label) else int(label[i])
            _add_metric_sample(buckets[name_i], gt_mask, anomaly_map, label_i)

    per_domain = {}
    n_valid = 0
    n_total = len(ordered_names)

    for name in ordered_names:
        if name not in buckets:
            if print_per_domain:
                reason = skipped_domains.get(name, "not included")
                print(f"  [domain={name}] skipped ({reason})")
            continue

        try:
            m = _finalize_full_metric_bucket(buckets[name])
            if int(m.get("n", 0)) <= 0:
                raise ValueError("no valid samples")
            per_domain[name] = m
            n_valid += 1

            if print_per_domain:
                print(
                    f"  [domain={name}] n={m['n']} "
                    f"mAUROC_sp_max={_fmt5(m['mAUROC_sp_max'])} "
                    f"mAP_sp_max={_fmt5(m['mAP_sp_max'])} "
                    f"mF1_max_sp_max={_fmt5(m['mF1_max_sp_max'])} "
                    f"mAUPRO_px={_fmt5(m['mAUPRO_px'])} "
                    f"mAUROC_px={_fmt5(m['mAUROC_px'])} "
                    f"mAP_px={_fmt5(m['mAP_px'])} "
                    f"mF1_max_px={_fmt5(m['mF1_max_px'])} "
                    f"mIoU_max_px={_fmt5(m['mIoU_max_px'])}"
                )
        except Exception as e:
            if print_per_domain:
                print(f"  [domain={name}] skipped: {e}")

    macro = {k: _nanmean_metric([m[k] for m in per_domain.values()]) for k in FULL_METRIC_KEYS}
    return {
        "macro": macro,
        "per_domain": per_domain,
        "n_valid": n_valid,
        "n_total": n_total,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--class_name", type=str, default="hunhe_png")
    parser.add_argument("--mvtec_root", type=str, default="./mvtec")
    parser.add_argument("--domains_dir", type=str, default="./outputs/domains")
    parser.add_argument("--pseudo_domain_json", type=str, default="", help="直接读取 pseudo-domain JSON manifest 训练/评估；为空时保留旧 txt 逻辑")
    parser.add_argument("--ckpt_dir", type=str, default="./checkpoints")

    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--micro_batch_size", type=int, default=4, help="Memory-only micro-batch size for teacher backward. 0 disables; objective is unchanged.")
    parser.add_argument("--num_workers", type=int, default=8)

    # 评估专用
    parser.add_argument("--eval_num_workers", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=8)

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--dro_eta", type=float, default=0.08)
    parser.add_argument("--q_min_floor", type=float, default=0.005)
    parser.add_argument("--q_max_cap", type=float, default=0.90)
    parser.add_argument("--best_metric", type=str, default="mAUROC_sp_max", choices=FULL_METRIC_KEYS)

    parser.add_argument("--film_emb_dim", type=int, default=256)
    parser.add_argument("--film_hidden", type=int, default=1024)
    parser.add_argument("--film_dropout", type=float, default=0.0)

    parser.add_argument("--cluster_dir", type=str, default="")
    parser.add_argument("--pseudo_script", type=str, default="pseudo-domain_discover_json.py")
    parser.add_argument("--cluster_infer_method", type=str, default="hybrid", choices=["hybrid", "hdbscan", "centers"])
    parser.add_argument("--cluster_router_device", type=str, default="cpu", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--cluster_hdbscan_min_conf", type=float, default=0.0)
    parser.add_argument("--cluster_router_cache_dir", type=str, default="")
    parser.add_argument("--cluster_router_cache_flush_interval", type=int, default=16)
    parser.add_argument("--no_cluster_router_cache", action="store_true")

    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--print_per_domain", action="store_true")

    parser.add_argument("--seed", type=int, default=111)

    # 默认 deterministic=True，只有显式传 --non_deterministic 才关闭
    parser.add_argument("--deterministic", dest="deterministic", action="store_true")
    parser.add_argument("--non_deterministic", dest="deterministic", action="store_false")
    parser.set_defaults(deterministic=True)

    # 默认 amp=False
    parser.add_argument("--amp", action="store_true")

    args = parser.parse_args()

    os.makedirs(args.ckpt_dir, exist_ok=True)
    teacher_ckpt = os.path.join(args.ckpt_dir, f"teacher_cond_{args.class_name}.pth")
    teacher_best_ckpt = os.path.join(args.ckpt_dir, f"teacher_cond_{args.class_name}_best.pth")

    setup_seed(args.seed, deterministic=args.deterministic)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    print("Deterministic:", args.deterministic)
    print("AMP:", args.amp)
    print("GroupDRO eta:", args.dro_eta)
    print("GroupDRO q_min_floor:", args.q_min_floor)
    print("GroupDRO q_max_cap:", args.q_max_cap)
    print("Best teacher metric:", args.best_metric)
    print("Train num_workers:", args.num_workers)
    print("Micro batch size:", args.micro_batch_size)
    print("Eval  num_workers:", args.eval_num_workers)
    print("Eval  batch_size :", args.eval_batch_size)
    print("Cluster router method:", args.cluster_infer_method)
    print("Cluster router device:", args.cluster_router_device)
    print("Cluster router cache:", not args.no_cluster_router_cache)
    if args.cluster_router_cache_dir:
        print("Cluster router cache dir:", args.cluster_router_cache_dir)
    print("Pseudo-domain source:", "json" if args.pseudo_domain_json else "txt")
    if args.pseudo_domain_json:
        print("Pseudo-domain JSON:", args.pseudo_domain_json)
    else:
        print("Domains dir:", args.domains_dir)
    args.domain_source = "json" if args.pseudo_domain_json else "txt"

    data_transform, gt_transform = get_data_transforms(args.image_size, args.image_size)
    root_path = os.path.join(args.mvtec_root, args.class_name)

    train_data = MVTecTrainPseudoDomainDataset(
        root=root_path,
        transform=data_transform,
        domain_dir=args.domains_dir,
        pseudo_domain_json=args.pseudo_domain_json,
        strict=True
    )

    # probe loader 只给 infer_z_channels 用，避免影响正式训练顺序
    probe_loader = build_loader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=False,
        seed=args.seed + 100003,
        num_workers=args.num_workers,
        deterministic=args.deterministic,
        pin_memory=True,
        drop_last=False,
    )

    # 正式训练 loader：独立随机源
    train_loader = build_loader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
        num_workers=args.num_workers,
        deterministic=args.deterministic,
        pin_memory=True,
        drop_last=False,
    )

    domain_name_to_id = get_domain_name_to_id(train_data, args.domains_dir, args.pseudo_domain_json)
    print(f"Domain mapping size: {len(domain_name_to_id)}")

    # 合并 domain 测试集：一个 loader，batch 内可混多个 domain
    merged_domain_test_loader = build_merged_domain_test_loader(
        root_path=root_path,
        domains_dir=args.domains_dir,
        domain_name_to_id=domain_name_to_id,
        data_transform=data_transform,
        gt_transform=gt_transform,
        batch_size=args.eval_batch_size,
        num_workers=args.eval_num_workers,
        pin_memory=True,
        strict_mask=True,
        pseudo_domain_json=args.pseudo_domain_json,
    )

    merged_ds = merged_domain_test_loader.dataset
    print(
        f"Built merged domain test loader: "
        f"total_txts={len(getattr(merged_ds, 'all_domain_names', []))}, "
        f"valid_domains={len(getattr(merged_ds, 'valid_domain_names', []))}, "
        f"samples={len(merged_ds)}"
    )

    encoder, bn = wide_resnet50_2(pretrained=True)
    encoder = encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    id_to_domain_name = domain_id_to_name_map(domain_name_to_id)
    cluster_router = ClusterDomainRouter.from_manifest(
        pseudo_domain_json=args.pseudo_domain_json,
        domain_name_to_id=domain_name_to_id,
        cluster_dir=args.cluster_dir,
        pseudo_script=args.pseudo_script,
        method=args.cluster_infer_method,
        feature_device=args.cluster_router_device,
        hdbscan_min_conf=args.cluster_hdbscan_min_conf,
        cache=(not args.no_cluster_router_cache),
        cache_dir=args.cluster_router_cache_dir,
        cache_flush_interval=args.cluster_router_cache_flush_interval,
    ) if args.pseudo_domain_json else None

    bn = bn.to(device)
    decoder = de_wide_resnet50_2(pretrained=False).to(device)

    num_domains = int(getattr(train_data, "num_domains"))

    # 用 probe_loader，不污染 train_loader 的随机顺序
    C = infer_z_channels(encoder, bn, probe_loader, device)
    print("num_domains:", num_domains, "z_channels:", C)

    film = DomainFiLM(
        num_domains=num_domains,
        channels=C,
        emb_dim=args.film_emb_dim,
        hidden=args.film_hidden,
        dropout=args.film_dropout,
        init_identity=True
    ).to(device)

    teacher = ConditionalTeacher(bn=bn, decoder=decoder, film=film).to(device)

    opt = torch.optim.Adam(
        list(teacher.bn.parameters()) +
        list(teacher.decoder.parameters()) +
        list(teacher.film.parameters()),
        lr=args.lr,
        betas=(0.5, 0.999)
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.startswith("cuda")))

    domain_q = torch.full(
        (num_domains,),
        fill_value=1.0 / max(1, num_domains),
        dtype=torch.float32,
        device=device,
    )
    eval_metric_max = {k: float("nan") for k in FULL_METRIC_KEYS}
    latest_metrics = None
    best_info = None

    for ep in range(1, args.epochs + 1):
        teacher.train()
        loss_list = []
        raw_loss_list = []

        for img, domain_id in train_loader:
            domain_q, loss_value, raw_loss_value = train_teacher_batch_with_optional_microbatch(
                encoder=encoder,
                teacher=teacher,
                img_cpu=img,
                domain_id_cpu=domain_id,
                device=device,
                opt=opt,
                scaler=scaler,
                domain_q=domain_q,
                args=args,
            )

            loss_list.append(float(loss_value))
            raw_loss_list.append(float(raw_loss_value))

        q_max = float(domain_q.max().detach().cpu().item()) if domain_q.numel() > 0 else float("nan")
        q_min = float(domain_q.min().detach().cpu().item()) if domain_q.numel() > 0 else float("nan")
        print(
            f"[Teacher] epoch [{ep:03d}/{args.epochs}] "
            f"loss_dro={np.mean(loss_list):.4f} "
            f"loss_raw={np.mean(raw_loss_list):.4f} "
            f"q_min={q_min:.4f} q_max={q_max:.4f} "
            f"eta={args.dro_eta:.4f}"
        )

        if (ep % args.eval_every == 0) or (ep == args.epochs):
            teacher.eval()
            eval_out = evaluate_full_metrics_over_merged_loader(
                encoder=encoder,
                model=teacher,
                domain_loader=merged_domain_test_loader,
                device=device,
                model_kind="teacher",
                print_per_domain=args.print_per_domain,
                cluster_router=cluster_router,
                id_to_domain_name=id_to_domain_name,
            )
            metrics = eval_out["macro"]
            n_valid = eval_out["n_valid"]
            n_total = eval_out["n_total"]
            eval_metric_max = _update_metric_max(eval_metric_max, metrics)
            print(
                f"[Teacher Eval@{ep}] Domain-MacroAvg over {n_valid}/{n_total} txts: "
                f"mAUROC_px={_fmt5(metrics['mAUROC_px'])}, "
                f"mAUROC_sp_max={_fmt5(metrics['mAUROC_sp_max'])}, "
                f"mAUPRO_px={_fmt5(metrics['mAUPRO_px'])}"
            )
            _print_eval_metric_table(f"Teacher@{ep}", metrics, eval_metric_max)
            latest_metrics = dict(metrics)

            if _is_better_teacher(metrics, best_info, primary_key=args.best_metric):
                best_info = {
                    "best_epoch": int(ep),
                    "best_metric": args.best_metric,
                    "best_primary": _metric_value(metrics, args.best_metric),
                    "best_score": _teacher_selection_score(metrics),
                    "best_metrics": {k: float(metrics.get(k, float("nan"))) for k in FULL_METRIC_KEYS},
                    "best_domain_q": domain_q.detach().cpu().tolist(),
                }
                torch.save(
                    build_teacher_payload(
                        teacher=teacher,
                        domain_q=domain_q,
                        args=args,
                        num_domains=num_domains,
                        z_channels=C,
                        domain_name_to_id=domain_name_to_id,
                        epoch=ep,
                        latest_metrics=metrics,
                        metric_max=eval_metric_max,
                        best_info=best_info,
                        checkpoint_type="best",
                        cluster_router=cluster_router,
                    ),
                    teacher_best_ckpt,
                )
                print(
                    f"[Best Teacher] epoch={ep} "
                    f"{args.best_metric}={_fmt5(best_info['best_primary'])} "
                    f"score={_fmt5(best_info['best_score'])} "
                    f"-> saved: {teacher_best_ckpt}"
                )

        if (ep % args.save_every == 0) or (ep == args.epochs):
            torch.save(
                build_teacher_payload(
                    teacher=teacher,
                    domain_q=domain_q,
                    args=args,
                    num_domains=num_domains,
                    z_channels=C,
                    domain_name_to_id=domain_name_to_id,
                    epoch=ep,
                    latest_metrics=latest_metrics,
                    metric_max=eval_metric_max,
                    best_info=best_info,
                    checkpoint_type="latest",
                    cluster_router=cluster_router,
                ),
                teacher_ckpt,
            )
            print("Saved latest:", teacher_ckpt)
            if best_info is not None:
                print(
                    f"Current best: epoch={best_info['best_epoch']} "
                    f"{best_info['best_metric']}={_fmt5(best_info['best_primary'])} "
                    f"best_ckpt={teacher_best_ckpt}"
                )

    print("Done. Teacher latest ckpt:", teacher_ckpt)
    print("Done. Teacher best ckpt:", teacher_best_ckpt)


if __name__ == "__main__":
    main()