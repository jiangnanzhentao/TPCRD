import os

# 必须在 import torch 之前
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import random
import argparse
import numpy as np

import torch
import torch.backends.cudnn as cudnn

from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score

from dataset import get_data_transforms
from dataset import MVTecTrainPseudoDomainDataset

from resnet import wide_resnet50_2
from de_resnet import de_wide_resnet50_2

from cond_modules import (
    DomainFiLM, ConditionalTeacher, StudentNoDomain, ClusterDomainRouter,
    rd4ad_cosine_loss, distill_feats_loss,
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



def build_cluster_router(args, teacher_meta: dict, domain_name_to_id: dict):
    if not args.pseudo_domain_json:
        return None
    cluster_dir = args.cluster_dir or str(teacher_meta.get("cluster_dir", "") or "")
    pseudo_script = args.pseudo_script or str(teacher_meta.get("pseudo_script", "pseudo-domain_discover_json.py") or "pseudo-domain_discover_json.py")
    method = args.cluster_infer_method or str(teacher_meta.get("cluster_infer_method", "hybrid") or "hybrid")
    feature_device = args.cluster_router_device or str(teacher_meta.get("cluster_router_device", "cpu") or "cpu")
    hdbscan_min_conf = float(args.cluster_hdbscan_min_conf if args.cluster_hdbscan_min_conf is not None else teacher_meta.get("cluster_hdbscan_min_conf", 0.0))
    return ClusterDomainRouter.from_manifest(
        pseudo_domain_json=args.pseudo_domain_json,
        domain_name_to_id=domain_name_to_id,
        cluster_dir=cluster_dir,
        pseudo_script=pseudo_script,
        method=method,
        feature_device=feature_device,
        hdbscan_min_conf=hdbscan_min_conf,
        cache=(not args.no_cluster_router_cache),
        cache_dir=args.cluster_router_cache_dir or str(teacher_meta.get("cluster_router_cache_dir", "") or ""),
        cache_flush_interval=args.cluster_router_cache_flush_interval,
    )


def _nanmean(xs):
    xs2 = [x for x in xs if (x == x)]
    if len(xs2) == 0:
        return float("nan")
    return float(sum(xs2) / len(xs2))


def feature_l2_sample_losses(ref_feats, pred_feats):
    """
    Lightweight per-sample teacher reconstruction quality for reliability-aware distillation.

    It is used only to compute r_i = exp(-tau * L_teacher(x_i)).
    The original RD loss and KD loss are kept unchanged to preserve the behavior
    of this training script.
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


def reliability_from_teacher_loss(teacher_sample_losses, tau: float, floor: float):
    """
    Lightweight reliability gate for KD.

    r_i = floor + (1 - floor) * exp(-tau * L_teacher(x_i))

    A non-zero floor keeps the update close to the original KD objective and
    avoids hurting the current strong baseline, while still downweighting samples
    where the conditional teacher reconstructs poorly.
    """
    tau = float(tau)
    floor = float(floor)
    floor = min(max(floor, 0.0), 1.0)
    exponent = (-tau * teacher_sample_losses.detach()).clamp(min=-60.0, max=0.0)
    raw = torch.exp(exponent)
    return (floor + (1.0 - floor) * raw).detach()


def distill_feats_sample_losses(teacher_feats, student_feats):
    """
    Return per-sample KD losses using the original distill_feats_loss definition.

    This keeps the KD metric identical to the existing code while making the
    reliability weighting faithful to the paper objective:
        L_KD = mean_i r_i * L_dist(x_i).
    """
    if len(teacher_feats) != len(student_feats):
        raise ValueError(f"Feature list length mismatch: {len(teacher_feats)} vs {len(student_feats)}")
    if len(teacher_feats) == 0:
        raise ValueError("Empty feature lists for distillation.")

    batch_size = int(teacher_feats[0].shape[0])
    losses = []
    for i in range(batch_size):
        t_i = [feat[i:i + 1] for feat in teacher_feats]
        s_i = [feat[i:i + 1] for feat in student_feats]
        losses.append(distill_feats_loss(t_i, s_i))
    return torch.stack(losses, dim=0)


def _init_metric_bucket():
    # Store arrays instead of flattened Python lists to avoid RAM blow-up during
    # multi-class pixel metric evaluation. Metrics are computed from the same
    # gt/anomaly map values in _finalize_metric_bucket.
    return {
        "gt_list_sp": [],
        "pr_list_sp": [],
        "aupro_list": [],
        "masks": [],
        "amaps": [],
    }


def _finalize_metric_bucket(bucket):
    if len(bucket["gt_list_sp"]) == 0:
        return float("nan"), float("nan"), float("nan")
    masks = np.stack(bucket["masks"], axis=0).astype(np.uint8, copy=False)
    amaps = np.stack(bucket["amaps"], axis=0).astype(np.float32, copy=False)
    auroc_px = float(round(roc_auc_score(masks.reshape(-1), amaps.reshape(-1)), 3))
    auroc_sp = float(round(roc_auc_score(bucket["gt_list_sp"], bucket["pr_list_sp"]), 3))
    aupro = float(np.mean(bucket["aupro_list"])) if len(bucket["aupro_list"]) > 0 else float("nan")
    aupro = float(round(aupro, 3)) if (aupro == aupro) else float("nan")
    bucket["masks"].clear()
    bucket["amaps"].clear()
    return auroc_px, auroc_sp, aupro


@torch.no_grad()
def macro_eval_student_over_merged_loader(
    encoder: torch.nn.Module,
    student: StudentNoDomain,
    domain_loader: torch.utils.data.DataLoader,
    device: str,
    print_per_domain: bool = False,
    cluster_router: ClusterDomainRouter = None,
    id_to_domain_name: dict = None,
):
    """
    merged-domain 版 student 评估：
      1) 所有 domain 样本合并到一个 DataLoader
      2) batch 内混合多个 domain 一起前向
      3) 指标按 domain_name 分桶统计，最后做 macro average
    """
    from test import cal_anomaly_map, compute_pro

    encoder.eval()
    student.bn.eval()
    student.decoder.eval()

    dataset = domain_loader.dataset
    all_domain_names = list(getattr(dataset, "all_domain_names", []))
    valid_domain_names = list(getattr(dataset, "valid_domain_names", []))
    skipped_domains = dict(getattr(dataset, "skipped_domains", {}))

    buckets = {name: _init_metric_bucket() for name in valid_domain_names}
    if id_to_domain_name is None:
        id_to_domain_name = {}

    for batch in domain_loader:
        if len(batch) >= 7:
            img, gt, label, _, domain_name, _domain_id, img_paths = batch
        else:
            img, gt, label, _, domain_name, _domain_id = batch
            img_paths = None

        img = img.to(device, non_blocking=True)

        feats = encoder(img)
        if cluster_router is not None:
            routed_domain_id = cluster_router.predict_paths(img_paths, output_device=device)
        else:
            routed_domain_id = None
        s_outs, _ = student(feats)

        bs = img.shape[0]
        for i in range(bs):
            if routed_domain_id is not None:
                rid = int(routed_domain_id[i].detach().cpu().item())
                name_i = id_to_domain_name.get(rid, f"domain_{rid}")
            else:
                name_i = str(domain_name[i])
            if name_i not in buckets:
                continue

            feats_i = [x[i:i + 1] for x in feats]
            outs_i = [x[i:i + 1] for x in s_outs]

            anomaly_map, _ = cal_anomaly_map(feats_i, outs_i, img.shape[-1], amap_mode='a')
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)

            gt_i = gt[i:i + 1].clone()
            gt_i[gt_i > 0.5] = 1
            gt_i[gt_i <= 0.5] = 0
            gt_np = gt_i.cpu().numpy().astype(int)

            label_i = int(label[i].item()) if torch.is_tensor(label) else int(label[i])

            bucket = buckets[name_i]

            if label_i != 0:
                try:
                    bucket["aupro_list"].append(
                        compute_pro(
                            gt_np.squeeze(0),
                            anomaly_map[np.newaxis, :, :]
                        )
                    )
                except Exception:
                    pass

            bucket["masks"].append(np.squeeze(gt_np).astype(np.uint8, copy=False))
            bucket["amaps"].append(np.asarray(anomaly_map, dtype=np.float32))
            bucket["gt_list_sp"].append(int(np.max(gt_np)))
            bucket["pr_list_sp"].append(float(np.max(anomaly_map)))

    pxs, sps, aps = [], [], []
    n_total = len(all_domain_names) if len(all_domain_names) > 0 else len(valid_domain_names)
    n_valid = 0

    ordered_names = all_domain_names if len(all_domain_names) > 0 else valid_domain_names

    for name in ordered_names:
        if name not in buckets:
            if print_per_domain:
                reason = skipped_domains.get(name, "not included")
                print(f"  [domain={name}] skipped ({reason})")
            continue

        try:
            auroc_px, auroc_sp, aupro_px = _finalize_metric_bucket(buckets[name])
            pxs.append(auroc_px)
            sps.append(auroc_sp)
            aps.append(aupro_px)
            n_valid += 1

            if print_per_domain:
                print(
                    f"  [domain={name}] "
                    f"PixelAUROC={auroc_px:.3f} "
                    f"SampleAUROC={auroc_sp:.3f} "
                    f"PixelAUPRO={aupro_px:.3f}"
                )
        except Exception as e:
            if print_per_domain:
                print(f"  [domain={name}] skipped: {e}")

    return _nanmean(pxs), _nanmean(sps), _nanmean(aps), n_valid, n_total


def build_student_ckpt_meta(args, teacher_ckpt, teacher_meta, train_data, domain_name_to_id):
    return {
        "class_name": args.class_name,
        "mvtec_root": args.mvtec_root,
        "domains_dir": args.domains_dir,
        "pseudo_domain_json": args.pseudo_domain_json,
        "domain_source": getattr(args, "domain_source", "json" if args.pseudo_domain_json else "txt"),
        "teacher_ckpt": teacher_ckpt,
        "teacher_meta": teacher_meta,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "eval_batch_size": args.eval_batch_size,
        "eval_num_workers": args.eval_num_workers,
        "epochs": args.epochs,
        "lr": args.lr,
        "kd_weight": args.kd_weight,
        "reliability_tau": args.reliability_tau,
        "reliability_floor": args.reliability_floor,
        "eval_every": args.eval_every,
        "save_every": args.save_every,
        "seed": args.seed,
        "deterministic": bool(args.deterministic),
        "amp": bool(args.amp),
        "num_domains": int(getattr(train_data, "num_domains")),
        "domain_name_to_id": domain_name_to_id,
        "test_domain_assignment": "cluster_model_online" if teacher_meta.get("router_type") == "cluster_model_router" else "dataset_domain",
        "router_type": teacher_meta.get("router_type", ""),
        "cluster_dir": teacher_meta.get("cluster_dir", ""),
        "cluster_infer_method": teacher_meta.get("cluster_infer_method", ""),
        "cluster_router_cache_enabled": not bool(getattr(args, "no_cluster_router_cache", False)),
        "cluster_router_cache_dir": args.cluster_router_cache_dir or str(teacher_meta.get("cluster_router_cache_dir", "") or ""),
    }


def save_student_ckpt(path, student, meta, best_info=None):
    payload = {
        "bn": student.bn.state_dict(),
        "decoder": student.decoder.state_dict(),
        "meta": dict(meta),
    }
    if best_info is not None:
        payload["best_info"] = dict(best_info)
    torch.save(payload, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--class_name", type=str, default="texture")
    parser.add_argument("--mvtec_root", type=str, default="./mvtec")
    parser.add_argument("--domains_dir", type=str, default="./outputs/domains")
    parser.add_argument("--pseudo_domain_json", type=str, default="", help="直接读取 pseudo-domain JSON manifest 训练/评估；为空时保留旧 txt 逻辑")
    parser.add_argument("--cluster_dir", type=str, default="")
    parser.add_argument("--pseudo_script", type=str, default="pseudo-domain_discover_json.py")
    parser.add_argument("--cluster_infer_method", type=str, default="hybrid", choices=["hybrid", "hdbscan", "centers"])
    parser.add_argument("--cluster_router_device", type=str, default="cpu", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--cluster_hdbscan_min_conf", type=float, default=0.0)
    parser.add_argument("--cluster_router_cache_dir", type=str, default="")
    parser.add_argument("--cluster_router_cache_flush_interval", type=int, default=16)
    parser.add_argument("--no_cluster_router_cache", action="store_true")
    parser.add_argument("--ckpt_dir", type=str, default="./checkpoints")

    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)

    # merged-domain 评估专用
    parser.add_argument("--eval_num_workers", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=8)

    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--kd_weight", type=float, default=1.3)
    parser.add_argument("--reliability_tau", type=float, default=8.0)
    parser.add_argument("--reliability_floor", type=float, default=0.5)

    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--print_per_domain", action="store_true")

    parser.add_argument("--seed", type=int, default=111)

    # 默认 deterministic=True，只有显式传 --non_deterministic 才关闭
    parser.add_argument("--deterministic", dest="deterministic", action="store_true")
    parser.add_argument("--non_deterministic", dest="deterministic", action="store_false")
    parser.set_defaults(deterministic=True)

    # 默认 amp=False
    parser.add_argument("--amp", action="store_true")

    # teacher ckpt path
    parser.add_argument("--teacher_ckpt", type=str, default="")
    args = parser.parse_args()

    os.makedirs(args.ckpt_dir, exist_ok=True)
    default_teacher = os.path.join(args.ckpt_dir, f"teacher_cond_{args.class_name}_best.pth")
    teacher_ckpt = args.teacher_ckpt if args.teacher_ckpt else default_teacher
    student_ckpt = os.path.join(args.ckpt_dir, f"student_distill_{args.class_name}.pth")

    setup_seed(args.seed, deterministic=args.deterministic)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    print("Deterministic:", args.deterministic)
    print("AMP:", args.amp)
    print("Reliability tau:", args.reliability_tau)
    print("Reliability floor:", args.reliability_floor)
    print("Train num_workers:", args.num_workers)
    print("Eval  num_workers:", args.eval_num_workers)
    print("Eval  batch_size :", args.eval_batch_size)
    print("Pseudo-domain source:", "json" if args.pseudo_domain_json else "txt")
    if args.pseudo_domain_json:
        print("Pseudo-domain JSON:", args.pseudo_domain_json)
        print("Cluster router method:", args.cluster_infer_method)
        print("Cluster router device:", args.cluster_router_device)
        print("Cluster router cache:", not args.no_cluster_router_cache)
        if args.cluster_router_cache_dir:
            print("Cluster router cache dir:", args.cluster_router_cache_dir)
    else:
        print("Domains dir:", args.domains_dir)
    args.domain_source = "json" if args.pseudo_domain_json else "txt"

    data_transform, gt_transform = get_data_transforms(args.image_size, args.image_size)
    root_path = os.path.join(args.mvtec_root, args.class_name)

    # train dataset: (img, domain_id)
    train_data = MVTecTrainPseudoDomainDataset(
        root=root_path,
        transform=data_transform,
        domain_dir=args.domains_dir,
        pseudo_domain_json=args.pseudo_domain_json,
        strict=True
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

    # merged-domain test loader
    domain_name_to_id = get_domain_name_to_id(train_data, args.domains_dir, args.pseudo_domain_json)

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

    # build encoder (frozen)
    encoder, _bn_tmp = wide_resnet50_2(pretrained=True)
    encoder = encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    # load teacher ckpt
    if not os.path.isfile(teacher_ckpt):
        raise FileNotFoundError(f"Teacher ckpt not found: {teacher_ckpt}")
    ckpt = torch.load(teacher_ckpt, map_location="cpu")
    meta = ckpt.get("meta", {})

    num_domains = int(meta.get("num_domains", getattr(train_data, "num_domains")))
    z_channels = int(meta.get("z_channels", 2048))
    film_emb_dim = int(meta.get("film_emb_dim", 256))
    film_hidden = int(meta.get("film_hidden", 1024))
    film_dropout = float(meta.get("film_dropout", 0.0))

    # 一致性检查：teacher ckpt 与当前数据集尽量对齐
    train_num_domains = int(getattr(train_data, "num_domains"))
    if num_domains != train_num_domains:
        raise ValueError(
            f"Teacher ckpt num_domains ({num_domains}) != current train_data.num_domains ({train_num_domains})"
        )

    meta_domain_name_to_id = meta.get("domain_name_to_id", None)
    if isinstance(meta_domain_name_to_id, dict):
        meta_domain_name_to_id = {str(k): int(v) for k, v in meta_domain_name_to_id.items()}
        current_domain_name_to_id = {str(k): int(v) for k, v in domain_name_to_id.items()}
        if meta_domain_name_to_id != current_domain_name_to_id:
            raise ValueError(
                "Teacher ckpt domain_name_to_id does not match current dataset/domain mapping."
            )

    # rebuild teacher modules & load
    _, bn_t = wide_resnet50_2(pretrained=True)
    bn_t = bn_t.to(device)
    decoder_t = de_wide_resnet50_2(pretrained=False).to(device)
    film = DomainFiLM(
        num_domains=num_domains,
        channels=z_channels,
        emb_dim=film_emb_dim,
        hidden=film_hidden,
        dropout=film_dropout,
        init_identity=False
    ).to(device)

    teacher = ConditionalTeacher(bn=bn_t, decoder=decoder_t, film=film).to(device)
    teacher.bn.load_state_dict(ckpt["bn"], strict=True)
    teacher.decoder.load_state_dict(ckpt["decoder"], strict=True)
    teacher.film.load_state_dict(ckpt["film"], strict=True)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    id_to_domain_name = domain_id_to_name_map(domain_name_to_id)
    cluster_router = build_cluster_router(args, meta, domain_name_to_id)

    # build student & init from teacher bn/decoder
    _, bn_s = wide_resnet50_2(pretrained=True)
    bn_s = bn_s.to(device)
    decoder_s = de_wide_resnet50_2(pretrained=False).to(device)

    bn_s.load_state_dict(teacher.bn.state_dict(), strict=True)
    decoder_s.load_state_dict(teacher.decoder.state_dict(), strict=True)

    student = StudentNoDomain(bn=bn_s, decoder=decoder_s).to(device)

    opt = torch.optim.Adam(
        list(student.bn.parameters()) + list(student.decoder.parameters()),
        lr=args.lr,
        betas=(0.5, 0.999)
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.startswith("cuda")))

    student_meta = build_student_ckpt_meta(
        args=args,
        teacher_ckpt=teacher_ckpt,
        teacher_meta=meta,
        train_data=train_data,
        domain_name_to_id=domain_name_to_id,
    )

    best_sp = -1.0
    best_state = None

    for ep in range(1, args.epochs + 1):
        student.train()
        rd_list, kd_raw_list, kd_w_list, r_list, tea_list = [], [], [], [], []

        for img, domain_id in train_loader:
            img = img.to(device, non_blocking=True)
            domain_id = domain_id.to(device, non_blocking=True).long()

            with torch.no_grad():
                feats = encoder(img)
                t_outs, _ = teacher(feats, domain_id)
                teacher_sample_losses = feature_l2_sample_losses(feats, t_outs)
                reliability = reliability_from_teacher_loss(
                    teacher_sample_losses=teacher_sample_losses,
                    tau=args.reliability_tau,
                    floor=args.reliability_floor,
                )

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(args.amp and device.startswith("cuda"))):
                s_outs, _ = student(feats)
                loss_rd = rd4ad_cosine_loss(feats, s_outs)

                # Paper-faithful sample reliability-aware distillation:
                #   L_KD = mean_i r_i * L_dist(x_i)
                # The per-sample L_dist(x_i) is computed with the original
                # distill_feats_loss to avoid changing the KD metric itself.
                kd_sample_losses = distill_feats_sample_losses(t_outs, s_outs)
                loss_kd_raw = kd_sample_losses.mean()
                loss_kd = (reliability.to(kd_sample_losses.device) * kd_sample_losses).mean()
                loss = loss_rd + args.kd_weight * loss_kd

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            rd_list.append(float(loss_rd.item()))
            kd_raw_list.append(float(loss_kd_raw.item()))
            kd_w_list.append(float(loss_kd.item()))
            r_list.append(float(reliability.mean().item()))
            tea_list.append(float(teacher_sample_losses.mean().item()))

        print(
            f"[Student] epoch [{ep:03d}/{args.epochs}] "
            f"loss_rd={np.mean(rd_list):.4f} "
            f"loss_kd_raw={np.mean(kd_raw_list):.4f} "
            f"loss_kd_w={np.mean(kd_w_list):.4f} "
            f"r_mean={np.mean(r_list):.4f} "
            f"teacher_loss={np.mean(tea_list):.4f} "
            f"kd_w={args.kd_weight:.3f} "
            f"eff_kd_w={args.kd_weight * np.mean(r_list):.3f}"
        )

        # 每 eval_every 轮：输出 merged domains 宏平均
        if (ep % args.eval_every == 0) or (ep == args.epochs):
            student.bn.eval()
            student.decoder.eval()

            mean_px, mean_sp, mean_ap, n_valid, n_total = macro_eval_student_over_merged_loader(
                encoder=encoder,
                student=student,
                domain_loader=merged_domain_test_loader,
                device=device,
                print_per_domain=args.print_per_domain,
                cluster_router=cluster_router,
                id_to_domain_name=id_to_domain_name,
            )
            print(
                f"[Student Eval@{ep}] Domain-MacroAvg over {n_valid}/{n_total} txts: "
                f"PixelAUROC={mean_px:.3f}, SampleAUROC={mean_sp:.3f}, PixelAUPRO={mean_ap:.3f}"
            )

            save_student_ckpt(
                path=student_ckpt,
                student=student,
                meta=student_meta,
                best_info={
                    "latest_eval_epoch": ep,
                    "latest_mean_px": mean_px,
                    "latest_mean_sp": mean_sp,
                    "latest_mean_ap": mean_ap,
                }
            )

            if mean_sp == mean_sp and mean_sp > best_sp:
                best_sp = mean_sp
                best_state = {
                    "bn": {k: v.detach().cpu() for k, v in student.bn.state_dict().items()},
                    "decoder": {k: v.detach().cpu() for k, v in student.decoder.state_dict().items()},
                    "epoch": ep,
                    "mean_px": mean_px,
                    "mean_sp": mean_sp,
                    "mean_ap": mean_ap,
                }

            student.bn.train()
            student.decoder.train()

        if (ep % args.save_every == 0) or (ep == args.epochs):
            save_student_ckpt(
                path=student_ckpt,
                student=student,
                meta=student_meta,
                best_info=None if best_state is None else {
                    "best_epoch": best_state["epoch"],
                    "best_mean_px": best_state["mean_px"],
                    "best_mean_sp": best_state["mean_sp"],
                    "best_mean_ap": best_state["mean_ap"],
                }
            )
            print("Saved:", student_ckpt)

    if best_state is not None:
        best_path = student_ckpt.replace(".pth", "_best.pth")
        torch.save(
            {
                "bn": best_state["bn"],
                "decoder": best_state["decoder"],
                "meta": student_meta,
                "best_info": {
                    "epoch": best_state["epoch"],
                    "mean_px": best_state["mean_px"],
                    "mean_sp": best_state["mean_sp"],
                    "mean_ap": best_state["mean_ap"],
                }
            },
            best_path
        )
        print(
            f"[Best] epoch={best_state['epoch']} "
            f"PixelAUROC={best_state['mean_px']:.3f}, "
            f"SampleAUROC={best_state['mean_sp']:.3f}, "
            f"PixelAUPRO={best_state['mean_ap']:.3f} "
            f"-> saved: {best_path}"
        )

    print("Done. Student ckpt:", student_ckpt)


if __name__ == "__main__":
    main()