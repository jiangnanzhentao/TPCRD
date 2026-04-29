#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_TPCRD.py

TPCRD global pseudo-domain training entry.

This file implements the full intended workflow without modifying the public
MVTec AD directory layout and without creating/consuming additional txt domain
files for training:

  1) collect train/good images from one or more MVTec classes;
  2) extract features for all collected train/good samples;
  3) run ONE global PCA + HDBSCAN clustering over all train/good samples;
  4) write one global JSON manifest containing train pseudo-domain labels and raw test paths;
  5) train a cluster router inside train_teacher.py using only train/good pseudo-domain labels;
  6) train the conditional teacher once from that JSON;
  7) train the student once from the same JSON and teacher checkpoint.

Test samples are NOT assigned pseudo-domain labels during JSON generation.
During teacher/student evaluation, test domain ids are predicted online by the
cluster router trained only on train/good pseudo labels.

Important:
  - No images are copied or moved.
  - No txt pseudo-domain category files are required.
  - train_teacher.py / train_student.py are invoked as-is; their internal
    training losses, evaluation and checkpoint logic are not reimplemented here.
  - pseudo-domain_discover_json.py is used only as a feature/clustering library.

Examples:
  python train_TPCRD.py --mvtec_root /path/to/mvtec --class_name all --out_dir ./outputs/tpcrd_all

  python train_TPCRD.py --mvtec_root /path/to/mvtec --class_names bottle cable capsule \
      --experiment_name selected3 --out_dir ./outputs/tpcrd_selected3

  python train_TPCRD.py --mvtec_root /path/to/mvtec --class_name all \
      --skip_json --pseudo_domain_json ./outputs/tpcrd_all/json/training_manifest.json
"""

import argparse
import csv
import importlib.util
import json
import os
import shlex
import shutil
import subprocess
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

try:
    from joblib import dump
except Exception:  # pragma: no cover
    dump = None

try:
    from sklearn.metrics import silhouette_score
except Exception:  # pragma: no cover
    silhouette_score = None


MVTEC_AD_CLASSES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
]

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def canon_path(p) -> str:
    return os.path.abspath(os.path.normpath(str(p).strip().replace("\\", os.sep).replace("/", os.sep)))


def as_path(value: Optional[str]) -> Optional[Path]:
    if value is None or str(value).strip() == "":
        return None
    return Path(value).expanduser()


def resolve_script(script_name_or_path: str, anchor: Path) -> Path:
    raw = Path(script_name_or_path).expanduser()
    candidates: List[Path]
    if raw.is_absolute():
        candidates = [raw]
    else:
        candidates = [anchor / raw, Path.cwd() / raw, raw]

    for p in candidates:
        p = p.resolve()
        if p.is_file():
            return p

    raise FileNotFoundError(
        f"Script not found: {script_name_or_path}\nTried:\n  - "
        + "\n  - ".join(str(p.resolve()) for p in candidates)
    )


def run_cmd(cmd: List[str], cwd: Path, dry_run: bool = False) -> None:
    printable = " ".join(shlex.quote(str(x)) for x in cmd)
    print("\n" + "=" * 110)
    print(f"[TPCRD] RUN: {printable}")
    print(f"[TPCRD] CWD: {cwd}")
    print("=" * 110, flush=True)

    if dry_run:
        return

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:64,garbage_collection_threshold:0.8")
    subprocess.run([str(x) for x in cmd], cwd=str(cwd), env=env, check=True)


def split_extra(extra: str) -> List[str]:
    return shlex.split(extra) if extra else []


def add_flag(cmd: List[str], flag: str, value) -> None:
    if value is not None and str(value) != "":
        cmd.extend([flag, str(value)])


def add_bool(cmd: List[str], flag: str, enabled: bool) -> None:
    if enabled:
        cmd.append(flag)


def load_pseudo_module(pseudo_script: Path):
    spec = importlib.util.spec_from_file_location("tpcrd_pseudo_domain_discover_json", str(pseudo_script))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import pseudo module from: {pseudo_script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def detect_mvtec_classes(mvtec_root: Path) -> List[str]:
    if not mvtec_root.is_dir():
        raise FileNotFoundError(f"mvtec_root not found: {mvtec_root}")

    detected = []
    for child in sorted(mvtec_root.iterdir()):
        if not child.is_dir():
            continue
        if (child / "train" / "good").is_dir() and (child / "test").is_dir():
            detected.append(child.name)

    canonical = [c for c in MVTEC_AD_CLASSES if c in detected]
    custom = [c for c in detected if c not in canonical]
    return canonical + custom


def parse_classes(args, mvtec_root: Path) -> Tuple[List[str], str]:
    if args.class_names:
        raw = list(args.class_names)
    else:
        raw = [x.strip() for x in str(args.class_name or "all").split(",") if x.strip()]

    if len(raw) == 0:
        raw = ["all"]

    if len(raw) == 1 and raw[0].lower() in {"all", "*"}:
        classes = detect_mvtec_classes(mvtec_root)
        if not classes:
            raise FileNotFoundError(f"No MVTec-style class folders found under: {mvtec_root}")
        default_exp = "all"
    else:
        classes = []
        for name in raw:
            class_root = mvtec_root / name
            if not (class_root / "train" / "good").is_dir():
                raise FileNotFoundError(f"Missing train/good for class '{name}': {class_root / 'train' / 'good'}")
            if not (class_root / "test").is_dir():
                raise FileNotFoundError(f"Missing test folder for class '{name}': {class_root / 'test'}")
            classes.append(name)
        default_exp = classes[0] if len(classes) == 1 else "multi"

    experiment_name = args.experiment_name.strip() if args.experiment_name else default_exp
    return classes, experiment_name


def list_images(folder: Path) -> List[str]:
    paths: List[str] = []
    if not folder.is_dir():
        return paths
    for ext in IMAGE_EXTS:
        paths.extend(str(p) for p in folder.rglob(f"*{ext}"))
    return sorted(set(canon_path(p) for p in paths))


def path_rel(path: str, base: Path) -> str:
    try:
        return os.path.relpath(canon_path(path), canon_path(base)).replace(os.sep, "/")
    except Exception:
        return str(path).replace("\\", "/")


def make_base_record(path: str, mvtec_root: Path, class_name: str, split: str) -> dict:
    return {
        "path_abs": canon_path(path),
        "path": canon_path(path),
        "path_rel_to_mvtec_root": path_rel(path, mvtec_root),
        "source_class": str(class_name),
        "class_name": str(class_name),
        "split": split.replace("\\", "/"),
    }


def collect_train_records(mvtec_root: Path, classes: Sequence[str]) -> List[dict]:
    records: List[dict] = []
    for cls in classes:
        folder = mvtec_root / cls / "train" / "good"
        for p in list_images(folder):
            records.append(make_base_record(p, mvtec_root, cls, "train/good"))
    if len(records) == 0:
        raise RuntimeError(f"No train/good images found for classes={list(classes)} under mvtec_root={mvtec_root}")
    return records


def collect_test_records(mvtec_root: Path, classes: Sequence[str]) -> List[dict]:
    records: List[dict] = []
    for cls in classes:
        test_root = mvtec_root / cls / "test"
        if not test_root.is_dir():
            continue
        for defect_dir in sorted([p for p in test_root.iterdir() if p.is_dir()]):
            split = f"test/{defect_dir.name}"
            for p in list_images(defect_dir):
                rec = make_base_record(p, mvtec_root, cls, split)
                rec["img_type"] = defect_dir.name
                rec["is_anomaly"] = int(defect_dir.name != "good")
                records.append(rec)
    return records


def safe_membership_vectors(pseudo, clusterer, y_ref: np.ndarray, n_samples: int) -> np.ndarray:
    valid_ids = sorted([int(c) for c in set(y_ref.tolist()) if int(c) != -1])

    try:
        M = pseudo.hdbscan.all_points_membership_vectors(clusterer)
        M = np.asarray(M, dtype=np.float32)
    except Exception:
        M = np.zeros((0, 0), dtype=np.float32)

    if M.ndim != 2 or M.shape[0] != n_samples:
        M = np.zeros((n_samples, len(valid_ids)), dtype=np.float32)
        id2col = {cid: i for i, cid in enumerate(valid_ids)}
        for i, yy in enumerate(y_ref):
            yy = int(yy)
            if yy != -1 and yy in id2col:
                M[i, id2col[yy]] = 1.0

    return M


def soft_join_safe(
    pseudo,
    Z: np.ndarray,
    y: np.ndarray,
    clusterer,
    tau_high: float = 0.6,
    tau_low: float = 0.3,
    nn_k: int = 10,
    nn_ratio: float = 0.7,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    if Z.shape[0] == 0:
        return y, np.zeros((0, Z.shape[1]), dtype=np.float32), []

    y_ref = y.copy()
    valid_ids = sorted([int(c) for c in set(y_ref.tolist()) if int(c) != -1])
    if len(valid_ids) == 0:
        return y_ref, np.zeros((0, Z.shape[1]), dtype=np.float32), []

    M = safe_membership_vectors(pseudo, clusterer, y_ref, Z.shape[0])
    max_prob = M.max(axis=1) if (M.ndim == 2 and M.shape[1] > 0) else np.zeros(Z.shape[0], dtype=np.float32)

    is_conf = (y_ref != -1) & (max_prob >= tau_high)
    is_amb = (y_ref == -1) & (max_prob >= tau_low)

    if is_amb.any() and is_conf.any():
        from sklearn.neighbors import NearestNeighbors

        nn = NearestNeighbors(n_neighbors=min(nn_k, int(is_conf.sum())), metric="euclidean").fit(Z[is_conf])
        _, idx = nn.kneighbors(Z[is_amb], return_distance=True)
        conf_labels = y_ref[is_conf]
        amb_idx = np.where(is_amb)[0]
        for k, nn_indices in enumerate(idx):
            labs = conf_labels[nn_indices]
            vals, cnts = np.unique(labs, return_counts=True)
            j = int(cnts.argmax())
            maj, ratio = vals[j], cnts[j] / len(labs)
            if ratio >= nn_ratio:
                y_ref[amb_idx[k]] = maj

    # Center update. If a cluster has no confident weight, use median center.
    valid_ids = sorted([int(c) for c in set(y_ref.tolist()) if int(c) != -1])
    w = np.zeros(Z.shape[0], dtype=np.float32)
    w[is_conf] = 1.0
    if is_amb.any() and max_prob.size:
        w_amb = (np.clip(max_prob[is_amb], tau_low, tau_high) - tau_low) / max(1e-6, (tau_high - tau_low))
        w[is_amb] = w_amb.astype(np.float32)

    centers = []
    for cid in valid_ids:
        m = y_ref == cid
        ZZ = Z[m]
        ww = w[m]
        if ZZ.shape[0] == 0:
            centers.append(np.zeros((Z.shape[1],), dtype=np.float32))
        elif float(ww.sum()) > 1e-6:
            centers.append(((ZZ * ww[:, None]).sum(axis=0) / (ww.sum() + 1e-6)).astype(np.float32))
        else:
            centers.append(np.median(ZZ, axis=0).astype(np.float32))

    centers_arr = np.stack(centers, axis=0) if centers else np.zeros((0, Z.shape[1]), dtype=np.float32)
    return y_ref, centers_arr, valid_ids


def extract_feature_matrix(pseudo, records: List[dict], args, device: str, desc: str):
    vgg = extract_feature_matrix.vgg
    X, purity, kept_records = [], [], []
    for rec in tqdm(records, desc=desc):
        p = rec["path_abs"]
        try:
            x, pu = pseudo.build_image_feature(
                p,
                vgg,
                device,
                args.pseudo_patches,
                args.pseudo_patch,
                list(args.pseudo_scales),
                use_retinex=(not args.pseudo_no_retinex),
                rotations=args.pseudo_rotations,
                use_mask=(not args.pseudo_no_mask_padding),
            )
            X.append(x)
            purity.append(float(pu))
            kept_records.append(rec)
        except Exception as e:
            print(f"[TPCRD][WARN] feature extraction failed: {p} | {e}")

    if len(X) == 0:
        raise RuntimeError(f"No features were extracted for {desc}.")

    return np.stack(X, axis=0).astype(np.float32), np.asarray(purity, dtype=np.float32), kept_records


extract_feature_matrix.vgg = None  # type: ignore[attr-defined]


def normalize_nearest_confidence(raw_conf: np.ndarray) -> np.ndarray:
    raw_conf = np.asarray(raw_conf, dtype=np.float32)
    if raw_conf.size == 0:
        return raw_conf
    finite = np.isfinite(raw_conf)
    if not finite.any():
        return np.ones_like(raw_conf, dtype=np.float32)
    vals = raw_conf[finite]
    lo, hi = float(vals.min()), float(vals.max())
    if abs(hi - lo) < 1e-12:
        out = np.ones_like(raw_conf, dtype=np.float32)
    else:
        out = (raw_conf - lo) / (hi - lo)
    out[~finite] = 0.0
    return out.astype(np.float32)


def compute_cluster_stats(samples: List[dict]) -> Dict[str, dict]:
    stats: Dict[str, dict] = {}
    for rec in samples:
        cid = int(rec.get("cluster_id", rec.get("label", -1)))
        if cid == -1:
            continue
        key = str(cid)
        if key not in stats:
            stats[key] = {
                "cluster_id": cid,
                "domain_name": f"domain_{cid}",
                "count_total": 0,
                "count_by_split": {},
                "count_by_class": {},
            }
        split = str(rec.get("split", "unknown"))
        cls = str(rec.get("source_class", rec.get("class_name", "unknown")))
        stats[key]["count_total"] += 1
        stats[key]["count_by_split"][split] = stats[key]["count_by_split"].get(split, 0) + 1
        stats[key]["count_by_class"][cls] = stats[key]["count_by_class"].get(cls, 0) + 1
    return dict(sorted(stats.items(), key=lambda kv: int(kv[0])))


def write_json_outputs(
    json_dir: Path,
    train_samples: List[dict],
    test_samples: List[dict],
    mvtec_root: Path,
    classes: Sequence[str],
    experiment_name: str,
    cluster_summary: dict,
) -> Path:
    json_dir.mkdir(parents=True, exist_ok=True)
    all_samples = train_samples + test_samples
    # Cluster/domain statistics are computed from train pseudo-labels only.
    # Test records intentionally do not carry cluster_id/domain_id; evaluation routes
    # them online through the cluster router to avoid test pseudo-label leakage.
    cluster_stats = compute_cluster_stats(train_samples)

    counts_by_split: Dict[str, int] = {}
    counts_by_class: Dict[str, int] = {}
    for rec in all_samples:
        split = str(rec.get("split", "unknown"))
        cls = str(rec.get("source_class", rec.get("class_name", "unknown")))
        counts_by_split[split] = counts_by_split.get(split, 0) + 1
        counts_by_class[cls] = counts_by_class.get(cls, 0) + 1

    manifest = {
        "schema_version": "tpcrd_training_manifest/v4_global_train_cluster_cluster_router",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "experiment_name": experiment_name,
        "class_name": experiment_name,
        "class_names": list(classes),
        "data_root": str(mvtec_root),
        "mvtec_root": str(mvtec_root),
        "cluster_scope": "global_all_selected_train_good_only",
        "test_domain_assignment": "cluster_model_online_in_teacher_student_eval",
        "cluster_summary": cluster_summary,
        "paths": {
            "cluster_dir": str(json_dir.parent / "cluster"),
            "scaler": str(json_dir.parent / "cluster" / "scaler.joblib"),
            "pca": str(json_dir.parent / "cluster" / "pca.joblib"),
            "hdbscan": str(json_dir.parent / "cluster" / "hdbscan.joblib"),
            "centers": str(json_dir.parent / "cluster" / "centers.npy"),
            "valid_ids": str(json_dir.parent / "cluster" / "valid_ids.npy"),
        },
        "counts": {
            "samples_total": len(all_samples),
            "train_good": len(train_samples),
            "test": len(test_samples),
            "clusters": len(cluster_stats),
            "classes": len(classes),
            "by_split": counts_by_split,
            "by_class": counts_by_class,
        },
        "cluster_stats": cluster_stats,
        "samples": all_samples,
    }

    manifest_path = json_dir / "training_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    with open(json_dir / "train_good_clustered.json", "w", encoding="utf-8") as f:
        json.dump(train_samples, f, ensure_ascii=False, indent=2)
    with open(json_dir / "test_all_unrouted.json", "w", encoding="utf-8") as f:
        json.dump(test_samples, f, ensure_ascii=False, indent=2)
    with open(json_dir / "test_good_unrouted.json", "w", encoding="utf-8") as f:
        json.dump([x for x in test_samples if x.get("split") == "test/good"], f, ensure_ascii=False, indent=2)
    with open(json_dir / "test_bad_unrouted.json", "w", encoding="utf-8") as f:
        json.dump([x for x in test_samples if x.get("split") != "test/good"], f, ensure_ascii=False, indent=2)

    groups: Dict[str, dict] = {}
    for rec in train_samples:
        cid = int(rec.get("cluster_id", -1))
        if cid == -1:
            continue
        key = str(cid)
        if key not in groups:
            groups[key] = {"cluster_id": cid, "domain_name": f"domain_{cid}", "samples": []}
        groups[key]["samples"].append(rec)
    groups = dict(sorted(groups.items(), key=lambda kv: int(kv[0])))
    with open(json_dir / "domain_groups.json", "w", encoding="utf-8") as f:
        json.dump(groups, f, ensure_ascii=False, indent=2)

    index = {
        "training_manifest": str(manifest_path),
        "train_good_clustered": str(json_dir / "train_good_clustered.json"),
        "test_all_unrouted": str(json_dir / "test_all_unrouted.json"),
        "test_good_unrouted": str(json_dir / "test_good_unrouted.json"),
        "test_bad_unrouted": str(json_dir / "test_bad_unrouted.json"),
        "domain_groups": str(json_dir / "domain_groups.json"),
    }
    with open(json_dir / "json_index.json", "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    return manifest_path


def write_csv(path: Path, rows: List[dict], columns: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})


def run_global_pseudo_domain_discovery(args, pseudo_script: Path, mvtec_root: Path, classes: Sequence[str], out_dir: Path, experiment_name: str) -> Path:
    pseudo = load_pseudo_module(pseudo_script)
    pseudo.set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[TPCRD][GLOBAL-CLUSTER] Device: {device}")
    print(f"[TPCRD][GLOBAL-CLUSTER] Classes ({len(classes)}): {list(classes)}")

    cluster_dir = out_dir / "cluster"
    json_dir = out_dir / "json"
    cluster_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    train_records = collect_train_records(mvtec_root, classes)
    test_records = collect_test_records(mvtec_root, classes)
    print(f"[TPCRD][GLOBAL-CLUSTER] train/good samples: {len(train_records)}")
    print(f"[TPCRD][GLOBAL-CLUSTER] test samples      : {len(test_records)}")

    if len(train_records) < 2:
        raise RuntimeError("At least two train/good images are required for PCA/HDBSCAN clustering.")

    extract_feature_matrix.vgg = pseudo.VGGFeat().eval().to(device)  # type: ignore[attr-defined]

    X_train, purity_train, train_records = extract_feature_matrix(
        pseudo, train_records, args, device, desc="[TPCRD][GLOBAL-CLUSTER] extracting train/good"
    )

    print("[TPCRD][GLOBAL-CLUSTER] Fitting PCA on all selected train/good samples ...")
    ss, pca, Z_train = pseudo.fit_pca(X_train, out_dim=args.pseudo_out_dim)

    print("[TPCRD][GLOBAL-CLUSTER] Running one global HDBSCAN over all selected train/good samples ...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        clusterer, y_base, prob_base = pseudo.run_hdbscan(
            Z_train,
            min_cluster_size=args.pseudo_min_cluster_size,
            min_samples=args.pseudo_min_samples,
            metric=args.pseudo_metric,
        )

    y_train = y_base.copy()
    prob_final = np.asarray(prob_base, dtype=np.float32).copy()
    prob_patch_noise = np.full(y_train.shape[0], np.nan, dtype=np.float32)

    if not args.pseudo_no_noise_recluster:
        print("[TPCRD][GLOBAL-CLUSTER] Re-cluster noise subset ...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            y_train, prob_patch_noise = pseudo.recluster_noise(
                Z_train, y_train, args.pseudo_noise_min_cluster, args.pseudo_noise_min_samples
            )

    centers_arr, valid_ids = pseudo.compute_centers(Z_train, y_train)
    if not args.pseudo_no_refine and len(valid_ids) > 0:
        print("[TPCRD][GLOBAL-CLUSTER] Soft-join remaining ambiguous samples ...")
        y_train, centers_arr, valid_ids = soft_join_safe(
            pseudo,
            Z_train,
            y_train,
            clusterer,
            tau_high=args.pseudo_tau_high,
            tau_low=args.pseudo_tau_low,
            nn_k=args.pseudo_nn_k,
            nn_ratio=args.pseudo_nn_ratio,
        )

    # If HDBSCAN found nothing, keep the pipeline usable by creating one global pseudo-domain.
    if len([c for c in set(y_train.tolist()) if int(c) != -1]) == 0:
        print("[TPCRD][WARN] HDBSCAN produced no valid clusters. Falling back to one global domain_0.")
        y_train = np.zeros(Z_train.shape[0], dtype=np.int32)
        prob_final = np.ones(Z_train.shape[0], dtype=np.float32)
        centers_arr = np.median(Z_train, axis=0, keepdims=True).astype(np.float32)
        valid_ids = [0]
    else:
        centers_arr, valid_ids = pseudo.compute_centers(Z_train, y_train)

    # Optionally assign remaining train noise to nearest valid center so teacher/student receive all train/good samples.
    if not args.pseudo_keep_train_noise:
        noise_idx = np.where(y_train == -1)[0]
        if len(noise_idx) > 0:
            mapped, conf = pseudo.assign_to_centers(Z_train[noise_idx], centers_arr, valid_ids)
            y_train[noise_idx] = mapped
            prob_final[noise_idx] = normalize_nearest_confidence(conf)
            print(f"[TPCRD][GLOBAL-CLUSTER] Assigned {len(noise_idx)} remaining train noise samples to nearest centers.")

    # Apply noise probabilities where available.
    mask_noise_patch = ~np.isnan(prob_patch_noise)
    if mask_noise_patch.shape == prob_final.shape:
        prob_final[mask_noise_patch] = prob_patch_noise[mask_noise_patch]

    # Recompute centers after final train-noise assignment.
    centers_arr, valid_ids = pseudo.compute_centers(Z_train, y_train)
    if len(valid_ids) == 0:
        raise RuntimeError("No valid pseudo-domain center remains after clustering.")

    sil = -1.0
    mask_valid = y_train != -1
    if silhouette_score is not None and int(mask_valid.sum()) >= 3 and len(set(y_train[mask_valid].tolist())) >= 2:
        try:
            sil = float(silhouette_score(Z_train[mask_valid], y_train[mask_valid], metric="euclidean"))
        except Exception:
            sil = -1.0
    print(f"[TPCRD][GLOBAL-CLUSTER] Clusters: {len(valid_ids)} | Silhouette excl. noise: {sil:.4f}")

    if dump is not None:
        dump(ss, cluster_dir / "scaler.joblib")
        dump(pca, cluster_dir / "pca.joblib")
        dump(clusterer, cluster_dir / "hdbscan.joblib")
    np.save(cluster_dir / "centers.npy", centers_arr)
    np.save(cluster_dir / "valid_ids.npy", np.asarray(valid_ids, dtype=np.int32))

    train_samples: List[dict] = []
    for rec, cid, pr, pu in zip(train_records, y_train, prob_final, purity_train):
        cid = int(cid)
        if cid == -1:
            continue
        out = dict(rec)
        out.update({
            "cluster_id": cid,
            "label": cid,
            "domain_name": f"domain_{cid}",
            "probability": float(pr) if np.isfinite(pr) else 0.0,
            "purity": float(pu),
            "assignment_source": "global_train_cluster",
        })
        train_samples.append(out)

    # Test records are intentionally kept unrouted here.
    # Do NOT extract test features or assign nearest pseudo-domain centers during JSON generation.
    # Teacher/student evaluation predicts test domain ids online with the cluster router trained only
    # from train/good pseudo labels.
    test_samples: List[dict] = []
    for rec in test_records:
        out = dict(rec)
        out.update({
            "domain_name": "__unrouted__",
            "domain_id": -1,
            "assignment_source": "unrouted_test_router_required",
        })
        # Remove any accidental precomputed pseudo-domain keys from old manifests/runs.
        for k in ("cluster_id", "label", "pred_cluster", "cluster", "confidence", "probability", "purity"):
            out.pop(k, None)
        test_samples.append(out)

    print(f"[TPCRD][GLOBAL-CLUSTER] Test samples are left unrouted: {len(test_samples)}. Cluster router will assign domains online during evaluation.")

    # CSV side outputs are diagnostic only; downstream training uses JSON.
    write_csv(
        cluster_dir / "train_good.csv",
        train_samples,
        ["path_abs", "source_class", "split", "cluster_id", "probability", "purity", "assignment_source"],
    )
    write_csv(
        out_dir / "test_all_unrouted.csv",
        test_samples,
        ["path_abs", "source_class", "split", "img_type", "is_anomaly", "domain_name", "domain_id", "assignment_source"],
    )

    cluster_summary = {
        "n_train_good_input": len(train_records),
        "n_train_good_labeled": len(train_samples),
        "n_test_input": len(test_records),
        "n_test_unrouted": len(test_samples),
        "n_clusters_found": len(valid_ids),
        "valid_cluster_ids": [int(x) for x in valid_ids],
        "silhouette_train_excl_noise": sil,
        "feature_extractor": "VGG19 + traditional texture features from pseudo-domain_discover_json.py",
        "pca_components": int(getattr(pca, "n_components_", args.pseudo_out_dim)),
        "min_cluster_size": int(args.pseudo_min_cluster_size),
        "min_samples": int(args.pseudo_min_samples),
        "metric": str(args.pseudo_metric),
        "noise_recluster": not bool(args.pseudo_no_noise_recluster),
        "refine": not bool(args.pseudo_no_refine),
        "remaining_train_noise_kept": bool(args.pseudo_keep_train_noise),
    }

    with open(cluster_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(cluster_summary, f, ensure_ascii=False, indent=2)

    meta = {
        "schema_version": "tpcrd_global_train_cluster_meta/v3_cluster_router",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "experiment_name": experiment_name,
        "class_names": list(classes),
        "mvtec_root": str(mvtec_root),
        "patches": args.pseudo_patches,
        "patch": args.pseudo_patch,
        "scales": list(args.pseudo_scales),
        "no_retinex": bool(args.pseudo_no_retinex),
        "rotations": int(args.pseudo_rotations),
        "mask_padding": not bool(args.pseudo_no_mask_padding),
        "out_dim_effective": int(getattr(pca, "n_components_", args.pseudo_out_dim)),
    }
    with open(cluster_dir / "model_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    manifest_path = write_json_outputs(
        json_dir=json_dir,
        train_samples=train_samples,
        test_samples=test_samples,
        mvtec_root=mvtec_root,
        classes=classes,
        experiment_name=experiment_name,
        cluster_summary=cluster_summary,
    )

    # The old txt domain path is intentionally not created. Remove stale legacy dir if present.
    legacy_domains = out_dir / "domains"
    if legacy_domains.is_dir() and not args.keep_generated_domain_txt:
        shutil.rmtree(legacy_domains)

    print(f"[TPCRD][GLOBAL-CLUSTER] JSON manifest saved: {manifest_path}")
    return manifest_path


def load_manifest_summary(manifest_path: Path) -> Dict:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Pseudo-domain JSON manifest not found: {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    samples = manifest.get("samples", []) if isinstance(manifest, dict) else []
    counts = manifest.get("counts", {}) if isinstance(manifest, dict) else {}
    cluster_stats = manifest.get("cluster_stats", {}) if isinstance(manifest, dict) else {}
    return {
        "schema_version": manifest.get("schema_version") if isinstance(manifest, dict) else None,
        "experiment_name": manifest.get("experiment_name") if isinstance(manifest, dict) else None,
        "classes": manifest.get("class_names") if isinstance(manifest, dict) else None,
        "num_samples": len(samples) if isinstance(samples, list) else None,
        "counts": counts,
        "num_clusters": len(cluster_stats) if isinstance(cluster_stats, dict) else None,
    }


def find_teacher_ckpt(ckpt_dir: Path, experiment_name: str, explicit_teacher_ckpt: str = "") -> Path:
    if explicit_teacher_ckpt:
        p = Path(explicit_teacher_ckpt).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Explicit teacher checkpoint not found: {p}")
        return p

    best = (ckpt_dir / f"teacher_cond_{experiment_name}_best.pth").resolve()
    latest = (ckpt_dir / f"teacher_cond_{experiment_name}.pth").resolve()
    if best.is_file():
        return best
    if latest.is_file():
        print(f"[TPCRD][WARN] Best teacher ckpt not found, using latest: {latest}")
        return latest
    raise FileNotFoundError(
        "Teacher checkpoint not found. Expected one of:\n"
        f"  - {best}\n"
        f"  - {latest}"
    )


def build_teacher_cmd(args, teacher_script: Path, experiment_name: str, manifest_path: Path, domains_dir: Path, ckpt_dir: Path) -> List[str]:
    cmd = [sys.executable, str(teacher_script)]
    add_flag(cmd, "--class_name", experiment_name)
    add_flag(cmd, "--mvtec_root", args.mvtec_root)
    add_flag(cmd, "--domains_dir", domains_dir)
    add_flag(cmd, "--pseudo_domain_json", manifest_path)
    add_flag(cmd, "--ckpt_dir", ckpt_dir)

    add_flag(cmd, "--image_size", args.image_size)
    add_flag(cmd, "--batch_size", args.teacher_batch_size)
    add_flag(cmd, "--micro_batch_size", args.teacher_micro_batch_size)
    add_flag(cmd, "--num_workers", args.teacher_num_workers)
    add_flag(cmd, "--eval_batch_size", args.teacher_eval_batch_size)
    add_flag(cmd, "--eval_num_workers", args.teacher_eval_num_workers)
    add_flag(cmd, "--epochs", args.teacher_epochs)
    add_flag(cmd, "--lr", args.teacher_lr)
    add_flag(cmd, "--dro_eta", args.dro_eta)
    add_flag(cmd, "--q_min_floor", args.q_min_floor)
    add_flag(cmd, "--q_max_cap", args.q_max_cap)
    add_flag(cmd, "--best_metric", args.teacher_best_metric)
    add_flag(cmd, "--film_emb_dim", args.film_emb_dim)
    add_flag(cmd, "--film_hidden", args.film_hidden)
    add_flag(cmd, "--film_dropout", args.film_dropout)
    add_flag(cmd, "--cluster_dir", Path(manifest_path).resolve().parent.parent / "cluster")
    add_flag(cmd, "--pseudo_script", args.pseudo_script)
    add_flag(cmd, "--cluster_infer_method", args.cluster_infer_method)
    add_flag(cmd, "--cluster_router_device", args.cluster_router_device)
    add_flag(cmd, "--cluster_hdbscan_min_conf", args.cluster_hdbscan_min_conf)
    add_flag(cmd, "--cluster_router_cache_dir", args.cluster_router_cache_dir)
    add_flag(cmd, "--cluster_router_cache_flush_interval", args.cluster_router_cache_flush_interval)
    add_bool(cmd, "--no_cluster_router_cache", args.no_cluster_router_cache)
    add_flag(cmd, "--eval_every", args.teacher_eval_every)
    add_flag(cmd, "--save_every", args.teacher_save_every)
    add_flag(cmd, "--seed", args.seed)
    add_bool(cmd, "--print_per_domain", args.print_per_domain)
    add_bool(cmd, "--amp", args.amp)
    add_bool(cmd, "--non_deterministic", args.non_deterministic)
    cmd.extend(split_extra(args.teacher_extra))
    return cmd


def build_student_cmd(args, student_script: Path, experiment_name: str, manifest_path: Path, domains_dir: Path, ckpt_dir: Path, teacher_ckpt: Path) -> List[str]:
    cmd = [sys.executable, str(student_script)]
    add_flag(cmd, "--class_name", experiment_name)
    add_flag(cmd, "--mvtec_root", args.mvtec_root)
    add_flag(cmd, "--domains_dir", domains_dir)
    add_flag(cmd, "--pseudo_domain_json", manifest_path)
    add_flag(cmd, "--ckpt_dir", ckpt_dir)
    add_flag(cmd, "--teacher_ckpt", teacher_ckpt)
    add_flag(cmd, "--cluster_dir", Path(manifest_path).resolve().parent.parent / "cluster")
    add_flag(cmd, "--pseudo_script", args.pseudo_script)
    add_flag(cmd, "--cluster_infer_method", args.cluster_infer_method)
    add_flag(cmd, "--cluster_router_device", args.cluster_router_device)
    add_flag(cmd, "--cluster_hdbscan_min_conf", args.cluster_hdbscan_min_conf)
    add_flag(cmd, "--cluster_router_cache_dir", args.cluster_router_cache_dir)
    add_flag(cmd, "--cluster_router_cache_flush_interval", args.cluster_router_cache_flush_interval)
    add_bool(cmd, "--no_cluster_router_cache", args.no_cluster_router_cache)

    add_flag(cmd, "--image_size", args.image_size)
    add_flag(cmd, "--batch_size", args.student_batch_size)
    add_flag(cmd, "--num_workers", args.student_num_workers)
    add_flag(cmd, "--eval_batch_size", args.student_eval_batch_size)
    add_flag(cmd, "--eval_num_workers", args.student_eval_num_workers)
    add_flag(cmd, "--epochs", args.student_epochs)
    add_flag(cmd, "--lr", args.student_lr)
    add_flag(cmd, "--kd_weight", args.kd_weight)
    add_flag(cmd, "--reliability_tau", args.reliability_tau)
    add_flag(cmd, "--reliability_floor", args.reliability_floor)
    add_flag(cmd, "--eval_every", args.student_eval_every)
    add_flag(cmd, "--save_every", args.student_save_every)
    add_flag(cmd, "--seed", args.seed)
    add_bool(cmd, "--print_per_domain", args.print_per_domain)
    add_bool(cmd, "--amp", args.amp)
    add_bool(cmd, "--non_deterministic", args.non_deterministic)
    cmd.extend(split_extra(args.student_extra))
    return cmd


def write_pipeline_record(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Global pseudo-domain JSON + TPCRD teacher/student training.")

    parser.add_argument("--mvtec_root", type=str, default="./mvtec")
    parser.add_argument("--class_name", type=str, default="all", help="Class name, comma-separated names, or all.")
    parser.add_argument("--class_names", type=str, nargs="*", default=None, help="Explicit class list, e.g. bottle cable capsule.")
    parser.add_argument("--experiment_name", type=str, default="", help="Checkpoint/log namespace. Defaults to all, class name, or multi.")
    parser.add_argument("--out_dir", type=str, default="./outputs/tpcrd")
    parser.add_argument("--ckpt_dir", type=str, default="./checkpoints")
    parser.add_argument("--domains_dir", type=str, default="", help="Compatibility placeholder for old txt API; unused in JSON mode.")
    parser.add_argument("--pseudo_domain_json", type=str, default="", help="Existing global training_manifest.json.")

    parser.add_argument("--pseudo_script", type=str, default="pseudo-domain_discover_json.py")
    parser.add_argument("--teacher_script", type=str, default="train_teacher.py")
    parser.add_argument("--student_script", type=str, default="train_student.py")
    parser.add_argument("--work_dir", type=str, default="", help="Subprocess cwd. Defaults to current cwd.")

    parser.add_argument("--skip_json", action="store_true")
    parser.add_argument("--skip_teacher", action="store_true")
    parser.add_argument("--skip_student", action="store_true")
    parser.add_argument("--json_only", action="store_true")
    parser.add_argument("--teacher_only", action="store_true")
    parser.add_argument("--student_only", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--keep_generated_domain_txt", action="store_true")

    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--non_deterministic", action="store_true")
    parser.add_argument("--print_per_domain", action="store_true")

    # Global pseudo-domain discovery options.
    parser.add_argument("--pseudo_patches", type=int, default=12)
    parser.add_argument("--pseudo_patch", type=int, default=128)
    parser.add_argument("--pseudo_scales", type=float, nargs="+", default=[1.0, 0.7])
    parser.add_argument("--pseudo_no_retinex", action="store_true")
    parser.add_argument("--pseudo_rotations", type=int, default=4, choices=[1, 4])
    parser.add_argument("--pseudo_no_mask_padding", action="store_true")
    parser.add_argument("--pseudo_out_dim", type=int, default=256)
    parser.add_argument("--pseudo_min_cluster_size", type=int, default=20)
    parser.add_argument("--pseudo_min_samples", type=int, default=5)
    parser.add_argument("--pseudo_metric", type=str, default="euclidean", choices=["euclidean", "manhattan"])
    parser.add_argument("--pseudo_no_refine", action="store_true")
    parser.add_argument("--pseudo_tau_high", type=float, default=0.6)
    parser.add_argument("--pseudo_tau_low", type=float, default=0.3)
    parser.add_argument("--pseudo_nn_k", type=int, default=10)
    parser.add_argument("--pseudo_nn_ratio", type=float, default=0.7)
    parser.add_argument("--pseudo_no_noise_recluster", action="store_true")
    parser.add_argument("--pseudo_noise_min_cluster", type=int, default=12)
    parser.add_argument("--pseudo_noise_min_samples", type=int, default=4)
    parser.add_argument("--pseudo_keep_train_noise", action="store_true", help="Keep cluster=-1 train samples in JSON instead of assigning them to nearest center. Default assigns to nearest center.")


    # Teacher options.
    parser.add_argument("--teacher_epochs", type=int, default=200)
    parser.add_argument("--teacher_lr", type=float, default=0.005)
    parser.add_argument("--teacher_batch_size", type=int, default=8)
    parser.add_argument("--teacher_micro_batch_size", type=int, default=4, help="Memory-only micro-batch for teacher backward; keeps logical GroupDRO batch unchanged.")
    parser.add_argument("--teacher_num_workers", type=int, default=8)
    parser.add_argument("--teacher_eval_batch_size", type=int, default=8)
    parser.add_argument("--teacher_eval_num_workers", type=int, default=8)
    parser.add_argument("--teacher_eval_every", type=int, default=5)
    parser.add_argument("--teacher_save_every", type=int, default=10)
    parser.add_argument("--teacher_best_metric", type=str, default="mAUROC_sp_max")
    parser.add_argument("--dro_eta", type=float, default=0.08)
    parser.add_argument("--q_min_floor", type=float, default=0.005)
    parser.add_argument("--q_max_cap", type=float, default=0.90)
    parser.add_argument("--film_emb_dim", type=int, default=256)
    parser.add_argument("--film_hidden", type=int, default=1024)
    parser.add_argument("--film_dropout", type=float, default=0.0)
    parser.add_argument("--cluster_infer_method", type=str, default="hybrid", choices=["hybrid", "hdbscan", "centers"])
    parser.add_argument("--cluster_router_device", type=str, default="cpu", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--cluster_hdbscan_min_conf", type=float, default=0.0)
    parser.add_argument("--cluster_router_cache_dir", type=str, default="")
    parser.add_argument("--cluster_router_cache_flush_interval", type=int, default=16)
    parser.add_argument("--no_cluster_router_cache", action="store_true")
    parser.add_argument("--teacher_extra", type=str, default="")

    # Student options.
    parser.add_argument("--student_epochs", type=int, default=120)
    parser.add_argument("--student_lr", type=float, default=0.005)
    parser.add_argument("--student_batch_size", type=int, default=8)
    parser.add_argument("--student_num_workers", type=int, default=8)
    parser.add_argument("--student_eval_batch_size", type=int, default=8)
    parser.add_argument("--student_eval_num_workers", type=int, default=8)
    parser.add_argument("--student_eval_every", type=int, default=10)
    parser.add_argument("--student_save_every", type=int, default=10)
    parser.add_argument("--kd_weight", type=float, default=1.3)
    parser.add_argument("--reliability_tau", type=float, default=8.0)
    parser.add_argument("--reliability_floor", type=float, default=0.5)
    parser.add_argument("--teacher_ckpt", type=str, default="")
    parser.add_argument("--student_extra", type=str, default="")

    args = parser.parse_args()

    if args.json_only:
        args.skip_teacher = True
        args.skip_student = True
    if args.teacher_only:
        args.skip_student = True
    if args.student_only:
        args.skip_teacher = True
        args.skip_json = True

    anchor = Path(__file__).resolve().parent
    work_dir = (as_path(args.work_dir) or Path.cwd()).resolve()
    pseudo_script = resolve_script(args.pseudo_script, anchor)
    teacher_script = resolve_script(args.teacher_script, anchor)
    student_script = resolve_script(args.student_script, anchor)

    mvtec_root = Path(args.mvtec_root).expanduser().resolve()
    classes, experiment_name = parse_classes(args, mvtec_root)

    out_dir = Path(args.out_dir).expanduser().resolve()
    if Path(args.out_dir).as_posix().rstrip("/") in {"outputs/tpcrd", "./outputs/tpcrd"}:
        out_dir = (out_dir / experiment_name).resolve()
    ckpt_dir = Path(args.ckpt_dir).expanduser().resolve()
    domains_dir = Path(args.domains_dir).expanduser().resolve() if args.domains_dir else (out_dir / "_unused_domains_json_mode").resolve()
    manifest_path = Path(args.pseudo_domain_json).expanduser().resolve() if args.pseudo_domain_json else (out_dir / "json" / "training_manifest.json")

    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    domains_dir.mkdir(parents=True, exist_ok=True)

    print("[TPCRD] Configuration")
    print(json.dumps({
        "experiment_name": experiment_name,
        "classes": classes,
        "mvtec_root": str(mvtec_root),
        "out_dir": str(out_dir),
        "json_manifest": str(manifest_path),
        "ckpt_dir": str(ckpt_dir),
        "pseudo_script": str(pseudo_script),
        "teacher_script": str(teacher_script),
        "student_script": str(student_script),
        "cluster_scope": "global_all_selected_train_good_only",
        "test_domain_assignment": "cluster_model_online_in_teacher_student_eval",
        "skip_json": bool(args.skip_json),
        "skip_teacher": bool(args.skip_teacher),
        "skip_student": bool(args.skip_student),
    }, ensure_ascii=False, indent=2))

    if not args.skip_json:
        if args.dry_run:
            print(f"[TPCRD] DRY RUN: would generate global JSON at {manifest_path}")
            return
        manifest_path = run_global_pseudo_domain_discovery(
            args=args,
            pseudo_script=pseudo_script,
            mvtec_root=mvtec_root,
            classes=classes,
            out_dir=out_dir,
            experiment_name=experiment_name,
        )

    if args.dry_run:
        print(f"[TPCRD] DRY RUN: would use JSON manifest: {manifest_path}")
        return

    summary = load_manifest_summary(manifest_path)
    print("[TPCRD] JSON summary")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if not args.skip_teacher:
        teacher_cmd = build_teacher_cmd(
            args=args,
            teacher_script=teacher_script,
            experiment_name=experiment_name,
            manifest_path=manifest_path,
            domains_dir=domains_dir,
            ckpt_dir=ckpt_dir,
        )
        run_cmd(teacher_cmd, cwd=work_dir, dry_run=False)

    teacher_ckpt = None
    if not args.skip_student:
        teacher_ckpt = find_teacher_ckpt(ckpt_dir, experiment_name, args.teacher_ckpt)
        student_cmd = build_student_cmd(
            args=args,
            student_script=student_script,
            experiment_name=experiment_name,
            manifest_path=manifest_path,
            domains_dir=domains_dir,
            ckpt_dir=ckpt_dir,
            teacher_ckpt=teacher_ckpt,
        )
        run_cmd(student_cmd, cwd=work_dir, dry_run=False)
    elif args.teacher_ckpt:
        teacher_ckpt = Path(args.teacher_ckpt).expanduser().resolve()

    record = {
        "schema_version": "tpcrd_pipeline_record/v4_global_train_cluster_cluster_router",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "experiment_name": experiment_name,
        "classes": classes,
        "mvtec_root": str(mvtec_root),
        "out_dir": str(out_dir),
        "pseudo_domain_json": str(manifest_path),
        "manifest_summary": summary,
        "ckpt_dir": str(ckpt_dir),
        "teacher_ckpt": str(teacher_ckpt) if teacher_ckpt else str((ckpt_dir / f"teacher_cond_{experiment_name}_best.pth").resolve()),
        "student_ckpt": str((ckpt_dir / f"student_distill_{experiment_name}.pth").resolve()),
        "stages": {
            "json": not args.skip_json,
            "teacher": not args.skip_teacher,
            "student": not args.skip_student,
        },
    }
    record_path = out_dir / "tpcrd_pipeline_record.json"
    write_pipeline_record(record_path, record)

    print("\n[TPCRD] Pipeline finished.")
    print(f"[TPCRD] JSON manifest : {manifest_path}")
    print(f"[TPCRD] Teacher ckpt   : {record['teacher_ckpt']}")
    print(f"[TPCRD] Student ckpt   : {record['student_ckpt']}")
    print(f"[TPCRD] Run record     : {record_path}")


if __name__ == "__main__":
    main()
