import os
import json
import glob
import importlib.util
import hashlib
import time
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score


# -------------------------
# RD4AD losses
# -------------------------
def rd4ad_cosine_loss(teacher_feats: List[torch.Tensor], student_feats: List[torch.Tensor]) -> torch.Tensor:
    assert len(teacher_feats) == len(student_feats), "teacher_feats and student_feats must have same length"
    cos = nn.CosineSimilarity()
    loss = 0.0
    for a, b in zip(teacher_feats, student_feats):
        loss = loss + torch.mean(
            1.0 - cos(
                a.view(a.shape[0], -1),
                b.view(b.shape[0], -1),
            )
        )
    return loss


def distill_feats_loss(teacher_outs: List[torch.Tensor], student_outs: List[torch.Tensor]) -> torch.Tensor:
    return rd4ad_cosine_loss(teacher_outs, student_outs)


# -------------------------
# FiLM conditioning
# -------------------------
class DomainFiLM(nn.Module):
    """
    z' = (1 + gamma_d) * z + beta_d
    z: (B,C,H,W), domain_id: (B,)
    """
    def __init__(
        self,
        num_domains: int,
        channels: int,
        emb_dim: int = 128,
        hidden: int = 512,
        dropout: float = 0.0,
        init_identity: bool = True,
    ):
        super().__init__()
        self.num_domains = int(num_domains)
        self.channels = int(channels)

        self.embed = nn.Embedding(self.num_domains, emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden, 2 * channels),
        )

        if init_identity:
            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, z: torch.Tensor, domain_id: torch.Tensor) -> torch.Tensor:
        if domain_id.dtype != torch.long:
            domain_id = domain_id.long()

        if domain_id.numel() > 0:
            dmin = int(domain_id.min().item())
            dmax = int(domain_id.max().item())
            if dmin < 0 or dmax >= self.num_domains:
                raise ValueError(f"domain_id out of range: min={dmin}, max={dmax}, num_domains={self.num_domains}")

        e = self.embed(domain_id)            # (B, emb_dim)
        params = self.mlp(e)                 # (B, 2C)
        gamma, beta = params.chunk(2, dim=1)
        gamma = gamma.view(-1, self.channels, 1, 1)
        beta = beta.view(-1, self.channels, 1, 1)
        return (1.0 + gamma) * z + beta


class ConditionalTeacher(nn.Module):
    """
    T(x,d): feats -> bn -> FiLM(d) -> decoder
    """
    def __init__(self, bn: nn.Module, decoder: nn.Module, film: DomainFiLM):
        super().__init__()
        self.bn = bn
        self.decoder = decoder
        self.film = film

    def forward(self, teacher_feats: List[torch.Tensor], domain_id: torch.Tensor):
        z = self.bn(teacher_feats)
        zc = self.film(z, domain_id)
        outs = self.decoder(zc)
        return outs, zc


class StudentNoDomain(nn.Module):
    """
    S(x): feats -> bn -> decoder
    """
    def __init__(self, bn: nn.Module, decoder: nn.Module):
        super().__init__()
        self.bn = bn
        self.decoder = decoder

    def forward(self, teacher_feats: List[torch.Tensor]):
        z = self.bn(teacher_feats)
        outs = self.decoder(z)
        return outs, z




class ClusterDomainRouter:
    def __init__(
        self,
        cluster_dir: str,
        domain_name_to_id: Dict[str, int],
        pseudo_script: str = "pseudo-domain_discover_json.py",
        method: str = "hybrid",
        feature_device: str = "cpu",
        hdbscan_min_conf: float = 0.0,
        cache: bool = True,
        cache_dir: str = "",
        cache_flush_interval: int = 16,
    ):
        self.cluster_dir = _canon_path(cluster_dir)
        self.domain_name_to_id = {str(k): int(v) for k, v in dict(domain_name_to_id).items()}
        self.cluster_to_domain_id = self._build_cluster_to_domain_id(self.domain_name_to_id)
        if len(self.cluster_to_domain_id) == 0:
            raise ValueError("ClusterDomainRouter requires domain names like domain_<cluster_id>.")
        self.method = str(method or "hybrid").lower()
        if self.method not in {"hybrid", "hdbscan", "centers"}:
            raise ValueError(f"Unsupported cluster infer method: {self.method}")
        self.feature_device = self._resolve_feature_device(feature_device)
        self.hdbscan_min_conf = float(hdbscan_min_conf)
        self.cache_enabled = bool(cache)
        self.cache_flush_interval = max(1, int(cache_flush_interval))
        self.cache_dir_arg = str(cache_dir or "")
        self.cache: Dict[str, Dict] = {}
        self.cache_dirty = 0
        self.cache_path = ""
        self.cache_fingerprint = ""

        self.scaler = self._load_joblib("scaler.joblib")
        self.pca = self._load_joblib("pca.joblib")
        self.clusterer = self._load_joblib("hdbscan.joblib", required=False)
        self.centers = self._load_npy("centers.npy").astype(np.float32)
        self.valid_ids = self._load_npy("valid_ids.npy").astype(np.int32).tolist()
        if self.centers.ndim != 2 or len(self.valid_ids) != int(self.centers.shape[0]):
            raise ValueError(
                f"Invalid centers/valid_ids in {self.cluster_dir}: "
                f"centers={self.centers.shape}, valid_ids={len(self.valid_ids)}"
            )

        self.meta = self._load_meta()
        self.patches = int(self.meta.get("patches", 12))
        self.patch = int(self.meta.get("patch", 128))
        self.scales = [float(x) for x in self.meta.get("scales", [1.0, 0.7])]
        self.no_retinex = bool(self.meta.get("no_retinex", False))
        self.rotations = int(self.meta.get("rotations", 4))
        self.use_mask = bool(self.meta.get("mask_padding", True))

        self.pseudo_script = self._resolve_pseudo_script(pseudo_script)
        self.pseudo = self._load_pseudo_module(self.pseudo_script)
        self.vgg = self.pseudo.VGGFeat().eval().to(self.feature_device)
        for p in self.vgg.parameters():
            p.requires_grad_(False)

        self._init_persistent_cache()

        print(
            f"[ClusterRouter] cluster_dir={self.cluster_dir} method={self.method} "
            f"feature_device={self.feature_device} domains={len(self.domain_name_to_id)} "
            f"cache={'on' if self.cache_enabled else 'off'} cache_file={self.cache_path if self.cache_enabled else ''}"
        )

    @classmethod
    def from_manifest(
        cls,
        pseudo_domain_json: str,
        domain_name_to_id: Dict[str, int],
        cluster_dir: str = "",
        pseudo_script: str = "pseudo-domain_discover_json.py",
        method: str = "hybrid",
        feature_device: str = "cpu",
        hdbscan_min_conf: float = 0.0,
        cache: bool = True,
        cache_dir: str = "",
        cache_flush_interval: int = 16,
    ):
        pseudo_domain_json = _canon_path(pseudo_domain_json)
        inferred_cluster_dir = ""
        if os.path.isfile(pseudo_domain_json):
            try:
                with open(pseudo_domain_json, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                if isinstance(payload, dict):
                    paths = payload.get("paths", {})
                    if isinstance(paths, dict):
                        inferred_cluster_dir = str(paths.get("cluster_dir", "") or "")
            except Exception:
                inferred_cluster_dir = ""
        if not cluster_dir:
            cluster_dir = inferred_cluster_dir
        if not cluster_dir:
            cluster_dir = os.path.join(os.path.dirname(os.path.dirname(pseudo_domain_json)), "cluster")
        return cls(
            cluster_dir=cluster_dir,
            domain_name_to_id=domain_name_to_id,
            pseudo_script=pseudo_script,
            method=method,
            feature_device=feature_device,
            hdbscan_min_conf=hdbscan_min_conf,
            cache=cache,
            cache_dir=cache_dir,
            cache_flush_interval=cache_flush_interval,
        )

    @staticmethod
    def _build_cluster_to_domain_id(domain_name_to_id: Dict[str, int]) -> Dict[int, int]:
        out: Dict[int, int] = {}
        for name, did in domain_name_to_id.items():
            s = str(name)
            if s.startswith("domain_"):
                try:
                    out[int(s.split("domain_", 1)[1])] = int(did)
                except Exception:
                    pass
        return out

    @staticmethod
    def _resolve_feature_device(feature_device: str) -> str:
        feature_device = str(feature_device or "cpu").lower()
        if feature_device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if feature_device.startswith("cuda") and not torch.cuda.is_available():
            return "cpu"
        return feature_device

    def _load_joblib(self, name: str, required: bool = True):
        path = os.path.join(self.cluster_dir, name)
        if not os.path.isfile(path):
            if required:
                raise FileNotFoundError(f"Missing cluster artifact: {path}")
            return None
        from joblib import load
        return load(path)

    def _load_npy(self, name: str) -> np.ndarray:
        path = os.path.join(self.cluster_dir, name)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing cluster artifact: {path}")
        return np.load(path)

    def _load_meta(self) -> dict:
        path = os.path.join(self.cluster_dir, "model_meta.json")
        if not os.path.isfile(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _resolve_pseudo_script(pseudo_script: str) -> str:
        candidates = []
        if pseudo_script:
            candidates.append(Path(str(pseudo_script)).expanduser())
        here = Path(__file__).resolve().parent
        if pseudo_script:
            candidates.append(here / str(pseudo_script))
        candidates.append(here / "pseudo-domain_discover_json.py")
        candidates.append(Path.cwd() / "pseudo-domain_discover_json.py")
        for p in candidates:
            try:
                rp = p.resolve()
                if rp.is_file():
                    return str(rp)
            except Exception:
                pass
        raise FileNotFoundError(f"Could not resolve pseudo feature script: {pseudo_script}")

    @staticmethod
    def _load_pseudo_module(path: str):
        spec = importlib.util.spec_from_file_location("tpcrd_pseudo_feature_module", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import pseudo feature script: {path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def _artifact_stats_for_cache(self) -> Dict[str, Dict[str, int]]:
        out = {}
        for name in ("scaler.joblib", "pca.joblib", "hdbscan.joblib", "centers.npy", "valid_ids.npy", "model_meta.json"):
            path = os.path.join(self.cluster_dir, name)
            if os.path.isfile(path):
                st = os.stat(path)
                out[name] = {"size": int(st.st_size), "mtime_ns": int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9)))}
        return out

    def _make_cache_fingerprint(self) -> str:
        payload = {
            "cluster_dir": self.cluster_dir,
            "artifacts": self._artifact_stats_for_cache(),
            "domain_name_to_id": sorted((str(k), int(v)) for k, v in self.domain_name_to_id.items()),
            "method": self.method,
            "hdbscan_min_conf": float(self.hdbscan_min_conf),
            "patches": int(self.patches),
            "patch": int(self.patch),
            "scales": [float(x) for x in self.scales],
            "no_retinex": bool(self.no_retinex),
            "rotations": int(self.rotations),
            "use_mask": bool(self.use_mask),
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return hashlib.sha1(raw).hexdigest()[:20]

    def _init_persistent_cache(self):
        if not self.cache_enabled:
            return
        self.cache_fingerprint = self._make_cache_fingerprint()
        cache_dir = self.cache_dir_arg.strip()
        if not cache_dir:
            cache_dir = os.path.join(self.cluster_dir, "router_cache")
        cache_dir = _canon_path(cache_dir)
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_path = os.path.join(cache_dir, f"cluster_router_cache_{self.cache_fingerprint}.json")
        self.cache = {}
        if os.path.isfile(self.cache_path):
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                if isinstance(payload, dict) and payload.get("fingerprint") == self.cache_fingerprint:
                    entries = payload.get("entries", {})
                    if isinstance(entries, dict):
                        self.cache = {str(k): dict(v) for k, v in entries.items() if isinstance(v, dict)}
            except Exception as e:
                print(f"[ClusterRouter][WARN] Failed to load cache {self.cache_path}: {e}")
                self.cache = {}
        print(f"[ClusterRouter] loaded {len(self.cache)} cached route entries")

    @staticmethod
    def _file_signature(path: str) -> Dict[str, int]:
        try:
            st = os.stat(path)
            return {"size": int(st.st_size), "mtime_ns": int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9)))}
        except Exception:
            return {"size": -1, "mtime_ns": -1}

    def _get_cached_route(self, path: str):
        if not self.cache_enabled:
            return None
        cp = _canon_path(path)
        rec = self.cache.get(cp)
        if not isinstance(rec, dict):
            return None
        sig = self._file_signature(cp)
        if int(rec.get("size", -2)) != int(sig.get("size", -1)):
            return None
        if int(rec.get("mtime_ns", -2)) != int(sig.get("mtime_ns", -1)):
            return None
        try:
            cid = int(rec.get("cluster_id"))
            did = int(rec.get("domain_id"))
        except Exception:
            return None
        if cid not in self.cluster_to_domain_id:
            return None
        if int(self.cluster_to_domain_id[cid]) != did:
            return None
        return cid, did

    def _store_cached_route(self, path: str, cluster_id: int, domain_id: int):
        if not self.cache_enabled:
            return
        cp = _canon_path(path)
        sig = self._file_signature(cp)
        self.cache[cp] = {
            "cluster_id": int(cluster_id),
            "domain_id": int(domain_id),
            "size": int(sig.get("size", -1)),
            "mtime_ns": int(sig.get("mtime_ns", -1)),
            "updated_at": int(time.time()),
        }
        self.cache_dirty += 1
        if self.cache_dirty >= self.cache_flush_interval:
            self.save_cache()

    def save_cache(self):
        if not self.cache_enabled or not self.cache_path:
            return
        payload = {
            "schema_version": "tpcrd_cluster_router_cache/v1",
            "fingerprint": self.cache_fingerprint,
            "cluster_dir": self.cluster_dir,
            "method": self.method,
            "hdbscan_min_conf": float(self.hdbscan_min_conf),
            "updated_at": int(time.time()),
            "entries": self.cache,
        }
        tmp = self.cache_path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
            os.replace(tmp, self.cache_path)
            self.cache_dirty = 0
        except Exception as e:
            print(f"[ClusterRouter][WARN] Failed to save cache {self.cache_path}: {e}")
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except Exception:
                pass

    def _path_to_z(self, path: str) -> np.ndarray:
        x, _ = self.pseudo.build_image_feature(
            path,
            self.vgg,
            self.feature_device,
            self.patches,
            self.patch,
            list(self.scales),
            use_retinex=(not self.no_retinex),
            rotations=self.rotations,
            use_mask=self.use_mask,
        )
        X = np.asarray(x, dtype=np.float32).reshape(1, -1)
        Xn = self.scaler.transform(X)
        Z = self.pca.transform(Xn).astype(np.float32)
        return Z

    def _nearest_cluster(self, Z: np.ndarray) -> int:
        diff = self.centers.astype(np.float32) - Z.reshape(1, -1).astype(np.float32)
        d = np.sqrt(np.sum(diff * diff, axis=1))
        idx = int(np.argmin(d))
        return int(self.valid_ids[idx])

    def _hdbscan_cluster(self, Z: np.ndarray) -> Tuple[int, float]:
        if self.clusterer is None:
            return -1, 0.0
        try:
            import hdbscan
            labels, strengths = hdbscan.approximate_predict(self.clusterer, Z)
            cid = int(labels[0])
            conf = float(strengths[0]) if len(strengths) else 0.0
            return cid, conf
        except Exception:
            return -1, 0.0

    def predict_cluster_id(self, path: str) -> int:
        path = _canon_path(path)
        cached = self._get_cached_route(path)
        if cached is not None:
            return int(cached[0])

        Z = self._path_to_z(path)
        cid = -1

        if self.method in {"hybrid", "hdbscan"}:
            pred, conf = self._hdbscan_cluster(Z)
            if pred != -1 and pred in self.cluster_to_domain_id and conf >= self.hdbscan_min_conf:
                cid = int(pred)
            elif self.method == "hdbscan":
                cid = -1

        if cid == -1 and self.method in {"hybrid", "centers"}:
            cid = self._nearest_cluster(Z)

        if cid == -1 or cid not in self.cluster_to_domain_id:
            cid = self._nearest_cluster(Z)

        did = int(self.cluster_to_domain_id[int(cid)])
        self._store_cached_route(path, int(cid), did)
        return int(cid)

    @torch.no_grad()
    def predict_paths(self, paths, output_device: str = "cpu") -> torch.Tensor:
        if paths is None:
            raise ValueError("ClusterDomainRouter needs image paths from the test dataset.")
        if isinstance(paths, (str, bytes)):
            paths = [paths]
        domain_ids = []
        for p in list(paths):
            if isinstance(p, bytes):
                p = p.decode("utf-8")
            cp = _canon_path(str(p))
            cached = self._get_cached_route(cp)
            if cached is not None:
                domain_ids.append(int(cached[1]))
                continue
            cid = self.predict_cluster_id(cp)
            if cid not in self.cluster_to_domain_id:
                cid = self._nearest_cluster(self._path_to_z(cp))
            did = int(self.cluster_to_domain_id[int(cid)])
            self._store_cached_route(cp, int(cid), did)
            domain_ids.append(did)
        if self.cache_enabled and self.cache_dirty > 0:
            self.save_cache()
        return torch.tensor(domain_ids, dtype=torch.long, device=output_device)


@torch.no_grad()
def infer_z_channels(encoder: nn.Module, bn: nn.Module, train_loader, device: str) -> int:
    encoder.eval()
    bn.eval()
    img, *_ = next(iter(train_loader))
    img = img.to(device, non_blocking=True)
    feats = encoder(img)
    z = bn(feats)
    return int(z.shape[1])



# -------------------------
# TXT/JSON-based test subset (MVTec style)
# -------------------------
def _canon_path(p: str) -> str:
    p = str(p).strip().replace("\\", os.sep).replace("/", os.sep)
    return os.path.abspath(os.path.normpath(p))


def _tail_from_train_test(p: str) -> Optional[str]:
    parts = _canon_path(p).split(os.sep)
    for k in ("train", "test"):
        if k in parts:
            i = parts.index(k)
            return os.path.join(*parts[i:])
    return None


def _safe_int(v, default: int = -1) -> int:
    try:
        if v is None:
            return default
        if isinstance(v, str) and v.strip() == "":
            return default
        return int(float(v))
    except Exception:
        return default


def resolve_img_path(line: str, root: str) -> str:
    root = _canon_path(root)
    raw = line.strip()
    if not raw:
        return ""

    raw_norm = raw.replace("\\", os.sep).replace("/", os.sep)

    p0 = _canon_path(raw_norm)
    if os.path.isfile(p0):
        return p0

    # root/class 以及 root 的上一级都尝试，兼容“不移动公共数据集”的相对路径
    for base in (root, os.path.dirname(root), os.getcwd()):
        p1 = _canon_path(os.path.join(base, raw_norm))
        if os.path.isfile(p1):
            return p1

    tail = _tail_from_train_test(raw_norm)
    if tail is not None:
        for base in (root, os.path.dirname(root), os.getcwd()):
            p2 = _canon_path(os.path.join(base, tail))
            if os.path.isfile(p2):
                return p2

    return p0


def read_txt_lines(fp: str) -> List[str]:
    encs = ("utf-8-sig", "utf-8", "gbk")
    last = None
    for enc in encs:
        try:
            out = []
            with open(fp, "r", encoding=enc) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    out.append(line)
            return out
        except Exception as e:
            last = e
    raise last


def _load_json_payload(json_path: str):
    json_path = _canon_path(json_path)
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"pseudo_domain_json not found: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _iter_json_records(payload) -> List[dict]:
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]

    if not isinstance(payload, dict):
        return []

    if isinstance(payload.get("samples"), list):
        return [x for x in payload["samples"] if isinstance(x, dict)]

    records: List[dict] = []
    for _, v in payload.items():
        if isinstance(v, dict) and isinstance(v.get("samples"), list):
            records.extend([x for x in v["samples"] if isinstance(x, dict)])
    return records


def _path_candidates_from_record(rec: dict) -> List[str]:
    vals = []
    for k in ("path_abs", "path", "path_rel_to_data_parent", "image_path", "img_path"):
        v = rec.get(k, None)
        if v is not None and str(v).strip():
            vals.append(str(v).strip())
    return vals


def _resolve_record_path(rec: dict, root: str, manifest_data_root: Optional[str] = None) -> str:
    root = _canon_path(root) if root else os.getcwd()
    base_roots = [root, os.path.dirname(root), os.getcwd()]
    if manifest_data_root:
        mroot = _canon_path(manifest_data_root)
        base_roots.extend([mroot, os.path.dirname(mroot)])

    uniq_bases = []
    seen = set()
    for b in base_roots:
        if b and b not in seen:
            uniq_bases.append(b)
            seen.add(b)

    first_fallback = ""
    for raw in _path_candidates_from_record(rec):
        raw_norm = raw.replace("\\", os.sep).replace("/", os.sep)

        p0 = _canon_path(raw_norm)
        if not first_fallback:
            first_fallback = p0
        if os.path.isfile(p0):
            return p0

        for base in uniq_bases:
            p1 = _canon_path(os.path.join(base, raw_norm))
            if os.path.isfile(p1):
                return p1

        tail = _tail_from_train_test(raw_norm)
        if tail is not None:
            cls = str(rec.get("source_class", rec.get("class_name", ""))).strip()
            if cls and cls not in ("all", "unknown"):
                for base in uniq_bases:
                    p_cls = _canon_path(os.path.join(base, cls, tail))
                    if os.path.isfile(p_cls):
                        return p_cls
            for base in uniq_bases:
                p2 = _canon_path(os.path.join(base, tail))
                if os.path.isfile(p2):
                    return p2

    return first_fallback


def _record_split(rec: dict) -> str:
    return str(rec.get("split", "")).strip().replace("\\", "/")


def _record_cluster_id(rec: dict) -> int:
    for k in ("cluster_id", "label", "pred_cluster", "cluster"):
        if k in rec:
            cid = _safe_int(rec.get(k), -1)
            if cid != -1:
                return cid
    return -1


def _domain_name_from_cluster(cluster_id: int) -> str:
    return f"domain_{int(cluster_id)}"


def _is_test_record(rec: dict, path: str = "") -> bool:
    split = _record_split(rec)
    if split.startswith("test/"):
        return True
    tail = _tail_from_train_test(path) if path else None
    return tail is not None and tail.split(os.sep)[0] == "test"


def _label_type_from_test_path_or_split(path: str, split: str) -> Tuple[int, str]:
    split = split.replace("\\", "/")
    if split.startswith("test/"):
        img_type = split.split("/", 1)[1] or "unknown"
    else:
        parts = _canon_path(path).split(os.sep)
        img_type = "unknown"
        if "test" in parts:
            i = parts.index("test")
            if i + 1 < len(parts):
                img_type = parts[i + 1]

    label = 0 if img_type == "good" else 1
    return label, img_type


def _find_mask_for_test_image(img_path: str, img_type: str, strict_mask: bool = True):
    if img_type == "good":
        return 0

    parts = _canon_path(img_path).split(os.sep)
    mask_roots = []

    if "test" in parts:
        i = parts.index("test")
        if i > 0:
            class_root = os.sep.join(parts[:i])
            mask_roots.append(os.path.join(class_root, "ground_truth", img_type))

    # 兜底：root/test/type/xxx.png -> root/ground_truth/type/xxx.png 的常见 MVTec 结构
    parent_type = os.path.basename(os.path.dirname(img_path))
    if parent_type and parent_type != img_type:
        parent_root = os.path.dirname(os.path.dirname(os.path.dirname(img_path)))
        mask_roots.append(os.path.join(parent_root, "ground_truth", img_type))

    stem = os.path.splitext(os.path.basename(img_path))[0]
    basename = os.path.basename(img_path)

    candidates = []
    for mask_dir in mask_roots:
        candidates.extend([
            os.path.join(mask_dir, f"{stem}_mask.png"),
            os.path.join(mask_dir, f"{stem}.png"),
            os.path.join(mask_dir, basename),
        ])
        candidates.extend(sorted(glob.glob(os.path.join(mask_dir, f"{stem}*"))))

    seen = set()
    for cand in candidates:
        cand = _canon_path(cand)
        if cand in seen:
            continue
        seen.add(cand)
        if os.path.isfile(cand):
            return cand

    if strict_mask:
        roots = ", ".join(mask_roots) if mask_roots else "<unresolved>"
        raise FileNotFoundError(f"Mask not found for image: {img_path}\nExpected in: {roots}")
    return 0


class SubsetMVTecFromTxt(torch.utils.data.Dataset):
    """
    Build MVTec-like TEST subset from txt.
    Returns: img, gt, label(0/1), img_type
    """
    def __init__(self, root: str, img_list: List[str], transform, gt_transform, strict_mask: bool = True):
        super().__init__()
        self.root = _canon_path(root)
        self.transform = transform
        self.gt_transform = gt_transform
        self.strict_mask = strict_mask

        abs_imgs = []
        for line in img_list:
            p = resolve_img_path(line, self.root)
            if not p:
                continue
            if os.sep + "test" + os.sep not in p:
                continue
            if os.path.isfile(p):
                abs_imgs.append(p)

        abs_imgs = sorted(list(dict.fromkeys(abs_imgs)))
        if len(abs_imgs) == 0:
            raise ValueError("No valid test images found in this txt after resolving paths.")

        img_paths, gt_paths, labels, types = [], [], [], []

        by_type: Dict[str, List[str]] = {}
        for p in abs_imgs:
            defect_type = os.path.basename(os.path.dirname(p))  # test/<defect_type>/xxx.png
            by_type.setdefault(defect_type, []).append(p)

        for defect_type, paths in by_type.items():
            paths = sorted(paths)
            if defect_type == "good":
                img_paths.extend(paths)
                gt_paths.extend([0] * len(paths))
                labels.extend([0] * len(paths))
                types.extend(["good"] * len(paths))
            else:
                for ip in paths:
                    img_paths.append(ip)
                    labels.append(1)
                    types.append(defect_type)
                    gt_paths.append(_find_mask_for_test_image(ip, defect_type, strict_mask=strict_mask))

        self.img_paths = img_paths
        self.gt_paths = gt_paths
        self.labels = labels
        self.types = types
        assert len(self.img_paths) == len(self.gt_paths) == len(self.labels) == len(self.types)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx: int):
        img_path = self.img_paths[idx]
        gt = self.gt_paths[idx]
        label = self.labels[idx]
        img_type = self.types[idx]

        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        if gt == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)

        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"
        return img, gt, label, img_type


class SubsetMVTecFromJson(torch.utils.data.Dataset):
    """
    Build MVTec-like TEST subset from pseudo-domain JSON.
    Returns: img, gt, label(0/1), img_type
    """
    def __init__(
        self,
        root: str,
        records: List[dict],
        transform,
        gt_transform,
        strict_mask: bool = True,
        manifest_data_root: Optional[str] = None,
    ):
        super().__init__()
        self.root = _canon_path(root)
        self.transform = transform
        self.gt_transform = gt_transform
        self.strict_mask = strict_mask

        img_paths, gt_paths, labels, types = [], [], [], []
        seen = set()

        for rec in records:
            p = _resolve_record_path(rec, self.root, manifest_data_root=manifest_data_root)
            if not p or not os.path.isfile(p):
                continue
            if not _is_test_record(rec, p):
                continue
            if p in seen:
                continue
            seen.add(p)

            split = _record_split(rec)
            label, img_type = _label_type_from_test_path_or_split(p, split)
            gt = 0 if label == 0 else _find_mask_for_test_image(p, img_type, strict_mask=strict_mask)

            img_paths.append(p)
            gt_paths.append(gt)
            labels.append(label)
            types.append(img_type)

        if len(img_paths) == 0:
            raise ValueError("No valid test images found in pseudo-domain JSON after resolving paths.")

        self.img_paths = img_paths
        self.gt_paths = gt_paths
        self.labels = labels
        self.types = types
        assert len(self.img_paths) == len(self.gt_paths) == len(self.labels) == len(self.types)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx: int):
        img_path = self.img_paths[idx]
        gt = self.gt_paths[idx]
        label = self.labels[idx]
        img_type = self.types[idx]

        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        if gt == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)

        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"
        return img, gt, label, img_type


def list_domain_txts(domains_dir: str) -> List[str]:
    return sorted(glob.glob(os.path.join(domains_dir, "*.txt")))


def build_domain_test_loaders(
    root_path: str,
    domains_dir: str,
    data_transform,
    gt_transform,
    batch_size: int = 1,
    num_workers: int = 0,
    pin_memory: bool = True,
    strict_mask: bool = True,
) -> Dict[str, torch.utils.data.DataLoader]:
    txt_files = list_domain_txts(domains_dir)
    if len(txt_files) == 0:
        raise FileNotFoundError(f"No txt files found in: {domains_dir}")

    loaders = {}
    for fp in txt_files:
        name = os.path.splitext(os.path.basename(fp))[0]
        lines = read_txt_lines(fp)
        try:
            ds = SubsetMVTecFromTxt(
                root=root_path,
                img_list=lines,
                transform=data_transform,
                gt_transform=gt_transform,
                strict_mask=strict_mask
            )
        except Exception:
            continue

        loaders[name] = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
        )

    if len(loaders) == 0:
        raise RuntimeError("No valid domain test loaders could be built from txt files.")
    return loaders


class MergedDomainTestDatasetFromTxts(torch.utils.data.Dataset):
    """
    把多个 domain txt 合并成一个测试集。
    返回:
        img, gt, label, img_type, domain_name, domain_id
    """
    def __init__(
        self,
        root: str,
        domains_dir: str,
        domain_name_to_id: Dict[str, int],
        transform,
        gt_transform,
        strict_mask: bool = True,
    ):
        super().__init__()
        self.root = _canon_path(root)
        self.domains_dir = _canon_path(domains_dir)
        self.transform = transform
        self.gt_transform = gt_transform
        self.strict_mask = strict_mask
        self.domain_name_to_id = {str(k): int(v) for k, v in domain_name_to_id.items()}

        txt_files = list_domain_txts(self.domains_dir)
        if len(txt_files) == 0:
            raise FileNotFoundError(f"No txt files found in: {self.domains_dir}")

        self.all_domain_names: List[str] = [
            os.path.splitext(os.path.basename(fp))[0] for fp in txt_files
        ]
        self.valid_domain_names: List[str] = []
        self.skipped_domains: Dict[str, str] = {}

        self.img_paths: List[str] = []
        self.gt_paths: List[object] = []
        self.labels: List[int] = []
        self.types: List[str] = []
        self.domain_names: List[str] = []
        self.domain_ids: List[int] = []

        for fp in txt_files:
            name = os.path.splitext(os.path.basename(fp))[0]

            if name not in self.domain_name_to_id:
                self.skipped_domains[name] = "no id mapping"
                continue

            try:
                lines = read_txt_lines(fp)
                ds = SubsetMVTecFromTxt(
                    root=self.root,
                    img_list=lines,
                    transform=self.transform,
                    gt_transform=self.gt_transform,
                    strict_mask=self.strict_mask,
                )
            except Exception as e:
                self.skipped_domains[name] = str(e)
                continue

            dom_id = int(self.domain_name_to_id[name])

            self.valid_domain_names.append(name)
            self.img_paths.extend(ds.img_paths)
            self.gt_paths.extend(ds.gt_paths)
            self.labels.extend(ds.labels)
            self.types.extend(ds.types)
            self.domain_names.extend([name] * len(ds))
            self.domain_ids.extend([dom_id] * len(ds))

        if len(self.img_paths) == 0:
            raise RuntimeError("No valid merged domain test samples could be built from txt files.")

        assert (
            len(self.img_paths)
            == len(self.gt_paths)
            == len(self.labels)
            == len(self.types)
            == len(self.domain_names)
            == len(self.domain_ids)
        )

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx: int):
        img_path = self.img_paths[idx]
        gt = self.gt_paths[idx]
        label = self.labels[idx]
        img_type = self.types[idx]
        domain_name = self.domain_names[idx]
        domain_id = self.domain_ids[idx]

        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        if gt == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)

        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"
        return img, gt, label, img_type, domain_name, domain_id, img_path


class MergedDomainTestDatasetFromJson(torch.utils.data.Dataset):
    """
    JSON 模式下的 merged test dataset。

    重要：本类不读取、不使用 JSON 中为 test 样本预先写入的 cluster_id/domain_id。
    测试样本的伪域必须在评估时由 cluster model router 在线预测，避免 test pseudo-domain
    label leakage。本 dataset 只负责提供 image / mask / anomaly label。

    返回占位格式以兼容旧评估接口：
        img, gt, label, img_type, "__unrouted__", -1
    """
    def __init__(
        self,
        root: str,
        pseudo_domain_json: str,
        domain_name_to_id: Dict[str, int],
        transform,
        gt_transform,
        strict_mask: bool = True,
    ):
        super().__init__()
        self.root = _canon_path(root)
        self.pseudo_domain_json = _canon_path(pseudo_domain_json)
        self.transform = transform
        self.gt_transform = gt_transform
        self.strict_mask = strict_mask
        self.domain_name_to_id = {str(k): int(v) for k, v in domain_name_to_id.items()}

        payload = _load_json_payload(self.pseudo_domain_json)
        manifest_data_root = payload.get("data_root") if isinstance(payload, dict) else None
        records = _iter_json_records(payload)

        ordered = sorted(self.domain_name_to_id.items(), key=lambda kv: int(kv[1]))
        self.all_domain_names: List[str] = [str(k) for k, _ in ordered]
        self.valid_domain_names: List[str] = list(self.all_domain_names)
        self.skipped_domains: Dict[str, str] = {}

        self.img_paths: List[str] = []
        self.gt_paths: List[object] = []
        self.labels: List[int] = []
        self.types: List[str] = []
        self.domain_names: List[str] = []
        self.domain_ids: List[int] = []

        seen = set()
        for rec in records:
            p = _resolve_record_path(rec, self.root, manifest_data_root=manifest_data_root)
            if not p or not os.path.isfile(p) or not _is_test_record(rec, p):
                continue
            p = _canon_path(p)
            if p in seen:
                continue
            seen.add(p)

            split = _record_split(rec)
            label, img_type = _label_type_from_test_path_or_split(p, split)
            gt = 0 if label == 0 else _find_mask_for_test_image(p, img_type, strict_mask=self.strict_mask)

            self.img_paths.append(p)
            self.gt_paths.append(gt)
            self.labels.append(label)
            self.types.append(img_type)
            self.domain_names.append("__unrouted__")
            self.domain_ids.append(-1)

        if len(self.img_paths) == 0:
            raise RuntimeError("No valid test samples could be built from pseudo-domain JSON.")

        assert (
            len(self.img_paths)
            == len(self.gt_paths)
            == len(self.labels)
            == len(self.types)
            == len(self.domain_names)
            == len(self.domain_ids)
        )

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx: int):
        img_path = self.img_paths[idx]
        gt = self.gt_paths[idx]
        label = self.labels[idx]
        img_type = self.types[idx]
        domain_name = self.domain_names[idx]
        domain_id = self.domain_ids[idx]

        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        if gt == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)

        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"
        return img, gt, label, img_type, domain_name, domain_id, img_path

def build_merged_domain_test_loader(
    root_path: str,
    domains_dir: str,
    domain_name_to_id: Dict[str, int],
    data_transform,
    gt_transform,
    batch_size: int = 8,
    num_workers: int = 0,
    pin_memory: bool = True,
    strict_mask: bool = True,
    pseudo_domain_json: str = "",
) -> torch.utils.data.DataLoader:
    if pseudo_domain_json:
        ds = MergedDomainTestDatasetFromJson(
            root=root_path,
            pseudo_domain_json=pseudo_domain_json,
            domain_name_to_id=domain_name_to_id,
            transform=data_transform,
            gt_transform=gt_transform,
            strict_mask=strict_mask,
        )
    else:
        ds = MergedDomainTestDatasetFromTxts(
            root=root_path,
            domains_dir=domains_dir,
            domain_name_to_id=domain_name_to_id,
            transform=data_transform,
            gt_transform=gt_transform,
            strict_mask=strict_mask,
        )

    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return loader


def get_domain_name_to_id(train_data, domains_dir: str = "", pseudo_domain_json: str = "") -> Dict[str, int]:
    """
    尽量从训练集对象拿到 domain_name -> domain_id 映射。
    JSON 模式下训练集会暴露 domain_name_to_id，因此不会依赖任何 txt 文件。
    """
    if hasattr(train_data, "domain_name_to_id"):
        d = getattr(train_data, "domain_name_to_id")
        if isinstance(d, dict) and len(d) > 0:
            return {str(k): int(v) for k, v in d.items()}

    if hasattr(train_data, "domains") and isinstance(getattr(train_data, "domains"), (list, tuple)):
        names = list(getattr(train_data, "domains"))
        return {str(n): int(i) for i, n in enumerate(names)}

    # fallback: sorted txt filenames，保留旧逻辑
    txts = list_domain_txts(domains_dir)
    names = [os.path.splitext(os.path.basename(x))[0] for x in txts]
    return {n: i for i, n in enumerate(names)}


# -------------------------
# Evaluation: conditional teacher & macro-avg
# -------------------------
@torch.no_grad()
def evaluation_conditional_teacher(
    encoder: nn.Module,
    bn: nn.Module,
    decoder: nn.Module,
    film: nn.Module,
    dataloader,
    device: str,
    domain_id: int,
) -> Tuple[float, float, float]:
    """
    teacher 评估：与 test.evaluation 同逻辑，只是 outputs = decoder( film(bn(inputs), domain_id) )
    """
    # 复用你项目里的 cal_anomaly_map / compute_pro，确保一致
    from test import cal_anomaly_map, compute_pro

    bn.eval()
    decoder.eval()
    film.eval()
    encoder.eval()

    gt_list_px, pr_list_px = [], []
    gt_list_sp, pr_list_sp = [], []
    aupro_list = []

    for img, gt, label, _ in dataloader:
        img = img.to(device, non_blocking=True)

        inputs = encoder(img)
        z = bn(inputs)
        dom = torch.full((img.shape[0],), int(domain_id), dtype=torch.long, device=device)
        zc = film(z, dom)
        outputs = decoder(zc)

        anomaly_map, _ = cal_anomaly_map(inputs, outputs, img.shape[-1], amap_mode='a')
        anomaly_map = gaussian_filter(anomaly_map, sigma=4)

        gt = gt.clone()
        gt[gt > 0.5] = 1
        gt[gt <= 0.5] = 0

        if label.item() != 0:
            try:
                aupro_list.append(
                    compute_pro(
                        gt.squeeze(0).cpu().numpy().astype(int),
                        anomaly_map[np.newaxis, :, :]
                    )
                )
            except Exception:
                pass

        gt_list_px.extend(gt.cpu().numpy().astype(int).ravel())
        pr_list_px.extend(anomaly_map.ravel())
        gt_list_sp.append(np.max(gt.cpu().numpy().astype(int)))
        pr_list_sp.append(np.max(anomaly_map))

    auroc_px = float(round(roc_auc_score(gt_list_px, pr_list_px), 3))
    auroc_sp = float(round(roc_auc_score(gt_list_sp, pr_list_sp), 3))
    aupro = float(np.mean(aupro_list)) if len(aupro_list) > 0 else float("nan")
    aupro = float(round(aupro, 3)) if (aupro == aupro) else float("nan")
    return auroc_px, auroc_sp, aupro


def _nanmean(xs: List[float]) -> float:
    xs2 = [x for x in xs if (x == x)]
    if len(xs2) == 0:
        return float("nan")
    return float(sum(xs2) / len(xs2))


def _init_metric_bucket() -> Dict[str, List[float]]:
    return {
        "gt_list_px": [],
        "pr_list_px": [],
        "gt_list_sp": [],
        "pr_list_sp": [],
        "aupro_list": [],
    }


def _finalize_metric_bucket(bucket: Dict[str, List[float]]) -> Tuple[float, float, float]:
    auroc_px = float(round(roc_auc_score(bucket["gt_list_px"], bucket["pr_list_px"]), 3))
    auroc_sp = float(round(roc_auc_score(bucket["gt_list_sp"], bucket["pr_list_sp"]), 3))
    aupro = float(np.mean(bucket["aupro_list"])) if len(bucket["aupro_list"]) > 0 else float("nan")
    aupro = float(round(aupro, 3)) if (aupro == aupro) else float("nan")
    return auroc_px, auroc_sp, aupro


@torch.no_grad()
def macro_eval_teacher_over_merged_loader(
    encoder: nn.Module,
    teacher: "ConditionalTeacher",
    domain_loader: torch.utils.data.DataLoader,
    device: str,
    print_per_domain: bool = False,
) -> Tuple[float, float, float, int, int]:
    """
    使用合并后的 domain test loader 进行评估。
    关键点：
      1) 模型前向按 batch 跑
      2) anomaly map / PRO / AUROC 统计按样本、按 domain 分桶
      3) 指标口径与原串行版保持一致
    """
    from test import cal_anomaly_map, compute_pro

    teacher.bn.eval()
    teacher.decoder.eval()
    teacher.film.eval()
    encoder.eval()

    dataset = domain_loader.dataset
    all_domain_names = list(getattr(dataset, "all_domain_names", []))
    valid_domain_names = list(getattr(dataset, "valid_domain_names", []))
    skipped_domains = dict(getattr(dataset, "skipped_domains", {}))

    buckets: Dict[str, Dict[str, List[float]]] = {
        name: _init_metric_bucket() for name in valid_domain_names
    }

    for batch in domain_loader:
        if len(batch) >= 7:
            img, gt, label, _, domain_name, domain_id, _img_path = batch
        else:
            img, gt, label, _, domain_name, domain_id = batch

        img = img.to(device, non_blocking=True)
        domain_id = domain_id.to(device, non_blocking=True).long()

        inputs = encoder(img)
        z = teacher.bn(inputs)
        zc = teacher.film(z, domain_id)
        outputs = teacher.decoder(zc)

        bs = img.shape[0]
        for i in range(bs):
            name_i = str(domain_name[i])
            if name_i not in buckets:
                continue

            inputs_i = [x[i:i + 1] for x in inputs]
            outputs_i = [x[i:i + 1] for x in outputs]

            anomaly_map, _ = cal_anomaly_map(inputs_i, outputs_i, img.shape[-1], amap_mode='a')
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

            bucket["gt_list_px"].extend(gt_np.ravel())
            bucket["pr_list_px"].extend(anomaly_map.ravel())
            bucket["gt_list_sp"].append(np.max(gt_np))
            bucket["pr_list_sp"].append(np.max(anomaly_map))

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
                dom_id = getattr(dataset, "domain_name_to_id", {}).get(name, None)
                if dom_id is None:
                    print(f"  [domain={name}] PixelAUROC={auroc_px:.3f} SampleAUROC={auroc_sp:.3f} PixelAUPRO={aupro_px:.3f}")
                else:
                    print(f"  [domain={name}] id={int(dom_id)} PixelAUROC={auroc_px:.3f} SampleAUROC={auroc_sp:.3f} PixelAUPRO={aupro_px:.3f}")
        except Exception as e:
            if print_per_domain:
                print(f"  [domain={name}] skipped: {e}")

    return _nanmean(pxs), _nanmean(sps), _nanmean(aps), n_valid, n_total


@torch.no_grad()
def macro_eval_teacher_over_txts(
    encoder: nn.Module,
    teacher: "ConditionalTeacher",
    domain_loaders: Dict[str, torch.utils.data.DataLoader],
    domain_name_to_id: Dict[str, int],
    device: str,
    print_per_domain: bool = False,
) -> Tuple[float, float, float, int, int]:
    pxs, sps, aps = [], [], []
    n_total = 0
    n_valid = 0

    for name, loader in domain_loaders.items():
        n_total += 1
        if name not in domain_name_to_id:
            if print_per_domain:
                print(f"  [domain={name}] skipped (no id mapping)")
            continue
        dom_id = int(domain_name_to_id[name])

        try:
            auroc_px, auroc_sp, aupro_px = evaluation_conditional_teacher(
                encoder, teacher.bn, teacher.decoder, teacher.film, loader, device, dom_id
            )
            pxs.append(auroc_px)
            sps.append(auroc_sp)
            aps.append(aupro_px)
            n_valid += 1
            if print_per_domain:
                print(f"  [domain={name}] id={dom_id} PixelAUROC={auroc_px:.3f} SampleAUROC={auroc_sp:.3f} PixelAUPRO={aupro_px:.3f}")
        except Exception as e:
            if print_per_domain:
                print(f"  [domain={name}] skipped: {e}")

    return _nanmean(pxs), _nanmean(sps), _nanmean(aps), n_valid, n_total


@torch.no_grad()
def macro_eval_student_over_txts(
    evaluation_fn,
    encoder: nn.Module,
    bn: nn.Module,
    decoder: nn.Module,
    domain_loaders: Dict[str, torch.utils.data.DataLoader],
    device: str,
    print_per_domain: bool = False,
) -> Tuple[float, float, float, int, int]:
    pxs, sps, aps = [], [], []
    n_total = 0
    n_valid = 0

    for name, loader in domain_loaders.items():
        n_total += 1
        try:
            auroc_px, auroc_sp, aupro_px = evaluation_fn(encoder, bn, decoder, loader, device)
            pxs.append(float(auroc_px))
            sps.append(float(auroc_sp))
            aps.append(float(aupro_px))
            n_valid += 1
            if print_per_domain:
                print(f"  [domain={name}] PixelAUROC={auroc_px:.3f} SampleAUROC={auroc_sp:.3f} PixelAUPRO={aupro_px:.3f}")
        except Exception as e:
            if print_per_domain:
                print(f"  [domain={name}] skipped: {e}")

    return _nanmean(pxs), _nanmean(sps), _nanmean(aps), n_valid, n_total