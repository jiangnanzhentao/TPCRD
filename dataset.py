from torchvision import transforms
from PIL import Image
import os
import json
import torch
import glob
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST, ImageFolder
import numpy as np
from typing import Dict, List, Tuple, Optional


def get_data_transforms(size, isize):
    mean_train = [0.485, 0.456, 0.406]
    std_train = [0.229, 0.224, 0.225]
    data_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.CenterCrop(isize),
        transforms.Normalize(mean=mean_train, std=std_train)
    ])
    gt_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.CenterCrop(isize),
        transforms.ToTensor()
    ])
    return data_transforms, gt_transforms


class MVTecDataset(torch.utils.data.Dataset):
    """
    原 RD4AD 的 MVTecDataset：用于 test（返回 4 个值）
    返回: img, gt, label(0 good / 1 anomaly), img_type
    """
    def __init__(self, root, transform, gt_transform, phase):
        if phase == 'train':
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'test')
            self.gt_path = os.path.join(root, 'ground_truth')

        self.transform = transform
        self.gt_transform = gt_transform

        # load dataset
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()

    def load_dataset(self):
        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                gt_paths = glob.glob(os.path.join(self.gt_path, defect_type) + "/*.png")
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"
        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        if gt == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)

        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"
        return img, gt, label, img_type



# ----------------------------
#  伪域标签读取 + 训练集 Dataset
# ----------------------------

def _canon_path(p: str) -> str:
    # 统一成绝对路径 + normpath，并把 / \ 归一化
    p = str(p).strip().replace("\\", os.sep).replace("/", os.sep)
    return os.path.abspath(os.path.normpath(p))


def _tail_from_train_test(abs_path: str) -> Optional[str]:
    """
    从路径中截取 'train/...' 或 'test/...' 作为鲁棒匹配 key。
    用于解决 txt/json 里路径前缀与当前机器数据根目录不完全一致的问题。
    """
    parts = _canon_path(abs_path).split(os.sep)
    for k in ("train", "test"):
        if k in parts:
            i = parts.index(k)
            return os.path.join(*parts[i:])  # e.g. train/good/xxx.png
    return None


def _read_txt_lines(fp: str) -> List[str]:
    """
    兼容常见编码：utf-8/utf-8-sig/gbk。逐行返回非空行。
    """
    encodings = ("utf-8-sig", "utf-8", "gbk")
    last_err = None
    for enc in encodings:
        try:
            with open(fp, "r", encoding=enc) as f:
                lines = []
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith("#"):
                        continue
                    lines.append(line)
                return lines
        except Exception as e:
            last_err = e
    raise last_err


def _safe_int(v, default: int = -1) -> int:
    try:
        if v is None:
            return default
        if isinstance(v, str) and v.strip() == "":
            return default
        return int(float(v))
    except Exception:
        return default


def _load_json_payload(json_path: str):
    json_path = _canon_path(json_path)
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"pseudo_domain_json not found: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _iter_json_records(payload) -> List[dict]:
    """
    兼容 pseudo-domain_discover_json.py 导出的几种 JSON：
      - training_manifest.json: dict，含 samples
      - domain_groups.json: dict，每个 cluster 下含 samples
      - train_good_clustered.json / infer_test_*.json: list[dict]
    """
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
    """
    尽量把 JSON 中的路径解析到当前机器的数据文件：
      1) 原路径/绝对路径存在则直接使用；
      2) 若路径中含 train/test 尾巴，则接到 root 或 root 的上一级；
      3) 若 JSON 里有 data_root，也尝试 data_root 及其上一级；
      4) 最后返回规范化后的第一个候选路径，便于报错定位。
    """
    root = _canon_path(root) if root else os.getcwd()
    base_roots = [root, os.path.dirname(root), os.getcwd()]
    if manifest_data_root:
        mroot = _canon_path(manifest_data_root)
        base_roots.extend([mroot, os.path.dirname(mroot)])

    # 去重，保留顺序
    uniq_bases = []
    seen = set()
    for b in base_roots:
        if b and b not in seen:
            uniq_bases.append(b)
            seen.add(b)

    raw_candidates = _path_candidates_from_record(rec)
    first_fallback = ""

    for raw in raw_candidates:
        raw_norm = raw.replace("\\", os.sep).replace("/", os.sep)

        # 绝对/相对原样检查
        p0 = _canon_path(raw_norm)
        if not first_fallback:
            first_fallback = p0
        if os.path.isfile(p0):
            return p0

        # 直接拼到候选根
        for base in uniq_bases:
            p1 = _canon_path(os.path.join(base, raw_norm))
            if os.path.isfile(p1):
                return p1

        # 截取 train/test 尾巴后拼根目录。多类别全局 JSON 中不同类别会有相同
        # train/good/000.png，因此优先尝试 source_class/class_name + tail，避免跨类别误匹配。
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


def _split_allowed(split: str, include_splits: Optional[Tuple[str, ...]]) -> bool:
    if include_splits is None:
        return True
    split = split.replace("\\", "/")
    return split in {s.replace("\\", "/") for s in include_splits}


def build_pseudo_domain_maps_from_json(
    json_path: str,
    root: str,
    include_splits: Optional[Tuple[str, ...]] = ("train/good",),
) -> Tuple[Dict[str, int], Dict[str, int], List[str], Dict[str, int], Dict[int, int]]:
    """
    从 JSON manifest 建立训练伪域映射，不要求把图片移动到同一个数据目录，也不生成/读取 txt。

    返回：
      abs_map: 当前机器绝对路径 -> compact domain_id
      tail_map: train/test 尾巴 -> compact domain_id
      domain_names: domain_id -> domain_{cluster_id}
      domain_name_to_id: domain_{cluster_id} -> compact domain_id
      cluster_to_domain_id: 原 cluster_id -> compact domain_id
    """
    payload = _load_json_payload(json_path)
    records = _iter_json_records(payload)
    manifest_data_root = payload.get("data_root") if isinstance(payload, dict) else None

    selected = []
    for rec in records:
        split = _record_split(rec)
        if not _split_allowed(split, include_splits):
            continue
        cid = _record_cluster_id(rec)
        if cid == -1:
            continue
        p = _resolve_record_path(rec, root=root, manifest_data_root=manifest_data_root)
        if not p or not os.path.isfile(p):
            continue
        selected.append((p, cid))

    if len(selected) == 0:
        raise RuntimeError(
            "No valid labeled training samples were found in pseudo_domain_json. "
            f"json_path={json_path}, include_splits={include_splits}"
        )

    cluster_ids = sorted({cid for _, cid in selected})
    cluster_to_domain_id = {cid: i for i, cid in enumerate(cluster_ids)}
    domain_names = [_domain_name_from_cluster(cid) for cid in cluster_ids]
    domain_name_to_id = {name: i for i, name in enumerate(domain_names)}

    abs_map: Dict[str, int] = {}
    tail_map: Dict[str, int] = {}

    for path, cid in selected:
        domain_id = cluster_to_domain_id[cid]
        ap = _canon_path(path)

        if ap in abs_map and abs_map[ap] != domain_id:
            raise ValueError(f"Duplicate image assigned to multiple JSON pseudo-domains: {ap}")
        abs_map[ap] = domain_id

        tail = _tail_from_train_test(ap)
        if tail is not None:
            # 多类别 MVTec 常见同名文件，例如 carpet/train/good/000.png 与
            # grid/train/good/000.png 的 tail 都是 train/good/000.png。JSON 模式训练
            # 直接使用绝对路径列表，不依赖 tail_map；这里对冲突 tail 做歧义丢弃，
            # 避免把不同类别的同名图片错误绑定到同一伪域。
            if tail in tail_map and tail_map[tail] != domain_id:
                tail_map.pop(tail, None)
            else:
                tail_map[tail] = domain_id

    return abs_map, tail_map, domain_names, domain_name_to_id, cluster_to_domain_id


def build_pseudo_domain_maps(domain_dir: str) -> Tuple[Dict[str, int], Dict[str, int], List[str]]:
    """
    兼容旧逻辑：读取 outputs/domains 下的若干 txt。
      - 每个 txt 文件代表一个伪域类别
      - txt 内每行是该伪域下的图片路径（整行即路径，文件名可含空格）
    """
    domain_dir = _canon_path(domain_dir)
    txt_files = sorted(glob.glob(os.path.join(domain_dir, "*.txt")))
    if len(txt_files) == 0:
        raise FileNotFoundError(f"No .txt files found in domain_dir={domain_dir}")

    abs_map: Dict[str, int] = {}
    tail_map: Dict[str, int] = {}
    domain_names: List[str] = []

    for domain_id, fp in enumerate(txt_files):
        name = os.path.splitext(os.path.basename(fp))[0]
        domain_names.append(name)

        for raw_line in _read_txt_lines(fp):
            ap = _canon_path(raw_line)

            # 同一图片被分到多个域，直接报错（避免 silent bug）
            if ap in abs_map and abs_map[ap] != domain_id:
                raise ValueError(f"Duplicate image in multiple domains: {ap}")
            abs_map[ap] = domain_id

            tail = _tail_from_train_test(ap)
            if tail is not None:
                if tail in tail_map and tail_map[tail] != domain_id:
                    raise ValueError(f"Duplicate tail in multiple domains: {tail}")
                tail_map[tail] = domain_id

    return abs_map, tail_map, domain_names


class MVTecTrainPseudoDomainDataset(torch.utils.data.Dataset):
    """
    训练用 Dataset：返回 (img, domain_id)

    新逻辑：
    - 若传入 pseudo_domain_json，则直接从 JSON 中读取 train/good 样本路径与 cluster_id；
    - 不扫描公共数据集根目录，不要求把多类图像移动到同一个 class/train/good 下；
    - domain_id 仍然压缩为 0..num_domains-1，确保 FiLM / GroupDRO 逻辑不变。

    旧逻辑：
    - 未传 pseudo_domain_json 时，仍按 outputs/domains/*.txt 建立伪域标签。
    """
    def __init__(
        self,
        root: str,
        transform,
        domain_dir: str = "outputs/domains",
        pseudo_domain_json: str = "",
        json_train_splits: Tuple[str, ...] = ("train/good",),
        img_exts: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp"),
        strict: bool = False,   # 保留参数以兼容旧调用
        drop_unlabeled: bool = True,
        default_domain_id: int = 0
    ):
        super().__init__()
        self.root = _canon_path(root)
        self.transform = transform
        self.strict = strict
        self.drop_unlabeled = drop_unlabeled
        self.default_domain_id = default_domain_id
        self.pseudo_domain_json = str(pseudo_domain_json or "")
        self.domain_source = "json" if self.pseudo_domain_json else "txt"

        self.domain_name_to_id: Dict[str, int] = {}
        self.cluster_to_domain_id: Dict[int, int] = {}
        self.domain_id_to_cluster: Dict[int, int] = {}

        if self.pseudo_domain_json:
            (
                self.abs_map,
                self.tail_map,
                self.domain_names,
                self.domain_name_to_id,
                self.cluster_to_domain_id,
            ) = build_pseudo_domain_maps_from_json(
                json_path=self.pseudo_domain_json,
                root=self.root,
                include_splits=json_train_splits,
            )
            self.domain_id_to_cluster = {int(v): int(k) for k, v in self.cluster_to_domain_id.items()}
            self.num_domains = len(self.domain_names)
            self.img_paths = sorted(self.abs_map.keys())
            self.domain_ids = [self.abs_map[p] for p in self.img_paths]
            print(
                f"[PseudoDomainDataset:JSON] loaded {len(self.img_paths)} train samples "
                f"from {self.pseudo_domain_json}; num_domains={self.num_domains}."
            )
            return

        self.abs_map, self.tail_map, self.domain_names = build_pseudo_domain_maps(domain_dir)
        self.num_domains = len(self.domain_names)
        self.domain_name_to_id = {str(name): int(i) for i, name in enumerate(self.domain_names)}

        train_root = os.path.join(self.root, "train")
        if not os.path.isdir(train_root):
            raise FileNotFoundError(f"train folder not found: {train_root}")

        img_paths: List[str] = []
        for ext in img_exts:
            img_paths.extend(glob.glob(os.path.join(train_root, "**", f"*{ext}"), recursive=True))
        img_paths = sorted(set(map(_canon_path, img_paths)))

        if len(img_paths) == 0:
            raise FileNotFoundError(f"No training images found under: {train_root}")

        raw_domain_ids = [self._get_domain_id(p) for p in img_paths]

        labeled = [(p, d) for p, d in zip(img_paths, raw_domain_ids) if d >= 0]
        unlabeled = [p for p, d in zip(img_paths, raw_domain_ids) if d < 0]

        if drop_unlabeled:
            if len(labeled) == 0:
                raise RuntimeError("No labeled training samples remain after dropping unlabeled images.")
            self.img_paths = [p for p, _ in labeled]
            self.domain_ids = [d for _, d in labeled]
            print(f"[PseudoDomainDataset:TXT] dropped {len(unlabeled)} unlabeled images, kept {len(self.img_paths)} images.")
        else:
            self.img_paths = img_paths
            self.domain_ids = [
                d if d >= 0 else default_domain_id
                for d in raw_domain_ids
            ]
            print(f"[PseudoDomainDataset:TXT] assigned default_domain_id={default_domain_id} to {len(unlabeled)} unlabeled images.")

    def _get_domain_id(self, img_path: str) -> int:
        ap = _canon_path(img_path)
        if ap in self.abs_map:
            return self.abs_map[ap]

        tail = _tail_from_train_test(ap)
        if tail is not None and tail in self.tail_map:
            return self.tail_map[tail]

        return -1

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx: int):
        img_path = self.img_paths[idx]
        domain_id = self.domain_ids[idx]

        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        domain_id = torch.tensor(domain_id, dtype=torch.long)
        return img, domain_id


# ----------------------------
# 原 load_data 保持不变（MNIST/CIFAR10/FashionMNIST/retina）
# ----------------------------
def load_data(dataset_name='mnist', normal_class=0, batch_size='16'):

    if dataset_name == 'cifar10':
        img_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

        os.makedirs("./Dataset/CIFAR10/train", exist_ok=True)
        dataset = CIFAR10('./Dataset/CIFAR10/train', train=True, download=True, transform=img_transform)
        print("Cifar10 DataLoader Called...")
        print("All Train Data: ", dataset.data.shape)
        dataset.data = dataset.data[np.array(dataset.targets) == normal_class]
        dataset.targets = [normal_class] * dataset.data.shape[0]
        print("Normal Train Data: ", dataset.data.shape)

        os.makedirs("./Dataset/CIFAR10/test", exist_ok=True)
        test_set = CIFAR10("./Dataset/CIFAR10/test", train=False, download=True, transform=img_transform)
        print("Test Train Data:", test_set.data.shape)

    elif dataset_name == 'mnist':
        img_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ])

        os.makedirs("./Dataset/MNIST/train", exist_ok=True)
        dataset = MNIST('./Dataset/MNIST/train', train=True, download=True, transform=img_transform)
        print("MNIST DataLoader Called...")
        print("All Train Data: ", dataset.data.shape)
        dataset.data = dataset.data[np.array(dataset.targets) == normal_class]
        dataset.targets = [normal_class] * dataset.data.shape[0]
        print("Normal Train Data: ", dataset.data.shape)

        os.makedirs("./Dataset/MNIST/test", exist_ok=True)
        test_set = MNIST("./Dataset/MNIST/test", train=False, download=True, transform=img_transform)
        print("Test Train Data:", test_set.data.shape)

    elif dataset_name == 'fashionmnist':
        img_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ])

        os.makedirs("./Dataset/FashionMNIST/train", exist_ok=True)
        dataset = FashionMNIST('./Dataset/FashionMNIST/train', train=True, download=True, transform=img_transform)
        print("FashionMNIST DataLoader Called...")
        print("All Train Data: ", dataset.data.shape)
        dataset.data = dataset.data[np.array(dataset.targets) == normal_class]
        dataset.targets = [normal_class] * dataset.data.shape[0]
        print("Normal Train Data: ", dataset.data.shape)

        os.makedirs("./Dataset/FashionMNIST/test", exist_ok=True)
        test_set = FashionMNIST("./Dataset/FashionMNIST/test", train=False, download=True, transform=img_transform)
        print("Test Train Data:", test_set.data.shape)

    elif dataset_name == 'retina':
        data_path = 'Dataset/OCT2017/train'
        orig_transform = transforms.Compose([
            transforms.Resize([128, 128]),
            transforms.ToTensor()
        ])

        dataset = ImageFolder(root=data_path, transform=orig_transform)

        test_data_path = 'Dataset/OCT2017/test'
        test_set = ImageFolder(root=test_data_path, transform=orig_transform)

    else:
        raise Exception(
            "You enter {} as dataset, which is not a valid dataset for this repository!".format(dataset_name)
        )

    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
    )
    return train_dataloader, test_dataloader