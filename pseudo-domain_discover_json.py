#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-end pipeline:
1) 原味无监督聚类（train/good；含：噪声二次聚类 + refine软加入 + 概率补丁合并）
2) 保存聚类模型 artifacts，用于后续实时聚类路由
3) domains/ 按簇写 domain_{cluster_id}.txt：
   - 聚类完成先写入 train/good（覆盖）
   - 路径写为相对到 data_root 的上一级目录（Windows 反斜杠），不写 -1 噪声

输出（默认在 --out_dir 下）：
- cluster/: scaler.joblib, pca.joblib, hdbscan.joblib, centers.npy, valid_ids.npy, model_meta.json, train_good.csv, summary.json
- domains/domain_{cluster_id}.txt

依赖：torch, torchvision, scikit-learn, hdbscan, scikit-image, opencv-python, pandas, tqdm, joblib
"""

import os, glob, json, csv, argparse
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import cv2
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import vgg19, VGG19_Weights

from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.color import rgb2lab
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
import hdbscan
from joblib import dump, load
from scipy.spatial.distance import cdist

                                   
def set_seed(seed=42):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def list_images(folder: str)->List[str]:
    exts=("*.png","*.jpg","*.jpeg","*.bmp","*.tif","*.tiff"); paths=[]
    for e in exts: paths.extend(glob.glob(os.path.join(folder,"**",e), recursive=True))
    return sorted(paths)

def read_rgb(path: str)->np.ndarray:
    data=np.fromfile(path, dtype=np.uint8)
    img=cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None: img=cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None: raise RuntimeError(f"Failed to read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def safe_imread_rgb(path: str) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    img  = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None: img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None: raise FileNotFoundError(f"Failed to read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def pil_from_rgb(rgb: np.ndarray) -> Image.Image:
    return Image.fromarray(rgb)

def to_backslash(p:str)->str:
    return p.replace('/', '\\')

def rel_to_parent(p: str, root: str)->str:
    """返回相对到 data_root 的上一级目录的路径（Windows 反斜杠）"""
    parent=os.path.dirname(os.path.abspath(root))
    ap=os.path.abspath(p)
    try: rel=os.path.relpath(ap, parent)
    except Exception: rel=p
    return to_backslash(rel)

def retinex_simplified(rgb: np.ndarray)->np.ndarray:
    eps=1e-6
    blur=cv2.GaussianBlur(rgb.astype(np.float32)+eps,(0,0),15)
    out=np.log(rgb.astype(np.float32)+eps)-np.log(blur)
    out=out-out.min(); out=out/(out.max()+1e-6)
    return (out*255).clip(0,255).astype(np.uint8)


                                                              
def texture_mask(rgb: np.ndarray, win:int=31, sat_thr:float=0.05, var_scale:float=0.5,
                 sobel_scale:float=1.5)->np.ndarray:
    gray=cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)/255.0
    k=max(5, win|1)
    m=cv2.blur(gray,(k,k)); m2=cv2.blur(gray*gray,(k,k))
    var=np.maximum(0.0, m2 - m*m)
    v_med=float(np.median(var)); v_std=float(np.std(var)+1e-6)
    mask_var = var > (v_med + var_scale * v_std)

    hsv=cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    sat=(hsv[...,1].astype(np.float32))/255.0
    mask_sat = sat > sat_thr

    gx=cv2.Sobel(gray, cv2.CV_32F, 1,0, ksize=3)
    gy=cv2.Sobel(gray, cv2.CV_32F, 0,1, ksize=3)
    mag=cv2.magnitude(gx,gy)
    mag=cv2.GaussianBlur(mag,(0,0),3)
    g_thr = max(5.0/255.0, float(np.median(mag))*sobel_scale)
    mask_grad = mag > g_thr

    mask = (mask_var | mask_grad) & mask_sat

    mask = (mask.astype(np.uint8)*255)
    k2=max(3,(k//2)|1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((k2,k2),np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((k2,k2),np.uint8))

              
    num, lbl = cv2.connectedComponents(mask)
    if num>1:
        areas=[int((lbl==i).sum()) for i in range(1,num)]
        best=1+int(np.argmax(areas))
        mask=(lbl==best).astype(np.uint8)*255
    return mask>0

def sample_patches_masked(rgb: np.ndarray, mask: np.ndarray, n_patches:int, patch:int)->List[np.ndarray]:
    H,W=mask.shape; out=[]; need=n_patches
    cov = cv2.boxFilter(mask.astype(np.float32), ddepth=-1, ksize=(patch,patch), normalize=True)
    cov = np.pad(cov, ((patch//2,patch//2),(patch//2,patch//2)), mode='constant')
    cov = cov[0:H,0:W]
    ys,xs = np.where(cov >= 0.7)
    idx = np.arange(len(ys))
    if len(idx)==0: return []
    np.random.shuffle(idx)
    for t in idx:
        y=int(ys[t]); x=int(xs[t])
        y0=max(0, y-patch//2); x0=max(0, x-patch//2)
        if y0+patch>H or x0+patch>W: continue
        crop=rgb[y0:y0+patch, x0:x0+patch]
        out.append(crop)
        if len(out)>=need: break
    return out


                                                              
class VGGFeat(nn.Module):
    def __init__(self, layers=("features.2","features.7","features.16","features.25","features.34")):
        super().__init__()
        m=vgg19(weights=VGG19_Weights.IMAGENET1K_V1); m.eval()
        for p in m.parameters(): p.requires_grad=False
        self.backbone=m; self.layers=layers
    def forward(self,x):
        feats={}; cur=x
        for name,layer in self.backbone.features._modules.items():
            cur=layer(cur); key=f"features.{name}"
            if key in self.layers: feats[key]=cur
        return feats

@torch.no_grad()
def extract_deep_stats_batch(model: VGGFeat, patches_rgb: List[np.ndarray], device="cpu"):
    if len(patches_rgb)==0: return np.zeros((0,1),dtype=np.float32)
    tfm=T.Compose([T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    B=32; out=[]
    for i in range(0,len(patches_rgb),B):
        chunk=patches_rgb[i:i+B]
        tens=torch.stack([tfm(Image.fromarray(p)) for p in chunk],0).to(device).float()
        feats=model(tens); stats=[]
        for k in model.layers:
            f=feats[k]; mean=f.mean(dim=(2,3)); std=f.std(dim=(2,3))
            stats.append(mean); stats.append(std)
        vec=torch.cat(stats,dim=1).detach().cpu().numpy()
        out.append(vec)
    return np.concatenate(out,axis=0).astype(np.float32)

def deep_feats_for_patches(model: VGGFeat, patch_list: List[np.ndarray], rotations:int, device:str)->np.ndarray:
    ks=[0] if rotations==1 else [0,1,2,3]
    rot_patches=[]
    for p in patch_list:
        for k in ks: rot_patches.append(np.rot90(p,k))
    all_vecs=extract_deep_stats_batch(model, rot_patches, device=device)
    R=len(ks); D=all_vecs.shape[1] if all_vecs.ndim==2 else 1
    if all_vecs.shape[0]!=len(patch_list)*R:
        return np.zeros((len(patch_list),D),dtype=np.float32)
    return all_vecs.reshape(len(patch_list),R,D).mean(axis=1)

def lbp_hist(gray: np.ndarray, P=8, R=1.0, bins=59)->np.ndarray:
    lbp=local_binary_pattern(gray,P=P,R=R,method='uniform')
    hist,_=np.histogram(lbp,bins=bins,range=(0,bins),density=True)
    return hist.astype(np.float32)

def glcm_feats(gray: np.ndarray, distances=None, angles=None)->np.ndarray:
    if distances is None: distances=[1,2,4]
    if angles is None: angles=[0,np.pi/4,np.pi/2,3*np.pi/4]
    g=(gray/8).astype(np.uint8)
    gl=graycomatrix(g, distances=distances, angles=angles, levels=32, symmetric=True, normed=True)
    props=['contrast','dissimilarity','homogeneity','energy','correlation','ASM']
    feats=[graycoprops(gl,p).ravel() for p in props]
    return np.concatenate(feats).astype(np.float32)

def radial_power_spectrum(gray: np.ndarray, n_bins=32)->np.ndarray:
    F=np.fft.fftshift(np.abs(np.fft.fft2(gray))); H,W=F.shape; cy,cx=H//2,W//2
    Y,X=np.ogrid[:H,:W]; R=np.sqrt((Y-cy)**2+(X-cx)**2); max_r=int(R.max())
    bins=np.linspace(0,max_r,n_bins+1); hist=np.zeros(n_bins,dtype=np.float64)
    for i in range(n_bins):
        m=(R>=bins[i])&(R<bins[i+1])
        if m.any(): hist[i]=F[m].mean()
    hist=hist/(hist.sum()+1e-8)
    return hist.astype(np.float32)

def lab_color_stats(rgb: np.ndarray)->np.ndarray:
    lab=rgb2lab(rgb/255.0); a=lab[...,1].astype(np.float32); b=lab[...,2].astype(np.float32)
    return np.array([a.mean(),a.std(),b.mean(),b.std()],dtype=np.float32)

def extract_traditional_feats(patch_rgb: np.ndarray)->np.ndarray:
    gray=cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2GRAY)
    lbp=lbp_hist(gray); gl=glcm_feats(gray); rps=radial_power_spectrum(gray); lab=lab_color_stats(patch_rgb)
    return np.concatenate([lbp,gl,rps,lab]).astype(np.float32)


                                                              
def geometric_median(X: np.ndarray, eps=1e-5, max_iter=200)->np.ndarray:
    y=X.mean(axis=0)
    for _ in range(max_iter):
        D=np.linalg.norm(X-y,axis=1)+1e-12; W=1.0/D; T=(X*W[:,None]).sum(axis=0)/W.sum()
        if np.linalg.norm(y-T)<eps: return T
        y=T
    return y

def patch_aggregate(features: np.ndarray)->Tuple[np.ndarray,float]:
    med=geometric_median(features)
    d=np.linalg.norm(features-med[None,:],axis=1)
    mad=np.median(np.abs(d-np.median(d)))
    return med.astype(np.float32), float(mad)

def build_image_feature_from_rgb(
    rgb: np.ndarray, vgg:VGGFeat, device:str, n_patches:int, patch:int, scales:List[float],
    use_retinex=True, rotations=4, use_mask=True
)->Tuple[np.ndarray,float]:
    if use_retinex: rgb=retinex_simplified(rgb)
    base_mask = texture_mask(rgb, win=max(31, (patch//3)|1)) if use_mask else np.ones(rgb.shape[:2],dtype=bool)
    feats=[]
    for s in scales:
        h=int(rgb.shape[0]*s); w=int(rgb.shape[1]*s)
        res=cv2.resize(rgb,(w,h),interpolation=cv2.INTER_AREA) if s!=1.0 else rgb
        msk=cv2.resize(base_mask.astype(np.uint8)*255,(w,h),interpolation=cv2.INTER_NEAREST)>0
        patches=sample_patches_masked(res, msk, n_patches=max(1,n_patches//len(scales)), patch=patch)
        if len(patches)==0:
            for _ in range(max(1,n_patches//len(scales))):
                y=np.random.randint(0,h-patch+1); x=np.random.randint(0,w-patch+1)
                patches.append(res[y:y+patch,x:x+patch])
        deep_arr=deep_feats_for_patches(vgg, patches, rotations, device)
        trad_arr=np.stack([extract_traditional_feats(p) for p in patches],axis=0)
        feats.append(np.concatenate([deep_arr,trad_arr],axis=1))
    feats_arr=np.concatenate(feats,axis=0) if len(feats)>0 else np.zeros((1,1),dtype=np.float32)
    agg,purity=patch_aggregate(feats_arr)
    return agg,purity

def build_image_feature(path:str, vgg:VGGFeat, device:str, n_patches:int, patch:int, scales:List[float],
                        use_retinex=True, rotations=4, use_mask=True)->Tuple[np.ndarray,float]:
    rgb=read_rgb(path)
    return build_image_feature_from_rgb(rgb, vgg, device, n_patches, patch, scales, use_retinex, rotations, use_mask)


                                                             
def fit_pca(X: np.ndarray, out_dim=256):
    ss=StandardScaler(); Xn=ss.fit_transform(X)
    eff_dim = int(min(out_dim, Xn.shape[0]-1, Xn.shape[1])); eff_dim=max(2, eff_dim)
    pca=PCA(n_components=eff_dim, svd_solver='auto', random_state=42)
    Z=pca.fit_transform(Xn); return ss,pca,Z

def run_hdbscan(Z: np.ndarray, min_cluster_size=20, min_samples=5, metric='euclidean'):
    clusterer=hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,
                              metric=metric, cluster_selection_method='eom', prediction_data=True)
    labels=clusterer.fit_predict(Z)
    prob=getattr(clusterer,'probabilities_', np.ones(len(Z),dtype=np.float32))
    return clusterer, labels, prob

def compute_centers(Z: np.ndarray, y: np.ndarray)->Tuple[np.ndarray,List[int]]:
    valid_ids=sorted([c for c in set(y) if c!=-1]); centers=[]
    for c in valid_ids: centers.append(np.median(Z[y==c],axis=0))
    centers_arr=np.stack(centers,axis=0) if len(centers)>0 else np.zeros((0,Z.shape[1]),dtype=np.float32)
    return centers_arr, valid_ids

def assign_to_centers(Z: np.ndarray, centers: np.ndarray, valid_ids: List[int])->Tuple[np.ndarray,np.ndarray]:
    if Z.shape[0]==0 or centers.shape[0]==0:
        return np.array([],dtype=np.int32), np.array([],dtype=np.float32)
    D=cdist(Z, centers, metric='euclidean')
    lbl=D.argmin(axis=1); conf=1.0/(D.min(axis=1)+1e-6)
    mapped=np.array([valid_ids[i] for i in lbl], dtype=np.int32)
    return mapped, conf.astype(np.float32)


                                                             


def _membership_matrix_or_hard(clusterer, Z: np.ndarray, y_ref: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    valid_ids = sorted([c for c in set(y_ref) if c != -1])
    if len(valid_ids) == 0:
        return np.zeros((Z.shape[0], 0), dtype=np.float32), valid_ids
    try:
        M = hdbscan.all_points_membership_vectors(clusterer)
        M = np.asarray(M, dtype=np.float32)
        if M.ndim != 2 or M.shape[0] != Z.shape[0] or M.shape[1] == 0:
            raise ValueError(f"invalid membership matrix shape: {M.shape}")
        return M, valid_ids
    except Exception:
        id2col = {cid: i for i, cid in enumerate(valid_ids)}
        M = np.zeros((Z.shape[0], len(valid_ids)), dtype=np.float32)
        for i, yy in enumerate(y_ref):
            if yy != -1 and yy in id2col:
                M[i, id2col[yy]] = 1.0
        return M, valid_ids

def soft_join(Z: np.ndarray, y: np.ndarray, clusterer: hdbscan.HDBSCAN,
              tau_high=0.6, tau_low=0.3, nn_k=10, nn_ratio=0.7)\
        ->Tuple[np.ndarray, np.ndarray, List[int]]:
    if Z.shape[0] == 0:
        return y, np.zeros((0, Z.shape[1]), dtype=np.float32), []
    y_ref = y.copy()
    M, valid_ids = _membership_matrix_or_hard(clusterer, Z, y_ref)
    max_prob = M.max(axis=1) if M.ndim == 2 and M.size else np.zeros(Z.shape[0], dtype=np.float32)
    is_conf = (y_ref != -1) & (max_prob >= tau_high)
    is_amb = (y_ref == -1) & (max_prob >= tau_low)
    if is_amb.any() and is_conf.any():
        nn = NearestNeighbors(n_neighbors=min(nn_k, int(is_conf.sum())), metric='euclidean').fit(Z[is_conf])
        _, I = nn.kneighbors(Z[is_amb], return_distance=True)
        conf_labels = y_ref[is_conf]
        amb_idx = np.where(is_amb)[0]
        for k, idxs in enumerate(I):
            labs = conf_labels[idxs]
            vals, cnts = np.unique(labs, return_counts=True)
            j = int(cnts.argmax())
            maj, ratio = vals[j], cnts[j] / len(labs)
            if ratio >= nn_ratio:
                y_ref[amb_idx[k]] = maj
    w = np.zeros(Z.shape[0], dtype=np.float32)
    w[is_conf] = 1.0
    if is_amb.any() and M.ndim == 2 and M.size:
        w_amb = (np.clip(max_prob[is_amb], tau_low, tau_high) - tau_low) / max(1e-6, (tau_high - tau_low))
        w[is_amb] = w_amb
    centers = []
    for cid in valid_ids:
        m = (y_ref == cid)
        if m.sum() == 0:
            centers.append(np.zeros((Z.shape[1],), dtype=np.float32))
            continue
        ZZ = Z[m]
        ww = w[m]
        c = (ZZ * ww[:, None]).sum(axis=0) / (ww.sum() + 1e-6) if ww.sum() > 1e-6 else np.median(ZZ, axis=0)
        centers.append(c)
    centers_arr = np.stack(centers, axis=0) if len(centers) else np.zeros((0, Z.shape[1]), dtype=np.float32)
    return y_ref, centers_arr, valid_ids

def recluster_noise(Z: np.ndarray, y: np.ndarray, min_cluster_size:int, min_samples:int)\
        -> Tuple[np.ndarray, np.ndarray]:
    mask = (y == -1)
    y_out = y.copy()
    prob_patch = np.full(y.shape[0], np.nan, dtype=np.float32)
    if mask.sum() < max(5, min_cluster_size):
        return y_out, prob_patch
    clusterer2 = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,
                                 metric='euclidean', cluster_selection_method='eom', prediction_data=True)
    y2 = clusterer2.fit_predict(Z[mask])
    if (y2 != -1).sum() == 0:
        return y_out, prob_patch
    new_lbl_start = (max([c for c in set(y) if c != -1]) + 1) if (y != -1).any() else 0
    idx_noise = np.where(mask)[0]
    probs2 = getattr(clusterer2, 'probabilities_', np.ones(len(y2), dtype=np.float32))
    cur = new_lbl_start
    for c in sorted(set(y2)):
        if c == -1: continue
        sub = (y2 == c)
        y_out[idx_noise[sub]] = cur
        prob_patch[idx_noise[sub]] = probs2[sub].astype(np.float32)
        cur += 1
    return y_out, prob_patch


                                                             
def write_domains_for_train(domain_dir: Path, data_root: str, paths: List[str], labels: np.ndarray):
    """按簇覆盖写入 train/good（不写 -1）"""
    domain_dir.mkdir(parents=True, exist_ok=True)
    buckets: Dict[int, List[str]] = {}
    for p, c in zip(paths, labels):
        c = int(c)
        if c == -1:              
            continue
        buckets.setdefault(c, []).append(rel_to_parent(p, data_root))
    for cid, plist in buckets.items():
        fpath = domain_dir / f"domain_{cid}.txt"
        with open(fpath, 'w', encoding='utf-8') as f:
            for line in sorted(plist):
                f.write(line + "\n")

def append_domains_for_split(domain_dir: Path, data_root: str, rows: List[Tuple[str,int,float]]):
    """将推理得到的 (abs_path, cluster_id, confidence) 追加到对应簇文件"""
    domain_dir.mkdir(parents=True, exist_ok=True)
    buckets: Dict[int, List[str]] = {}
    for p, cid, _ in rows:
        buckets.setdefault(int(cid), []).append(rel_to_parent(p, data_root))
    for cid, plist in buckets.items():
        fpath = domain_dir / f"domain_{cid}.txt"
        with open(fpath, 'a', encoding='utf-8') as f:
            for line in sorted(plist):
                f.write(line + "\n")


                                                                 
def cluster_train_good(data_root: str, out_dir: str,
                       patches=12, patch=128, scales=(1.0,0.7), no_retinex=False,
                       rotations=4, no_mask_padding=False,
                       out_dim=256, min_cluster_size=20, min_samples=5, metric='euclidean',
                       refine=True, tau_high=0.6, tau_low=0.3, nn_k=10, nn_ratio=0.7,
                       noise_recluster=True, noise_min_cluster=12, noise_min_samples=4,
                       domain_dir_path: Optional[Path] = None) -> Path:
    """原味聚类 + 软加入 + 噪声二次聚类；仅 train/good；保存 train_good.csv；domains 覆盖写入 train/good"""
    set_seed(42)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    out_dir=Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    d_train=os.path.join(data_root,'train','good')
    train_imgs=list_images(d_train)
    print(f"[CLUSTER] Train normal: {len(train_imgs)}")
    if len(train_imgs) < 2:
        raise RuntimeError("训练数据不足以进行 PCA/聚类，请检查 train/good 是否有至少两张图片")

    vgg=VGGFeat().eval().to(device)

    X_tr=[]; P_tr=[]
    for p in tqdm(train_imgs, desc='[CLUSTER] extracting train/good'):
        try:
            x,pu=build_image_feature(p, vgg, device, patches, patch, list(scales),
                                     use_retinex=(not no_retinex), rotations=rotations,
                                     use_mask=(not no_mask_padding))
            X_tr.append(x); P_tr.append(pu)
        except Exception as e:
            print(f"[WARN] {p} failed: {e}")
    if len(X_tr)==0:
        raise RuntimeError("未成功提取任何特征，请检查图片/依赖环境")
    X_tr=np.stack(X_tr,axis=0); P_tr=np.array(P_tr,dtype=np.float32)

    print("[CLUSTER] Fitting PCA ...")
    ss,pca,Z_tr = fit_pca(X_tr, out_dim=out_dim)

    print("[CLUSTER] HDBSCAN clustering ...")
    clusterer, y_tr, prob_tr = run_hdbscan(Z_tr, min_cluster_size=min_cluster_size, min_samples=min_samples, metric=metric)

                        
    y_base = y_tr.copy()
    prob_base = prob_tr.copy()

                                                
    if noise_recluster:
        print("[CLUSTER] Re-cluster noise subset ...")
        y_tr, prob_patch_noise = recluster_noise(Z_tr, y_tr, noise_min_cluster, noise_min_samples)
    else:
        prob_patch_noise = np.full(y_tr.shape[0], np.nan, dtype=np.float32)

                                                 
    centers_arr, valid_ids = compute_centers(Z_tr, y_tr)
    if refine:
        print("[CLUSTER] Soft-join (-1) ...")
        y_tr_ref, centers_ref, valid_ids = soft_join(Z_tr, y_tr, clusterer,
                                                     tau_high=tau_high, tau_low=tau_low,
                                                     nn_k=nn_k, nn_ratio=nn_ratio)
        y_tr = y_tr_ref
        centers_arr = centers_ref

                                                           
    M, orig_valid_ids = _membership_matrix_or_hard(clusterer, Z_tr, y_base)
    id2col = {cid: i for i, cid in enumerate(orig_valid_ids)}

    prob_patch_refine = np.full(y_tr.shape[0], np.nan, dtype=np.float32)
    if M.ndim == 2 and M.shape[0] == y_tr.shape[0] and len(id2col) > 0:
        was_noise = (y_base == -1)
        now_label = y_tr
        for cid in orig_valid_ids:
            col = id2col[cid]
            if col >= M.shape[1]:
                continue
            idx = np.where(was_noise & (now_label == cid))[0]
            if len(idx) > 0:
                prob_patch_refine[idx] = M[idx, col].astype(np.float32)

                                             
    prob_final = prob_base.copy()
    mask_noise = ~np.isnan(prob_patch_noise); prob_final[mask_noise] = prob_patch_noise[mask_noise]
    mask_refine = ~np.isnan(prob_patch_refine); prob_final[mask_refine] = prob_patch_refine[mask_refine]

                             
    mask=(y_tr!=-1); sil=-1.0
    if mask.sum()>=3 and len(set(y_tr[mask]))>=2:
        sil=float(silhouette_score(Z_tr[mask], y_tr[mask], metric='euclidean'))
    print(f"[CLUSTER] Silhouette (excl. noise): {sil:.4f}")

            
    dump(ss, out_dir/"scaler.joblib"); dump(pca, out_dir/"pca.joblib"); dump(clusterer, out_dir/"hdbscan.joblib")
    np.save(out_dir/"centers.npy", centers_arr); np.save(out_dir/"valid_ids.npy", np.array(valid_ids,dtype=np.int32))
    meta={
        "patches":patches,"patch":patch,"scales":list(scales),"no_retinex":bool(no_retinex),
        "rotations":rotations,"out_dim_effective":int(pca.n_components_),"min_cluster_size":min_cluster_size,
        "min_samples":min_samples,"metric":metric,"mask_padding":not no_mask_padding,
        "refine":bool(refine),"tau_high":tau_high,"tau_low":tau_low,
        "nn_k":nn_k,"nn_ratio":nn_ratio,
        "noise_recluster":bool(noise_recluster),"noise_min_cluster":noise_min_cluster,"noise_min_samples":noise_min_samples,
        "data_root":os.path.abspath(data_root)
    }
    with open(out_dir/"model_meta.json","w",encoding="utf-8") as f: json.dump(meta,f,ensure_ascii=False,indent=2)

                         
    with open(out_dir/"train_good.csv","w",newline='',encoding='utf-8') as f:
        w=csv.writer(f); w.writerow(["path","cluster","probability","purity"])
        for p,c,pr,pu in zip(train_imgs,y_tr,prob_final,P_tr):
            w.writerow([p,int(c),float(pr),float(pu)])

                                           
    if domain_dir_path is not None:
        write_domains_for_train(domain_dir_path, data_root, train_imgs, y_tr)
        print(f"[DOMAIN] Wrote per-cluster train/good to {domain_dir_path}")

    summary={
        "train_good":len(train_imgs),
        "n_clusters_found":len([c for c in set(y_tr) if c!=-1]),
        "silhouette_train_excl_noise":sil,
        "domains_dir":str(domain_dir_path) if domain_dir_path else ""
    }
    with open(out_dir/"summary.json","w",encoding="utf-8") as f: json.dump(summary,f,ensure_ascii=False,indent=2)
    print("[CLUSTER] Saved to:", str(out_dir))
    return out_dir


                                                              


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="./out_cluster")
    ap.add_argument("--patches", type=int, default=12)
    ap.add_argument("--patch", type=int, default=128)
    ap.add_argument("--scales", type=float, nargs="+", default=[1.0, 0.7])
    ap.add_argument("--no_retinex", action="store_true")
    ap.add_argument("--rotations", type=int, default=4, choices=[1, 4])
    ap.add_argument("--no_mask_padding", action="store_true")
    ap.add_argument("--out_dim", type=int, default=256)
    ap.add_argument("--min_cluster_size", type=int, default=20)
    ap.add_argument("--min_samples", type=int, default=5)
    ap.add_argument("--metric", type=str, default="euclidean", choices=["euclidean", "manhattan"])
    ap.add_argument("--no_refine", action="store_true")
    ap.add_argument("--tau_high", type=float, default=0.6)
    ap.add_argument("--tau_low", type=float, default=0.3)
    ap.add_argument("--nn_k", type=int, default=10)
    ap.add_argument("--nn_ratio", type=float, default=0.7)
    ap.add_argument("--no_noise_recluster", action="store_true")
    ap.add_argument("--noise_min_cluster", type=int, default=12)
    ap.add_argument("--noise_min_samples", type=int, default=4)
    args = ap.parse_args()
    cluster_train_good(
        data_root=args.data_root,
        out_dir=args.out_dir,
        patches=args.patches,
        patch=args.patch,
        scales=tuple(args.scales),
        no_retinex=args.no_retinex,
        rotations=args.rotations,
        no_mask_padding=args.no_mask_padding,
        out_dim=args.out_dim,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        metric=args.metric,
        refine=(not args.no_refine),
        tau_high=args.tau_high,
        tau_low=args.tau_low,
        nn_k=args.nn_k,
        nn_ratio=args.nn_ratio,
        noise_recluster=(not args.no_noise_recluster),
        noise_min_cluster=args.noise_min_cluster,
        noise_min_samples=args.noise_min_samples,
        domain_dir_path=None,
    )
