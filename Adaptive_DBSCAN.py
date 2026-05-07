"""
Adaptive_DBSCAN (paper version)
===============================
EN Wei et al., 2024 《Adaptive marine traffic behaviour pattern recognition based on
multidimensional dynamic time warping》 EN。

------------------------------------------------------------------------------
EN“EN”（EN）
------------------------------------------------------------------------------
AIS EN（EN） → EN TS → EN DTW EN D
→ EN POA Table 3 EN“EN/EN”EN eps/minPts → DBSCAN(metric=precomputed) EN。

------------------------------------------------------------------------------
EN（EN）
------------------------------------------------------------------------------
- EN（trajectory）：
  EN shape=(T, 4) EN：
    [lat, lon, sog, cog]
  EN：
    lat/lon : EN（EN）
    sog     : Speed Over Ground（EN）
    cog     : Course Over Ground（EN，0~360）

- TS / TSz：
  TS  EN “EN”（EN d EN）。
  TSz EN “z-score EN”，EN DTW EN，EN。

- D：
  DTW EN，shape=(N, N)，N EN：
    D[i, j] = DTW(TSz[i], TSz[j]) / max(len_i, len_j)
  EN：D[i,i]=0；EN >0。

- DBSCAN EN：
  eps    : EN（EN DTW）
  minPts : EN（min_samples）

------------------------------------------------------------------------------
EN POA Table 3（EN“paper/POA EN”）EN
------------------------------------------------------------------------------
EN D EN“EN”EN eps。

EN cut-off distance（dc，EN eps EN）EN：
  Num_i(dc) = |{ j != i  EN  D[i,j] < dc }|
  Mpt(dc)   = mean_i Num_i(dc)
  P(dc)     = Mpt(dc) / dc

EN P(dc) EN（EN）：
  G_i = |P_i - P_{i-1}| / step
EN dc EN Eps（eps）：
  eps = Eps = argmax_i G_i EN dc

minPts EN/EN：
  - EN MinPt = Eps * Gmax（EN eps_gmax / paper EN）
  - EN，EN：
      minPts = round(mean(Num_i(eps))) EN round(median(Num_i(eps)))
    EN minpts_stat EN（EN select_eps_minpts_from_D_paper）。

------------------------------------------------------------------------------
“EN”（strictness）EN：EN
------------------------------------------------------------------------------
EN POA Table 3 EN：
  - EN valid EN：D > 0 EN finite（EN 0 EN NaN/Inf）
  - EN：D < dc（EN <=）
  - EN（j != i）

------------------------------------------------------------------------------
DTW EN（EN）
------------------------------------------------------------------------------
DTW EN O(T1*T2)。EN dtaidistance EN exact DTW，
EN（EN max(n1,n2)），EN Sakoe–Chiba window（EN）。
EN、EN，EN D EN。

EN：
  1. EN (lat, lon, sog, cog)
  2. EN DTW EN
  3. EN Wei et al. (2024) Table 3（POA）EN cut-off distance dc：
     - valid = D EN >0 EN finite EN
     - LD = min(valid), HD = max(valid)
     - Size = ceil((HD - LD) / ΔS)，EN i=0..Size：
         dc = LD + i*ΔS
        Num = count_nonzero(D < dc EN j≠i, axis=1)
         Mpt = mean(Num)
         P = Mpt / dc
         DenPower EN (P, dc)
     - EN：G_i = |P_i - P_{i-1}| / ΔS，EN：
         Eps = dc_i
         MinPt = Eps * Gmax
     - EN：eps = Eps，minPts = round(mean/median(Num))（EN/EN）
  5. EN DBSCAN EN，EN labels

EN：
  labels  (EN=N)
  eps, minPts
"""

import numpy as np
from typing import List, Tuple
import math
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from concurrent.futures import ThreadPoolExecutor, as_completed
from dtaidistance import dtw_ndim

FEATURE_MODE_DEFAULT = "delta_cog"

def get_feature_tag(feature_mode: str) -> str:
    """
    EN（EN DTW EN）。
    EN：dtw_cache.py EN Adaptive_DBSCAN.FEATURE_TAG EN meta。

    EN feature_tag？
    - DTW EN“EN hash”EN；
    - EN（EN delta_cog EN pca_latlon），
      EN CSV EN，EN；
    - EN，EN meta，EN。
    """
    m = (feature_mode or "").strip().lower()
    if m in ("pca_latlon", "pca", "pca_reduced"):
        return "pca_latlon,sog,delta_cog@v1"
    if m in ("delta", "delta_cog", "dcog"):
        return "lat,lon,sog,delta_cog@v1"
    return f"unknown_feature_mode:{feature_mode}"

FEATURE_TAG = get_feature_tag(FEATURE_MODE_DEFAULT)


def _forward_fill_nonfinite(arr: np.ndarray, fill_first: float = 0.0) -> np.ndarray:
    """
    EN 1D EN NaN/Inf EN“EN”EN，EN。
    - EN，EN fill_first
    - EN，EN fill_first
    """
    x = np.asarray(arr, dtype=float).copy()
    if x.size == 0:
        return x
    if not np.isfinite(x[0]):
        x[0] = float(fill_first)
    for i in range(1, x.size):
        if not np.isfinite(x[i]):
            x[i] = x[i - 1]
    x[~np.isfinite(x)] = float(fill_first)
    return x


def _latlon_to_local_xy_meters(
    lat: np.ndarray,
    lon: np.ndarray,
    lat0: float,
    lon0: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    EN（equirectangular / EN）：
      - EN (lat0, lon0)
      - y_m = (lat - lat0) * 111132.92
      - x_m = (lon - lon0) * 111412.84 * cos(lat0)

    EN：EN“EN”EN“EN”，EN lon EN cos(lat) EN。

    EN：
    - EN“1EN”EN（EN cos(lat)）。
    - DTW EN“EN”，EN。
    - EN equirectangular EN (lat,lon) EN (x_m,y_m)。
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    lat0 = float(lat0)
    lon0 = float(lon0)
    lat0_rad = math.radians(lat0)
    y_m = (lat - lat0) * 111132.92
    x_m = (lon - lon0) * (111412.84 * math.cos(lat0_rad))
    return x_m, y_m


def build_TS_features(
    trajectories: List[np.ndarray],
    feature_mode: str = FEATURE_MODE_DEFAULT,
    dt_s: float = 60.0,
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
    """
    EN（EN DTW EN，EN）：

    EN：
      - EN N×4: [lat, lon, sog, cog]

    feature_mode:
      - "pca_latlon": N×4 -> N×4: [pca_latlon1, pca_latlon2, sog, delta_cog]
          ✅ EN lat/lon EN：
          - EN lat/lon EN z-score EN
          - EN lat/lon EN PCA EN（EN 2 EN）
          - sog EN delta_cog EN（EN）
          - EN：PCAEN lat/lon（2EN） + sog（1EN） + delta_cog（1EN）
      - "delta_cog": N×4 -> [x_abs, y_abs, sog, delta_cog]
          delta_cog = ΔCOG（deg），EN [-180, 180]，EN=0

    EN（z-score，EN）：
      - DTW EN N×4 EN f（EN）
      - EN (lat, lon) EN (x_m, y_m)，EN：
            x_abs = x_m
            y_abs = y_m
      - z-score EN：mu EN sigma EN all_feat EN：
          mu = mean(all_feat_mat, axis=0)
          sigma = std(all_feat_mat, axis=0) + 1e-12
        EN mu/sigma EN：
          TSz = (f - mu) / sigma

    EN（EN）：
    - DTW EN“EN” f(t)，EN；
      EN，EN (x_abs,y_abs)、EN sog、EN（delta_cog）。
    - EN dx/dy/dsog EN，EN DTW EN。
    """
    global FEATURE_TAG
    FEATURE_TAG = get_feature_tag(feature_mode)
    mode = (feature_mode or "").strip().lower()
    if mode in ("delta", "dcog"):
        mode = "delta_cog"
    if mode in ("pca", "pca_reduced"):
        mode = "pca_latlon"

    if mode not in ("delta_cog", "pca_latlon"):
        raise ValueError(f"Unsupported feature_mode={feature_mode!r}. Supported: delta_cog, pca_latlon")

    if mode == "pca_latlon":
        all_lat = []
        all_lon = []
        all_trajs_data = []
        
        for traj in trajectories:
            lat = np.asarray(traj[:, 0], dtype=float)
            lon = np.asarray(traj[:, 1], dtype=float)
            sog = np.asarray(traj[:, 2], dtype=float)
            cog = np.asarray(traj[:, 3], dtype=float)
            
            lat = _forward_fill_nonfinite(lat, fill_first=0.0)
            lon = _forward_fill_nonfinite(lon, fill_first=0.0)
            sog = _forward_fill_nonfinite(sog, fill_first=0.0)
            cog = _forward_fill_nonfinite(cog, fill_first=0.0)
            
            all_lat.append(lat)
            all_lon.append(lon)
            all_trajs_data.append((lat, lon, sog, cog))
        
        all_lat_flat = np.concatenate(all_lat) if all_lat else np.array([])
        all_lon_flat = np.concatenate(all_lon) if all_lon else np.array([])
        
        mu_lat = float(np.nanmean(all_lat_flat)) if all_lat_flat.size > 0 else 0.0
        sigma_lat = float(np.nanstd(all_lat_flat)) + 1e-12 if all_lat_flat.size > 0 else 1.0
        mu_lon = float(np.nanmean(all_lon_flat)) if all_lon_flat.size > 0 else 0.0
        sigma_lon = float(np.nanstd(all_lon_flat)) + 1e-12 if all_lon_flat.size > 0 else 1.0
        
        latlon_standardized = []
        for lat, lon in zip(all_lat, all_lon):
            lat_norm = (lat - mu_lat) / sigma_lat
            lon_norm = (lon - mu_lon) / sigma_lon
            latlon_standardized.append(np.stack([lat_norm, lon_norm], axis=1))
        
        latlon_all = np.vstack(latlon_standardized) if latlon_standardized else np.zeros((0, 2))
        
        pca = PCA(n_components=2)
        if latlon_all.shape[0] > 0:
            pca.fit(latlon_all)
        else:
            pca.components_ = np.eye(2)
            pca.mean_ = np.zeros(2)
        
        TS = []
        all_feat = []
        for lat, lon, sog, cog in all_trajs_data:
            lat_norm = (lat - mu_lat) / sigma_lat
            lon_norm = (lon - mu_lon) / sigma_lon
            latlon_norm = np.stack([lat_norm, lon_norm], axis=1)
            
            latlon_pca = pca.transform(latlon_norm)
            
            delta_cog = np.diff(cog)
            delta_cog = (delta_cog + 180.0) % 360.0 - 180.0
            delta_cog = np.insert(delta_cog, 0, 0.0)
            delta_cog = np.nan_to_num(delta_cog, nan=0.0, posinf=0.0, neginf=0.0)
            
            f = np.column_stack([latlon_pca, sog, delta_cog])
            TS.append(f)
            all_feat.append(f)
        
        if len(all_feat) > 0:
            all_feat_mat = np.vstack(all_feat)
            mu = np.nanmean(all_feat_mat, axis=0)
            sigma = np.nanstd(all_feat_mat, axis=0) + 1e-12
        else:
            mu = np.zeros(4, dtype=float)
            sigma = np.ones(4, dtype=float)
        
        print("\n📊 EN（pca_latlon；lat/lon PCAEN + sog/delta_cog）:")
        print("  feature_mode:", "pca_latlon")
        print(f"  lat/lon EN: lat(μ={mu_lat:.6f}, σ={sigma_lat:.6f}), lon(μ={mu_lon:.6f}, σ={sigma_lon:.6f})")
        if latlon_all.shape[0] > 0:
            explained_var = pca.explained_variance_ratio_
            print(f"  PCA EN: PC1={explained_var[0]:.4f}, PC2={explained_var[1]:.4f}, EN={explained_var.sum():.4f}")
        print(f"  EN mu:    [pca_latlon1={mu[0]:.6f}, pca_latlon2={mu[1]:.6f}, sog={mu[2]:.6f}, delta_cog={mu[3]:.6f}]")
        print(f"  EN sigma: [pca_latlon1={sigma[0]:.6f}, pca_latlon2={sigma[1]:.6f}, sog={sigma[2]:.6f}, delta_cog={sigma[3]:.6f}]")
        
        labels = ["pca_latlon1", "pca_latlon2", "sog", "delta_cog"]
        issues = []
        for lbl, s in zip(labels, sigma):
            if s < 0.01:
                issues.append(f"{lbl} sigmaEN({s:.6f}), EN")
            elif s > 10.0:
                issues.append(f"{lbl} sigmaEN({s:.2f}), EN")
        if issues:
            print("  ⚠️ EN:")
            for iss in issues:
                print(f"    - {iss}")
        else:
            print("  ✅ EN")
        
        TSz = [(f - mu) / sigma for f in TS]
        return TSz, mu, sigma

    first_lats = []
    first_lons = []
    for traj in trajectories:
        if len(traj) > 0:
            first_lats.append(float(traj[0, 0]))
            first_lons.append(float(traj[0, 1]))
    if len(first_lats) > 0:
        lat_ref = float(np.median(first_lats))
        lon_ref = float(np.median(first_lons))
    else:
        lat_ref = 30.0
        lon_ref = 120.0

    TSz = []
    all_feat = []
    all_delta = []
    for traj in trajectories:
        lat, lon, sog, cog = traj.T

        lat = _forward_fill_nonfinite(lat, fill_first=0.0)
        lon = _forward_fill_nonfinite(lon, fill_first=0.0)
        sog = _forward_fill_nonfinite(sog, fill_first=0.0)
        cog = _forward_fill_nonfinite(cog, fill_first=0.0)

        x_m, y_m = _latlon_to_local_xy_meters(lat, lon, lat0=lat_ref, lon0=lon_ref)
        x_m = np.nan_to_num(x_m, nan=0.0, posinf=0.0, neginf=0.0)
        y_m = np.nan_to_num(y_m, nan=0.0, posinf=0.0, neginf=0.0)
        x_abs = x_m
        y_abs = y_m

        delta_cog = np.diff(cog)
        delta_cog = (delta_cog + 180.0) % 360.0 - 180.0

        delta_cog = np.insert(delta_cog, 0, 0.0)
        delta_cog = np.nan_to_num(delta_cog, nan=0.0, posinf=0.0, neginf=0.0)

        f = np.stack([x_abs, y_abs, sog, delta_cog], axis=1)
        d4 = delta_cog

        TSz.append(f)
        all_feat.append(f)

        dx = np.diff(x_abs, prepend=x_abs[0] if len(x_abs) > 0 else 0.0)
        dy = np.diff(y_abs, prepend=y_abs[0] if len(y_abs) > 0 else 0.0)
        dsog = np.diff(sog, prepend=sog[0] if len(sog) > 0 else 0.0)
        dx = np.nan_to_num(dx, nan=0.0, posinf=0.0, neginf=0.0)
        dy = np.nan_to_num(dy, nan=0.0, posinf=0.0, neginf=0.0)
        dsog = np.nan_to_num(dsog, nan=0.0, posinf=0.0, neginf=0.0)
        d4 = np.nan_to_num(d4, nan=0.0, posinf=0.0, neginf=0.0)
        all_delta.append(np.stack([dx, dy, dsog, d4], axis=1))

    if len(all_feat) > 0:
        all_feat_mat = np.vstack(all_feat)
        mu = np.nanmean(all_feat_mat, axis=0)
        sigma = np.nanstd(all_feat_mat, axis=0) + 1e-12
    else:
        mu = np.zeros(4, dtype=float)
        sigma = np.ones(4, dtype=float)

    all_delta_mat = np.vstack(all_delta) if len(all_delta) > 0 else np.zeros((0, 4), dtype=float)
    mu_delta_debug = (np.nanmean(all_delta_mat, axis=0)) if all_delta_mat.size else np.zeros(4, dtype=float)
    sigma_delta_debug = (np.nanstd(all_delta_mat, axis=0) + 1e-12) if all_delta_mat.size else np.ones(4, dtype=float)
    
    print("\n📊 EN（mu/sigma estimated from absolute features）:")
    last_name = "delta_cog"
    print("  feature_mode:", mode)
    print("  note:", "lat/lon EN (x_m,y_m)，EN (x_abs,y_abs)，EN")
    print(f"  EN (lat_ref, lon_ref): ({lat_ref:.6f}, {lon_ref:.6f}) [EN，EN]")
    print(
        "  mu (EN，EN):    "
        f"[x_abs={mu[0]:.6f}, y_abs={mu[1]:.6f}, sog={mu[2]:.6f}, {last_name}={mu[3]:.6f}]"
    )
    print(
        "  sigma (EN，EN): "
        f"[x_abs={sigma[0]:.6f}, y_abs={sigma[1]:.6f}, sog={sigma[2]:.6f}, {last_name}={sigma[3]:.6f}]"
    )
    print(
        "  mu_delta (EN，EN，EN): "
        f"[dx={mu_delta_debug[0]:.6f}, dy={mu_delta_debug[1]:.6f}, dsog={mu_delta_debug[2]:.6f}, {last_name}={mu_delta_debug[3]:.6f}]"
    )
    print(
        "  sigma_delta (EN，EN，EN): "
        f"[dx={sigma_delta_debug[0]:.6f}, dy={sigma_delta_debug[1]:.6f}, dsog={sigma_delta_debug[2]:.6f}, {last_name}={sigma_delta_debug[3]:.6f}]"
    )
    labels = ['x_abs', 'y_abs', 'sog', last_name]
    issues = []
    for i, (lbl, s) in enumerate(zip(labels, sigma)):
        if s < 0.01:
            issues.append(f"{lbl} sigmaEN({s:.6f}), EN")
        elif s > 10.0:
            issues.append(f"{lbl} sigmaEN({s:.2f}), EN")
    if issues:
        print("  ⚠️ EN:")
        for iss in issues:
            print(f"    - {iss}")
    else:
        print("  ✅ EN")
    
    TSz = [(f - mu) / sigma for f in TSz]
    return TSz, mu, sigma



def compute_dtw_distance(
    traj1: np.ndarray,
    traj2: np.ndarray,
    window_ratio: float = 0.1
) -> float:
    """
    EN DTW EN（EN fastdtw EN DTW）。

    EN：
    - EN fastdtw（EN DTW）EN；
    - fastdtw EN Sakoe–Chiba window EN window_ratio EN（EN radius EN），
      EN window_ratio EN，EN。

    EN
    ----
    traj1 : np.ndarray
        EN，shape=(T1, d)
    traj2 : np.ndarray
        EN，shape=(T2, d)
    window_ratio : float
        EN，EN 0.1（EN 10%）

    EN
    ----
    float
        EN DTW EN（EN warping path length）

    EN：
    - EN DTW EN/EN；
    - EN“EN”EN，EN eps EN。
    """
    traj1 = np.asarray(traj1, dtype=np.float64)
    traj2 = np.asarray(traj2, dtype=np.float64)

    n1 = len(traj1)
    n2 = len(traj2)

    try:
        distance, path = fastdtw(traj1, traj2, dist=euclidean)
        norm = max(1, len(path))
        dist_norm = float(distance) / float(norm)
    except Exception:
        dist_norm = float("inf")

    if not np.isfinite(dist_norm):
        raise ValueError(
            "DTW distance is non-finite (inf/nan). "
            f"n1={n1}, n2={n2}. "
            "Please check input trajectories for non-finite values."
        )

    return float(dist_norm)


def build_distance_matrix(TSz: List[np.ndarray], verbose=True, n_jobs=-1) -> np.ndarray:
    """
    EN D (NxN)
    - EN (i<j)，EN。
    - EN（ThreadPoolExecutor）：
      EN DTW EN GIL；EN。
    n_jobs: -1ENCPUEN，1EN
    """
    N = len(TSz)
    D = np.zeros((N, N), dtype=float)
    
    pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    def compute_pair(ij):
        i, j = ij
        d = compute_dtw_distance(TSz[i], TSz[j])
        return (i, j, d)
    
    if n_jobs == 1 or n_jobs is None:
        for i, j, d in map(compute_pair, pairs):
            D[i, j] = D[j, i] = d
            if verbose and (i + 1) % 10 == 0:
                print(f"  DTW EN: {i+1}/{N}")
    else:
        import os
        num_workers = min(os.cpu_count() or 1, len(pairs)) if n_jobs == -1 else n_jobs
        completed = 0
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(compute_pair, pair): pair for pair in pairs}
            for future in as_completed(futures):
                i, j, d = future.result()
                D[i, j] = D[j, i] = d
                completed += 1
                if verbose and completed % 10 == 0:
                    print(f"  DTW EN: {completed}/{len(pairs)}")
    
    return D


def select_eps_minpts_from_D_paper(
    D: np.ndarray,
    delta_s: float,
    verbose=True,
    minpts_stat: str = "mean",
    two_stage: bool = False,
    coarse_steps: int = 150,
    refine_steps: int = 300,
    refine_window_mult: float = 5.0,
) -> Tuple[float, int]:
    """
    Wei et al. (2024) Table 3 (POA) EN：
      1) valid = D EN >0 EN finite EN
         LD = min(valid), HD = max(valid)
      2) Size = ceil((HD - LD) / ΔS)，EN i=0..Size:
           dc = LD + i*ΔS
           Num = count_nonzero(D < dc EN j≠i, axis=1)
           Mpt = mean(Num)
           P = Mpt / dc  (dc=0 EN)
           DenPower.append([P, dc])
      3) EN i=1..M-1:
           Gi = abs(DenPower[i][0] - DenPower[i-1][0]) / ΔS
           EN Gi > Gmax：EN Gmax=Gi, Eps=DenPower[i][1], MinPt = Eps * Gmax
      4) ✅ EN（EN）：
         - Eps EN Table 3（EN）EN
         - EN minPts EN MinPt=Eps*Gmax
         - EN：EN eps=Eps EN，EN Num（strict '<' EN，EN valid=(D>0 & finite)）
           EN minPts = round(mean(Num)) EN round(median(Num))（EN minpts_stat EN，EN mean）
         - EN/EN，EN
      5) verbose EN：EN Eps、MinPt、Gmax、EN LD/HD/Size
    
    Two-stage (coarse→refine)（EN，EN two_stage=True）：
      - coarse: EN [LD,HD] EN coarse_steps EN，EN eps_coarse（EN）
      - refine: EN eps_coarse ± refine_window_mult*ΔS_coarse EN，EN refine_steps EN eps_final

    EN：
      - EN “delta_s=None/<=0 ⇒ EN(target_steps)” EN；
        EN two_stage=False，EN delta_s>0。

    EN：
      “EN D，EN DBSCAN EN eps EN minPts？”

    EN（EN）：
    - valid EN (D > 0) EN finite；EN 0/NaN/Inf EN LD/HD EN。
    - EN D < dc（EN <=），EN（j != i）。
    - eps EN POA Table 3：EN density power EN。
    - minPts EN（minpts_stat）：
        * "eps_gmax"/"paper"：minPts = Eps * Gmax（EN）
        * "mean"（EN）：minPts = round(mean(Num_i(Eps)))
        * "median"：minPts = round(median(Num_i(Eps)))
    """
    D = np.asarray(D, dtype=float)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError(f"POA Table3 EN D（N×N），EN: {getattr(D, 'shape', None)}")
    valid = D[(D > 0) & np.isfinite(D)]
    if valid.size == 0:
        return 1.0, 4

    LD = float(np.min(valid))
    HD = float(np.max(valid))
    range_val = HD - LD

    n = int(D.shape[0])
    offdiag = ~np.eye(n, dtype=bool)
    
    def _scan_eps(
        *,
        step: float,
        ld: float,
        hd: float,
    ) -> Tuple[float, float, float, int]:
        """
        Scan POA density power curve with fixed step.
        Returns: (Eps, Gmax, MinPt(Eps*Gmax), Size)
        """
        step = float(step)
        if not (step > 0):
            raise ValueError(f"delta_s/step must be > 0, got {step!r}")
        range_val2 = float(hd - ld)
        Size2 = int(math.ceil(range_val2 / step)) if range_val2 > 0 else 0

        DenPower: List[Tuple[float, float]] = []
        for ii in range(Size2 + 1):
            dc = float(ld + ii * step)
            Num = np.count_nonzero((D > 0) & np.isfinite(D) & (D < dc) & offdiag, axis=1)
            Mpt = float(np.mean(Num))
            P = float(Mpt / max(dc, 1e-12))
            DenPower.append((P, float(dc)))

        Gmax2 = 0.0
        Eps2 = float(DenPower[0][1]) if DenPower else float(ld)
        MinPt2 = 0.0
        for ii in range(1, len(DenPower)):
            Gi = abs(DenPower[ii][0] - DenPower[ii - 1][0]) / step
            if Gi > Gmax2:
                Gmax2 = float(Gi)
                Eps2 = float(DenPower[ii][1])
                MinPt2 = float(Eps2 * Gmax2)
        return float(Eps2), float(Gmax2), float(MinPt2), int(Size2)

    if two_stage:
        cs = int(coarse_steps)
        rs = int(refine_steps)
        if cs < 5:
            raise ValueError(f"coarse_steps must be >= 5, got {cs}")
        if rs < 10:
            raise ValueError(f"refine_steps must be >= 10, got {rs}")
        if verbose:
            print(
                "  [Two-stage] note: delta_s is NOT used as the scan step here. "
                "Scan resolution is controlled by coarse_steps/refine_steps/window_mult."
            )
        coarse_step = max(range_val / float(cs), 1e-6)
        if verbose:
            print(f"  [Two-stage] coarse_steps={cs}, coarse_step={coarse_step:.6f}")
        eps_coarse, _, _, _ = _scan_eps(step=coarse_step, ld=LD, hd=HD)

        win_mult = float(refine_window_mult)
        if not np.isfinite(win_mult) or win_mult <= 0:
            raise ValueError(f"refine_window_mult must be > 0, got {refine_window_mult!r}")
        half_w = win_mult * coarse_step
        ld2 = max(LD, float(eps_coarse - half_w))
        hd2 = min(HD, float(eps_coarse + half_w))
        range2 = float(hd2 - ld2)
        refine_step = max(range2 / float(rs), 1e-6)
        if verbose:
            print(f"  [Two-stage] eps_coarse={eps_coarse:.6f}, window=[{ld2:.6f},{hd2:.6f}], refine_steps={rs}, refine_step={refine_step:.6f}")
        Eps, Gmax, MinPt, Size = _scan_eps(step=refine_step, ld=ld2, hd=hd2)
        step = refine_step
        if verbose:
            print(f"  [Two-stage] eps_final={Eps:.6f}, Gmax={Gmax:.6f}")
    else:
        step = float(delta_s)
        if not (step > 0):
            raise ValueError(
                f"delta_s must be > 0 when two_stage=False. (Old adaptive-step logic has been removed.) Got {delta_s!r}"
            )
        if verbose:
            print(f"  [EN] delta_s={step:.4f}")

    if not two_stage:
        Eps, Gmax, MinPt, Size = _scan_eps(step=step, ld=LD, hd=HD)


    stat = (minpts_stat or "mean").strip().lower()
    
    Num_eps = np.count_nonzero((D > 0) & np.isfinite(D) & (D < float(Eps)) & offdiag, axis=1)
    mpt_mean = float(np.mean(Num_eps)) if Num_eps.size else 0.0
    mpt_median = float(np.median(Num_eps)) if Num_eps.size else 0.0
    
    if stat in ("eps_gmax", "paper", "gmax", "e*g"):
        minPts_raw = MinPt
        stat_used = "Eps*Gmax"
    elif stat in ("median", "med", "p50"):
        minPts_raw = mpt_median
        stat_used = "median"
    else:
        minPts_raw = mpt_mean
        stat_used = "mean"
    
    minPts_int_raw = int(round(minPts_raw))
    minPts_int = max(1, minPts_int_raw)
    if verbose:
        print(
            "  [POA Table3] step=%.6f, LD=%.4f, HD=%.4f, Size=%d | Eps=%.4f, MinPt(Eps*Gmax)=%.4f, Gmax=%.6f"
            % (step, LD, HD, Size, Eps, MinPt, Gmax)
        )
        if minPts_int != minPts_int_raw:
            print(f"  ⚠️ [minPts] round EN {minPts_int_raw}，EN DBSCAN EN minPts>=1，EN {minPts_int}")
        if stat_used == "Eps*Gmax":
            print(
                "  [minPts] use=%s (MinPt=Eps*Gmax) -> %d"
                % (stat_used, minPts_int)
            )
        else:
            print(
                "  [minPts@eps] mean(Num)=%.4f, median(Num)=%.4f, use=%s -> %d"
                % (mpt_mean, mpt_median, stat_used, minPts_int)
            )
    return float(Eps), int(minPts_int)





def density_power_curve_poa_minmax_strict(
    D: np.ndarray,
    delta_s: float,
    *,
    two_stage: bool = False,
    coarse_steps: int = 150,
    refine_steps: int = 300,
    refine_window_mult: float = 5.0,
):
    """
    Strict POA Table 3 curve (EN select_eps_minpts_from_D_paper EN DenPower EN)：
      - valid = (D > 0 & finite)
      - LD = min(valid), HD = max(valid)
      - Size = ceil((HD - LD) / ΔS)
      - dc = LD + i*ΔS
      - Num = count_nonzero(D < dc EN j≠i, axis=1)
      - Mpt = mean(Num)
      - P = Mpt / dc
    Returns (dc_list, mpt_list, p_list).
    
    EN“EN”，EN POA Table 3 EN：
      x EN：dc（EN eps）
      y1 ：Mpt(dc) = mean(Num_i(dc))
      y2 ：P(dc)   = Mpt(dc) / dc
    """
    D = np.asarray(D, dtype=float)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError(f"POA Table3 curve EN D（N×N），EN: {getattr(D, 'shape', None)}")
    valid = D[(D > 0) & np.isfinite(D)]
    if valid.size == 0:
        return [], [], []

    LD = float(np.min(valid))
    HD = float(np.max(valid))
    range_val = HD - LD
    
    def _curve_for_range(ld: float, hd: float, step: float):
        range_val2 = float(hd - ld)
        Size2 = int(math.ceil(range_val2 / step)) if range_val2 > 0 else 0
        dc_list2, mpt_list2, p_list2 = [], [], []
        n = int(D.shape[0])
        offdiag = ~np.eye(n, dtype=bool)
        for ii in range(Size2 + 1):
            dc = float(ld + ii * step)
            Num = np.count_nonzero((D > 0) & np.isfinite(D) & (D < dc) & offdiag, axis=1)
            Mpt = float(np.mean(Num))
            P = float(Mpt / max(dc, 1e-12))
            dc_list2.append(dc)
            mpt_list2.append(Mpt)
            p_list2.append(P)
        return dc_list2, mpt_list2, p_list2, Size2

    if two_stage:
        cs = int(coarse_steps)
        rs = int(refine_steps)
        if cs < 5 or rs < 10:
            return [], [], []
        coarse_step = max(range_val / float(cs), 1e-6)
        dc_c, mpt_c, p_c, _ = _curve_for_range(LD, HD, coarse_step)
        if len(p_c) < 2:
            return dc_c, mpt_c, p_c
        g = np.abs(np.diff(np.asarray(p_c))) / coarse_step
        k = int(np.argmax(g)) + 1
        eps_coarse = float(dc_c[k])

        win_mult = float(refine_window_mult)
        if not np.isfinite(win_mult) or win_mult <= 0:
            win_mult = 5.0
        half_w = win_mult * coarse_step
        ld2 = max(LD, float(eps_coarse - half_w))
        hd2 = min(HD, float(eps_coarse + half_w))
        refine_step = max(float(hd2 - ld2) / float(rs), 1e-6)
        dc_f, mpt_f, p_f, _ = _curve_for_range(ld2, hd2, refine_step)
        return dc_f, mpt_f, p_f

    step = float(delta_s)
    if not (step > 0):
        raise ValueError(f"delta_s must be > 0 for POA curve, got {delta_s!r}")
    
    dc_list, mpt_list, p_list, _ = _curve_for_range(LD, HD, step)
    return dc_list, mpt_list, p_list


def dbscan_from_distance(D: np.ndarray, eps: float, minPts: int) -> np.ndarray:
    """
    EN DBSCAN，EN precomputed EN

    EN：
    - sklearn.DBSCAN EN labels EN [-1, 0..K-1]（-1 EN）
    - EN +1，EN 1 EN，EN 0。
      EN/EN：0=noise，1..K=clusters。
    """
    db = DBSCAN(eps=eps, min_samples=minPts, metric="precomputed", n_jobs=-1)
    labels = db.fit_predict(D)
    labels = labels + 1
    labels[labels == 0] = 0
    return labels


def adaptive_dbscan_paper(TSz: List[np.ndarray], delta_s: float, verbose=True):

    if verbose:
        print("📘 [Adaptive-DBSCAN] EN")

    D = build_distance_matrix(TSz, verbose=verbose)

    eps, minPts = select_eps_minpts_from_D_paper(D, delta_s=delta_s, verbose=verbose)

    labels = dbscan_from_distance(D, eps, minPts)

    if verbose:
        K = int(labels.max())
        noise_ratio = float(np.mean(labels == 0))
        print(f"✅ EN: K={K}, EN={noise_ratio:.2%}, eps={eps:.3f}, minPts={minPts}")
    return labels, eps, minPts


def run_amtbra_dbscan_paper(
    TS_raw: List[np.ndarray],
    delta_s: float,
    verbose=True,
    n_jobs=-1,
    D: "np.ndarray | None" = None,
    feature_mode: str = FEATURE_MODE_DEFAULT,
    dt_s: float = 60.0,
    minpts_stat: str = "mean",
    two_stage: bool = False,
    coarse_steps: int = 150,
    refine_steps: int = 300,
    refine_window_mult: float = 5.0,
):
    """
    ✅ EN“EN”（EN run_complete_pipeline.py EN）

    EN：
      TS_raw : List[np.ndarray]
        EN（EN (T,4)=[lat,lon,sog,cog]）

    EN（EN）：
      1) build_TS_features：EN (lat,lon,sog,cog) EN DTW EN TSz（EN）
      2) build_distance_matrix：EN TSz EN DTW，EN D（EN D EN）
      3) select_eps_minpts_from_D_paper：EN POA Table 3 EN D EN eps/minPts（EN minpts_stat）
      4) dbscan_from_distance：EN DBSCAN(metric=precomputed) EN labels

    EN：
      labels, eps, minPts, D, mu, sigma
    - mu/sigma EN build_TS_features EN（EN/EN）。
    """
    TSz, mu, sigma = build_TS_features(TS_raw, feature_mode=feature_mode, dt_s=dt_s)
    if D is None:
        D = build_distance_matrix(TSz, verbose=verbose, n_jobs=n_jobs)
    else:
        if verbose:
            print(f"📦 ENDTWEN，EN: {D.shape}")
    
    if verbose:
        valid_D = D[(D > 0) & np.isfinite(D)]
        print("\n📏 EN:")
        print(f"  mean: {valid_D.mean():.4f}")
        print(f"  std:  {valid_D.std():.4f}")
        print(f"  min:  {valid_D.min():.4f}")
        print(f"  max:  {valid_D.max():.4f}")
        print(f"  LD(min valid): {valid_D.min():.4f}")
        print(f"  HD(max valid): {valid_D.max():.4f}")
    
    eps, minPts = select_eps_minpts_from_D_paper(
        D,
        delta_s=delta_s,
        verbose=verbose,
        minpts_stat=str(minpts_stat),
        two_stage=bool(two_stage),
        coarse_steps=int(coarse_steps),
        refine_steps=int(refine_steps),
        refine_window_mult=float(refine_window_mult),
    )
    labels = dbscan_from_distance(D, eps, minPts)
    if verbose:
        K = int(labels.max())
        noise_ratio = float(np.mean(labels == 0))
        print(f"✅ [Paper] K={K}, EN={noise_ratio:.2%}, eps={eps:.3f}, minPts={minPts}")
    return labels, eps, minPts, D, mu, sigma


def drl_select_eps_minpts(
    D: np.ndarray,
    model_path: "str | None" = None,
    agent: "object | None" = None,
    verbose: bool = False,
) -> Tuple[float, int]:
    """
    EN DRL EN (eps, minPts) EN。

    EN：
    - EN Adaptive_DBSCAN EN，EN；
    - EN drl_dbscan.drl_select_eps_minpts，EN；
    - EN：EN D，EN。

    EN
    ----
    D : np.ndarray, shape=(N, N)
        DTW EN
    model_path : str, optional
        DRL EN（EN，EN）
    agent : PPOAgent, optional
        EN DRL EN，EN
    verbose : bool
        EN

    EN
    ----
    eps : float
        DRL EN eps EN
    minPts : int
        DRL EN minPts EN
    """
    try:
        from drl_dbscan import drl_select_eps_minpts as _drl_select
        return _drl_select(D, model_path=model_path, agent=agent, verbose=verbose)
    except ImportError as e:
        raise ImportError(
            "EN drl_dbscan EN。EN drl_dbscan.py EN，"
            "EN（torch, scipy EN）。"
        ) from e


def compare_feature_modes_paper(
    TS_raw: List[np.ndarray],
    modes: Tuple[str, ...] = ("delta_cog", "pca_latlon"),
    delta_s: float = None,
    n_jobs: int = -1,
    dt_s: float = 60.0,
    verbose: bool = True,
) -> dict:
    """
    EN“EN + DBSCAN + EN(DBI′/CP′/SP′)”EN。

    EN dict：
      {
        mode: {
          "eps": float, "minPts": int,
          "K": int, "noise_ratio": float,
          "DBI'": float, "CP'": float, "SP'": float
        }, ...
      }
    """
    from evaluate import evaluate_clusters

    results = {}
    for mode in modes:
        if verbose:
            print("\n" + "=" * 60)
            print(f"🔬 [Compare] feature_mode={mode}（paper）")
            print("=" * 60)
        labels, eps, minPts, D, _, _ = run_amtbra_dbscan_paper(
            TS_raw,
            delta_s=delta_s,
            verbose=verbose,
            n_jobs=n_jobs,
            D=None,
            feature_mode=mode,
            dt_s=dt_s,
        )
        dbi, cp, sp = evaluate_clusters(D, labels)
        K = int(labels.max()) if labels.size else 0
        noise_ratio = float(np.mean(labels == 0)) if labels.size else 1.0
        results[str(mode)] = {
            "eps": float(eps),
            "minPts": int(minPts),
            "K": int(K),
            "noise_ratio": float(noise_ratio),
            "DBI'": float(dbi),
            "CP'": float(cp),
            "SP'": float(sp),
        }
        if verbose:
            print(f"✅ [Compare][{mode}] eps={eps:.4f}, minPts={minPts}, K={K}, noise={noise_ratio:.2%}")
            print(f"   DBI′={dbi:.4f}, CP′={cp:.4f}, SP′={sp:.4f}")
    return results


if __name__ == "__main__":
    np.random.seed(0)
    trajs = []
    for k in range(5):
        lat = np.linspace(30, 30.05, 40) + np.random.randn(40) * 1e-4
        lon = np.linspace(120, 120.05, 40) + np.random.randn(40) * 1e-4
        sog = np.linspace(10, 11, 40)
        cog = np.linspace(0, 90, 40)
        trajs.append(np.stack([lat, lon, sog, cog], axis=1))

    TSz, mu, sigma = build_TS_features(trajs)
    labels, eps, minPts = adaptive_dbscan_paper(TSz, delta_s=0.07)