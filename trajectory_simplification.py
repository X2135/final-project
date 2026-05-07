from typing import List, Tuple
import math
import numpy as np
import atexit

_MERGED_LENGTHS: List[int] = []

_SACSB_DEBUG_MAX = 5
_SACSB_DEBUG_COUNT = 0

def _print_merge_length_stats():
    """
    Print merged length distribution and short-trajectory warning after the run.
    This is kept inside the simplification module so that the "paper run" does not
    need to modify the pipeline to get the required diagnostics.
    """
    if not _MERGED_LENGTHS:
        return
    arr = np.asarray(_MERGED_LENGTHS, dtype=float)
    if arr.size == 0:
        return
    mn = int(np.min(arr))
    med = int(np.median(arr))
    p90 = int(np.quantile(arr, 0.90))
    mx = int(np.max(arr))
    short_thr = 3
    short_cnt = int(np.sum(arr < short_thr))
    ratio = float(short_cnt / max(1, arr.size))
    print("\n[Simplify] merge EN: "
          f"min={mn}, median={med}, p90={p90}, max={mx}")
    if short_cnt > 0:
        print(f"⚠️ [Simplify] merge EN < {short_thr} EN: {short_cnt}/{int(arr.size)} ({ratio:.2%})，"
              "EN/EN")

atexit.register(_print_merge_length_stats)

def _deg_to_meter_scale(lat_ref_deg: float) -> Tuple[float, float]:
    phi = math.radians(lat_ref_deg)
    lat_m = 111132.92 - 559.82 * math.cos(2 * phi) + 1.175 * math.cos(4 * phi) - 0.0023 * math.cos(6 * phi)
    lon_m = 111412.84 * math.cos(phi) - 93.5 * math.cos(3 * phi) + 0.118 * math.cos(5 * phi)
    return float(max(1.0, lat_m)), float(max(1.0, lon_m))

def _to_xy_meters(lat: np.ndarray, lon: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    if lat.size == 0:
        return lat.copy(), lon.copy()
    lat_ref = float(np.nanmean(lat))
    sy, sx = _deg_to_meter_scale(lat_ref)
    x = (lon - lon[0]) * sx
    y = (lat - lat[0]) * sy
    return x, y

def _point_seg_dist(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    vx, vy = bx - ax, by - ay
    wx, wy = px - ax, py - ay
    vv = vx * vx + vy * vy
    if vv <= 1e-12:
        return math.hypot(px - ax, py - ay)
    t = (wx * vx + wy * vy) / vv
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    cx, cy = ax + t * vx, ay + t * vy
    return math.hypot(px - cx, py - cy)

def _dp_indices(x: np.ndarray, y: np.ndarray, eps_m: float) -> List[int]:
    n = len(x)
    if n <= 2:
        return [0, n - 1] if n > 1 else [0]
    keep = np.zeros(n, dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        a, b = stack.pop()
        ax, ay, bx, by = x[a], y[a], x[b], y[b]
        imax, dmax = -1, -1.0
        for i in range(a + 1, b):
            d = _point_seg_dist(x[i], y[i], ax, ay, bx, by)
            if d > dmax:
                dmax, imax = d, i
        if dmax > eps_m and imax != -1:
            keep[imax] = True
            stack.append((a, imax))
            stack.append((imax, b))
    return np.flatnonzero(keep).tolist()

def _cog_angle_diff(cog1: float, cog2: float) -> float:
    """
    EN COG EN（EN），EN。
    
    Args:
        cog1: EN（EN，EN [0, 360]）
        cog2: EN（EN，EN [0, 360]）
    
    Returns:
        EN（EN），EN [-180, 180]
    
    Notes:
        - EN COG = 360 EN 0°（AIS EN）
        - EN NaN EN：EN NaN，EN nanmean/nanstd EN，EN“EN 0”EN
        - EN [-180°, 180°] EN，EN：
          * cog1=10°, cog2=350° → -20° (EN 340°)
          * cog1=350°, cog2=10° → 20°
    """
    if not (np.isfinite(cog1) and np.isfinite(cog2)):
        return float("nan")
    
    cog1 = 0.0 if cog1 == 360.0 else cog1
    cog2 = 0.0 if cog2 == 360.0 else cog2
    
    cog1 = cog1 % 360.0
    cog2 = cog2 % 360.0
    
    diff = cog2 - cog1
    diff = ((diff + 180.0) % 360.0) - 180.0
    
    return float(diff)

def _course_variation_from_adjacent_cog(cog: np.ndarray, i: int) -> float:
    """
    EN AIS EN COG EN Course_Variation EN theta（EN）。
    
    Args:
        cog: COG EN（EN，EN [0, 360]）
        i: EN（EN i%2==0 EN i!=0 EN；EN i-2 EN，EN i>=2）
    
    Returns:
        theta（EN，EN）；EN NaN（EN nanmean/nanstd）
    
    Notes:
        - EN AIS EN COG EN：EN，EN lat/lon EN bearing EN
        - EN Course_Variation(Lat[i-2:i], Lon[i-2:i])（Python EN
          EN），EN**EN**EN AIS COG EN：
          theta = |COG[i] - COG[i-1]|（EN [-180, 180] EN）
        - COG=360 EN 0°；NaN/EN NaN，EN
    """
    if i < 2:
        return float("nan")

    cog_prev = float(cog[i - 1])
    cog_curr = float(cog[i])
    d = _cog_angle_diff(cog_prev, cog_curr)
    return float(abs(d)) if np.isfinite(d) else float("nan")

_course_variation_3pt = _course_variation_from_adjacent_cog

def _sacsb_indices(
    lat: np.ndarray,
    lon: np.ndarray,
    sog: np.ndarray,
    cog: np.ndarray,
    alpha: float = 2.2,
    min_sep: int = 1,
) -> List[int]:
    """
    SACSB EN（EN）：EN（key-point selector）。

    EN（EN i EN“EN”EN）：
      - SACSB EN“EN”，EN COG/SOG EN
      - EN i%2==0 EN i!=0 EN COG_Variation，EN i（EN 2,4,6,...）
      - EN：μc,σc EN theta；μv,σv EN SOG（nanmean/nanstd）
      - EN（EN remove/delete）：
        EN theta EN [μc-ασc, μc+ασc] EN SOG[i] EN [μv-ασv, μv+ασv]，
        EN i→i+1 EN，EN“EN”，EN (i+1) EN keep_indices（EN i+1 < N）。

    Notes（EN COG EN，EN）：
      - EN AIS EN COG EN，EN [-180,180]
      - COG=360 EN 0°（AIS EN）
      - NaN/EN：EN NaN EN nanmean/nanstd EN，EN/EN
    """
    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)
    sog = np.asarray(sog, float)
    cog = np.asarray(cog, float)
    N = int(len(lat))
    if N == 0:
        return []
    COG_Variation: List[Tuple[int, float]] = []

    for orig_i in range(N):
        if orig_i % 2 == 0:
            if orig_i == 0:
                continue
            theta = _course_variation_from_adjacent_cog(cog, orig_i)
            COG_Variation.append((orig_i, theta))

    cv = np.asarray([t for _, t in COG_Variation], dtype=float)
    mu_c = float(np.nanmean(cv)) if cv.size else 0.0
    sg_c = float(np.nanstd(cv)) if cv.size else 0.0
    if not np.isfinite(mu_c):
        mu_c = 0.0
    if (not np.isfinite(sg_c)) or sg_c <= 0.0:
        sg_c = 1e-12

    mu_v = float(np.nanmean(sog)) if sog.size else 0.0
    sg_v = float(np.nanstd(sog)) if sog.size else 0.0
    if not np.isfinite(mu_v):
        mu_v = 0.0
    if (not np.isfinite(sg_v)) or sg_v <= 0.0:
        sg_v = 1e-12

    keep_indices: set[int] = set()
    for orig_i, theta in COG_Variation:
        theta_out = bool(
            np.isfinite(theta)
            and (theta < (mu_c - alpha * sg_c) or theta > (mu_c + alpha * sg_c))
        )
        v_i = float(sog[orig_i]) if 0 <= orig_i < len(sog) else float("nan")
        sog_out = bool(
            np.isfinite(v_i)
            and (v_i < (mu_v - alpha * sg_v) or v_i > (mu_v + alpha * sg_v))
        )
        if theta_out or sog_out:
            kp = int(orig_i + 1)
            if kp < N:
                keep_indices.add(kp)

    try:
        idx_key = sorted(keep_indices)
        print(
            "[SACSB] "
            f"N={N}, len(COG_Variation)={len(COG_Variation)}, "
            f"len(idx_sacsb_keypoints)={len(idx_key)}"
        )
    except Exception:
        pass

    global _SACSB_DEBUG_COUNT
    if keep_indices and _SACSB_DEBUG_COUNT < _SACSB_DEBUG_MAX:
        try:
            print(f"[SACSB] keypoints orig idx (first10)={sorted(keep_indices)[:10]}")
        except Exception:
            pass
        _SACSB_DEBUG_COUNT += 1

    return sorted(keep_indices)

def sacs_b_simplify(
    lat: List[float],
    lon: List[float],
    sog: List[float],
    cog: List[float],
    ship_length_m: float,
    alpha: float = 1.0,
    dp_epsilon_scale: float = 0.8,
) -> Tuple[List[float], List[float], List[int]]:
    """
    Full SACSB simplification (Wei et al., 2024)
      - Stage 1: SACSB (pseudo-code strict mode): delete points when COG variation or SOG exceeds μ ± α·σ
      - Stage 2: DP (ε=0.8×L) position simplification on ORIGINAL position sequence (lat/lon only)
      - Paper reproduction: SACSB and DP run concurrently and the final kept set is merge/union of indices
    """
    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)
    sog = np.asarray(sog, float)
    cog = np.asarray(cog, float)
    n = len(lat)
    if n == 0:
        return [], [], []
    if n == 1:
        return [float(lat[0])], [float(lon[0])], [0]

    idx_sacsb = _sacsb_indices(lat, lon, sog, cog, alpha=alpha, min_sep=1)

    x_all, y_all = _to_xy_meters(lat, lon)
    eps_m = float(max(1.0, dp_epsilon_scale * float(ship_length_m)))
    idx_dp = _dp_indices(x_all, y_all, eps_m)

    idx_merge = sorted(set(idx_sacsb).union(idx_dp))
    if idx_merge:
        if idx_merge[0] != 0:
            idx_merge = [0] + idx_merge
        if idx_merge[-1] != n - 1:
            idx_merge.append(n - 1)
    else:
        idx_merge = [0, n - 1] if n > 1 else [0]

    try:
        ratio = (len(idx_merge) / max(1, n))
        print(
            "[Simplify] "
            f"N={n}, DP_kept={len(idx_dp)}, SACSB_keypoints={len(idx_sacsb)}, "
            f"merge_kept={len(idx_merge)}, compression={(1.0 - ratio):.2%}"
        )
        _MERGED_LENGTHS.append(int(len(idx_merge)))
    except Exception:
        pass

    lat_s = lat[idx_merge].astype(float).tolist()
    lon_s = lon[idx_merge].astype(float).tolist()
    return lat_s, lon_s, idx_merge

if __name__ == "__main__":
    np.random.seed(0)
    lats = np.r_[np.linspace(30, 30.1, 60), np.linspace(30.1, 30.2, 60)]
    lons = np.r_[np.linspace(120, 120.1, 60), np.linspace(120.1, 120.05, 60)]
    lats += (np.random.randn(lats.size) * 5e-5)
    lons += (np.random.randn(lons.size) * 5e-5)
    sogs = 12.0 + 0.5 * np.sin(np.linspace(0, 6.28, lats.size))
    cogs = np.r_[np.linspace(20, 40, 60), np.linspace(40, 110, 60)]

    L = 100.0
    lat_s, lon_s, kept = sacs_b_simplify(
        lats.tolist(), lons.tolist(), sogs.tolist(), cogs.tolist(), ship_length_m=L
    )
    print(f"Original points = {len(lats)}, Simplified = {len(kept)}, rate = {(1 - len(kept)/len(lats))*100:.1f}%")
    print("Indices:", kept)