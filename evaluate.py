"""
Paper-strict clustering evaluation (Wei et al., 2024).

This module provides two evaluators:

- evaluate_clusters_paper():
  Paper-strict implementation aligned with Wei et al. (2024)
  Eq.(15)–(18) + centre trajectory Eq.(21)–(22).

- evaluate_clusters_pairwise_approx():
  Debug-only legacy approximation based on a precomputed distance matrix.
  NOT paper definition.

Important:
  - Input is the feature sequences (e.g. TSz), NOT a precomputed distance matrix.
  - labels==0 is treated as noise and ignored (clusters are labels>0).
"""

from typing import Callable, List, Tuple
import numpy as np


def _pad_to_len_paper(ts: np.ndarray, m: int) -> np.ndarray:
    """
    Paper padding Eq.(21)-(22) (feature layout: [x, y, v, w]):
      - if len(ts) < m:
        * position dims (0,1): extend with last value
        * motion dims   (2,3): pad with 0
      - if len(ts) >= m: truncate/keep first m points (keep as-is)
    """
    ts = np.asarray(ts, dtype=float)
    if m <= 0:
        return ts[:0].reshape((0, ts.shape[1] if ts.ndim == 2 else 4))
    if ts.ndim != 2 or ts.shape[1] < 4:
        raise ValueError("TS element must be a 2D array with at least 4 columns.")

    n = int(len(ts))
    if n == m:
        return ts
    if n > m:
        return ts[:m]

    x_last = float(ts[-1, 0]) if n > 0 and np.isfinite(ts[-1, 0]) else 0.0
    y_last = float(ts[-1, 1]) if n > 0 and np.isfinite(ts[-1, 1]) else 0.0
    pad = np.zeros((m - n, ts.shape[1]), dtype=float)
    pad[:, 0] = x_last
    pad[:, 1] = y_last
    return np.vstack([ts, pad])


def evaluate_clusters_pairwise_approx(D: np.ndarray, labels: np.ndarray) -> Tuple[float, float, float]:
    """
    Pairwise-approx evaluation (NOT paper definition).

    This is the legacy diagnostic metric based on a precomputed distance matrix.
    Kept only for debugging; DO NOT use for paper results.
    """
    if D is None or D.ndim != 2:
        return float("nan"), float("nan"), float("nan")
    uniq = sorted([k for k in np.unique(labels) if k > 0])
    K = len(uniq)
    if K < 2:
        return float("nan"), float("nan"), float("nan")

    cluster_indices = [np.where(labels == k)[0] for k in uniq]

    def _avg_intra_distance(idx: np.ndarray) -> float:
        if idx.size <= 1:
            return 0.0
        sub = D[np.ix_(idx, idx)]
        n = idx.size
        s = np.sum(sub) - np.sum(np.diag(sub))
        return float(s / max(1, n * (n - 1)))

    def _avg_inter_distance(idx_a: np.ndarray, idx_b: np.ndarray) -> float:
        if idx_a.size == 0 or idx_b.size == 0:
            return np.inf
        sub = D[np.ix_(idx_a, idx_b)]
        return float(np.mean(sub))

    intra = np.array([_avg_intra_distance(idx) for idx in cluster_indices], dtype=float)
    sizes = np.array([len(idx) for idx in cluster_indices], dtype=float)
    total = float(np.sum(sizes))
    cp_prime = float(np.sum(intra * (sizes / max(1.0, total))))

    inter_mat = np.full((K, K), np.inf, dtype=float)
    for i in range(K):
        for j in range(i + 1, K):
            m = _avg_inter_distance(cluster_indices[i], cluster_indices[j])
            inter_mat[i, j] = inter_mat[j, i] = m
    sp_prime = float(np.min(inter_mat[np.isfinite(inter_mat)])) if np.isfinite(inter_mat).any() else float("nan")

    db_vals = []
    for i in range(K):
        vals = []
        for j in range(K):
            if i == j:
                continue
            mij = inter_mat[i, j]
            if np.isfinite(mij) and mij > 0:
                vals.append((intra[i] + intra[j]) / mij)
        if vals:
            db_vals.append(max(vals))
    dbi_prime = float(np.mean(db_vals)) if db_vals else float("nan")
    return dbi_prime, cp_prime, sp_prime


def evaluate_clusters_paper(
    TS: List[np.ndarray],
    labels: np.ndarray,
    dtw_dist: Callable[[np.ndarray, np.ndarray], float],
    pad_mode: str = "paper",
) -> Tuple[float, float, float]:
    """
    Paper-strict evaluation (Wei et al., 2024 Eq.(15)-(18) + Eq.(21)-(22)).

    Args:
        TS: List of trajectories (each is N×4 feature sequence).
        labels: cluster labels (1..K for clusters, 0 for noise). Noise is ignored.
        dtw_dist: DTW distance function consistent with clustering (same window/normalization).
        pad_mode: "paper" only (kept for explicitness).

    Returns:
        (DBI_prime, CP_prime, SP_prime). If number of valid clusters < 2, returns (nan,nan,nan).

    Strictness (per request):
        - Centre-centre DTW distances D(c_i, c_j) that are non-finite or <= 0 make the
          corresponding term unusable; in strict mode we DO NOT "skip-and-average".
          Instead, SP′ and DBI′ become NaN if any required centre-centre term is invalid.
    """
    if pad_mode != "paper":
        raise ValueError("Only pad_mode='paper' is supported in this paper-strict implementation.")

    labels = np.asarray(labels)
    uniq = sorted([int(k) for k in np.unique(labels) if int(k) > 0])
    if len(uniq) < 2:
        return float("nan"), float("nan"), float("nan")

    centres = {}
    cbar = {}
    for k in uniq:
        idx = np.where(labels == k)[0]
        if idx.size == 0:
            continue
        members = [TS[int(i)] for i in idx]
        lengths = [int(len(t)) for t in members]
        m = int(max(lengths)) if lengths else 0
        if m <= 0:
            continue

        padded = [_pad_to_len_paper(t, m) for t in members]
        centre = np.mean(np.stack(padded, axis=0), axis=0)
        centres[k] = centre

        dists = []
        ok = True
        for t in padded:
            d = float(dtw_dist(t, centre))
            if not np.isfinite(d):
                ok = False
                break
            dists.append(d)
        cbar[k] = float(np.mean(dists)) if (ok and dists) else float("nan")

    valid_clusters = [k for k in uniq if k in centres and np.isfinite(cbar.get(k, np.nan))]
    K = len(valid_clusters)
    if K < 2:
        return float("nan"), float("nan"), float("nan")

    CP_prime = float(np.mean([cbar[k] for k in valid_clusters]))

    dc = {}
    sum_pair = 0.0
    all_pairs_valid = True
    for a_i in range(K):
        for a_j in range(a_i + 1, K):
            ki = valid_clusters[a_i]
            kj = valid_clusters[a_j]
            ci = centres[ki]
            cj = centres[kj]
            L = max(int(len(ci)), int(len(cj)))
            ci_p = _pad_to_len_paper(ci, L)
            cj_p = _pad_to_len_paper(cj, L)
            d = float(dtw_dist(ci_p, cj_p))
            if not (np.isfinite(d) and d > 0):
                all_pairs_valid = False
                break
            dc[(ki, kj)] = d
            dc[(kj, ki)] = d
            sum_pair += d
        if not all_pairs_valid:
            break

    if all_pairs_valid:
        SP_prime = float((2.0 * sum_pair) / max(1.0, (K * K - K)))
    else:
        SP_prime = float("nan")

    if not all_pairs_valid:
        DBI_prime = float("nan")
    else:
        db_terms = []
        for ki in valid_clusters:
            vals = []
            for kj in valid_clusters:
                if kj == ki:
                    continue
                dij = dc[(ki, kj)]
                vals.append((cbar[ki] + cbar[kj]) / dij)
            db_terms.append(max(vals))
        DBI_prime = float(np.mean(db_terms)) if db_terms else float("nan")

    return DBI_prime, CP_prime, SP_prime


evaluate_clusters_NOT_PAPER = evaluate_clusters_pairwise_approx


