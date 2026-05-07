#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Recompute DTW matrices for 5 datasets x 4 feature modes, then compare metrics.

Goal
----
Use the reproduced paper-style clustering strategy (POA parameter selection),
but recompute DTW matrices into a NEW directory for these feature modes:
  - delta_cog
  - yaw_rate
  - behavior
  - pca_latlon

Datasets (directory inputs) are expected to be folders containing pruned_dataset.csv.

Output
------
1) Recomputed DTW matrices in:
   <output_root>/dtw/<feature>/<dataset_name>_dtw_matrix.npy
2) Detailed CSV:
   <output_root>/feature_metrics_detailed.csv
3) Summary CSV:
   <output_root>/feature_metrics_summary.csv
4) Console table (English):
   Feature | DBI' | CP' | SP' | K | Noise Ratio | Core Ratio
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.cluster import DBSCAN

from load_ais import load_ais_data, clip_sog_kn
from data_pre_processing import interpolate_traj
from trajectory_simplification import sacs_b_simplify
from Adaptive_DBSCAN import (
    build_TS_features,
    build_distance_matrix,
    select_eps_minpts_from_D_paper,
)
from evaluate import evaluate_clusters_NOT_PAPER


FEATURES = ["delta_cog", "yaw_rate", "behavior", "pca_latlon"]


def _prepare_simplified_trajectories(csv_path: Path) -> List[np.ndarray]:
    """
    Reuse the main-pipeline style preprocessing + simplification to build TS_raw.
    """
    trajs = load_ais_data(str(csv_path))
    ship_ids = [sid for sid, t in trajs.items() if len(t["lat"]) >= 50]

    processed: List[np.ndarray] = []
    lengths: List[float] = []
    for sid in ship_ids:
        ship_data = trajs[sid]
        Lm = ship_data.get("length_m", np.nan)
        try:
            Lm = float(Lm)
        except Exception:
            Lm = float("nan")
        if (not np.isfinite(Lm)) or (Lm <= 0.0):
            continue

        traj_interp = interpolate_traj(ship_data, fixed_dt_s=60)
        traj_interp["sog"] = clip_sog_kn(traj_interp["sog"], 0.0, 45.0)
        tr = np.stack(
            [
                np.asarray(traj_interp["lat"], dtype=float),
                np.asarray(traj_interp["lon"], dtype=float),
                np.asarray(traj_interp["sog"], dtype=float),
                np.asarray(traj_interp["cog"], dtype=float),
            ],
            axis=1,
        )
        processed.append(tr)
        lengths.append(Lm)

    simplified: List[np.ndarray] = []
    for i, tr in enumerate(processed):
        lat, lon, sog, cog = tr[:, 0], tr[:, 1], tr[:, 2], tr[:, 3]
        lat_s, lon_s, kept = sacs_b_simplify(lat, lon, sog, cog, ship_length_m=lengths[i], alpha=1.0)
        simp = np.stack([lat[kept], lon[kept], sog[kept], cog[kept]], axis=1)
        simplified.append(simp)
    return simplified


def _to_local_xy(lat: np.ndarray, lon: np.ndarray, lat0: float, lon0: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert lat/lon to local meter coordinates around (lat0, lon0).
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    lat0_rad = np.deg2rad(float(lat0))
    y_m = (lat - float(lat0)) * 111132.92
    x_m = (lon - float(lon0)) * (111412.84 * np.cos(lat0_rad))
    return x_m, y_m


def _zscore_ts(ts_list: List[np.ndarray]) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
    """
    Global z-score over concatenated points of all trajectories.
    """
    if not ts_list:
        return [], np.zeros(4, dtype=float), np.ones(4, dtype=float)
    all_feat = np.vstack(ts_list)
    mu = np.nanmean(all_feat, axis=0)
    sigma = np.nanstd(all_feat, axis=0) + 1e-12
    tsz = [(f - mu) / sigma for f in ts_list]
    return tsz, mu, sigma


def _build_custom_features(
    trajectories: List[np.ndarray],
    mode: str,
    dt_s: float = 60.0,
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
    """
    Build custom TS features for modes not implemented in Adaptive_DBSCAN:
      - yaw_rate
      - behavior
    Each trajectory output is (T,4), then globally z-scored.
    """
    mode_l = (mode or "").strip().lower()
    if mode_l == "behavior_only":
        mode_l = "behavior"
    if mode_l not in ("yaw_rate", "behavior"):
        raise ValueError(f"Unsupported custom mode: {mode}")

    first_lats = [float(tr[0, 0]) for tr in trajectories if len(tr) > 0]
    first_lons = [float(tr[0, 1]) for tr in trajectories if len(tr) > 0]
    lat_ref = float(np.median(first_lats)) if first_lats else 30.0
    lon_ref = float(np.median(first_lons)) if first_lons else 120.0

    ts: List[np.ndarray] = []
    for tr in trajectories:
        lat = np.asarray(tr[:, 0], dtype=float)
        lon = np.asarray(tr[:, 1], dtype=float)
        sog = np.asarray(tr[:, 2], dtype=float)
        cog = np.asarray(tr[:, 3], dtype=float)

        x_abs, y_abs = _to_local_xy(lat, lon, lat_ref, lon_ref)
        x_abs = np.nan_to_num(x_abs, nan=0.0, posinf=0.0, neginf=0.0)
        y_abs = np.nan_to_num(y_abs, nan=0.0, posinf=0.0, neginf=0.0)
        sog = np.nan_to_num(sog, nan=0.0, posinf=0.0, neginf=0.0)
        cog = np.nan_to_num(cog, nan=0.0, posinf=0.0, neginf=0.0)

        delta_cog = np.diff(cog)
        delta_cog = (delta_cog + 180.0) % 360.0 - 180.0
        delta_cog = np.insert(delta_cog, 0, 0.0)

        if mode_l == "yaw_rate":
            yaw_rate = delta_cog / max(float(dt_s), 1e-6)
            f = np.column_stack([x_abs, y_abs, sog, yaw_rate])
        else:
            dx = np.diff(x_abs, prepend=x_abs[0] if len(x_abs) else 0.0)
            dy = np.diff(y_abs, prepend=y_abs[0] if len(y_abs) else 0.0)
            dsog = np.diff(sog, prepend=sog[0] if len(sog) else 0.0)
            f = np.column_stack([dx, dy, dsog, delta_cog])

        f = np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0).astype(float)
        ts.append(f)

    return _zscore_ts(ts)


def _compute_core_ratio(D: np.ndarray, eps: float, minPts: int) -> Tuple[np.ndarray, int, float]:
    """
    Run DBSCAN (precomputed) and return:
      - labels in project convention (0=noise, 1..K)
      - K
      - core_ratio
    """
    db = DBSCAN(eps=float(eps), min_samples=int(minPts), metric="precomputed", n_jobs=-1)
    labels_raw = db.fit_predict(D)
    labels = labels_raw + 1
    labels[labels < 0] = 0
    K = int(labels.max()) if labels.size else 0
    core_ratio = float(len(getattr(db, "core_sample_indices_", [])) / max(1, D.shape[0]))
    return labels, K, core_ratio


def main() -> int:
    parser = argparse.ArgumentParser(description="Recompute DTW for 4 feature modes and compare metrics.")
    parser.add_argument(
        "--dataset-dirs",
        nargs="+",
        required=False,
        help="Dataset directories (each should contain pruned_dataset.csv). Not needed if --read-dtw-from-dir is used.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="dtw_cache/recomputed_feature_compare",
        help="Root directory for new DTW matrices and result CSVs.",
    )
    parser.add_argument(
        "--read-dtw-from-dir",
        type=str,
        default=None,
        help="Read pre-computed DTW matrices from this directory instead of recomputing. "
             "Format: path/to/dtw/ with subdirs: delta_cog/, yaw_rate/, behavior/, pca_latlon/",
    )
    parser.add_argument("--delta-s", type=float, default=0.04, help="POA fixed delta_s.")
    parser.add_argument("--two-stage", action="store_true", help="Use two-stage POA search.")
    parser.add_argument("--coarse-steps", type=int, default=150, help="Two-stage coarse steps.")
    parser.add_argument("--refine-steps", type=int, default=300, help="Two-stage refine steps.")
    parser.add_argument("--refine-window-mult", type=float, default=5.0, help="Two-stage refine window multiplier.")
    parser.add_argument(
        "--minpts-stat",
        type=str,
        default="mean",
        choices=["mean", "median", "paper", "eps_gmax"],
        help="minPts statistic mode in POA selector.",
    )
    parser.add_argument("--n-jobs", type=int, default=-1, help="DTW matrix compute workers.")
    args = parser.parse_args()
    
    if args.read_dtw_from_dir is None and not args.dataset_dirs:
        parser.error("Either --dataset-dirs or --read-dtw-from-dir must be provided.")
    if args.read_dtw_from_dir is not None and args.dataset_dirs:
        print("[INFO] Both --dataset-dirs and --read-dtw-from-dir provided. Using --read-dtw-from-dir (skip DTW compute).")

    repo_root = Path(__file__).resolve().parent
    out_root = (repo_root / args.output_root).resolve()
    dtw_root = out_root / "dtw"
    dtw_root.mkdir(parents=True, exist_ok=True)

    detailed_rows: List[Dict[str, object]] = []

    if args.read_dtw_from_dir is not None:
        print(f"\n[MODE] Reading pre-computed DTW matrices from: {args.read_dtw_from_dir}")
        dtw_read_root = Path(args.read_dtw_from_dir).resolve()
        if not dtw_read_root.exists():
            print(f"[ERROR] DTW directory does not exist: {dtw_read_root}")
            return 1

        dataset_names: set = set()
        for feature_dir in dtw_read_root.iterdir():
            if feature_dir.is_dir():
                for npy_file in feature_dir.glob("*_dtw_matrix.npy"):
                    dataset_name = npy_file.stem.replace("_dtw_matrix", "")
                    dataset_names.add(dataset_name)

        dataset_names = sorted(list(dataset_names))
        print(f"[INFO] Found {len(dataset_names)} datasets: {dataset_names}")

        for dataset_name in dataset_names:
            print(f"\n=== Dataset: {dataset_name} (reading pre-computed DTW) ===")
            for mode in FEATURES:
                print(f"  -> Feature mode: {mode}")
                try:
                    dtw_src_path = dtw_read_root / mode / f"{dataset_name}_dtw_matrix.npy"
                    if not dtw_src_path.exists():
                        print(f"     [WARN] DTW file not found: {dtw_src_path}")
                        detailed_rows.append({
                            "Dataset": dataset_name,
                            "Feature": mode,
                            "DTW Path": "",
                            "eps": np.nan,
                            "minPts": np.nan,
                            "DBI'": np.nan,
                            "CP'": np.nan,
                            "SP'": np.nan,
                            "K": 0,
                            "Noise Ratio": 1.0,
                            "Core Ratio": 0.0,
                        })
                        continue

                    D = np.load(dtw_src_path)
                    print(f"     loaded={dtw_src_path.name}, shape={D.shape}")

                    eps, minPts = select_eps_minpts_from_D_paper(
                        D,
                        delta_s=float(args.delta_s),
                        verbose=False,
                        minpts_stat=str(args.minpts_stat),
                        two_stage=bool(args.two_stage),
                        coarse_steps=int(args.coarse_steps),
                        refine_steps=int(args.refine_steps),
                        refine_window_mult=float(args.refine_window_mult),
                    )
                    labels, K, core_ratio = _compute_core_ratio(D, eps, minPts)
                    noise_ratio = float(np.mean(labels == 0)) if labels.size else 1.0
                    dbi_p, cp_p, sp_p = evaluate_clusters_NOT_PAPER(D, labels)

                    detailed_rows.append({
                        "Dataset": dataset_name,
                        "Feature": mode,
                        "DTW Path": str(dtw_src_path),
                        "eps": float(eps),
                        "minPts": int(minPts),
                        "DBI'": float(dbi_p) if np.isfinite(dbi_p) else np.nan,
                        "CP'": float(cp_p) if np.isfinite(cp_p) else np.nan,
                        "SP'": float(sp_p) if np.isfinite(sp_p) else np.nan,
                        "K": int(K),
                        "Noise Ratio": float(noise_ratio),
                        "Core Ratio": float(core_ratio),
                    })
                    print(
                        f"     K={K}, noise={noise_ratio:.2%}, "
                        f"DBI'={dbi_p:.4f}, CP'={cp_p:.4f}, SP'={sp_p:.4f}"
                    )
                except Exception as e:
                    print(f"     [WARN] failed on mode={mode}: {e}")
                    detailed_rows.append({
                        "Dataset": dataset_name,
                        "Feature": mode,
                        "DTW Path": "",
                        "eps": np.nan,
                        "minPts": np.nan,
                        "DBI'": np.nan,
                        "CP'": np.nan,
                        "SP'": np.nan,
                        "K": 0,
                        "Noise Ratio": 1.0,
                        "Core Ratio": 0.0,
                    })

    else:
        print(f"\n[MODE] Computing DTW matrices from scratch")
        for ds_dir_raw in args.dataset_dirs:
            ds_dir = Path(ds_dir_raw)
            if not ds_dir.is_absolute():
                ds_dir = (repo_root / ds_dir).resolve()
            dataset_name = ds_dir.name
            csv_path = ds_dir / "pruned_dataset.csv"
            if not csv_path.exists():
                print(f"[WARN] skip dataset: missing {csv_path}")
                continue

            print(f"\n=== Dataset: {dataset_name} ===")
            simplified = _prepare_simplified_trajectories(csv_path)
            if not simplified:
                print(f"[WARN] no valid trajectories after preprocess/simplify: {dataset_name}")
                continue
            print(f"Prepared trajectories: {len(simplified)}")

            for mode in FEATURES:
                print(f"  -> Feature mode: {mode}")
                try:
                    if mode in ("delta_cog", "pca_latlon"):
                        TSz, mu, sigma = build_TS_features(
                            simplified,
                            feature_mode=mode,
                            dt_s=60.0,
                        )
                    else:
                        TSz, mu, sigma = _build_custom_features(simplified, mode=mode, dt_s=60.0)

                    D = build_distance_matrix(TSz, verbose=False, n_jobs=int(args.n_jobs))

                    mode_dir = dtw_root / mode
                    mode_dir.mkdir(parents=True, exist_ok=True)
                    dtw_path = mode_dir / f"{dataset_name}_dtw_matrix.npy"
                    np.save(dtw_path, D)

                    eps, minPts = select_eps_minpts_from_D_paper(
                        D,
                        delta_s=float(args.delta_s),
                        verbose=False,
                        minpts_stat=str(args.minpts_stat),
                        two_stage=bool(args.two_stage),
                        coarse_steps=int(args.coarse_steps),
                        refine_steps=int(args.refine_steps),
                        refine_window_mult=float(args.refine_window_mult),
                    )
                    labels, K, core_ratio = _compute_core_ratio(D, eps, minPts)
                    noise_ratio = float(np.mean(labels == 0)) if labels.size else 1.0
                    dbi_p, cp_p, sp_p = evaluate_clusters_NOT_PAPER(D, labels)

                    detailed_rows.append(
                        {
                            "Dataset": dataset_name,
                            "Feature": mode,
                            "DTW Path": str(dtw_path),
                            "eps": float(eps),
                            "minPts": int(minPts),
                            "DBI'": float(dbi_p) if np.isfinite(dbi_p) else np.nan,
                            "CP'": float(cp_p) if np.isfinite(cp_p) else np.nan,
                            "SP'": float(sp_p) if np.isfinite(sp_p) else np.nan,
                            "K": int(K),
                            "Noise Ratio": float(noise_ratio),
                            "Core Ratio": float(core_ratio),
                        }
                    )
                    print(
                        f"     saved={dtw_path.name}, K={K}, noise={noise_ratio:.2%}, "
                        f"DBI'={dbi_p:.4f}, CP'={cp_p:.4f}, SP'={sp_p:.4f}"
                    )
                except Exception as e:
                    print(f"     [WARN] failed on mode={mode}: {e}")
                    detailed_rows.append(
                        {
                            "Dataset": dataset_name,
                            "Feature": mode,
                            "DTW Path": "",
                            "eps": np.nan,
                            "minPts": np.nan,
                            "DBI'": np.nan,
                            "CP'": np.nan,
                            "SP'": np.nan,
                            "K": 0,
                            "Noise Ratio": 1.0,
                            "Core Ratio": 0.0,
                        }
                    )

    out_root.mkdir(parents=True, exist_ok=True)
    detailed_csv = out_root / "feature_metrics_detailed.csv"
    with detailed_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "Dataset",
                "Feature",
                "DTW Path",
                "eps",
                "minPts",
                "DBI'",
                "CP'",
                "SP'",
                "K",
                "Noise Ratio",
                "Core Ratio",
            ],
        )
        writer.writeheader()
        writer.writerows(detailed_rows)

    summary_rows: List[Dict[str, object]] = []
    for mode in FEATURES:
        rows = [r for r in detailed_rows if r["Feature"] == mode]
        if not rows:
            continue
        def _arr(key: str) -> np.ndarray:
            vals = [float(r[key]) for r in rows if np.isfinite(float(r[key]))]
            return np.asarray(vals, dtype=float) if vals else np.asarray([], dtype=float)

        dbi = _arr("DBI'")
        cp = _arr("CP'")
        sp = _arr("SP'")
        k = _arr("K")
        noise = _arr("Noise Ratio")
        core = _arr("Core Ratio")
        summary_rows.append(
            {
                "Feature": mode,
                "DBI'": float(np.mean(dbi)) if dbi.size else np.nan,
                "CP'": float(np.mean(cp)) if cp.size else np.nan,
                "SP'": float(np.mean(sp)) if sp.size else np.nan,
                "K": int(round(float(np.mean(k)))) if k.size else 0,
                "Noise Ratio": float(np.mean(noise)) if noise.size else 1.0,
                "Core Ratio": float(np.mean(core)) if core.size else 0.0,
            }
        )

    summary_csv = out_root / "feature_metrics_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["Feature", "DBI'", "CP'", "SP'", "K", "Noise Ratio", "Core Ratio"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\nFeature | DBI' | CP' | SP' | K | Noise Ratio | Core Ratio")
    print("-" * 72)
    for r in summary_rows:
        dbi_v = float(r["DBI'"])
        cp_v = float(r["CP'"])
        sp_v = float(r["SP'"])
        dbi_s = f"{dbi_v:.4f}" if np.isfinite(dbi_v) else "nan"
        cp_s = f"{cp_v:.4f}" if np.isfinite(cp_v) else "nan"
        sp_s = f"{sp_v:.4f}" if np.isfinite(sp_v) else "nan"
        print(
            f"{r['Feature']} | {dbi_s} | {cp_s} | {sp_s} | {int(r['K'])} | "
            f"{float(r['Noise Ratio']):.2%} | {float(r['Core Ratio']):.2%}"
        )

    print("\nDataset | Feature | DBI' | CP' | SP' | K | Noise Ratio | Core Ratio")
    print("-" * 110)
    for r in detailed_rows:
        dbi_v = float(r["DBI'"])
        cp_v = float(r["CP'"])
        sp_v = float(r["SP'"])
        dbi_s = f"{dbi_v:.4f}" if np.isfinite(dbi_v) else "nan"
        cp_s = f"{cp_v:.4f}" if np.isfinite(cp_v) else "nan"
        sp_s = f"{sp_v:.4f}" if np.isfinite(sp_v) else "nan"
        print(
            f"{r['Dataset']} | {r['Feature']} | {dbi_s} | {cp_s} | {sp_s} | {int(r['K'])} | "
            f"{float(r['Noise Ratio']):.2%} | {float(r['Core Ratio']):.2%}"
        )

    print(f"\nSaved detailed results: {detailed_csv}")
    print(f"Saved summary results : {summary_csv}")
    print(f"Saved DTW matrices dir: {dtw_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

