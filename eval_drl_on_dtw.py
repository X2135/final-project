#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate trained DRL model on a single DTW matrix.
==================================================

Usage Examples
--------------
Run from project root, e.g.:

  python eval_drl_on_dtw.py --dtw-npy dtw_cache/delta_cog/pruned_dataset_dtw_matrix.npy

Or to evaluate a two-stage output directory (automatically parsing DTW path from summary.txt):

    python eval_drl_on_dtw.py --run-dir path/to/two_stage_output

The script will:
1. Load DTW distance matrix D
2. Use trained DRL model (models/drl_multi_from_teacher.pth) to select parameters
3. Run DBSCAN once with selected (eps, minPts) and print DBI' / CP' / SP' metrics
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np

from drl_dbscan import drl_select_eps_minpts, constrain_distance_matrix
from Adaptive_DBSCAN import dbscan_from_distance
from evaluate import evaluate_clusters_NOT_PAPER
from pretrain_drl_teacher import _abs_from_repo, _parse_dtw_path_from_summary, _resolve_run_dir


def _load_dtw_from_args(dtw_npy: Optional[str], run_dir: Optional[str]) -> np.ndarray:
    """
    Load a DTW distance matrix based on command-line arguments.

    Priority order:
    1. If --dtw-npy provided, load that .npy directly
    2. Otherwise if --run-dir provided, parse DTW .npy path from run_dir/summary.txt
    """
    if dtw_npy:
        p = Path(_abs_from_repo(dtw_npy))
        if not p.exists():
            raise FileNotFoundError(f"DTW .npy file not found: {p}")
        D = np.load(p)
        D = np.asarray(D, dtype=float)
        out_dir = Path("dtw_cache") / "eval_drl"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{p.stem}_eval.npy"
        np.save(out_path, D)
        print(f"✅ Loaded DTW from: {p} (shape={D.shape})")
        print(f"💾 Saved independent DTW copy for this evaluation: {out_path}")
        return D

    if run_dir:
        rd = _resolve_run_dir(run_dir)
        summ = rd / "summary.txt"
        pth = _parse_dtw_path_from_summary(summ)
        if not pth:
            raise ValueError(f"Could not parse DTW .npy path from {summ}")
        p = Path(_abs_from_repo(pth))
        if not p.exists():
            raise FileNotFoundError(f"DTW .npy file not found: {p}")
        D = np.load(p)
        D = np.asarray(D, dtype=float)
        out_dir = Path("dtw_cache") / "eval_drl"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{rd.name}_dtw_eval.npy"
        np.save(out_path, D)
        print(f"✅ Loaded DTW from run_dir={rd.name}, path={p} (shape={D.shape})")
        print(f"💾 Generated independent DTW cache for dataset: {out_path}")
        return D

    raise ValueError("Must provide either --dtw-npy or --run-dir.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate DRL model on a single DTW matrix.")
    parser.add_argument(
        "--dtw-npy",
        type=str,
        default=None,
        help="DTW distance matrix .npy file path (takes priority)",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="two-stage or paper output directory (contains run_*/summary.txt, will auto-parse DTW path)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/drl_multi_from_teacher.pth",
        help="Path to DRL model file (default: models/drl_multi_from_teacher.pth)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print DRL internal debug info like eps_pos",
    )
    parser.add_argument(
        "--minpts-stat",
        type=str,
        default="mean",
        choices=["mean", "median"],
        help="minPts calculation method: mean or median (default: mean)",
    )
    args = parser.parse_args()

    D = _load_dtw_from_args(args.dtw_npy, args.run_dir)

    eps, minPts = drl_select_eps_minpts(
        D,
        model_path=_abs_from_repo(args.model),
        minpts_stat=str(args.minpts_stat),
        verbose=bool(args.verbose),
    )
    print("\n================ DRL EN ================")
    print(f"eps   = {eps:.6f}")
    print(f"minPts= {minPts}")

    Dc = constrain_distance_matrix(D)

    labels = dbscan_from_distance(Dc, eps=eps, minPts=minPts)
    K = int(labels.max()) if labels.size else 0
    noise_ratio = float(np.mean(labels == 0)) if labels.size else 1.0

    dbi_p, cp_p, sp_p = evaluate_clusters_NOT_PAPER(Dc, labels)

    print("\n================ EN (DRL, NOT_PAPER) ================")
    print(f"K        = {K}")
    print(f"noise    = {noise_ratio:.2%}")
    print(f"DBI′     = {dbi_p:.4f}")
    print(f"CP′      = {cp_p:.4f}")
    print(f"SP′      = {sp_p:.4f}")
    print("====================================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

