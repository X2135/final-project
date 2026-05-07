#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train DRL on multiple DTW .npy matrices using drl_dbscan.train_drl_dbscan_multi.

Example usage:

    python train_drl_on_npy_list.py \\
        --train path/to/a.npy path/to/b.npy path/to/c.npy \\
        --model-save models/drl_split_train.pth \\
        --log logs/drl_split_train.csv

    # Or load a glob of DTW matrices
    python train_drl_on_npy_list.py \\
        --train-glob "dtw_cache/delta_cog/*_dtw_matrix.npy" \\
        --model-save models/drl_all.pth \\
        --log logs/drl_all.csv
"""
from __future__ import annotations

import argparse
import glob as glob_mod
from pathlib import Path

import numpy as np

from drl_dbscan import train_drl_dbscan_multi


def main() -> int:
    p = argparse.ArgumentParser(description="Train DRL on multiple DTW .npy matrices.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--train",
        nargs="+",
        help="Training DTW matrix .npy paths (can be multiple)",
    )
    src.add_argument(
        "--train-glob",
        type=str,
        metavar="PATTERN",
        help=(
            "Training DTW matrix glob pattern (relative to project root), e.g., "
            "'dtw_cache/delta_cog/*_dtw_matrix.npy'"
        ),
    )
    p.add_argument("--episodes", type=int, default=500, help="Number of training episodes (default: 500)")
    p.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    p.add_argument(
        "--minpts-stat",
        type=str,
        default="mean",
        choices=["mean", "median"],
        help="minPts calculation: mean or median (default: mean)",
    )
    p.add_argument(
        "--model-save",
        type=str,
        default="models/drl_from_npy_list.pth",
        help="Save path for trained PPO model",
    )
    p.add_argument(
        "--log",
        type=str,
        default=None,
        help="Optional: Training CSV log path",
    )
    args = p.parse_args()

    root = Path(__file__).resolve().parent
    if args.train_glob:
        pattern = args.train_glob
        if not Path(pattern).is_absolute():
            pattern = str(root / pattern)
        train_paths = sorted(glob_mod.glob(pattern))
        if not train_paths:
            raise FileNotFoundError(f"--train-glob matched no files: {args.train_glob!r} -> {pattern!r}")
    else:
        train_paths = list(args.train or [])

    # Load DTW distance matrices
    Ds = []
    for rel in train_paths:
        path = Path(rel)
        if not path.is_absolute():
            path = root / path
        if not path.exists():
            raise FileNotFoundError(f"DTW file does not exist: {path}")
        D = np.load(path)
        # Validate that matrix is square
        if D.ndim != 2 or D.shape[0] != D.shape[1]:
            raise ValueError(f"DTW matrix must be square: {path}, shape={D.shape}")
        Ds.append(np.asarray(D, dtype=float))
        print(f"✅ Loaded {path} shape={D.shape}")

    # Train DRL model
    train_drl_dbscan_multi(
        Ds,
        num_episodes=int(args.episodes),
        max_steps_per_episode=1,
        model_save_path=str(args.model_save),
        log_path=args.log,
        minpts_stat=str(args.minpts_stat),
        verbose=True,
        seed=int(args.seed),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
