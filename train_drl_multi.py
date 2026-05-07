#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train DRL (PPO) for DBSCAN parameter selection on MULTIPLE datasets.
====================================================================

Why
---
Your current DRL training (`--train-drl` in `run_complete_pipeline.py`) trains on ONE dataset,
so it easily overfits and does not generalize.

This script trains on a list of DTW distance matrices (or pipeline run dirs),
sampling one dataset per episode (contextual bandit across datasets).

Usage
-----
Option A: provide DTW .npy matrices directly:
  python train_drl_multi.py --dtw-npy A.npy B.npy C.npy --save ./drl_model_multi.pth

Option B: provide pipeline run dirs (it will parse summary.txt to locate dtw .npy):
  python train_drl_multi.py --run-dirs outputs/test_cleaning_seed/run_20260206_174317 outputs/test_cleaning_seed/run_20260226_154901 --save ./drl_model_multi.pth
"""

from __future__ import annotations

import argparse
import os
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np

from drl_dbscan import train_drl_dbscan_multi


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _default_out_dir() -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (_repo_root() / "outputs" / f"drl_multi_train_{ts}").resolve()


def _abs_from_repo(p: str) -> str:
    x = (p or "").strip()
    if not x:
        return x
    if os.path.isabs(x):
        return x
    return str((_repo_root() / x).resolve())


def _parse_dtw_path_from_summary(summary_path: Path) -> Optional[str]:
    text = summary_path.read_text(encoding="utf-8", errors="ignore")
    patterns = [
        r"ENDTWEN:\s*([^\s]+\.npy)\s*$",
        r"DTWEN:\s*([^\s]+\.npy)\s*$",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.MULTILINE)
        if m:
            return _abs_from_repo(m.group(1).strip())
    return None


def _resolve_run_dir(p: str) -> Path:
    """
    Accept either:
    - a concrete run dir containing summary.txt, OR
    - a parent directory containing run_*/summary.txt (choose latest by mtime).
    """
    path = Path(_abs_from_repo(p))
    if not path.exists():
        raise FileNotFoundError(f"run_dir does not exist: {path}")
    if path.is_file():
        if path.name == "summary.txt":
            return path.parent
        raise FileNotFoundError(f"Expected a directory or summary.txt, got file: {path}")

    summ = path / "summary.txt"
    if summ.exists():
        return path

    candidates = [pp for pp in path.glob("run_*/summary.txt") if pp.is_file()]
    if not candidates:
        raise FileNotFoundError(f"summary.txt not found in run_dir or any run_*/ under: {path}")
    candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return candidates[0].parent


def _plot_drl_diagnostics(csv_path: Path) -> None:
    """
    Generate 3 quick diagnostic plots next to the training CSV:
      1) reward curve (episode vs reward)
      2) K/num_clusters histogram
      3) eps_pos histogram

    This is intentionally lightweight and does not require pandas.
    """
    try:
        import csv as _csv
        import math as _math

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:
        print(f"⚠️ EN DRL EN（EN matplotlib）：{e}")
        return

    if not csv_path.exists():
        return

    episodes: List[int] = []
    rewards: List[float] = []
    ks: List[int] = []
    eps_pos: List[float] = []

    with csv_path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        r = _csv.DictReader(f)
        for row in r:
            try:
                ep = int(float(row.get("episode", "0") or "0"))
                rw = float(row.get("reward", "nan") or "nan")
                k = int(float(row.get("num_clusters", "0") or "0"))
                pos = float(row.get("eps_pos", "nan") or "nan")
            except Exception:
                continue
            if ep <= 0:
                continue
            if not _math.isfinite(rw):
                continue
            episodes.append(ep)
            rewards.append(float(rw))
            ks.append(int(k))
            if _math.isfinite(pos):
                eps_pos.append(float(pos))

    out_dir = csv_path.parent
    stem = csv_path.stem

    if episodes and rewards:
        plt.figure(figsize=(10, 4))
        plt.plot(episodes, rewards, linewidth=1.2)
        plt.xlabel("episode")
        plt.ylabel("reward")
        plt.title("DRL training reward curve")
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / f"{stem}_reward_curve.png", dpi=200)
        plt.close()

    if ks:
        plt.figure(figsize=(6, 4))
        plt.hist(ks, bins=min(30, max(5, len(set(ks)))), edgecolor="black", alpha=0.8)
        plt.xlabel("K (num_clusters)")
        plt.ylabel("count")
        plt.title("DRL K distribution")
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / f"{stem}_K_hist.png", dpi=200)
        plt.close()

    if eps_pos:
        plt.figure(figsize=(6, 4))
        plt.hist(eps_pos, bins=20, range=(0.0, 1.0), edgecolor="black", alpha=0.8)
        plt.xlabel("eps_pos in [0,1]")
        plt.ylabel("count")
        plt.title("DRL eps_pos distribution")
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / f"{stem}_eps_pos_hist.png", dpi=200)
        plt.close()

    print(f"🖼️ EN DRL EN: {out_dir}")


def main() -> int:
    p = argparse.ArgumentParser(description="Train DRL on multiple DTW matrices for better generalization.")
    p.add_argument("--dtw-npy", type=str, nargs="*", default=[], help="List of DTW .npy matrices (NxN).")
    p.add_argument("--run-dirs", type=str, nargs="*", default=[], help="List of pipeline run dirs (each contains summary.txt).")
    p.add_argument("--episodes", type=int, default=300, help="Number of training episodes.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory for this training run. If omitted, a new timestamped folder under ./outputs/ will be created.",
    )
    p.add_argument("--save", type=str, default="drl_model_multi.pth", help="Output model filename or path.")
    p.add_argument("--load", type=str, default=None, help="Optional model to load and continue training.")
    p.add_argument("--log", type=str, default="drl_train_multi.csv", help="Training log CSV filename or path.")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else _default_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    save_path = Path(args.save).expanduser()
    if not save_path.is_absolute():
        save_path = (out_dir / save_path.name).resolve()
    else:
        save_path = save_path.resolve()

    log_path = None
    if args.log:
        lp = Path(args.log).expanduser()
        if not lp.is_absolute():
            lp = (out_dir / lp.name).resolve()
        else:
            lp = lp.resolve()
        log_path = str(lp)

    dtw_paths: List[str] = []
    for x in args.dtw_npy:
        dtw_paths.append(_abs_from_repo(x))

    for rd in args.run_dirs:
        run_dir = _resolve_run_dir(rd)
        summ = run_dir / "summary.txt"
        pth = _parse_dtw_path_from_summary(summ)
        if not pth:
            raise ValueError(f"Cannot parse DTW .npy path from: {summ}")
        dtw_paths.append(pth)

    dtw_paths = list(dict.fromkeys(dtw_paths))
    if len(dtw_paths) < 2:
        raise ValueError(f"Need >=2 DTW matrices for multi-dataset training, got {len(dtw_paths)}")

    Ds: List[np.ndarray] = []
    for pth in dtw_paths:
        arr = np.load(pth)
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
            raise ValueError(f"DTW matrix must be NxN, got {arr.shape} from {pth}")
        Ds.append(np.asarray(arr, dtype=float))

    if not args.quiet:
        print(f"✅ Loaded {len(Ds)} DTW matrices for training.")
        for i, pth in enumerate(dtw_paths[:10]):
            print(f"  - [{i}] {pth}")
        print(f"📁 Training outputs will be written to: {out_dir}")
        print(f"💾 Model path: {save_path}")
        if log_path:
            print(f"🧾 Log CSV path: {log_path}")

    train_drl_dbscan_multi(
        Ds,
        num_episodes=int(args.episodes),
        max_steps_per_episode=1,
        model_save_path=str(save_path),
        model_load_path=str(Path(args.load).expanduser().resolve()) if args.load else None,
        log_path=log_path,
        verbose=not bool(args.quiet),
        seed=int(args.seed),
    )

    if log_path:
        try:
            _plot_drl_diagnostics(Path(log_path).resolve())
        except Exception as e:
            print(f"⚠️ EN DRL EN（EN）：{e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

