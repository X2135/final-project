#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch-evaluate multiple AIS testsets with multiple parameter-selection methods.

It runs `run_complete_pipeline.py` for each dataset under each method, then parses
`summary.txt` to extract key metrics and writes a single `results.csv` suitable
for paper tables.

Methods (default):
  - paper    : Wei et al. paper POA (single-stage eps scan)
  - improved : POA two-stage eps selection (coarse+refine)
  - drl      : DRL inference (load model; no training)

Outputs:
  out_root/
    results.csv
    runs/<dataset_tag>/<method>/run_YYYYmmdd_HHMMSS/...
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class RunResult:
    dataset_tag: str
    ais_path: str
    method: str
    out_base: str
    run_dir: str
    ok: bool
    error: str
    K: Optional[int]
    noise_ratio: Optional[float]
    eps: Optional[float]
    minPts: Optional[int]
    DBI_prime: Optional[float]
    CP_prime: Optional[float]
    SP_prime: Optional[float]


_RE_K_NOISE_EPS_MINPTS = re.compile(
    r"K=(?P<K>\d+)\s*,\s*EN=(?P<noise>[\d.]+)%\s*,\s*eps=(?P<eps>[-+eE0-9.]+)\s*,\s*minPts=(?P<minpts>\d+)"
)
_RE_DBI = re.compile(r"\bDBI[′']\s*:\s*(?P<val>[-+eE0-9.]+)")
_RE_CP = re.compile(r"\bCP[′']\s*:\s*(?P<val>[-+eE0-9.]+)")
_RE_SP = re.compile(r"\bSP[′']\s*:\s*(?P<val>[-+eE0-9.]+)")
_RE_MINPTS_MODE_1 = re.compile(r"\[minPts\]\s*use=(?P<mode>[A-Za-z0-9_*]+)")
_RE_MINPTS_MODE_2 = re.compile(r"\[minPts@eps\].*?\buse=(?P<mode>[A-Za-z0-9_*]+)\b")
_RE_TWO_STAGE = re.compile(r"\bPOA two-stage\b|\[Two-stage\]")


def _try_get_matplotlib():
    """
    Use project-local matplotlib setup (Agg backend + Chinese fonts).
    Returns (plt) or None if matplotlib is unavailable.
    """
    try:
        from visualization import _ensure_matplotlib  # type: ignore
        _, plt = _ensure_matplotlib()
        return plt
    except Exception:
        return None


def _read_results_csv(results_path: str) -> List[Dict[str, str]]:
    if not os.path.exists(results_path):
        return []
    with open(results_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return [dict(row) for row in r]


def _to_float(x: Optional[str]) -> Optional[float]:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None


def _to_int(x: Optional[str]) -> Optional[int]:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    try:
        return int(float(s))
    except Exception:
        return None


def _make_comparison_plots(results_path: str, out_root: str) -> None:
    """
    Generate method-comparison plots from results.csv:
      - grouped bar charts for DBI′, CP′, SP′, K, noise_ratio
      - one combined panel plot (5 subplots)
    """
    plt = _try_get_matplotlib()
    if plt is None:
        print("⚠️ matplotlib EN，EN（results.csv EN）。")
        return

    rows = _read_results_csv(results_path)
    if not rows:
        print("⚠️ results.csv EN，EN。")
        return

    rows_ok = [r for r in rows if str(r.get("ok", "")).strip() in ("1", "true", "True")]
    if not rows_ok:
        print("⚠️ results.csv EN ok=1 EN，EN。")
        return

    methods = sorted(set(r.get("method", "") for r in rows_ok if r.get("method")))
    datasets = sorted(set(r.get("dataset_tag", "") for r in rows_ok if r.get("dataset_tag")))
    if not methods or not datasets:
        print("⚠️ results.csv EN method/dataset_tag，EN。")
        return

    def val(dataset_tag: str, method: str, key: str):
        for r in rows_ok:
            if r.get("dataset_tag") == dataset_tag and r.get("method") == method:
                if key in ("K", "minPts"):
                    return _to_int(r.get(key))
                return _to_float(r.get(key))
        return None

    metrics = [
        ("DBI_prime", "DBI′ (↓)"),
        ("CP_prime", "CP′ (↓)"),
        ("SP_prime", "SP′ (↑)"),
        ("K", "K (#clusters)"),
        ("noise_ratio", "Noise ratio (↓)"),
    ]

    plots_dir = os.path.join(out_root, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    import numpy as np

    for key, title in metrics:
        X = np.arange(len(datasets), dtype=float)
        width = 0.8 / max(1, len(methods))

        fig = plt.figure(figsize=(max(10, 1.2 * len(datasets)), 5.2))
        ax = plt.gca()
        for i, m in enumerate(methods):
            ys = []
            for d in datasets:
                v = val(d, m, key)
                ys.append(np.nan if v is None else float(v))
            ax.bar(X + (i - (len(methods) - 1) / 2) * width, ys, width=width, label=m)

        ax.set_title(f"Method comparison: {title}")
        ax.set_xticks(X)
        ax.set_xticklabels(datasets, rotation=25, ha="right")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(ncols=min(len(methods), 4), frameon=False)

        if key == "noise_ratio":
            ax.set_ylim(0.0, 1.0)

        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, f"compare_{key}.png"), dpi=220)
        plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(max(12, 1.35 * len(datasets)), 8.5))
    axes = axes.flatten()
    for idx, (key, title) in enumerate(metrics):
        ax = axes[idx]
        X = np.arange(len(datasets), dtype=float)
        width = 0.8 / max(1, len(methods))
        for i, m in enumerate(methods):
            ys = []
            for d in datasets:
                v = val(d, m, key)
                ys.append(np.nan if v is None else float(v))
            ax.bar(X + (i - (len(methods) - 1) / 2) * width, ys, width=width, label=m)
        ax.set_title(title)
        ax.set_xticks(X)
        ax.set_xticklabels(datasets, rotation=25, ha="right")
        ax.grid(True, axis="y", alpha=0.25)
        if key == "noise_ratio":
            ax.set_ylim(0.0, 1.0)
    if len(metrics) < len(axes):
        ax = axes[len(metrics)]
        ax.axis("off")
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, loc="center", frameon=False, ncols=min(len(methods), 3))
    fig.suptitle("Batch evaluation comparison (paper / improved / drl)", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "compare_panel.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"📊 EN: {plots_dir}")


def _read_ais_list(path: str) -> List[str]:
    items: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            items.append(os.path.abspath(os.path.expanduser(s)))
    return items


def _dataset_tag_from_path(p: str) -> str:
    base = os.path.basename(p)
    if base.lower().endswith(".csv"):
        base = base[: -len(".csv")]
    base = re.sub(r"[^0-9A-Za-z._-]+", "_", base)
    return base[:80] if len(base) > 80 else base


def _latest_run_dir(out_base: str) -> Optional[str]:
    if not os.path.isdir(out_base):
        return None
    candidates = []
    for name in os.listdir(out_base):
        if name.startswith("run_"):
            full = os.path.join(out_base, name)
            if os.path.isdir(full):
                candidates.append(full)
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def _parse_summary(summary_path: str) -> Tuple[Optional[int], Optional[float], Optional[float], Optional[int], Optional[float], Optional[float], Optional[float]]:
    """
    Returns: (K, noise_ratio, eps, minPts, DBI', CP', SP')
    noise_ratio is in [0,1].
    """
    try:
        with open(summary_path, "r", encoding="utf-8", errors="ignore") as f:
            txt = f.read()
    except Exception:
        return (None, None, None, None, None, None, None)

    K = None
    noise_ratio = None
    eps = None
    minpts = None

    m = None
    for m0 in _RE_K_NOISE_EPS_MINPTS.finditer(txt):
        m = m0
    if m is not None:
        try:
            K = int(m.group("K"))
        except Exception:
            K = None
        try:
            noise_ratio = float(m.group("noise")) / 100.0
        except Exception:
            noise_ratio = None
        try:
            eps = float(m.group("eps"))
        except Exception:
            eps = None
        try:
            minpts = int(m.group("minpts"))
        except Exception:
            minpts = None

    def _last_float(regex: re.Pattern) -> Optional[float]:
        mm = None
        for x in regex.finditer(txt):
            mm = x
        if mm is None:
            return None
        try:
            return float(mm.group("val"))
        except Exception:
            return None

    dbi = _last_float(_RE_DBI)
    cp = _last_float(_RE_CP)
    sp = _last_float(_RE_SP)
    return (K, noise_ratio, eps, minpts, dbi, cp, sp)


def _parse_minpts_mode(summary_path: str) -> Tuple[Optional[str], Optional[bool]]:
    """
    Returns: (minpts_mode, two_stage)
      - minpts_mode: one of {'mean','median','Eps*Gmax', ...} if detectable
      - two_stage: whether POA two-stage was used (best-effort)
    """
    try:
        with open(summary_path, "r", encoding="utf-8", errors="ignore") as f:
            txt = f.read()
    except Exception:
        return (None, None)

    two_stage = bool(_RE_TWO_STAGE.search(txt))

    mode = None
    for m in _RE_MINPTS_MODE_2.finditer(txt):
        mode = m.group("mode")
    if mode is None:
        for m in _RE_MINPTS_MODE_1.finditer(txt):
            mode = m.group("mode")
    if mode is not None:
        mode = str(mode).strip()
    return (mode, two_stage)


def _run_cmd(cmd: List[str], log_path: str) -> int:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    return int(p.returncode)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--python",
        type=str,
        default=None,
        help="EN run_complete_pipeline.py EN Python EN（EN sys.executable）。"
             "EN numpy/sklearn EN，EN conda/venv python，EN /Users/lcx/miniconda3/envs/EN/bin/python",
    )
    ap.add_argument("--ais", nargs="*", default=[], help="AIS CSV EN（EN）")
    ap.add_argument("--ais-list", type=str, default=None, help="EN AIS CSV EN（EN#EN）")
    ap.add_argument("--out", type=str, default=None, help="EN（EN: ./outputs/batch_eval_EN）")
    ap.add_argument("--feature-mode", type=str, default="delta_cog", choices=["delta_cog", "pca_latlon"])
    ap.add_argument("--dtw-cache-dir", type=str, default=None, help="EN run_complete_pipeline.py EN --dtw-cache-dir")
    ap.add_argument("--traj-cache-dir", type=str, default=None, help="EN run_complete_pipeline.py EN --traj-cache-dir")
    ap.add_argument("--fixed-dt-s", type=int, default=60)
    ap.add_argument("--max-interp-points", type=int, default=None)
    ap.add_argument("--max-simplified-points", type=int, default=None)

    ap.add_argument("--delta-s", type=float, default=0.07, help="paper/improved EN ΔS（EN 0.07）")
    ap.add_argument(
        "--paper-delta-s",
        type=float,
        default=None,
        help="EN paper EN ΔS（EN --delta-s）。",
    )
    ap.add_argument("--minpts-stat", type=str, default="paper", choices=["mean", "median", "eps_gmax", "paper"])
    ap.add_argument(
        "--paper-minpts-stat",
        type=str,
        default=None,
        choices=["mean", "median", "eps_gmax", "paper"],
        help="EN paper EN --minpts-stat（EN --minpts-stat）。",
    )
    ap.add_argument(
        "--improved-minpts-stat",
        type=str,
        default=None,
        choices=["mean", "median", "eps_gmax", "paper"],
        help="EN improved(two-stage) EN --minpts-stat（EN --minpts-stat）。",
    )
    ap.add_argument("--improved-two-stage", action="store_true", default=True, help="improved EN two-stage（EN）")
    ap.add_argument("--poa-coarse-steps", type=int, default=150)
    ap.add_argument("--poa-refine-steps", type=int, default=300)
    ap.add_argument("--poa-refine-window-mult", type=float, default=5.0)

    ap.add_argument("--drl-model", type=str, default=None, help="DRL EN（EN --mode drl EN）")

    ap.add_argument("--paper-extra", type=str, default="", help="EN paper EN（EN，EN shell EN split）")
    ap.add_argument("--improved-extra", type=str, default="", help="EN improved EN")
    ap.add_argument("--drl-extra", type=str, default="", help="EN drl EN")
    ap.add_argument("--resume", action="store_true", help="EN summary.txt EN results EN")
    ap.add_argument("--no-plots", action="store_true", help="EN（EN plots/）")
    args = ap.parse_args()

    ais_paths: List[str] = [os.path.abspath(os.path.expanduser(p)) for p in (args.ais or [])]
    if args.ais_list:
        ais_paths.extend(_read_ais_list(args.ais_list))
    seen = set()
    ais_paths = [p for p in ais_paths if not (p in seen or seen.add(p))]
    if not ais_paths:
        print("❌ EN。EN --ais EN --ais-list")
        return 2

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = os.path.abspath(args.out) if args.out else os.path.abspath(os.path.join("./outputs", f"batch_eval_{ts}"))
    runs_root = os.path.join(out_root, "runs")
    os.makedirs(runs_root, exist_ok=True)

    methods = ["paper", "improved", "drl"]
    if "drl" in methods and not args.drl_model:
        print("❌ EN --drl-model EN drl EN（EN）。")
        return 2

    python = os.path.abspath(os.path.expanduser(args.python)) if args.python else (sys.executable or "python")
    runner = os.path.abspath(os.path.join(os.path.dirname(__file__), "run_complete_pipeline.py"))
    if not os.path.exists(runner):
        print(f"❌ EN run_complete_pipeline.py: {runner}")
        return 2

    results_path = os.path.join(out_root, "results.csv")
    log_dir = os.path.join(out_root, "logs")
    os.makedirs(log_dir, exist_ok=True)

    existing_keys = set()
    if args.resume and os.path.exists(results_path):
        try:
            with open(results_path, "r", encoding="utf-8") as f:
                rr = csv.DictReader(f)
                for row in rr:
                    ok = str(row.get("ok", "")).strip().lower()
                    if ok in ("1", "true", "yes", "y"):
                        existing_keys.add((row.get("dataset_tag"), row.get("method")))
        except Exception:
            existing_keys = set()

    fieldnames = [
        "dataset_tag",
        "ais_path",
        "method",
        "out_base",
        "run_dir",
        "ok",
        "error",
        "minpts_stat_arg",
        "minpts_mode_effective",
        "poa_two_stage_effective",
        "delta_s_arg",
        "K",
        "noise_ratio",
        "eps",
        "minPts",
        "DBI_prime",
        "CP_prime",
        "SP_prime",
    ]

    if not os.path.exists(results_path):
        with open(results_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()

    def _common_args() -> List[str]:
        out = [
            "--feature-mode",
            str(args.feature_mode),
            "--fixed-dt-s",
            str(int(args.fixed_dt_s)),
        ]
        if args.dtw_cache_dir:
            out += ["--dtw-cache-dir", os.path.abspath(args.dtw_cache_dir)]
        if args.traj_cache_dir:
            out += ["--traj-cache-dir", os.path.abspath(args.traj_cache_dir)]
        if args.max_interp_points is not None:
            out += ["--max-interp-points", str(int(args.max_interp_points))]
        if args.max_simplified_points is not None:
            out += ["--max-simplified-points", str(int(args.max_simplified_points))]
        return out

    def _paper_cmd(ais_path: str, out_base: str) -> List[str]:
        paper_minpts = str(args.paper_minpts_stat or args.minpts_stat)
        paper_delta_s = float(args.paper_delta_s) if args.paper_delta_s is not None else float(args.delta_s)
        cmd = [
            python,
            runner,
            "--mode",
            "paper",
            "--ais",
            ais_path,
            "--out",
            out_base,
            "--delta-s",
            str(float(paper_delta_s)),
            "--minpts-stat",
            str(paper_minpts),
        ]
        cmd += _common_args()
        if args.paper_extra:
            cmd += shlex.split(args.paper_extra)
        return cmd

    def _improved_cmd(ais_path: str, out_base: str) -> List[str]:
        improved_minpts = str(args.improved_minpts_stat or args.minpts_stat)
        cmd = [
            python,
            runner,
            "--mode",
            "paper",
            "--ais",
            ais_path,
            "--out",
            out_base,
            "--delta-s",
            str(float(args.delta_s)),
            "--minpts-stat",
            str(improved_minpts),
        ]
        cmd += _common_args()
        if args.improved_two_stage:
            cmd += [
                "--poa-two-stage",
                "--poa-coarse-steps",
                str(int(args.poa_coarse_steps)),
                "--poa-refine-steps",
                str(int(args.poa_refine_steps)),
                "--poa-refine-window-mult",
                str(float(args.poa_refine_window_mult)),
            ]
        if args.improved_extra:
            cmd += shlex.split(args.improved_extra)
        return cmd

    def _drl_cmd(ais_path: str, out_base: str) -> List[str]:
        cmd = [
            python,
            runner,
            "--mode",
            "drl",
            "--ais",
            ais_path,
            "--out",
            out_base,
            "--drl-model",
            os.path.abspath(args.drl_model),
        ] + _common_args()
        if args.drl_extra:
            cmd += shlex.split(args.drl_extra)
        return cmd

    cmd_builders = {"paper": _paper_cmd, "improved": _improved_cmd, "drl": _drl_cmd}

    for ais_path in ais_paths:
        if not os.path.exists(ais_path):
            print(f"⚠️ EN AIS EN: {ais_path}")
            continue
        dataset_tag = _dataset_tag_from_path(ais_path)
        for method in methods:
            key = (dataset_tag, method)
            if args.resume and key in existing_keys:
                print(f"⏭️ resume: EN {dataset_tag} / {method}")
                continue

            out_base = os.path.join(runs_root, dataset_tag, method)
            os.makedirs(out_base, exist_ok=True)
            log_path = os.path.join(log_dir, f"{dataset_tag}__{method}.log")
            cmd = cmd_builders[method](ais_path, out_base)

            print(f"\n▶ Running {dataset_tag} / {method}")
            print("  " + " ".join(shlex.quote(x) for x in cmd))
            rc = _run_cmd(cmd, log_path=log_path)

            run_dir = _latest_run_dir(out_base) or ""
            summary_path = os.path.join(run_dir, "summary.txt") if run_dir else ""
            ok = (rc == 0) and bool(run_dir) and os.path.exists(summary_path)
            error = "" if ok else f"returncode={rc}; log={log_path}"

            K, noise_ratio, eps, minpts, dbi, cp, sp = (None, None, None, None, None, None, None)
            if ok:
                K, noise_ratio, eps, minpts, dbi, cp, sp = _parse_summary(summary_path)
                if K is None and dbi is None and cp is None and sp is None:
                    ok = False
                    error = f"failed_to_parse_summary; summary={summary_path}; log={log_path}"

            row = {
                "dataset_tag": dataset_tag,
                "ais_path": ais_path,
                "method": method,
                "out_base": out_base,
                "run_dir": run_dir,
                "ok": "1" if ok else "0",
                "error": error,
                "minpts_stat_arg": "",
                "minpts_mode_effective": "",
                "poa_two_stage_effective": "",
                "delta_s_arg": "",
                "K": "" if K is None else str(int(K)),
                "noise_ratio": "" if noise_ratio is None else f"{float(noise_ratio):.6f}",
                "eps": "" if eps is None else f"{float(eps):.8f}",
                "minPts": "" if minpts is None else str(int(minpts)),
                "DBI_prime": "" if dbi is None else f"{float(dbi):.8f}",
                "CP_prime": "" if cp is None else f"{float(cp):.8f}",
                "SP_prime": "" if sp is None else f"{float(sp):.8f}",
            }
            if method == "paper":
                row["minpts_stat_arg"] = str(args.paper_minpts_stat or args.minpts_stat)
                row["delta_s_arg"] = f"{(float(args.paper_delta_s) if args.paper_delta_s is not None else float(args.delta_s)):.6f}"
            elif method == "improved":
                row["minpts_stat_arg"] = str(args.improved_minpts_stat or args.minpts_stat)
                row["delta_s_arg"] = f"{float(args.delta_s):.6f}"
            else:
                row["minpts_stat_arg"] = ""
                row["delta_s_arg"] = ""
            if ok and summary_path:
                mode_eff, two_stage_eff = _parse_minpts_mode(summary_path)
                if mode_eff is not None:
                    row["minpts_mode_effective"] = str(mode_eff)
                if two_stage_eff is not None:
                    row["poa_two_stage_effective"] = "1" if bool(two_stage_eff) else "0"

            with open(results_path, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writerow(row)

    print(f"\n✅ Done. results.csv = {results_path}")
    print(f"📁 runs/ = {runs_root}")
    if not bool(args.no_plots):
        try:
            _make_comparison_plots(results_path=results_path, out_root=out_root)
        except Exception as e:
            print(f"⚠️ EN（EN results.csv）：{e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

