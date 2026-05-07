"""
Sweep delta_s for paper reproduction (fixed-step POA) and summarize results.
===========================================================================

Goal
----
Run `run_complete_pipeline.py` multiple times with different `--delta-s` values,
store each run in a dedicated subfolder, and aggregate key metrics into CSV.

Example
-------
python sweep_delta_s.py \
  --ais /Users/lcx/Desktop/EN/EN/data/raw/ais_yuanli.csv \
  --feature-mode delta_cog \
  --out-root /Users/lcx/Desktop/EN/EN/outputs/delta_s_sweep

Default will run delta_s = 0.02, 0.04, ..., 0.16 (step=0.02) with:
  --mode paper
  --minpts-stat paper
  (no --poa-two-stage)
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


RUN_DIR_RE = re.compile(r"EN：\s*(.+)\s*$")


@dataclass
class SweepResult:
    delta_s: float
    out_base: str
    run_dir: str
    exit_code: int
    eps: Optional[float] = None
    minPts: Optional[int] = None
    K: Optional[int] = None
    noise_ratio: Optional[float] = None
    dbi_p: Optional[float] = None
    cp_p: Optional[float] = None
    sp_p: Optional[float] = None


def _make_unique_out_root(base: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    cand = base.parent / f"{base.name}_{ts}"
    if not cand.exists():
        return cand
    for i in range(2, 1000):
        cand2 = base.parent / f"{base.name}_{ts}_{i}"
        if not cand2.exists():
            return cand2
    return cand


def _parse_run_dir(stdout: str) -> Optional[str]:
    for line in reversed(stdout.splitlines()):
        m = RUN_DIR_RE.search(line)
        if m:
            return m.group(1).strip()
    return None


def _fallback_find_latest_run_dir(out_base: Path) -> Optional[Path]:
    run_dirs = [p for p in out_base.glob("run_*") if p.is_dir()]
    if not run_dirs:
        return None
    run_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return run_dirs[0]


def _parse_metrics_from_summary(summary_path: Path) -> Dict[str, Optional[float]]:
    text = summary_path.read_text(encoding="utf-8", errors="ignore")

    def find_float(pattern: str) -> Optional[float]:
        m = re.search(pattern, text, flags=re.MULTILINE)
        if not m:
            return None
        try:
            s = str(m.group(1)).strip()
            if s.lower() == "nan":
                return float("nan")
            return float(s)
        except Exception:
            return None

    m_final = None
    for m in re.finditer(
        r"EN.*?K\s*=\s*(\d+).*?EN\s*=\s*([0-9]*\.?[0-9]+)\s*%.*?eps\s*=\s*([0-9]*\.?[0-9]+).*?minPts\s*=\s*(\d+)",
        text,
        flags=re.MULTILINE,
    ):
        m_final = m

    if m_final is None:
        return {"eps": None, "minPts": None, "K": None, "noise_ratio": None, "dbi_p": None, "cp_p": None, "sp_p": None}

    K = int(m_final.group(1))
    noise_pct = float(m_final.group(2))
    eps = float(m_final.group(3))
    minPts = int(m_final.group(4))

    dbi_p = find_float(r"DBI.?\s*[:：]\s*([0-9]*\.?[0-9]+|nan)")
    cp_p = find_float(r"CP.?\s*[:：]\s*([0-9]*\.?[0-9]+|nan)")
    sp_p = find_float(r"SP.?\s*[:：]\s*([0-9]*\.?[0-9]+|nan)")

    return {
        "eps": eps,
        "minPts": float(minPts),
        "K": float(K),
        "noise_ratio": noise_pct / 100.0,
        "dbi_p": dbi_p,
        "cp_p": cp_p,
        "sp_p": sp_p,
    }


def _delta_grid(delta_s_list: Optional[List[float]]) -> List[float]:
    if delta_s_list:
        out = []
        for x in delta_s_list:
            try:
                out.append(float(x))
            except Exception:
                pass
        return out
    return [round(0.02 * i, 2) for i in range(1, 9)]


def run_sweep(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Sweep delta_s for paper reproduction.")
    p.add_argument("--ais", type=str, required=True, help="AIS CSV path")
    p.add_argument("--feature-mode", type=str, default="delta_cog", choices=["delta_cog", "pca_latlon"])
    p.add_argument(
        "--out-root",
        type=str,
        default="/Users/lcx/Desktop/EN/EN/outputs/delta_s_sweep",
        help="Root output folder (a timestamp suffix will be added).",
    )
    p.add_argument(
        "--delta-s",
        type=float,
        nargs="*",
        default=None,
        help="Explicit delta_s list. If omitted, uses 0.1..1.0.",
    )
    p.add_argument(
        "--pipeline-args",
        type=str,
        nargs="*",
        default=[],
        help="Extra args forwarded to run_complete_pipeline.py (e.g., --max-interp-points 20000).",
    )
    p.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip delta_s entries that already have a completed run_* folder with summary.txt under its out_base.",
    )
    args = p.parse_args(argv)

    repo_root = Path(__file__).resolve().parent
    out_root = _make_unique_out_root(Path(args.out_root))
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "logs").mkdir(parents=True, exist_ok=True)

    deltas = _delta_grid(list(args.delta_s) if args.delta_s else None)

    results: List[SweepResult] = []
    csv_path = out_root / "results.csv"

    def _flush_results_csv() -> None:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "delta_s",
                    "exit_code",
                    "K",
                    "noise_ratio",
                    "eps",
                    "minPts",
                    "dbi_p",
                    "cp_p",
                    "sp_p",
                    "out_base",
                    "run_dir",
                ],
            )
            w.writeheader()
            for r in results:
                w.writerow(asdict(r))

    try:
        for ds in deltas:
            ds = float(ds)
            ds_str = f"{ds:.2f}".rstrip("0").rstrip(".")
            ds_tag = ds_str.replace(".", "p")
            out_base = out_root / f"delta_s_{ds_tag}"
            out_base.mkdir(parents=True, exist_ok=True)
            log_path = out_root / "logs" / f"delta_s_{ds_tag}.log"

            if args.resume:
                latest = _fallback_find_latest_run_dir(out_base)
                if latest is not None and (latest / "summary.txt").exists():
                    rr = SweepResult(delta_s=ds, out_base=str(out_base), run_dir=str(latest.resolve()), exit_code=0)
                    m = _parse_metrics_from_summary(latest / "summary.txt")
                    rr.eps = m.get("eps")  # type: ignore[assignment]
                    rr.minPts = int(m["minPts"]) if m.get("minPts") is not None else None  # type: ignore[arg-type]
                    rr.K = int(m["K"]) if m.get("K") is not None else None  # type: ignore[arg-type]
                    rr.noise_ratio = float(m["noise_ratio"]) if m.get("noise_ratio") is not None else None  # type: ignore[arg-type]
                    rr.dbi_p = m.get("dbi_p")  # type: ignore[assignment]
                    rr.cp_p = m.get("cp_p")  # type: ignore[assignment]
                    rr.sp_p = m.get("sp_p")  # type: ignore[assignment]
                    results.append(rr)
                    print(f"[resume delta_s={ds_str}] K={rr.K} noise={rr.noise_ratio} eps={rr.eps} minPts={rr.minPts}")
                    _flush_results_csv()
                    continue

            cmd: List[str] = [
                sys.executable,
                str(repo_root / "run_complete_pipeline.py"),
                "--mode",
                "paper",
                "--ais",
                str(Path(args.ais).resolve()),
                "--out",
                str(out_base),
                "--feature-mode",
                str(args.feature_mode),
                "--delta-s",
                str(ds),
                "--minpts-stat",
                "paper",
            ]
            if args.pipeline_args:
                cmd += list(args.pipeline_args)

            if args.dry_run:
                print(" ".join(cmd))
                results.append(SweepResult(delta_s=ds, out_base=str(out_base), run_dir="", exit_code=0))
                _flush_results_csv()
                continue

            t0 = time.time()
            print(f"\n▶️  Running delta_s={ds_str} (log: {log_path})")
            with log_path.open("w", encoding="utf-8", errors="ignore") as lf:
                lf.write("[cmd]\n" + " ".join(cmd) + "\n\n")
                lf.flush()
                proc = subprocess.run(
                    cmd,
                    stdout=lf,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=os.environ.copy(),
                )
            dt = time.time() - t0

            latest = _fallback_find_latest_run_dir(out_base)
            run_dir_s = str(latest.resolve()) if latest else ""

            rr = SweepResult(delta_s=ds, out_base=str(out_base), run_dir=run_dir_s, exit_code=int(proc.returncode))
            try:
                if run_dir_s:
                    summ = Path(run_dir_s) / "summary.txt"
                    if summ.exists():
                        m = _parse_metrics_from_summary(summ)
                        rr.eps = m.get("eps")  # type: ignore[assignment]
                        rr.minPts = int(m["minPts"]) if m.get("minPts") is not None else None  # type: ignore[arg-type]
                        rr.K = int(m["K"]) if m.get("K") is not None else None  # type: ignore[arg-type]
                        rr.noise_ratio = float(m["noise_ratio"]) if m.get("noise_ratio") is not None else None  # type: ignore[arg-type]
                        rr.dbi_p = m.get("dbi_p")  # type: ignore[assignment]
                        rr.cp_p = m.get("cp_p")  # type: ignore[assignment]
                        rr.sp_p = m.get("sp_p")  # type: ignore[assignment]
            except Exception:
                pass

            results.append(rr)
            print(
                f"[delta_s={ds_str}] exit={rr.exit_code} time={dt:.1f}s "
                f"K={rr.K} noise={rr.noise_ratio} eps={rr.eps} minPts={rr.minPts}"
            )
            _flush_results_csv()
    except KeyboardInterrupt:
        print("\n🟡 Interrupted by user (Ctrl+C). Partial results have been saved.")
        _flush_results_csv()
        return 130

    print(f"\n✅ Sweep done. results.csv: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_sweep())

