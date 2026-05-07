#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from traditional_clustering import (
    kmeans_clustering,
    agglomerative_clustering,
    spectral_clustering,
)
from evaluate import evaluate_clusters_NOT_PAPER



def infer_target_k(dataset_name: str) -> int | None:
    if dataset_name.startswith("k=3"):
        return 3
    if dataset_name.startswith("k=7"):
        return 7
    if dataset_name.startswith("k=8"):
        return 8
    return None


def _score_row(row: Dict[str, object]) -> Tuple[float, float, float]:
    """EN：EN DBI，EN CP，EN -silhouette（EN silhouette EN）"""
    dbi = float(row["DBI_prime"]) if np.isfinite(row["DBI_prime"]) else float("inf")
    cp = float(row["CP_prime"]) if np.isfinite(row["CP_prime"]) else float("inf")
    sil = float(row.get("silhouette_score", np.nan))
    sil_penalty = -sil if np.isfinite(sil) else float("inf")
    return (dbi, cp, sil_penalty)


def evaluate_labels(D: np.ndarray, labels0: np.ndarray) -> Tuple[float, float, float]:
    labels_eval = np.asarray(labels0, dtype=int) + 1
    dbi_p, cp_p, sp_p = evaluate_clusters_NOT_PAPER(D, labels_eval)
    return (
        float(dbi_p) if np.isfinite(dbi_p) else np.nan,
        float(cp_p) if np.isfinite(cp_p) else np.nan,
        float(sp_p) if np.isfinite(sp_p) else np.nan,
    )


def tune_one_dataset(dtw_path: Path) -> List[Dict[str, object]]:
    dataset_name = dtw_path.parent.name
    D = np.asarray(np.load(dtw_path), dtype=float)
    N = int(D.shape[0])
    target_k = infer_target_k(dataset_name)
    if target_k is None:
        raise ValueError(f"Cannot infer target_k from dataset name: {dataset_name}")

    valid = D[(D > 0) & np.isfinite(D)]
    if valid.size == 0:
        q20 = q50 = q80 = 1.0
    else:
        q20, q50, q80 = np.quantile(valid, [0.2, 0.5, 0.8])

    rows_best: List[Dict[str, object]] = []

    best = None
    for n_comp in [3, 5, 8, 10, 12, 16]:
        n_comp_eff = min(max(2, n_comp), max(2, N - 1))
        labels, k_used, info = kmeans_clustering(
            D,
            k=int(target_k),
            n_components=n_comp_eff,
            random_state=42,
            verbose=False,
        )
        dbi_p, cp_p, sp_p = evaluate_labels(D, labels)
        row = {
            "dataset": dataset_name,
            "method": "K-Means",
            "eps": np.nan,
            "minPts": np.nan,
            "K": int(k_used),
            "noise_ratio": 0.0,
            "DBI_prime": dbi_p,
            "CP_prime": cp_p,
            "SP_prime": sp_p,
            "target_k": int(target_k),
            "silhouette_score": float(info.get("silhouette_score", np.nan)),
            "best_params": f"n_components={n_comp_eff}",
        }
        if best is None or _score_row(row) < _score_row(best):
            best = row
    rows_best.append(best)

    best = None
    for linkage in ["average", "complete", "single"]:
        labels, k_used, info = agglomerative_clustering(
            D,
            n_clusters=int(target_k),
            linkage=linkage,
            verbose=False,
        )
        dbi_p, cp_p, sp_p = evaluate_labels(D, labels)
        row = {
            "dataset": dataset_name,
            "method": "Agglomerative",
            "eps": np.nan,
            "minPts": np.nan,
            "K": int(k_used),
            "noise_ratio": 0.0,
            "DBI_prime": dbi_p,
            "CP_prime": cp_p,
            "SP_prime": sp_p,
            "target_k": int(target_k),
            "silhouette_score": float(info.get("silhouette_score", np.nan)),
            "best_params": f"linkage={linkage}",
        }
        if best is None or _score_row(row) < _score_row(best):
            best = row
    rows_best.append(best)

    best = None
    sigma_candidates = [
        max(1e-8, float(q20)),
        max(1e-8, float((q20 + q50) / 2.0)),
        max(1e-8, float(q50)),
        max(1e-8, float((q50 + q80) / 2.0)),
        max(1e-8, float(q80)),
    ]
    for sigma in sigma_candidates:
        labels, k_used, info = spectral_clustering(
            D,
            n_clusters=int(target_k),
            sigma=float(sigma),
            random_state=42,
            verbose=False,
        )
        dbi_p, cp_p, sp_p = evaluate_labels(D, labels)
        row = {
            "dataset": dataset_name,
            "method": "Spectral",
            "eps": np.nan,
            "minPts": np.nan,
            "K": int(k_used),
            "noise_ratio": 0.0,
            "DBI_prime": dbi_p,
            "CP_prime": cp_p,
            "SP_prime": sp_p,
            "target_k": int(target_k),
            "silhouette_score": float(info.get("silhouette_score", np.nan)),
            "best_params": f"sigma={sigma:.6f}",
        }
        if best is None or _score_row(row) < _score_row(best):
            best = row
    rows_best.append(best)

    return rows_best


def main() -> int:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Run tuned traditional clustering comparisons on user-provided DTW matrices.")
    parser.add_argument(
        "--dtw-npy",
        nargs="+",
        required=True,
        help="One or more DTW .npy matrices to evaluate.",
    )
    args = parser.parse_args()

    all_rows: List[Dict[str, object]] = []

    for rel in args.dtw_npy:
        p = Path(rel)
        if not p.is_absolute():
            p = root / p
        if not p.exists():
            print(f"[Skip] missing: {p}")
            continue
        print(f"[Tune] {p}")
        all_rows.extend(tune_one_dataset(p))

    if not all_rows:
        print("No valid datasets.")
        return 1

    df = pd.DataFrame(all_rows)
    df = df[
        [
            "dataset",
            "method",
            "eps",
            "minPts",
            "K",
            "noise_ratio",
            "DBI_prime",
            "CP_prime",
            "SP_prime",
            "target_k",
            "silhouette_score",
            "best_params",
        ]
    ]

    out_csv = root / "outputs" / "traditional_methods_comparison_selected_datasets_tuned.csv"
    out_md = root / "outputs" / "traditional_methods_comparison_selected_datasets_tuned.md"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_csv, index=False, float_format="%.4f", na_rep="nan")

    md_lines = [
        "# Traditional Methods Comparison (Tuned) on Selected Datasets\n\n",
        "| Dataset | Method | K | Noise Ratio | DBI' | CP' | SP' | silhouette | best_params |\n",
        "|---|---|---:|---:|---:|---:|---:|---:|---|\n",
    ]
    for _, r in df.iterrows():
        md_lines.append(
            "| {dataset} | {method} | {K} | {noise_ratio:.4f} | {dbi} | {cp} | {sp} | {sil} | {bp} |\n".format(
                dataset=r["dataset"],
                method=r["method"],
                K=int(r["K"]),
                noise_ratio=float(r["noise_ratio"]),
                dbi="nan" if pd.isna(r["DBI_prime"]) else f"{r['DBI_prime']:.4f}",
                cp="nan" if pd.isna(r["CP_prime"]) else f"{r['CP_prime']:.4f}",
                sp="nan" if pd.isna(r["SP_prime"]) else f"{r['SP_prime']:.4f}",
                sil="nan" if pd.isna(r["silhouette_score"]) else f"{r['silhouette_score']:.4f}",
                bp=r["best_params"],
            )
        )

    out_md.write_text("".join(md_lines), encoding="utf-8")

    print("\nSaved:")
    print(f"- {out_csv}")
    print(f"- {out_md}")
    print("\nPreview:")
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
