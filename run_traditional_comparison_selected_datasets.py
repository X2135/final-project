#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from traditional_clustering import traditional_clustering
from evaluate import evaluate_clusters_NOT_PAPER


METHODS = ["kmeans", "agglomerative", "spectral"]
METHOD_NAME = {
    "kmeans": "K-Means",
    "agglomerative": "Agglomerative",
    "spectral": "Spectral",
}


def infer_target_k(dataset_name: str) -> int | None:
    if dataset_name.startswith("k=3"):
        return 3
    if dataset_name.startswith("k=7"):
        return 7
    if dataset_name.startswith("k=8"):
        return 8
    return None


def run_one_dataset(dtw_path: Path) -> List[Dict[str, object]]:
    dataset_name = dtw_path.parent.name
    D = np.asarray(np.load(dtw_path), dtype=float)

    rows: List[Dict[str, object]] = []
    target_k = infer_target_k(dataset_name)

    for method in METHODS:
        labels, k_used, info = traditional_clustering(
            D,
            method=method,
            n_clusters=target_k,
            verbose=False,
        )
        labels_eval = np.asarray(labels, dtype=int) + 1
        dbi_p, cp_p, sp_p = evaluate_clusters_NOT_PAPER(D, labels_eval)

        rows.append(
            {
                "dataset": dataset_name,
                "method": METHOD_NAME[method],
                "eps": np.nan,
                "minPts": np.nan,
                "K": int(k_used),
                "noise_ratio": 0.0,
                "DBI_prime": float(dbi_p) if np.isfinite(dbi_p) else np.nan,
                "CP_prime": float(cp_p) if np.isfinite(cp_p) else np.nan,
                "SP_prime": float(sp_p) if np.isfinite(sp_p) else np.nan,
                "target_k": target_k,
                "silhouette_score": float(info.get("silhouette_score", np.nan)) if isinstance(info, dict) else np.nan,
            }
        )

    return rows


def main() -> int:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Run traditional clustering comparisons on user-provided DTW matrices.")
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
        print(f"[Run] {p}")
        all_rows.extend(run_one_dataset(p))

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
        ]
    ]

    out_csv = root / "outputs" / "traditional_methods_comparison_selected_datasets.csv"
    out_md = root / "outputs" / "traditional_methods_comparison_selected_datasets.md"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_csv, index=False, float_format="%.4f", na_rep="nan")

    md_lines = [
        "# Traditional Methods Comparison on Selected Datasets\n\n",
        "| Dataset | Method | eps | minPts | K | Noise Ratio | DBI' | CP' | SP' | target_k | silhouette |\n",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n",
    ]
    for _, r in df.iterrows():
        md_lines.append(
            "| {dataset} | {method} | {eps} | {minPts} | {K} | {noise_ratio:.4f} | {dbi} | {cp} | {sp} | {target_k} | {sil} |\n".format(
                dataset=r["dataset"],
                method=r["method"],
                eps="nan" if pd.isna(r["eps"]) else f"{r['eps']:.4f}",
                minPts="nan" if pd.isna(r["minPts"]) else f"{r['minPts']:.0f}",
                K=int(r["K"]),
                noise_ratio=float(r["noise_ratio"]),
                dbi="nan" if pd.isna(r["DBI_prime"]) else f"{r['DBI_prime']:.4f}",
                cp="nan" if pd.isna(r["CP_prime"]) else f"{r['CP_prime']:.4f}",
                sp="nan" if pd.isna(r["SP_prime"]) else f"{r['SP_prime']:.4f}",
                target_k="nan" if pd.isna(r["target_k"]) else f"{int(r['target_k'])}",
                sil="nan" if pd.isna(r["silhouette_score"]) else f"{r['silhouette_score']:.4f}",
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
