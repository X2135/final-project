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


def infer_target_k(dataset_name: str) -> int:
    if dataset_name.startswith("k=3"):
        return 3
    if dataset_name.startswith("k=7"):
        return 7
    if dataset_name.startswith("k=8"):
        return 8
    raise ValueError(f"Cannot infer k from dataset name: {dataset_name}")


def run_one_dataset(dtw_path: Path) -> List[Dict[str, object]]:
    dataset_name = dtw_path.parent.name
    D = np.asarray(np.load(dtw_path), dtype=float)
    target_k = infer_target_k(dataset_name)

    rows: List[Dict[str, object]] = []
    for method in METHODS:
        labels, k_used, info = traditional_clustering(
            D,
            method=method,
            n_clusters=int(target_k),
            verbose=False,
        )
        labels_eval = np.asarray(labels, dtype=int) + 1
        dbi_p, cp_p, sp_p = evaluate_clusters_NOT_PAPER(D, labels_eval)

        rows.append(
            {
                "dataset": dataset_name,
                "method": METHOD_NAME[method],
                "target_k": int(target_k),
                "K": int(k_used),
                "eps": np.nan,
                "minPts": np.nan,
                "noise_ratio": 0.0,
                "DBI_prime": float(dbi_p) if np.isfinite(dbi_p) else np.nan,
                "CP_prime": float(cp_p) if np.isfinite(cp_p) else np.nan,
                "SP_prime": float(sp_p) if np.isfinite(sp_p) else np.nan,
                "silhouette_score": float(info.get("silhouette_score", np.nan)) if isinstance(info, dict) else np.nan,
            }
        )

    return rows


def main() -> int:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Run fixed-k traditional clustering comparisons on user-provided DTW matrices.")
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
        print("No valid datasets found.")
        return 1

    df = pd.DataFrame(all_rows)[[
        "dataset", "method", "target_k", "K", "eps", "minPts", "noise_ratio", "DBI_prime", "CP_prime", "SP_prime", "silhouette_score"
    ]]

    out_csv = root / "outputs" / "traditional_methods_fixedk_5datasets.csv"
    out_md = root / "outputs" / "traditional_methods_fixedk_5datasets.md"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_csv, index=False, float_format="%.4f", na_rep="nan")

    lines = [
        "# Traditional Methods (Fixed-k) on 5 Datasets\n\n",
        "| Dataset | Method | target_k | K | Noise Ratio | DBI' | CP' | SP' | silhouette |\n",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|\n",
    ]
    for _, r in df.iterrows():
        lines.append(
            "| {d} | {m} | {tk} | {k} | {nr:.4f} | {dbi} | {cp} | {sp} | {sil} |\n".format(
                d=r["dataset"],
                m=r["method"],
                tk=int(r["target_k"]),
                k=int(r["K"]),
                nr=float(r["noise_ratio"]),
                dbi="nan" if pd.isna(r["DBI_prime"]) else f"{r['DBI_prime']:.4f}",
                cp="nan" if pd.isna(r["CP_prime"]) else f"{r['CP_prime']:.4f}",
                sp="nan" if pd.isna(r["SP_prime"]) else f"{r['SP_prime']:.4f}",
                sil="nan" if pd.isna(r["silhouette_score"]) else f"{r['silhouette_score']:.4f}",
            )
        )
    out_md.write_text("".join(lines), encoding="utf-8")

    print("\nUsed target_k mapping: inferred from each dataset name prefix (e.g. k=3 / k=7 / k=8)")
    print("\nSaved files:")
    print(f"- {out_csv}")
    print(f"- {out_md}")
    print("\nPreview:")
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
