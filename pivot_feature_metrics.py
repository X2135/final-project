#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert feature_metrics_detailed.csv to Pivot format.
Transform vertical arrangement into horizontal comparison table 
(each row is a dataset, columns are different metrics of 4 features).
"""

import pandas as pd
from pathlib import Path

repo_root = Path(__file__).resolve().parent
detailed_csv = repo_root / "dtw_cache/recomputed_feature_compare/feature_metrics_detailed.csv"

if not detailed_csv.exists():
    print(f"[ERROR] File not found: {detailed_csv}")
    exit(1)

df = pd.read_csv(detailed_csv)

metrics_to_pivot = ["K", "Noise Ratio", "Core Ratio", "eps", "minPts"]

for metric in metrics_to_pivot:
    pivot_df = df.pivot(index="Dataset", columns="Feature", values=metric)
    pivot_df = pivot_df[["delta_cog", "yaw_rate", "behavior", "pca_latlon"]]
    
    output_path = repo_root / f"dtw_cache/recomputed_feature_compare/pivot_{metric.replace(' ', '_')}.csv"
    pivot_df.to_csv(output_path, index=True)
    print(f"✅ Generated: {output_path.name}")
    print(pivot_df.to_string())
    print()

print("\n" + "="*80)
print("Comprehensive Pivot Table (K-value Comparison)")
print("="*80)
pivot_k = df.pivot(index="Dataset", columns="Feature", values="K")
print(pivot_k)

print("\n" + "="*80)
print("Comprehensive Pivot Table (Noise Ratio Comparison)")
print("="*80)
pivot_noise = df.pivot(index="Dataset", columns="Feature", values="Noise Ratio")
print((pivot_noise * 100).round(2).astype(str) + "%")

print("\n" + "="*80)
print("Comprehensive Pivot Table (eps Parameter Comparison)")
print("="*80)
pivot_eps = df.pivot(index="Dataset", columns="Feature", values="eps")
print(pivot_eps.round(4))

print("\nAll pivot CSVs saved to: dtw_cache/recomputed_feature_compare/pivot_*.csv")
