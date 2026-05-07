#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clean AIS dataset before entering run_complete_pipeline.

EN：
1. EN MMSI EN，EN；
2. EN、EN（EN），EN；
3. EN T∈[T_min,T_max]、L∈[L_min,L_max] EN；
4. EN CSV、EN、EN。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, Mapping, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def haversine_distance(lat1: np.ndarray | float, lon1: np.ndarray | float, lat2: np.ndarray | float, lon2: np.ndarray | float) -> np.ndarray:
    """
    Approximate Earth surface distance (meters).
    """
    R = 6371000.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def compute_track_metrics(df: pd.DataFrame) -> dict:
    """
    EN、EN、EN。
    """
    lat = df["lat"].to_numpy(dtype=float)
    lon = df["lon"].to_numpy(dtype=float)
    pts = len(df)
    length = 0.0
    if pts > 1:
        lat0 = lat[:-1]
        lon0 = lon[:-1]
        lat1 = lat[1:]
        lon1 = lon[1:]
        dist = haversine_distance(lat0, lon0, lat1, lon1)
        length = dist.sum()
    time_span = (
        (df["time"].max() - df["time"].min()).total_seconds()
        if "time" in df.columns and not df["time"].isna().all()
        else float("nan")
    )
    return {"points": pts, "length": float(length), "duration": float(time_span)}


def _pick_column(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    """
    EN df.columns EN“EN + EN”EN，EN。
    EN：EN，EN。
    """
    norm = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        key = str(cand).strip().lower()
        if key in norm:
            return norm[key]
    return None


def percentile_stats(values: Iterable[float]) -> dict:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return {}
    return {
        "min": float(np.min(arr)),
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(np.max(arr)),
    }


def print_distribution(name: str, stats: Mapping[str, float]) -> None:
    if not stats:
        print(f"{name} distribution: empty")
        return
    print(
        f"{name} → count={stats.get('count', 'N/A')}, "
        f"min={stats['min']:.1f}, mean={stats['mean']:.1f}, "
        f"p50={stats['p50']:.1f}, p90={stats['p90']:.1f}, max={stats['max']:.1f}"
    )


def aggregate_track_info(tracks: Mapping[int, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for mmsi, df in tracks.items():
        metrics = compute_track_metrics(df)
        rows.append({"MMSI": mmsi, **metrics})
    return pd.DataFrame(rows)


def build_tracks(df: pd.DataFrame) -> dict[int, pd.DataFrame]:
    tracks = {}
    for mmsi, group in df.groupby("MMSI", sort=False):
        group = group.sort_values("time", kind="mergesort")
        tracks[mmsi] = group.reset_index(drop=True)
    return tracks


def filter_tracks(
    tracks: Mapping[int, pd.DataFrame], t_min: int, t_max: int, l_min: float, l_max: float
) -> dict[int, pd.DataFrame]:
    filtered = {}
    for mmsi, df in tracks.items():
        pts = len(df)
        if pts == 0:
            continue
        metrics = compute_track_metrics(df)
        length = metrics["length"]
        if t_min <= pts <= t_max and l_min <= length <= l_max:
            filtered[mmsi] = df
    return filtered


def dump_filtered_rows_to_csv(df_out: pd.DataFrame, path: str | Path) -> None:
    """
    EN，EN/EN（EN CSV EN）。
    """
    df_out.to_csv(path, index=False, encoding="utf-8")


def plot_tracks(tracks: Mapping[int, pd.DataFrame], out_path: str | Path, title: str) -> None:
    plt.figure(figsize=(10, 8))
    for df in tracks.values():
        plt.plot(df["lon"], df["lat"], alpha=0.5, linewidth=0.8)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean AIS CSV for DTW+DBSCAN.")
    parser.add_argument("--input", "-i", type=str, required=True, help="EN AIS CSV EN")
    parser.add_argument(
        "--out",
        "-o",
        type=str,
        default="./cleaned_data",
        help="EN，EN clean CSV EN",
    )
    parser.add_argument(
        "--min-points",
        type=int,
        default=100,
        help="EN（T_min，EN100）",
    )
    parser.add_argument(
        "--max-len-quantile",
        type=float,
        default=95.0,
        help="EN，EN T_max（EN95%%）",
    )
    parser.add_argument(
        "--min-len-quantile",
        type=float,
        default=10.0,
        help="EN，EN L_min（EN10%%）",
    )
    parser.add_argument(
        "--max-len-upper-quantile",
        type=float,
        default=90.0,
        help="EN，EN L_max（EN90%%）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.out)

    df_raw = pd.read_csv(args.input, low_memory=False, dtype=str)
    orig_columns = df_raw.columns.tolist()

    mmsi_col = _pick_column(df_raw, ["MMSI", "mmsi"])
    time_col = _pick_column(df_raw, ["BaseDateTime", "time", "datetime", "timestamp"])
    lat_col = _pick_column(df_raw, ["LAT", "lat"])
    lon_col = _pick_column(df_raw, ["LON", "lon"])
    sog_col = _pick_column(df_raw, ["SOG", "sog"])
    cog_col = _pick_column(df_raw, ["COG", "cog"])

    missing = [name for name, col in [
        ("MMSI", mmsi_col),
        ("time/BaseDateTime", time_col),
        ("LAT/lat", lat_col),
        ("LON/lon", lon_col),
        ("SOG/sog", sog_col),
        ("COG/cog", cog_col),
    ] if col is None]
    if missing:
        raise ValueError(f"EN CSV EN: {missing}（EN，EN）")

    df_work = df_raw[[mmsi_col, time_col, lat_col, lon_col, sog_col, cog_col]].copy()
    df_work.columns = ["MMSI", "time", "lat", "lon", "sog", "cog"]
    df_work["time"] = pd.to_datetime(df_work["time"], errors="coerce")
    for c in ("lat", "lon", "sog", "cog"):
        df_work[c] = pd.to_numeric(df_work[c], errors="coerce")

    df_work = df_work.dropna(subset=["MMSI", "time", "lat", "lon", "sog", "cog"])

    original_tracks = build_tracks(df_work)
    stats_before = aggregate_track_info(original_tracks)
    print(f"EN: {len(original_tracks)}")
    pts_stats = percentile_stats(stats_before["points"])
    stats_dist = percentile_stats(stats_before["length"])
    print_distribution("EN", {**pts_stats, "count": len(stats_before)})
    print_distribution("EN", {**stats_dist, "count": len(stats_before)})

    t_min = args.min_points
    t_max = int(np.percentile(stats_before["points"], args.max_len_quantile))
    lengths = stats_before["length"]
    l_min = float(np.percentile(lengths, args.min_len_quantile))
    l_max = float(np.percentile(lengths, args.max_len_upper_quantile))

    print(f"EN：T_min={t_min}, T_max={t_max}, L_min={l_min:.1f}, L_max={l_max:.1f}")

    clean_tracks = filter_tracks(original_tracks, t_min, t_max, l_min, l_max)
    stats_after = aggregate_track_info(clean_tracks)
    print(f"EN: {len(clean_tracks)}")
    pts_stats_after = percentile_stats(stats_after["points"])
    stats_dist_after = percentile_stats(stats_after["length"])
    print_distribution("EN", {**pts_stats_after, "count": len(stats_after)})
    print_distribution("EN", {**stats_dist_after, "count": len(stats_after)})

    keep_mmsi = set(str(k) for k in clean_tracks.keys())
    keep_mask = (
        df_raw[mmsi_col].astype(str).isin(keep_mmsi)
        & df_raw[time_col].notna()
        & df_raw[lat_col].notna()
        & df_raw[lon_col].notna()
        & df_raw[sog_col].notna()
        & df_raw[cog_col].notna()
    )
    df_out = df_raw.loc[keep_mask, orig_columns]

    clean_csv_path = out_dir / "clean_ais_filtered.csv"
    dump_filtered_rows_to_csv(df_out, clean_csv_path)
    print(f"✅ EN CSV: {clean_csv_path}")

    img_path = out_dir / "cleaned_trajectories.png"
    plot_tracks(clean_tracks, img_path, title="Cleaned AIS trajectories")
    print(f"✅ EN: {img_path}（EN: {len(clean_tracks)}）")

    summary = {
        "input": args.input,
        "tracks_before": len(original_tracks),
        "tracks_after": len(clean_tracks),
        "t_min": t_min,
        "t_max": t_max,
        "length_min": l_min,
        "length_max": l_max,
    }
    with open(out_dir / "cleaning_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

