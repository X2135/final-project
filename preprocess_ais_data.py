#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AIS EN：EN

EN：
1. EN：EN >= 10 EN <= 10000
2. EN：SOG EN [0, 50] EN
3. EN：EN，EN < 100km
4. EN：EN，EN
5. EN：EN < 50%
"""

import os
import sys
import pandas as pd
import numpy as np
from load_ais import load_ais_data, wrap_lat, wrap_lon
import argparse


def calculate_distance_km(lat1, lon1, lat2, lon2):
    """EN（EN）"""
    from math import radians, cos, sin, asin, sqrt
    
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371
    return c * r


def filter_trajectory(traj_df, mmsi, verbose=False):
    """
    EN
    
    EN: (is_valid, reason)
    """
    if len(traj_df) < 10:
        return False, "EN < 10"
    
    if len(traj_df) > 10000:
        return False, "EN > 10000"
    
    if 'SOG' in traj_df.columns:
        sog = pd.to_numeric(traj_df['SOG'], errors='coerce')
        max_sog = sog.max()
        min_sog = sog.min()
        
        if pd.isna(max_sog) or max_sog > 50:
            return False, f"EN: max_SOG={max_sog:.1f}"
        
        if pd.notna(min_sog) and min_sog < 0:
            return False, f"EN: min_SOG={min_sog:.1f}"
    
    if 'LAT' in traj_df.columns and 'LON' in traj_df.columns:
        lat = pd.to_numeric(traj_df['LAT'], errors='coerce')
        lon = pd.to_numeric(traj_df['LON'], errors='coerce')
        
        if lat.isna().any() or lon.isna().any():
            missing_ratio = (lat.isna() | lon.isna()).sum() / len(traj_df)
            if missing_ratio > 0.5:
                return False, f"EN: {missing_ratio:.1%}"
        
        valid_mask = lat.notna() & lon.notna()
        if valid_mask.sum() > 1:
            valid_lat = lat[valid_mask].values
            valid_lon = lon[valid_mask].values
            
            for i in range(1, len(valid_lat)):
                dist_km = calculate_distance_km(
                    valid_lat[i-1], valid_lon[i-1],
                    valid_lat[i], valid_lon[i]
                )
                if dist_km > 100:
                    return False, f"EN: {dist_km:.1f}km at point {i}"
    
    if 'COG' in traj_df.columns:
        cog = pd.to_numeric(traj_df['COG'], errors='coerce')
        valid_cog = cog[cog.notna()]
        
        if len(valid_cog) > 0:
            if (valid_cog < 0).any() or (valid_cog >= 360).any():
                return False, "EN [0, 360)"
    
    if 'BaseDateTime' in traj_df.columns:
        try:
            times = pd.to_datetime(traj_df['BaseDateTime'], errors='coerce')
            if times.isna().sum() > len(times) * 0.5:
                return False, "EN"
            
            valid_times = times[times.notna()]
            if len(valid_times) > 1:
                time_diffs = valid_times.diff().dropna()
                if (time_diffs < pd.Timedelta(0)).any():
                    return False, "EN"
                if (time_diffs > pd.Timedelta('24h')).any():
                    return False, "EN（>24h）"
        except:
            pass
    
    required_cols = ['LAT', 'LON', 'SOG', 'COG']
    missing_counts = {}
    for col in required_cols:
        if col in traj_df.columns:
            missing = traj_df[col].isna().sum()
            missing_counts[col] = missing
    
    total_missing = sum(missing_counts.values())
    missing_ratio = total_missing / (len(traj_df) * len(required_cols))
    
    if missing_ratio > 0.5:
        return False, f"EN: {missing_ratio:.1%}"
    
    return True, "EN"


def preprocess_ais_data(
    input_path,
    output_path,
    min_points=10,
    max_points=10000,
    max_sog=50,
    max_jump_km=100,
    max_missing_ratio=0.5,
    verbose=True,
):
    """
    EN AIS EN：EN
    
    EN:
        input_path: EN CSV EN
        output_path: EN CSV EN
        min_points: EN
        max_points: EN
        max_sog: EN（EN）
        max_jump_km: EN（EN）
        max_missing_ratio: EN
        verbose: EN
    """
    print("=" * 60)
    print(f"EN AIS EN")
    print("=" * 60)
    print(f"EN: {input_path}")
    print(f"EN: {output_path}")
    print(f"\nEN:")
    print(f"  - EN: [{min_points}, {max_points}]")
    print(f"  - EN: [0, {max_sog}] EN")
    print(f"  - EN: {max_jump_km} km")
    print(f"  - EN: {max_missing_ratio:.1%}")
    print("=" * 60)
    
    print("\nEN 1/4: EN...")
    df = pd.read_csv(input_path, low_memory=False)
    print(f"  EN: {len(df):,}")
    print(f"  EN MMSI EN: {df['MMSI'].nunique()}")
    
    print("\nEN 2/4: EN MMSI EN...")
    grouped = df.groupby('MMSI')
    
    valid_trajectories = []
    removed_trajectories = []
    
    for mmsi, traj_df in grouped:
        is_valid, reason = filter_trajectory(traj_df, mmsi, verbose=False)
        
        if is_valid:
            valid_trajectories.append(traj_df)
        else:
            removed_trajectories.append({
                'mmsi': mmsi,
                'points': len(traj_df),
                'reason': reason
            })
            if verbose:
                print(f"  ❌ EN MMSI {mmsi}: {reason} (EN: {len(traj_df)})")
    
    print(f"\n  EN: {len(valid_trajectories)}")
    print(f"  EN: {len(removed_trajectories)}")
    
    print("\nEN 3/4: EN...")
    if valid_trajectories:
        df_filtered = pd.concat(valid_trajectories, ignore_index=True)
        print(f"  EN: {len(df_filtered):,}")
        print(f"  EN: {len(df_filtered)/len(df):.1%}")
    else:
        print("  ❌ EN！")
        return
    
    print("\nEN 4/4: EN...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_filtered.to_csv(output_path, index=False, encoding='utf-8')
    print(f"  ✅ EN: {output_path}")
    
    if removed_trajectories:
        removed_df = pd.DataFrame(removed_trajectories)
        removed_path = output_path.replace('.csv', '_removed.csv')
        removed_df.to_csv(removed_path, index=False, encoding='utf-8')
        print(f"  ✅ EN: {removed_path}")
        
        print("\nEN:")
        reason_counts = removed_df['reason'].value_counts()
        for reason, count in reason_counts.items():
            print(f"  - {reason}: {count} EN")
    
    print("\n" + "=" * 60)
    print("✅ EN！")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EN AIS EN：EN")
    parser.add_argument("--input", type=str, required=True, help="EN CSV EN")
    parser.add_argument("--output", type=str, required=True, help="EN CSV EN")
    parser.add_argument("--min-points", type=int, default=10, help="EN（EN 10）")
    parser.add_argument("--max-points", type=int, default=10000, help="EN（EN 10000）")
    parser.add_argument("--max-sog", type=float, default=50, help="EN（EN，EN 50）")
    parser.add_argument("--max-jump-km", type=float, default=100, help="EN（EN，EN 100）")
    parser.add_argument("--max-missing-ratio", type=float, default=0.5, help="EN（EN 0.5）")
    
    args = parser.parse_args()
    
    preprocess_ais_data(
        input_path=args.input,
        output_path=args.output,
        min_points=args.min_points,
        max_points=args.max_points,
        max_sog=args.max_sog,
        max_jump_km=args.max_jump_km,
        max_missing_ratio=args.max_missing_ratio,
        verbose=True,
    )
