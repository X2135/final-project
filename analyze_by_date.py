#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ENAISEN，EN，EN
"""

import pandas as pd
import os
from datetime import datetime

def analyze_and_filter_by_date(file_path, output_file=None):

    print("="*60)
    print("📊 AISEN")
    print("="*60)
    
    print(f"\n📁 EN: {file_path}")
    print("⏳ EN...")

    df = pd.read_csv(file_path, low_memory=False)
    
    print(f"✅ EN！EN: {len(df):,} EN")

    if 'Length' not in df.columns:
        raise KeyError("CSV EN Length EN（EN）。")

    before_len_filter = len(df)
    df['Length'] = pd.to_numeric(df['Length'], errors='coerce')
    df = df[df['Length'].notna() & (df['Length'] > 0)].copy()
    after_len_filter = len(df)
    print(
        f"\n📏 Length EN: EN Length EN >0 EN "
        f"({after_len_filter:,}/{before_len_filter:,}, EN {after_len_filter/max(1,before_len_filter):.2%})"
    )
    if after_len_filter == 0:
        raise ValueError("Length EN：EN CSV EN Length EN 0/EN。")
    
    print("\n⏰ EN...")
    time_candidates = [
        'BaseDateTime',
        'time',
        'datetime',
        'timestamp',
        'Base Date Time',
    ]
    time_col = None
    for cand in time_candidates:
        if cand in df.columns:
            if cand != 'BaseDateTime':
                df.rename(columns={cand: 'BaseDateTime'}, inplace=True)
            time_col = 'BaseDateTime'
            break
    if time_col is None:
        raise KeyError("CSV EN BaseDateTime/time/datetime/timestamp EN。")
    df['BaseDateTime'] = pd.to_datetime(df['BaseDateTime'])
    
    df['Date'] = df['BaseDateTime'].dt.date
    
    print("\n📈 EN...")
    daily_stats = df.groupby('Date').agg({
        'MMSI': ['count', 'nunique']
    }).reset_index()
    
    daily_stats.columns = ['Date', 'RecordCount', 'ShipCount']
    daily_stats = daily_stats.sort_values('ShipCount', ascending=False)
    
    print("\n" + "="*60)
    print("📅 EN（EN）:")
    print("="*60)
    print(f"{'EN':<15} {'EN':>12} {'EN':>10}")
    print("-"*60)
    
    for _, row in daily_stats.iterrows():
        print(f"{str(row['Date']):<15} {row['RecordCount']:>12,} {row['ShipCount']:>10,}")
    
    max_day = daily_stats.iloc[0]
    max_date = max_day['Date']
    max_ships = int(max_day['ShipCount'])
    max_records = int(max_day['RecordCount'])
    
    print("\n" + "="*60)
    print("🏆 EN:")
    print("="*60)
    print(f"  EN: {max_date}")
    print(f"  EN: {max_ships:,} EN")
    print(f"  EN: {max_records:,} EN")
    print(f"  EN: {max_records/max_ships:.1f} EN")
    print("="*60)
    
    print(f"\n🔍 EN {max_date} EN...")
    filtered_df = df[df['Date'] == max_date].copy()
    
    filtered_df = filtered_df.drop(columns=['Date'])
    
    print(f"✅ EN！EN {len(filtered_df):,} EN")
    
    if output_file:
        print(f"\n💾 EN: {output_file}")
        filtered_df.to_csv(output_file, index=False)
        
        file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
        print(f"✅ EN！EN: {file_size_mb:.2f} MB")
    
    return max_date, filtered_df


def main():

    import argparse

    parser = argparse.ArgumentParser(description="Analyze AIS CSV by date and export the day with most trajectories.")
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="./data/raw/ais_yuanli.csv",
        help="EN AIS CSV（EN data/raw/）",
    )
    parser.add_argument(
        "--out-dir",
        "-o",
        type=str,
        default="./data/by_date",
        help="EN（EN CSV EN）",
    )
    args = parser.parse_args()

    input_file = args.input
    if not os.path.exists(input_file):
        print(f"❌ EN: {input_file}")
        return

    os.makedirs(args.out_dir, exist_ok=True)

    max_date, filtered_df = analyze_and_filter_by_date(input_file)

    output_file = os.path.join(args.out_dir, f"ais_filtered_{max_date}.csv")
    print(f"\n💾 EN: {output_file}")
    filtered_df.to_csv(output_file, index=False)

    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"✅ EN！EN: {file_size_mb:.2f} MB")

    print("\n" + "="*60)
    print("✨ EN！")
    print("="*60)
    print(f"\nEN:")
    print(f"  📄 {output_file}")
    print(f"\nEN。")


if __name__ == "__main__":
    main()

