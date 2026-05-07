#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EN
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

try:
    rcParams['font.sans-serif'] = ['PingFang SC', 'Noto Sans CJK SC', 'SimHei', 'Arial Unicode MS']
    rcParams['axes.unicode_minus'] = False
except Exception:
    pass

from load_ais import load_ais_data, clip_sog_kn
from data_pre_processing import interpolate_traj
from trajectory_simplification import sacs_b_simplify


def save_single_traj_with_points(orig, proc, simp, save_path, title_prefix: str):
    """EN（EN/EN/EN，EN）EN。"""
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))

    lat_all = np.concatenate([
        np.asarray(orig['lat'], dtype=float),
        proc[:, 0],
        simp[:, 0] if len(simp) else np.asarray([], dtype=float),
    ])
    lon_all = np.concatenate([
        np.asarray(orig['lon'], dtype=float),
        proc[:, 1],
        simp[:, 1] if len(simp) else np.asarray([], dtype=float),
    ])
    lat_min, lat_max = float(np.min(lat_all)), float(np.max(lat_all))
    lon_min, lon_max = float(np.min(lon_all)), float(np.max(lon_all))
    dlat = max(1e-9, lat_max - lat_min)
    dlon = max(1e-9, lon_max - lon_min)

    ax = axes[0]
    ax.plot(orig['lon'], orig['lat'], '-', color='#2c7fb8', linewidth=1.5, alpha=0.6)
    ax.scatter(orig['lon'], orig['lat'], s=8, color='#2c7fb8', alpha=0.8)
    ax.set_xlim(lon_min - 0.05 * dlon, lon_max + 0.05 * dlon)
    ax.set_ylim(lat_min - 0.05 * dlat, lat_max + 0.05 * dlat)
    ax.set_title('EN (EN)')
    ax.set_xlabel('EN (°)')
    ax.set_ylabel('EN (°)')
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(proc[:, 1], proc[:, 0], '-', color='#d95f0e', linewidth=1.5, alpha=0.6)
    ax.scatter(proc[:, 1], proc[:, 0], s=8, color='#d95f0e', alpha=0.8)
    ax.set_xlim(lon_min - 0.05 * dlon, lon_max + 0.05 * dlon)
    ax.set_ylim(lat_min - 0.05 * dlat, lat_max + 0.05 * dlat)
    ax.set_title('EN (EN)')
    ax.set_xlabel('EN (°)')
    ax.set_ylabel('EN (°)')
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    if len(simp):
        ax.plot(simp[:, 1], simp[:, 0], '-o', color='#1a9850', linewidth=1.5, markersize=4, alpha=0.9)
        ax.scatter(simp[:, 1], simp[:, 0], s=12, color='#1a9850', alpha=0.95)
    ax.set_xlim(lon_min - 0.05 * dlon, lon_max + 0.05 * dlon)
    ax.set_ylim(lat_min - 0.05 * dlat, lat_max + 0.05 * dlat)
    ax.set_title(f'EN (EN{len(simp)}EN)')
    ax.set_xlabel('EN (°)')
    ax.set_ylabel('EN (°)')
    ax.grid(True, alpha=0.3)

    fig.suptitle(f'{title_prefix} - EN', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✅ EN: {save_path}")


def _plot_stage_all_points(latlon_list, save_path: str, title: str, pad_ratio: float = 0.12):
    """EN，EN。
    pad_ratio: EN，EN。
    """
    fig, ax = plt.subplots(1, 1, figsize=(20, 20))

    lat_all = []
    lon_all = []
    for lat, lon in latlon_list:
        if len(lat) == 0:
            continue
        lat_all.append(np.asarray(lat, dtype=float))
        lon_all.append(np.asarray(lon, dtype=float))
    if not lat_all:
        return
    lat_all = np.concatenate(lat_all)
    lon_all = np.concatenate(lon_all)
    lat_min, lat_max = float(np.min(lat_all)), float(np.max(lat_all))
    lon_min, lon_max = float(np.min(lon_all)), float(np.max(lon_all))
    dlat = max(1e-9, lat_max - lat_min)
    dlon = max(1e-9, lon_max - lon_min)

    for lat, lon in latlon_list:
        if len(lat) == 0:
            continue
        lat = np.asarray(lat, dtype=float)
        lon = np.asarray(lon, dtype=float)
        ax.plot(lon, lat, '-', linewidth=0.7, alpha=0.7, color='#1f77b4')
        ax.scatter(lon, lat, s=1, alpha=0.6, color='#1f77b4')

    ax.set_xlim(lon_min - pad_ratio * dlon, lon_max + pad_ratio * dlon)
    ax.set_ylim(lat_min - pad_ratio * dlat, lat_max + pad_ratio * dlat)
    ax.set_xlabel('EN (°)')
    ax.set_ylabel('EN (°)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✅ EN: {save_path}")


def plot_trajectory_comparison(original_trajs, processed_trajs, simplified_trajs, 
                               save_path, max_display=10):
    """
    EN、EN、EN
    
    Args:
        original_trajs: EN
        processed_trajs: EN
        simplified_trajs: EN
        save_path: EN
        max_display: EN
    """
    n_display = min(max_display, len(original_trajs))
    
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    
    all_lats = []
    all_lons = []
    
    for i in range(n_display):
        orig = original_trajs[i]
        all_lats.extend(orig['lat'])
        all_lons.extend(orig['lon'])
        
        proc = processed_trajs[i]
        all_lats.extend(proc[:, 0])
        all_lons.extend(proc[:, 1])
        
        simp = simplified_trajs[i]
        all_lats.extend(simp[:, 0])
        all_lons.extend(simp[:, 1])
    
    lat_min, lat_max = min(all_lats), max(all_lats)
    lon_min, lon_max = min(all_lons), max(all_lons)
    
    lat_margin = (lat_max - lat_min) * 0.05
    lon_margin = (lon_max - lon_min) * 0.05
    
    ax1 = axes[0]
    for i in range(n_display):
        orig = original_trajs[i]
        ax1.plot(orig['lon'], orig['lat'], '-', linewidth=1.5, alpha=0.6)
    ax1.set_xlim(lon_min - lon_margin, lon_max + lon_margin)
    ax1.set_ylim(lat_min - lat_margin, lat_max + lat_margin)
    ax1.set_xlabel('EN (°)', fontsize=12)
    ax1.set_ylabel('EN (°)', fontsize=12)
    ax1.set_title(f'EN (EN{n_display}EN)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    for i in range(n_display):
        proc = processed_trajs[i]
        ax2.plot(proc[:, 1], proc[:, 0], '-', linewidth=1.5, alpha=0.6)
    ax2.set_xlim(lon_min - lon_margin, lon_max + lon_margin)
    ax2.set_ylim(lat_min - lat_margin, lat_max + lat_margin)
    ax2.set_xlabel('EN (°)', fontsize=12)
    ax2.set_ylabel('EN (°)', fontsize=12)
    ax2.set_title(f'EN (EN+EN)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    ax3 = axes[2]
    for i in range(n_display):
        simp = simplified_trajs[i]
        ax3.plot(simp[:, 1], simp[:, 0], '-o', linewidth=1.5, markersize=3, alpha=0.6)
    ax3.set_xlim(lon_min - lon_margin, lon_max + lon_margin)
    ax3.set_ylim(lat_min - lat_margin, lat_max + lat_margin)
    ax3.set_xlabel('EN (°)', fontsize=12)
    ax3.set_ylabel('EN (°)', fontsize=12)
    ax3.set_title(f'EN (SACS-B)', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✅ EN: {save_path}")


def plot_single_trajectory_detail(original, processed, simplified, save_path, ship_id):
    """
    EN
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    all_lats = list(original['lat']) + list(processed[:, 0]) + list(simplified[:, 0])
    all_lons = list(original['lon']) + list(processed[:, 1]) + list(simplified[:, 1])
    lat_min, lat_max = min(all_lats), max(all_lats)
    lon_min, lon_max = min(all_lons), max(all_lons)
    lat_margin = (lat_max - lat_min) * 0.05
    lon_margin = (lon_max - lon_min) * 0.05
    
    ax1 = axes[0, 0]
    ax1.plot(original['lon'], original['lat'], 'b-', linewidth=2, alpha=0.6, label='EN')
    ax1.plot(processed[:, 1], processed[:, 0], 'r-', linewidth=1, alpha=0.8, label='EN')
    ax1.set_xlim(lon_min - lon_margin, lon_max + lon_margin)
    ax1.set_ylim(lat_min - lat_margin, lat_max + lat_margin)
    ax1.set_xlabel('EN (°)', fontsize=11)
    ax1.set_ylabel('EN (°)', fontsize=11)
    ax1.set_title('EN vs EN', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[0, 1]
    ax2.plot(processed[:, 1], processed[:, 0], 'r-', linewidth=2, alpha=0.4, label='EN')
    ax2.plot(simplified[:, 1], simplified[:, 0], 'g-o', linewidth=2, markersize=4, 
             alpha=0.8, label=f'EN (EN{len(simplified)}/{len(processed)}EN)')
    ax2.set_xlim(lon_min - lon_margin, lon_max + lon_margin)
    ax2.set_ylim(lat_min - lat_margin, lat_max + lat_margin)
    ax2.set_xlabel('EN (°)', fontsize=11)
    ax2.set_ylabel('EN (°)', fontsize=11)
    ax2.set_title('EN vs EN', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    ax3 = axes[1, 0]
    ax3.plot(range(len(original['sog'])), original['sog'], 'b-', linewidth=1, alpha=0.6, label='EN')
    ax3.plot(range(len(processed)), processed[:, 2], 'r-', linewidth=1, alpha=0.8, label='EN')
    ax3.set_xlabel('EN', fontsize=11)
    ax3.set_ylabel('EN (knots)', fontsize=11)
    ax3.set_title('EN(SOG)EN', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    ax4 = axes[1, 1]
    categories = ['EN', 'EN', 'EN']
    counts = [len(original['lat']), len(processed), len(simplified)]
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    bars = ax4.bar(categories, counts, color=colors, alpha=0.7)
    
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax4.set_ylabel('EN', fontsize=11)
    ax4.set_title('EN', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle(f'EN {ship_id} - EN', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✅ EN: {save_path}")


def main():
    """EN"""
    import argparse
    
    parser = argparse.ArgumentParser(description='EN')
    parser.add_argument('--ais', type=str, required=True,
                       help='AISEN')
    parser.add_argument('--output-dir', type=str, default='./outputs/comparison',
                       help='EN')
    parser.add_argument('--max-display', type=int, default=10,
                       help='EN')
    parser.add_argument('--detail-ship', type=int, default=None,
                       help='ENNEN（EN1EN）')
    parser.add_argument('--export-all', action='store_true',
                       help='EN（EN/EN/EN，EN）')
    parser.add_argument('--export-stage-all', action='store_true',
                       help='EN：EN/EN/EN，EN')
    parser.add_argument('--stage-pad', type=float, default=0.12,
                       help='EN（EN0.12，EN0.1~0.3）')
    
    args = parser.parse_args()
    
    print("="*70)
    print("🔍 EN")
    print("="*70)
    
    if not os.path.exists(args.ais):
        print(f"❌ EN: {args.ais}")
        return
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n📁 EN1: EN...")
    trajs = load_ais_data(args.ais)
    print(f"✅ EN {len(trajs)} EN")
    
    ship_ids = [sid for sid, t in trajs.items() if len(t['lat']) >= 50]
    print(f"✅ EN {len(ship_ids)} EN（EN≥50）")
    
    if len(ship_ids) == 0:
        print("❌ EN")
        return
    
    print("\n🔄 EN2: EN（EN+EN）...")
    original_trajs = []
    processed_trajs = []
    skipped_length_mmsi = []
    
    for i, sid in enumerate(ship_ids):
        ship_data = trajs[sid]
        Lm = ship_data.get("length_m", np.nan)
        try:
            Lm = float(Lm)
        except Exception:
            Lm = float("nan")
        if (not np.isfinite(Lm)) or (Lm <= 0.0):
            skipped_length_mmsi.append(sid)
            continue

        original_trajs.append(ship_data)
        
        traj_interp = interpolate_traj(ship_data, fixed_dt_s=60)
        traj_interp['sog'] = clip_sog_kn(traj_interp['sog'], 0.0, 45.0)
        
        traj = np.stack([
            np.array(traj_interp['lat']),
            np.array(traj_interp['lon']),
            np.array(traj_interp['sog']),
            np.array(traj_interp['cog']),
        ], axis=1)
        
        processed_trajs.append(traj)
        
        if (i + 1) % 10 == 0:
            print(f"  EN {i+1}/{len(ship_ids)} EN...")
    
    if skipped_length_mmsi:
        print(f"⚠️  EN {len(skipped_length_mmsi)} EN（Length EN/EN）。EN MMSI: {skipped_length_mmsi[:5]}")
    if len(processed_trajs) == 0:
        print("❌ EN Length EN/EN：EN CSV EN Length EN。")
        return
    print(f"✅ EN")
    
    print("\n✂️  EN3: EN（SACS-B）...")
    simplified_trajs = []
    
    for i, tr in enumerate(processed_trajs):
        lat, lon, sog, cog = tr[:, 0], tr[:, 1], tr[:, 2], tr[:, 3]
        Lm = float(original_trajs[i].get("length_m"))
        lat_s, lon_s, kept = sacs_b_simplify(lat, lon, sog, cog, ship_length_m=Lm, alpha=2.2)
        simp = np.stack([lat[kept], lon[kept], sog[kept], cog[kept]], axis=1)
        simplified_trajs.append(simp)
        
        if (i + 1) % 10 == 0:
            print(f"  EN {i+1}/{len(processed_trajs)} EN...")
    
    print(f"✅ EN")
    
    print("\n" + "="*70)
    print("📊 EN:")
    print("="*70)
    
    total_original = sum(len(t['lat']) for t in original_trajs)
    total_processed = sum(len(t) for t in processed_trajs)
    total_simplified = sum(len(t) for t in simplified_trajs)
    
    print(f"  EN: {len(original_trajs)}")
    print(f"  EN: {total_original:,}")
    print(f"  EN: {total_processed:,}")
    print(f"  EN: {total_simplified:,}")
    print(f"  EN: {total_processed/total_original:.2f}x")
    print(f"  EN: {total_simplified/total_processed:.2%}")
    print("="*70)
    
    print("\n📊 EN4: EN...")
    
    comparison_path = os.path.join(args.output_dir, 'trajectories_comparison.png')
    plot_trajectory_comparison(original_trajs, processed_trajs, simplified_trajs, 
                               comparison_path, max_display=args.max_display)
    
    if args.detail_ship is not None:
        ship_idx = args.detail_ship - 1
        if 0 <= ship_idx < len(ship_ids):
            detail_path = os.path.join(args.output_dir, 
                                      f'trajectory_detail_ship_{ship_ids[ship_idx]}.png')
            plot_single_trajectory_detail(
                original_trajs[ship_idx],
                processed_trajs[ship_idx],
                simplified_trajs[ship_idx],
                detail_path,
                ship_ids[ship_idx]
            )
        else:
            print(f"⚠️  EN: EN {args.detail_ship} EN (1-{len(ship_ids)})")
    else:
        if len(ship_ids) > 0:
            detail_path = os.path.join(args.output_dir, 
                                      f'trajectory_detail_ship_{ship_ids[0]}.png')
            plot_single_trajectory_detail(
                original_trajs[0],
                processed_trajs[0],
                simplified_trajs[0],
                detail_path,
                ship_ids[0]
            )

    if args.export_all:
        per_dir = os.path.join(args.output_dir, 'all_trajs')
        os.makedirs(per_dir, exist_ok=True)
        print(f"\n🖼️  EN（EN）... EN {len(processed_trajs)} EN")
        for i in range(len(processed_trajs)):
            mmsi = ship_ids[i]
            save_path = os.path.join(per_dir, f'traj_{i+1:03d}_ship_{mmsi}.png')
            save_single_traj_with_points(
                original_trajs[i],
                processed_trajs[i],
                simplified_trajs[i],
                save_path,
                title_prefix=f'EN {mmsi}'
            )

    if args.export_stage_all:
        stage_dir = os.path.join(args.output_dir, 'stage_all')
        os.makedirs(stage_dir, exist_ok=True)

        orig_latlon = [(t['lat'], t['lon']) for t in original_trajs]
        _plot_stage_all_points(
            orig_latlon,
            os.path.join(stage_dir, 'all_original.png'),
            'EN（EN）',
            pad_ratio=args.stage_pad,
        )

        proc_latlon = [(tr[:, 0], tr[:, 1]) for tr in processed_trajs]
        _plot_stage_all_points(
            proc_latlon,
            os.path.join(stage_dir, 'all_processed.png'),
            'EN（EN）',
            pad_ratio=args.stage_pad,
        )

        simp_latlon = [(tr[:, 0], tr[:, 1]) for tr in simplified_trajs]
        _plot_stage_all_points(
            simp_latlon,
            os.path.join(stage_dir, 'all_simplified.png'),
            'EN（EN）',
            pad_ratio=args.stage_pad,
        )
    
    print("\n" + "="*70)
    print("✨ EN！EN:")
    print("="*70)
    print(f"  📂 {os.path.abspath(args.output_dir)}")
    print(f"  📄 trajectories_comparison.png - EN")
    print(f"  📄 trajectory_detail_ship_*.png - EN")
    print("="*70)


if __name__ == '__main__':
    main()

