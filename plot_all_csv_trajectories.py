#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EN CSV EN
====================================
EN CSV EN，EN。

EN：
- EN CSV EN，EN（EN）。

EN（EN）：
- EN CSV EN，EN：
  output_dir/<relative_parent>/<csv_stem>_trajectories.png

EN（EN）：
- EN --inplace，EN CSV EN CSV EN：
  <csv_parent>/<csv_stem>_trajectories.png
"""

import os
import numpy as np
from pathlib import Path
from load_ais import load_ais_data
from visualization import plot_trajectories


def csv_to_trajectories(csv_path):
    """
    EN CSV EN
    
    EN
    ----
    csv_path : str
        CSV EN
    
    EN
    ----
    trajectories : list of np.ndarray
        EN shape = (T_i, 4)，EN [lat, lon, sog, cog]
    ship_count : int
        EN
    total_points : int
        EN
    """
    try:
        trajs_dict = load_ais_data(csv_path)
        trajectories = []
        
        for mmsi, traj_data in trajs_dict.items():
            lat = np.array(traj_data['lat'], dtype=float)
            lon = np.array(traj_data['lon'], dtype=float)
            sog = np.array(traj_data['sog'], dtype=float)
            cog = np.array(traj_data['cog'], dtype=float)
            
            traj = np.stack([lat, lon, sog, cog], axis=1)
            trajectories.append(traj)
        
        ship_count = len(trajectories)
        total_points = sum(len(t) for t in trajectories)
        
        return trajectories, ship_count, total_points
    except Exception as e:
        print(f"  ❌ EN: {e}")
        return None, 0, 0


def plot_all_csv_in_directory(
    input_dir: str = ".",
    output_dir: str = "./trajectory_plots",
    pattern: str = "*.csv",
    min_ships: int = 1,
    max_ships: int = None,
    inplace: bool = False,
    use_english: bool = False,
):
    """
    EN CSV EN，EN
    
    EN
    ----
    input_dir : str
        EN（EN）
    output_dir : str
        EN（EN ./trajectory_plots）
    pattern : str
        EN（EN "*.csv"）
    min_ships : int
        EN，EN（EN 1）
    max_ships : int
        EN，EN（EN None，EN）
    """
    input_path = Path(input_dir).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    
    if "**" in pattern:
        csv_files = sorted(input_path.glob(pattern))
    else:
        recursive_pattern = f"**/{pattern}"
        csv_files = sorted(input_path.glob(recursive_pattern))
    
    if not csv_files:
        print(f"❌ EN {input_path} EN {pattern} EN")
        return
    
    print("=" * 60)
    print(f"EN CSV EN")
    print("=" * 60)
    print(f"📁 EN: {input_path}")
    print(f"📁 EN: {output_path}")
    print(f"🔍 EN: {pattern}")
    print(f"📊 EN {len(csv_files)} EN CSV EN\n")
    
    success_count = 0
    skip_count = 0
    error_count = 0
    
    for idx, csv_file in enumerate(csv_files, 1):
        csv_name = csv_file.name
        csv_stem = csv_file.stem
        
        print(f"[{idx}/{len(csv_files)}] EN: {csv_name}")
        
        trajectories, ship_count, total_points = csv_to_trajectories(str(csv_file))
        
        if trajectories is None:
            error_count += 1
            print(f"  ⚠️  EN（EN）\n")
            continue
        
        if ship_count < min_ships:
            skip_count += 1
            print(f"  ⚠️  EN（EN {ship_count} < {min_ships}）\n")
            continue
        
        if max_ships is not None and ship_count > max_ships:
            skip_count += 1
            print(f"  ⚠️  EN（EN {ship_count} > {max_ships}）\n")
            continue
        
        if inplace:
            output_file = csv_file.parent / f"{csv_stem}_trajectories.png"
        else:
            try:
                rel_parent = csv_file.parent.relative_to(input_path)
            except Exception:
                rel_parent = Path(".")
            out_subdir = (output_path / rel_parent).resolve()
            out_subdir.mkdir(parents=True, exist_ok=True)
            output_file = out_subdir / f"{csv_stem}_trajectories.png"
        
        try:
            if use_english:
                title = f"{csv_stem}\n({ship_count} ships, {total_points:,} data points)"
            else:
                title = f"{csv_stem}\n（{ship_count} EN，{total_points:,} EN）"
            plot_trajectories(
                trajectories,
                str(output_file),
                max_count=None,
                title=title,
                figsize=(20, 20),
                pad_ratio=0.05,
                use_english=use_english,
            )
            
            success_count += 1
            print(f"  ✅ EN: {output_file.name}")
            print(f"     EN: {ship_count}, EN: {total_points:,}\n")
        except Exception as e:
            error_count += 1
            print(f"  ❌ EN: {e}\n")
    
    print("=" * 60)
    print("EN")
    print("=" * 60)
    print(f"✅ EN: {success_count} EN")
    print(f"⚠️  EN: {skip_count} EN")
    print(f"❌ EN: {error_count} EN")
    print(f"📁 EN: {output_path}")


def plot_single_csv(
    csv_path: str,
    output_dir: str = "./trajectory_plots",
    output_file: str | None = None,
    min_ships: int = 1,
    max_ships: int | None = None,
    inplace: bool = False,
    use_english: bool = False,
):
    """
    EN CSV EN（EN）。

    EN
    ----
    csv_path : str
        CSV EN
    output_dir : str
        EN（EN output_file EN）
    output_file : str | None
        EN PNG EN（EN，EN output_dir/<stem>_trajectories.png）
    min_ships / max_ships :
        EN（EN）
    """
    csv_p = Path(csv_path).expanduser().resolve()
    if not csv_p.exists():
        print(f"❌ CSV EN: {csv_p}")
        return

    out_dir_p = Path(output_dir).expanduser().resolve()
    out_dir_p.mkdir(parents=True, exist_ok=True)

    if output_file:
        out_file_p = Path(output_file).expanduser().resolve()
        out_file_p.parent.mkdir(parents=True, exist_ok=True)
    elif inplace:
        out_file_p = csv_p.parent / f"{csv_p.stem}_trajectories.png"
    else:
        try:
            rel_parent = csv_p.parent.relative_to(Path(".").resolve())
        except Exception:
            rel_parent = Path(".")
        out_subdir = (out_dir_p / rel_parent).resolve()
        out_subdir.mkdir(parents=True, exist_ok=True)
        out_file_p = out_subdir / f"{csv_p.stem}_trajectories.png"

    print("=" * 60)
    print("EN CSV EN")
    print("=" * 60)
    print(f"📄 EN: {csv_p}")
    print(f"🖼️ EN: {out_file_p}")

    trajectories, ship_count, total_points = csv_to_trajectories(str(csv_p))
    if trajectories is None:
        print("❌ EN，EN")
        return

    if ship_count < min_ships:
        print(f"⚠️ EN（EN {ship_count} < {min_ships}）")
        return
    if max_ships is not None and ship_count > max_ships:
        print(f"⚠️ EN（EN {ship_count} > {max_ships}）")
        return

    try:
        if use_english:
            title = f"{csv_p.stem}\n({ship_count} ships, {total_points:,} data points)"
        else:
            title = f"{csv_p.stem}\n（{ship_count} EN，{total_points:,} EN）"
        plot_trajectories(
            trajectories,
            str(out_file_p),
            max_count=None,
            title=title,
            figsize=(20, 20),
            pad_ratio=0.05,
            use_english=use_english,
        )
        print(f"✅ EN: {out_file_p}")
        print(f"   EN: {ship_count}, EN: {total_points:,}")
    except Exception as e:
        print(f"❌ EN: {e}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="EN CSV EN（EN）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EN:
  python plot_all_csv_trajectories.py --csv ./AIS_xxx.csv

  python plot_all_csv_trajectories.py
  
  python plot_all_csv_trajectories.py --input ./data --output ./plots
  
  python plot_all_csv_trajectories.py --pattern "*ais*.csv"
  
  python plot_all_csv_trajectories.py --min-ships 10 --max-ships 100
        """
    )
    
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="EN CSV EN（EN）",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="EN PNG EN（EN: --output/<csv_stem>_trajectories.png）",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="EN CSV EN（EN：--output-file > --inplace > --output）",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=".",
        help="EN（EN: EN）"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./trajectory_plots",
        help="EN（EN: ./trajectory_plots）"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.csv",
        help="EN（EN: *.csv）"
    )
    parser.add_argument(
        "--min-ships",
        type=int,
        default=1,
        help="EN，EN（EN: 1）"
    )
    parser.add_argument(
        "--max-ships",
        type=int,
        default=None,
        help="EN，EN（EN: EN）"
    )
    parser.add_argument(
        "--english",
        action="store_true",
        help="EN（EN: EN）"
    )
    
    args = parser.parse_args()

    if args.csv:
        plot_single_csv(
            csv_path=args.csv,
            output_dir=args.output,
            output_file=args.output_file,
            min_ships=args.min_ships,
            max_ships=args.max_ships,
            inplace=args.inplace,
            use_english=args.english,
        )
    else:
        plot_all_csv_in_directory(
            input_dir=args.input,
            output_dir=args.output,
            pattern=args.pattern,
            min_ships=args.min_ships,
            max_ships=args.max_ships,
            inplace=args.inplace,
            use_english=args.english,
        )


if __name__ == "__main__":
    main()

