#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import subprocess
import sys
from pathlib import Path


def find_dataset_folders(base_dir: str) -> list[tuple[str, str]]:
    """
    EN dataset EN CSV EN。
    
    EN: [(dataset_folder, csv_path), ...]
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"❌ EN: {base_dir}")
        return []
    
    results = []
    for item in base_path.iterdir():
        if not item.is_dir():
            continue
        
        name = item.name
        if name.endswith("_output") or name.startswith("run_"):
            continue
        
        if not name.endswith("_dataset"):
            continue
        
        csv_path = item / "pruned_dataset.csv"
        if not csv_path.exists():
            csv_files = list(item.glob("*.csv"))
            if csv_files:
                csv_path = csv_files[0]
                print(f"⚠️  {name}: EN pruned_dataset.csv，EN {csv_path.name}")
            else:
                print(f"⚠️  {name}: EN CSV EN，EN")
                continue
        
        results.append((str(item), str(csv_path)))
    
    return results


def run_pipeline(csv_path: str, output_dir: str, verbose: bool = True) -> bool:
    """
    EN。
    
    EN:
        csv_path: EN CSV EN
        output_dir: EN
        verbose: EN
    
    EN: EN
    """
    script_path = Path(__file__).parent / "run_complete_pipeline.py"
    if not script_path.exists():
        print(f"❌ EN: {script_path}")
        return False
    
    cmd = [
        sys.executable,
        str(script_path),
        "--ais", csv_path,
        "--out", output_dir,
        "--mode", "paper",
        "--feature-mode", "delta_cog",
        "--minpts-stat", "paper",
    ]
    
    if verbose:
        print(f"\n{'='*72}")
        print(f"EN: {' '.join(cmd)}")
        print(f"{'='*72}\n")
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,
            text=True,
        )
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"❌ EN (EN {e.returncode})")
        return False
    except Exception as e:
        print(f"❌ EN: {e}")
        return False


def main():
    base_dir = "outputs/two_stage"
    
    print("=" * 72)
    print("EN paper EN")
    print("=" * 72)
    print(f"EN: {base_dir}\n")
    
    datasets = find_dataset_folders(base_dir)
    
    if not datasets:
        print("❌ EN dataset EN")
        return
    
    print(f"✅ EN {len(datasets)} EN dataset EN:\n")
    for i, (folder, csv_path) in enumerate(datasets, 1):
        print(f"  {i}. {Path(folder).name}")
        print(f"     CSV: {Path(csv_path).name}")
    
    print("\n" + "=" * 72)
    response = input(f"EN？(y/n): ").strip().lower()
    if response != 'y':
        print("EN")
        return
    
    success_count = 0
    fail_count = 0
    
    for i, (folder, csv_path) in enumerate(datasets, 1):
        dataset_name = Path(folder).name
        print(f"\n{'='*72}")
        print(f"[{i}/{len(datasets)}] EN: {dataset_name}")
        print(f"{'='*72}")
        
        output_folder_name = dataset_name + "_output"
        output_dir = str(Path(base_dir) / output_folder_name)
        
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"EN: {csv_path}")
        print(f"EN: {output_dir}")
        
        success = run_pipeline(csv_path, output_dir, verbose=True)
        
        if success:
            print(f"\n✅ [{i}/{len(datasets)}] {dataset_name} EN")
            success_count += 1
        else:
            print(f"\n❌ [{i}/{len(datasets)}] {dataset_name} EN")
            fail_count += 1
    
    print("\n" + "=" * 72)
    print("EN")
    print("=" * 72)
    print(f"EN: {success_count}/{len(datasets)}")
    print(f"EN: {fail_count}/{len(datasets)}")
    print("=" * 72)


if __name__ == "__main__":
    main()
