#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wei et al. (2024)
Adaptive Marine Traffic Behaviour Pattern Recognition (AMTBRA)
—— EN
"""

import os
import sys
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from load_ais import wrap_lat, wrap_lon, wrap_cog, clip_sog_kn
from Adaptive_DBSCAN import run_amtbra_dbscan_paper
from Adaptive_DBSCAN import dbscan_from_distance
from Adaptive_DBSCAN import drl_select_eps_minpts
from Adaptive_DBSCAN import build_TS_features
from Adaptive_DBSCAN import compute_dtw_distance
from evaluate import evaluate_clusters_paper
from dtw_cache import build_distance_matrix_with_cache
from traditional_clustering import traditional_clustering

try:
    from Adaptive_DBSCAN import density_power_curve  # type: ignore
except Exception:
    density_power_curve = None

try:
    from Adaptive_DBSCAN import density_power_curve_poa_minmax_strict  # type: ignore
except Exception:
    density_power_curve_poa_minmax_strict = None



def _get_density_power_curve_for_plot(
    D: np.ndarray,
    *,
    delta_s: float,
    two_stage: bool = False,
    coarse_steps: int = 150,
    refine_steps: int = 300,
    refine_window_mult: float = 5.0,
):
    """
    EN (eps_list, avg_list, pow_list, curve_name)。
    EN POA Table 3 EN；EN；EN (None, None, None, None)。
    
    EN two-stage EN：EN two_stage=True EN，EN“EN”EN（EN eps EN）。
    """
    if density_power_curve_poa_minmax_strict is not None:
        dc_list, mpt_list, p_list = density_power_curve_poa_minmax_strict(
            D,
            delta_s=delta_s,
            two_stage=bool(two_stage),
            coarse_steps=int(coarse_steps),
            refine_steps=int(refine_steps),
            refine_window_mult=float(refine_window_mult),
        )
        return dc_list, mpt_list, p_list, "poa_strict"
    if density_power_curve is not None:
        eps_list, avg_list, pow_list = density_power_curve(D, delta_s=delta_s)
        return eps_list, avg_list, pow_list, "engineering"
    return None, None, None, None


os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

class TeeLogger:
    """EN"""
    def __init__(self, file_path):
        self.file = open(file_path, 'w', encoding='utf-8')
        self.stdout = sys.stdout
    
    def write(self, text):
        self.stdout.write(text)
        self.file.write(text)
        self.file.flush()
    
    def flush(self):
        self.stdout.flush()
        self.file.flush()
    
    def close(self):
        if hasattr(self, 'file') and self.file:
            self.file.close()
        sys.stdout = self.stdout


def dump_sequence_length_distribution(
    seqs,
    tag: str,
    out_dir: str,
    bins: int = 30,
):
    """
    EN（EN DTW EN）。

    - EN：count/min/mean/p50/p90/p95/max
    - EN：
      - len_stats_{tag}.json
      - len_hist_{tag}.csv
    """
    try:
        lens = []
        for s in (seqs or []):
            if s is None:
                continue
            try:
                lens.append(int(len(s)))
            except Exception:
                continue

        lens_np = np.asarray(lens, dtype=np.int64)
        if lens_np.size == 0:
            print(f"📏 [{tag}] EN：EN（size=0）")
            return

        def q(p: float) -> int:
            return int(np.quantile(lens_np, p))

        stats = {
            "tag": str(tag),
            "count": int(lens_np.size),
            "min": int(lens_np.min()),
            "mean": float(lens_np.mean()),
            "std": float(lens_np.std()),
            "p25": q(0.25),
            "p50": q(0.50),
            "p75": q(0.75),
            "p90": q(0.90),
            "p95": q(0.95),
            "p99": q(0.99),
            "max": int(lens_np.max()),
        }

        print(
            f"📏 [{tag}] EN："
            f"count={stats['count']}, min={stats['min']}, mean={stats['mean']:.1f}, "
            f"p50={stats['p50']}, p90={stats['p90']}, p95={stats['p95']}, max={stats['max']}"
        )

        os.makedirs(out_dir, exist_ok=True)

        bins_i = int(bins) if int(bins) > 0 else 30
        hist, edges = np.histogram(lens_np, bins=bins_i)
        df_hist = pd.DataFrame(
            {
                "bin_left": edges[:-1].astype(int),
                "bin_right": edges[1:].astype(int),
                "count": hist.astype(int),
            }
        )
        hist_path = os.path.join(out_dir, f"len_hist_{tag}.csv")
        df_hist.to_csv(hist_path, index=False, encoding="utf-8")

        import json as _json

        stats_path = os.path.join(out_dir, f"len_stats_{tag}.json")
        with open(stats_path, "w", encoding="utf-8") as f:
            _json.dump(stats, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"⚠️ [{tag}] EN: {e}")


def export_raw_trajs(raw_trajs, k=92, out_path="debug_raw_trajs.csv"):
    """
    EN k EN CSV EN
    
    EN
    ----
    raw_trajs : list of np.ndarray
        EN，EN shape = (T_i, 4)，EN [lat, lon, sog, cog]
    k : int
        EN，EN 92
    out_path : str
        EN CSV EN
    """
    k = min(k, len(raw_trajs))
    rows = []
    for traj_id in range(k):
        traj = raw_trajs[traj_id]
        for point_idx in range(len(traj)):
            lat, lon, sog, cog = traj[point_idx]
            rows.append({
                'traj_id': traj_id,
                'point_idx': point_idx,
                'lat': lat,
                'lon': lon,
                'sog': sog,
                'cog': cog
            })
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False, encoding='utf-8')
    print(f"✅ EN {k} EN: {out_path}")


def export_simplified_trajs(simplified_trajs, k=92, out_path="debug_simplified_trajs.csv"):
    """
    EN k EN CSV EN
    
    EN
    ----
    simplified_trajs : list of np.ndarray
        EN，EN shape = (T_j, 4)，EN [lat, lon, sog, cog]
    k : int
        EN，EN 92
    out_path : str
        EN CSV EN
    """
    k = min(k, len(simplified_trajs))
    rows = []
    for traj_id in range(k):
        traj = simplified_trajs[traj_id]
        for point_idx in range(len(traj)):
            lat, lon, sog, cog = traj[point_idx]
            rows.append({
                'traj_id': traj_id,
                'point_idx': point_idx,
                'lat': lat,
                'lon': lon,
                'sog': sog,
                'cog': cog
            })
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False, encoding='utf-8')
    print(f"✅ EN {k} EN: {out_path}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ais", type=str, required=True, help="AIS CSV EN")
    parser.add_argument("--out", type=str, default="./outputs", help="EN")
    parser.add_argument("--fixed-dt-s", type=int, default=60, help="EN（EN，EN60）")
    parser.add_argument("--traj-cache-dir", type=str, default=None, help="EN（EN ./traj_cache）")
    parser.add_argument("--force-recompute-traj", action="store_true", help="EN+EN（EN）")
    parser.add_argument(
        "--max-interp-points",
        type=int,
        default=None,
        help="EN（EN，EN；EN）",
    )
    parser.add_argument(
        "--max-simplified-points",
        type=int,
        default=None,
        help="EN（EN，EN；EN）。"
             "EN（EN）EN DTW/EN。",
    )
    parser.add_argument(
        "--delta-s",
        type=float,
        default=0.04,
        help="EN POA Table3 EN eps EN ΔS（EN 0.04）。EN two-stage EN。",
    )
    parser.add_argument(
        "--poa-two-stage",
        action="store_true",
        help="EN POA eps EN+EN（EN）。",
    )
    parser.add_argument(
        "--poa-coarse-steps",
        type=int,
        default=150,
        help="two-stage：EN（EN 150）。",
    )
    parser.add_argument(
        "--poa-refine-steps",
        type=int,
        default=300,
        help="two-stage：EN（EN 300）。",
    )
    parser.add_argument(
        "--poa-refine-window-mult",
        type=float,
        default=5.0,
        help="two-stage：EN（EN 5.0），EN=eps_coarse±mult*ΔS_coarse。",
    )
    parser.add_argument(
        "--feature-mode",
        type=str,
        default="delta_cog",
        choices=["delta_cog", "pca_latlon"],
        help="DTW EN：delta_cog / pca_latlon。EN delta_cog。",
    )
    parser.add_argument(
        "--minpts-stat",
        type=str,
        default="paper",
        choices=["mean", "median", "eps_gmax", "paper"],
        help="paper/POA EN minPts EN：mean/median（EN eps EN）EN eps_gmax/paper（EN MinPt=Eps*Gmax，EN paper）。",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="paper",
        choices=["paper", "drl", "dtw_only"],
        help="EN：'paper'=EN，'drl'=DRLEN，'dtw_only'=ENDTWEN（EN）",
    )
    parser.add_argument(
        "--english",
        action="store_true",
        help="EN（EN，EN：EN）",
    )
    parser.add_argument(
        "--train-drl",
        action="store_true",
        help="EN DRL EN（EN，EN --mode drl EN）",
    )
    parser.add_argument(
        "--drl-model",
        type=str,
        default="./drl_model.pth",
        help="DRL EN（EN，EN，EN: ./drl_model.pth）",
    )
    parser.add_argument(
        "--drl-episodes",
        type=int,
        default=50,
        help="DRL EN（EN --train-drl EN --mode drl EN，EN: 50）",
    )
    parser.add_argument(
        "--eps-sweep",
        action="store_true",
        help="(ENC) EN --mode drl EN eps sweep，EN K-ε EN out_dir（EN）",
    )
    parser.add_argument(
        "--eps-sweep-minpts",
        type=int,
        default=8,
        help="(ENC) eps sweep EN minPts（EN 8）",
    )
    parser.add_argument(
        "--eps-sweep-start",
        type=float,
        default=0.05,
        help="(ENC) eps sweep EN（EN 0.05；EN DRL EN D∈[0,1]）",
    )
    parser.add_argument(
        "--eps-sweep-end",
        type=float,
        default=0.95,
        help="(ENC) eps sweep EN（EN 0.95；EN DRL EN D∈[0,1]）",
    )
    parser.add_argument(
        "--eps-sweep-step",
        type=float,
        default=0.01,
        help="(ENC) eps sweep EN（EN 0.01）",
    )
    parser.add_argument("--eps", type=float, default=None, help="ENDBSCANENepsilon（EN）")
    parser.add_argument("--minpts", type=int, default=None, help="ENDBSCANENminPts（EN8，EN --eps EN8）")
    parser.add_argument("--force-recompute-dtw", action="store_true", help="ENDTWEN（EN）")
    parser.add_argument("--dtw-cache-dir", type=str, default=None, help="DTWEN（EN: ./dtw_cache，EN）")
    parser.add_argument(
        "--print-len-stats",
        action="store_true",
        help="EN（raw/simplified/TSz），EN DTW EN",
    )
    parser.add_argument(
        "--len-hist-bins",
        type=int,
        default=30,
        help="EN bins EN（EN 30）",
    )
    args = parser.parse_args()

    out_base = os.path.abspath(args.out)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(out_base, f"run_{run_id}")
    os.makedirs(out_dir, exist_ok=True)
    
    log_path = os.path.join(out_dir, "summary.txt")
    tee = TeeLogger(log_path)
    sys.stdout = tee
    sys.stderr = tee

    try:
        print("=" * 60)
        if args.mode == "paper":
            print("EN (EN)")
        elif args.mode == "drl":
            print("EN (DRL EN)")
        else:
            print("EN (EN DTW EN)")
        print("=" * 60)
        print(f"📌 EN: {args.mode}")
        print(f"🧩 EN: {args.feature_mode}")

        ais_file = args.ais
        if not os.path.exists(ais_file):
            print(f"❌ AISEN: {ais_file}")
            return

        print(f"✅ ENAISEN: {ais_file}")

        print("\n" + "=" * 40)
        print("EN1: EN")
        print("=" * 40)

        from load_ais import load_ais_data
        trajs = load_ais_data(ais_file)
        print(f"✅ EN {len(trajs)} EN")

        ship_ids = [sid for sid, t in trajs.items() if len(t["lat"]) >= 50]
        print(f"✅ EN {len(ship_ids)} EN")

        print("\n" + "=" * 40)
        print("EN2: EN")
        print("=" * 40)

        from data_pre_processing import interpolate_traj
        processed_trajectories = []
        processed_ship_lengths_m = []
        processed_ship_ids = []
        skipped_length_mmsi = []

        traj_cache_payload = None
        try:
            from traj_cache import load_traj_cache, save_traj_cache, make_params_dict  # type: ignore

            traj_cache_params = make_params_dict(
                fixed_dt_s=int(args.fixed_dt_s),
                min_traj_len=50,
                simp_alpha=1.0,
                dp_epsilon_scale=0.8,
                max_interp_points=args.max_interp_points,
                max_simplified_points=args.max_simplified_points,
            )
            if not bool(args.force_recompute_traj):
                traj_cache_payload = load_traj_cache(
                    csv_file_path=ais_file,
                    cache_dir=args.traj_cache_dir,
                    params=traj_cache_params,
                    verbose=True,
                )
        except Exception as _e:
            traj_cache_payload = None
            traj_cache_params = None
            print(f"⚠️ EN（EN）: {_e}")

        if traj_cache_payload is not None:
            processed_trajectories = traj_cache_payload.get("processed_trajectories", [])
            processed_ship_lengths_m = traj_cache_payload.get("processed_ship_lengths_m", [])
            processed_ship_ids = traj_cache_payload.get("processed_ship_ids", [])
            simplified_trajectories = traj_cache_payload.get("simplified_trajectories", [])
            skipped_length_mmsi = traj_cache_payload.get("skipped_length_mmsi", [])
            print(f"✅ EN+EN：processed={len(processed_trajectories)}, simplified={len(simplified_trajectories)}")

            raw_trajs = processed_trajectories
            simplified_trajs = simplified_trajectories

        else:
            for i, sid in enumerate(ship_ids):
                ship_data = trajs[sid]

                Lm = ship_data.get("length_m", np.nan)
                try:
                    Lm = float(Lm)
                except Exception:
                    Lm = float("nan")
                if (not np.isfinite(Lm)) or (Lm <= 0.0):
                    skipped_length_mmsi.append(sid)
                    print(
                        f"⚠️  EN {sid} ({i+1}/{len(ship_ids)}): Length EN/EN（{ship_data.get('length_m', None)!r}）"
                    )
                    continue

                print(f"EN {sid} ({i+1}/{len(ship_ids)})...")
                traj_interp = interpolate_traj(ship_data, fixed_dt_s=int(args.fixed_dt_s))
                traj_interp["sog"] = clip_sog_kn(traj_interp["sog"], 0.0, 45.0)

                try:
                    max_interp = args.max_interp_points
                    if max_interp is not None:
                        max_interp = int(max_interp)
                    n_interp = int(len(traj_interp.get("lat", [])))
                    if max_interp and n_interp > max_interp:
                        import numpy as _np

                        idx = _np.linspace(0, n_interp - 1, num=max_interp, dtype=int)
                        for k in ("time", "lat", "lon", "sog", "cog"):
                            v = traj_interp.get(k, None)
                            if v is None:
                                continue
                            arr = _np.asarray(v)
                            traj_interp[k] = arr[idx].tolist() if k == "time" else arr[idx]
                        print(f"  ✂️ EN: N_interp={n_interp} > {max_interp}，EN {max_interp}")
                except Exception as _e:
                    print(f"  ⚠️ max_interp_points EN（EN）：{_e}")

                traj = np.stack(
                    [
                        np.array(traj_interp["lat"]),
                        np.array(traj_interp["lon"]),
                        np.array(traj_interp["sog"]),
                        np.array(traj_interp["cog"]),
                    ],
                    axis=1,
                )
                processed_trajectories.append(traj)
                processed_ship_lengths_m.append(Lm)
                processed_ship_ids.append(sid)

        if skipped_length_mmsi:
            print(
                f"⚠️  EN {len(skipped_length_mmsi)} EN（Length EN/EN）。EN MMSI: {skipped_length_mmsi[:5]}"
            )
        if len(processed_trajectories) == 0:
            print("❌ EN Length EN/EN：EN CSV EN Length EN。")
            return
        print(
            f"✅ EN，EN {len(processed_trajectories)} EN（EN {len(skipped_length_mmsi)} EN）"
        )
        
        raw_trajs = processed_trajectories

        print("\n" + "=" * 40)
        print("EN3: EN")
        print("=" * 40)

        from trajectory_simplification import sacs_b_simplify

        simplified_trajectories = []
        ratios = []
        for i, tr in enumerate(processed_trajectories):
            lat, lon, sog, cog = tr[:, 0], tr[:, 1], tr[:, 2], tr[:, 3]
            ship_length_m = processed_ship_lengths_m[i]
            lat_s, lon_s, kept = sacs_b_simplify(lat, lon, sog, cog, ship_length_m=ship_length_m, alpha=1)

            try:
                max_simp = args.max_simplified_points
                if max_simp is not None:
                    max_simp = int(max_simp)
                if max_simp and len(kept) > max_simp:
                    import numpy as _np
                    pos = _np.linspace(0, len(kept) - 1, num=max_simp, dtype=int)
                    kept = [kept[int(p)] for p in pos]
                    if kept and kept[0] != 0:
                        kept[0] = 0
                    if kept and kept[-1] != (len(tr) - 1):
                        kept[-1] = (len(tr) - 1)
                    kept = sorted(set(int(x) for x in kept))
                    print(f"  ✂️ EN：kept={len(kept)}（EN <= {max_simp}）")
            except Exception as _e:
                print(f"  ⚠️ max_simplified_points EN（EN）：{_e}")

            simp = np.stack([lat[kept], lon[kept], sog[kept], cog[kept]], axis=1)
            simplified_trajectories.append(simp)
            ratio = (len(kept) / len(tr)) if len(tr) > 0 else 0.0
            ratios.append(ratio)
            sid = processed_ship_ids[i] if i < len(processed_ship_ids) else "unknown"
            print(f"  EN {i+1}/{len(processed_trajectories)} EN（MMSI={sid}），EN {len(kept)}，EN {1-ratio:.2%}")

        print(f"✅ EN，EN {len(simplified_trajectories)} EN")
        if ratios:
            print(f"\nEN：{np.mean([1 - r for r in ratios]):.2%}")
        
        simplified_trajs = simplified_trajectories

        try:
            if traj_cache_params is not None:
                save_traj_cache(
                    csv_file_path=ais_file,
                    cache_dir=args.traj_cache_dir,
                    params=traj_cache_params,
                    payload={
                        "processed_trajectories": processed_trajectories,
                        "processed_ship_ids": processed_ship_ids,
                        "processed_ship_lengths_m": processed_ship_lengths_m,
                        "simplified_trajectories": simplified_trajectories,
                        "skipped_length_mmsi": skipped_length_mmsi,
                    },
                    verbose=True,
                )
        except Exception as _e:
            print(f"⚠️ EN（EN）：{_e}")

        if getattr(args, "print_len_stats", False):
            dump_sequence_length_distribution(raw_trajs, tag="raw", out_dir=out_dir, bins=args.len_hist_bins)
            dump_sequence_length_distribution(simplified_trajs, tag="simplified", out_dir=out_dir, bins=args.len_hist_bins)

        print("\n" + "=" * 40)
        print("EN4: EN (EN)")
        print("=" * 40)

        all_list = [tr for tr in simplified_trajectories if tr is not None and len(tr) > 0]
        print(f"  EN: {len(all_list)} EN")
        if getattr(args, "print_len_stats", False) and args.mode != "dtw_only":
            dump_sequence_length_distribution(all_list, tag="simplified_all", out_dir=out_dir, bins=args.len_hist_bins)
        
        TSz_all = None
        D_all = None
        labels_all = np.array([], dtype=int)
        eps_all = np.nan
        minPts_all = np.nan
        
        if args.mode == "dtw_only":
            if all_list:
                print("\n[All] EN DTW EN（EN）...")
                TS_raw_all = [np.array(tr, dtype=float) for tr in all_list]
                TSz_all, mu_feat_all, sigma_feat_all = build_TS_features(
                    TS_raw_all,
                    feature_mode=str(args.feature_mode),
                    dt_s=float(args.fixed_dt_s),
                )
                
                cache_root = args.dtw_cache_dir if args.dtw_cache_dir is not None else "./dtw_cache"
                cache_dir = os.path.join(cache_root, "batch_dtw")
                os.makedirs(cache_dir, exist_ok=True)

                D_all = build_distance_matrix_with_cache(
                    TSz_all,
                    csv_file_path=ais_file,
                    cache_dir=cache_dir,
                    feature_mode=args.feature_mode,
                    dtw_method="fastdtw",
                    suffix=None,
                    verbose=True,
                    n_jobs=-1,
                    force_recompute=bool(args.force_recompute_dtw),
                )
                dataset_name = Path(ais_file).parent.name
                feat_dir = os.path.join(cache_dir, str(args.feature_mode))
                os.makedirs(feat_dir, exist_ok=True)
                out_path = os.path.join(feat_dir, f"{dataset_name}_dtw_matrix.npy")
                np.save(out_path, D_all)

                print(f"✅ EN AIS EN {ais_file} EN DTW EN。")
                print(f"   EN DTW EN: {os.path.abspath(out_path)}")
            else:
                print("⚠️ EN DTW EN")
            
            return
        
        if args.mode == "paper":
            delta_s = float(args.delta_s)
            if bool(args.poa_two_stage):
                print(
                    f"🔍 POA two-stage eps EN: coarse_steps={int(args.poa_coarse_steps)}, "
                    f"refine_steps={int(args.poa_refine_steps)}, window_mult={float(args.poa_refine_window_mult):.2f}"
                )
            else:
                print(f"🔍 POA EN: delta_s={delta_s:.4f}")

            if all_list:
                print("\n[All] EN (EN)...")
                TS_raw_all = [np.array(tr, dtype=float) for tr in all_list]
                TSz_all, mu_feat_all, sigma_feat_all = build_TS_features(
                    TS_raw_all,
                    feature_mode=str(args.feature_mode),
                    dt_s=float(args.fixed_dt_s),
                )
                if getattr(args, "print_len_stats", False):
                    dump_sequence_length_distribution(TS_raw_all, tag="TS_raw_all_paper", out_dir=out_dir, bins=args.len_hist_bins)
                    dump_sequence_length_distribution(TSz_all, tag="TSz_all_paper", out_dir=out_dir, bins=args.len_hist_bins)
                D_all = build_distance_matrix_with_cache(
                    TSz_all,
                    csv_file_path=ais_file,
                    cache_dir=args.dtw_cache_dir,
                    feature_mode=args.feature_mode,
                    dtw_method="fastdtw",
                    suffix="all",
                    verbose=True,
                    n_jobs=-1,
                    force_recompute=args.force_recompute_dtw
                )
                labels_all, eps_all, minPts_all, _, _, _ = run_amtbra_dbscan_paper(
                    TS_raw_all,
                    delta_s=delta_s,
                    verbose=True,
                    D=D_all,
                    feature_mode=str(args.feature_mode),
                    dt_s=float(args.fixed_dt_s),
                    minpts_stat=str(args.minpts_stat),
                    two_stage=bool(args.poa_two_stage),
                    coarse_steps=int(args.poa_coarse_steps),
                    refine_steps=int(args.poa_refine_steps),
                    refine_window_mult=float(args.poa_refine_window_mult),
                )
            else:
                print("⚠️ EN")
        
        elif args.mode == "drl":
            print("\n🤖 EN DRL EN")
            from drl_dbscan import train_drl_dbscan, constrain_distance_matrix
            from sklearn.cluster import DBSCAN as _SK_DBSCAN
            from sklearn.metrics import silhouette_score as _sk_silhouette_score

            def _checkB_print_reward_components(D: np.ndarray, eps: float, minPts: int, tag: str):
                """
                ✅ EN B：EN reward EN（sil/noise/core/K）
                EN：EN DRL EN（EN D EN constrain）。
                """
                try:
                    Dc = constrain_distance_matrix(D)
                    N = int(Dc.shape[0]) if isinstance(Dc, np.ndarray) and Dc.ndim == 2 else 0
                    db = _SK_DBSCAN(eps=float(eps), min_samples=int(minPts), metric="precomputed", n_jobs=-1)
                    labels = db.fit_predict(Dc)
                    noise_ratio = float(np.mean(labels == -1)) if N > 0 else 1.0
                    core_ratio = float(len(getattr(db, "core_sample_indices_", [])) / max(N, 1))
                    unique = set(int(x) for x in labels.tolist()) if N > 0 else set()
                    K = int(len(unique) - (1 if -1 in unique else 0))
                    sil = 0.0
                    if K >= 2 and N > K:
                        mask = labels != -1
                        if int(np.sum(mask)) >= 3 and len(set(labels[mask].tolist())) >= 2:
                            sil = float(_sk_silhouette_score(Dc[np.ix_(mask, mask)], labels[mask], metric="precomputed"))
                    print(
                        f"[Check B][{tag}] "
                        f"silhouette={sil:.4f}, noise_ratio={noise_ratio:.2%}, core_ratio={core_ratio:.2%}, K={K} "
                        f"(eps={float(eps):.4f}, minPts={int(minPts)})"
                    )
                except Exception as e:
                    print(f"⚠️ [Check B][{tag}] EN: {e}")

            def _checkC_eps_sweep(D: np.ndarray, tag: str):
                """
                ✅ EN C：EN minPts，EN eps，EN K-ε EN
                eps EN [0.05, 0.95]（EN DRL constrain EN D∈[0,1]）。
                """
                try:
                    Dc = constrain_distance_matrix(D)
                    N = int(Dc.shape[0]) if isinstance(Dc, np.ndarray) and Dc.ndim == 2 else 0
                    if N <= 1:
                        print(f"⚠️ [Check C][{tag}] EN，EN eps sweep")
                        return

                    eps_start = float(args.eps_sweep_start)
                    eps_end = float(args.eps_sweep_end)
                    eps_step = float(args.eps_sweep_step)
                    minPts_fix = int(args.eps_sweep_minpts)
                    if eps_step <= 0:
                        print(f"⚠️ [Check C][{tag}] eps_step<=0，EN")
                        return

                    eps_grid = np.arange(eps_start, eps_end + 1e-12, eps_step, dtype=float)
                    rows = []
                    for e in eps_grid:
                        db = _SK_DBSCAN(eps=float(e), min_samples=minPts_fix, metric="precomputed", n_jobs=-1)
                        labels = db.fit_predict(Dc)
                        noise_ratio = float(np.mean(labels == -1))
                        core_ratio = float(len(getattr(db, "core_sample_indices_", [])) / max(N, 1))
                        unique = set(int(x) for x in labels.tolist())
                        K = int(len(unique) - (1 if -1 in unique else 0))
                        rows.append(
                            {
                                "eps": float(e),
                                "minPts": int(minPts_fix),
                                "K": int(K),
                                "noise_ratio": float(noise_ratio),
                                "core_ratio": float(core_ratio),
                            }
                        )

                    df = pd.DataFrame(rows)
                    csv_path = os.path.join(out_dir, f"eps_sweep_{tag}.csv")
                    df.to_csv(csv_path, index=False, encoding="utf-8")
                    print(f"✅ [Check C][{tag}] EN: {csv_path}")

                    multi = df[df["K"] >= 2]
                    if len(multi) == 0:
                        print(f"📉 [Check C][{tag}] EN eps∈[{eps_start:.2f},{eps_end:.2f}] EN K<=1")
                    else:
                        print(
                            f"📈 [Check C][{tag}] EN："
                            f"eps≈[{multi['eps'].min():.2f},{multi['eps'].max():.2f}]（EN minPts={minPts_fix}）"
                        )

                    try:
                        import matplotlib.pyplot as plt

                        plt.figure(figsize=(8, 4))
                        plt.plot(df["eps"].values, df["K"].values, linewidth=2.0)
                        plt.xlabel("eps (constrained D in [0,1])")
                        plt.ylabel("K (num_clusters)")
                        plt.title(f"DBSCAN K vs eps (minPts={minPts_fix}) [{tag}]")
                        plt.grid(True, alpha=0.3)
                        plt.tight_layout()
                        png_path = os.path.join(out_dir, f"eps_sweep_{tag}.png")
                        plt.savefig(png_path, dpi=200)
                        plt.close()
                        print(f"✅ [Check C][{tag}] EN: {png_path}")
                    except Exception as e:
                        print(f"⚠️ [Check C][{tag}] EN（ENCSV）: {e}")
                except Exception as e:
                    print(f"⚠️ [Check C][{tag}] eps sweep EN: {e}")
            
            labels_all = np.array([], dtype=int)
            eps_all = np.nan
            minPts_all = np.nan
            D_all = None
            if all_list:
                print("\n[All] DRL EN...")
                TS_raw_all = [np.array(tr, dtype=float) for tr in all_list]
                TSz_all, mu_feat_all, sigma_feat_all = build_TS_features(
                    TS_raw_all,
                    feature_mode=str(args.feature_mode),
                    dt_s=float(args.fixed_dt_s),
                )
                if getattr(args, "print_len_stats", False):
                    dump_sequence_length_distribution(TS_raw_all, tag="TS_raw_all_drl", out_dir=out_dir, bins=args.len_hist_bins)
                    dump_sequence_length_distribution(TSz_all, tag="TSz_all_drl", out_dir=out_dir, bins=args.len_hist_bins)
                D_all = build_distance_matrix_with_cache(
                    TSz_all,
                    csv_file_path=ais_file,
                    cache_dir=args.dtw_cache_dir,
                    feature_mode=args.feature_mode,
                    dtw_method="fastdtw",
                    suffix="all",
                    verbose=True,
                    n_jobs=-1,
                    force_recompute=args.force_recompute_dtw
                )
                D_all = constrain_distance_matrix(D_all)
                if args.eps_sweep:
                    _checkC_eps_sweep(D_all, tag="all")

                if args.train_drl:
                    print(f"\n🔧 EN DRL EN（All），EN={args.drl_episodes} ...")
                    log_filename = "drl_train_all.csv"
                    agent_out = train_drl_dbscan(
                        D_all,
                        num_episodes=args.drl_episodes,
                        model_save_path=args.drl_model,
                        model_load_path=args.drl_model if os.path.exists(args.drl_model) else None,
                        log_path=os.path.join(out_dir, log_filename),
                        verbose=True,
                    )
                    eps_all, minPts_all = drl_select_eps_minpts(D_all, agent=agent_out, verbose=True)
                else:
                    eps_all, minPts_all = drl_select_eps_minpts(D_all, model_path=args.drl_model, verbose=True)

                print(f"  ✅ DRL EN: eps={eps_all:.4f}, minPts={minPts_all}")
                _checkB_print_reward_components(D_all, eps_all, minPts_all, tag="all")
                
                labels_all = dbscan_from_distance(D_all, eps_all, minPts_all)
                K_all = int(labels_all.max())
                noise_ratio_all = float(np.mean(labels_all == 0))
                print(f"  ✅ [DRL] K={K_all}, EN={noise_ratio_all:.2%}, eps={eps_all:.4f}, minPts={minPts_all}")
            else:
                print("⚠️ EN")

        if args.eps is not None:
            eps_override = float(args.eps)
            minpts_override = int(args.minpts) if args.minpts is not None else 8
            print(f"\n⚙️ EN: eps={eps_override}, minPts={minpts_override}")
            if D_all is not None and labels_all.size:
                labels_all = dbscan_from_distance(D_all, eps_override, minpts_override)
                eps_all, minPts_all = eps_override, minpts_override

        if labels_all.size:
            K_all = int(labels_all.max())
            noise_ratio_all = float(np.mean(labels_all == 0))
            mode_label = "Paper" if args.mode == "paper" else "DRL"
            print(
                f"\n✅ [All] EN ({mode_label}): K={K_all}, EN={noise_ratio_all:.2%}, eps={eps_all:.4f}, minPts={minPts_all}"
            )

        if D_all is not None and TSz_all is not None and len(TSz_all) > 0:
            try:
                print("\n" + "=" * 40)
                print("EN: EN (K-Means / Agglomerative / Spectral)")
                print("=" * 40)

                labels_km, k_km, info_km = traditional_clustering(
                    D_all,
                    method="kmeans",
                    n_clusters=None,
                    verbose=True,
                )
                print(f"[KMeans] EN: k={k_km}, silhouette={info_km.get('silhouette_score', 0.0):.4f}")
                labels_km_paper = labels_km + 1
                dbi_km, cp_km, sp_km = evaluate_clusters_paper(
                    TSz_all,
                    labels_km_paper,
                    dtw_dist=lambda a, b: compute_dtw_distance(a, b),
                    pad_mode="paper",
                )
                print(f"[KMeans]  DBI′: {dbi_km:.4f}")
                print(f"[KMeans]  CP′ : {cp_km:.4f}")
                print(f"[KMeans]  SP′ : {sp_km:.4f}")

                labels_agg, k_agg, info_agg = traditional_clustering(
                    D_all,
                    method="agglomerative",
                    n_clusters=None,
                    max_clusters=10,
                    verbose=True,
                )
                print(f"[Agglomerative] EN: k={k_agg}, silhouette={info_agg.get('silhouette_score', 0.0):.4f}")
                labels_agg_paper = labels_agg + 1
                dbi_agg, cp_agg, sp_agg = evaluate_clusters_paper(
                    TSz_all,
                    labels_agg_paper,
                    dtw_dist=lambda a, b: compute_dtw_distance(a, b),
                    pad_mode="paper",
                )
                print(f"[Agglomerative]  DBI′: {dbi_agg:.4f}")
                print(f"[Agglomerative]  CP′ : {cp_agg:.4f}")
                print(f"[Agglomerative]  SP′ : {sp_agg:.4f}")

                labels_spec, k_spec, info_spec = traditional_clustering(
                    D_all,
                    method="spectral",
                    n_clusters=None,
                    max_clusters=10,
                    verbose=True,
                )
                print(f"[Spectral] EN: k={k_spec}, silhouette={info_spec.get('silhouette_score', 0.0):.4f}")
                labels_spec_paper = labels_spec + 1
                dbi_spec, cp_spec, sp_spec = evaluate_clusters_paper(
                    TSz_all,
                    labels_spec_paper,
                    dtw_dist=lambda a, b: compute_dtw_distance(a, b),
                    pad_mode="paper",
                )
                print(f"[Spectral]  DBI′: {dbi_spec:.4f}")
                print(f"[Spectral]  CP′ : {cp_spec:.4f}")
                print(f"[Spectral]  SP′ : {sp_spec:.4f}")

            except Exception as e:
                print(f"⚠️ EN: {e}")

        print("\n" + "=" * 40)
        print("EN5: EN")
        print("=" * 40)

        try:
            print("Paper-strict evaluation: Eq.(15)-(18) + Eq.(21)-(22), centre trajectory + DTW.")
            print("DTW window: disabled (Sakoe–Chiba window code commented out)")
            if args.mode == "drl":
                print(
                    "ℹ️ DRL EN：EN DRL EN(constrain_distance_matrix)EN；"
                    "EN DBI′/CP′/SP′ EN，EN DTW EN。"
                )

            if TSz_all is not None and labels_all.size:
                dbi_o, cp_o, sp_o = evaluate_clusters_paper(
                    TSz_all,
                    labels_all,
                    dtw_dist=lambda a, b: compute_dtw_distance(a, b),
                    pad_mode="paper",
                )
                print(f"[All]  DBI′: {dbi_o:.4f}")
                print(f"[All]  CP′ : {cp_o:.4f}")
                print(f"[All]  SP′ : {sp_o:.4f}")
            else:
                print("⚠️ [All] EN")
        except Exception as e:
            print(f"⚠️ EN: {e}")

        print("\n" + "=" * 40)
        print("EN6: EN")
        print("=" * 40)

        os.makedirs(out_dir, exist_ok=True)
        from visualization import (
            plot_clusters_by_label,
            plot_distance_matrix_heatmap,
            summarize_and_save_metrics,
            plot_trajectories,
            plot_density_power,
        )

        try:
            if D_all is not None and labels_all.size:
                plot_distance_matrix_heatmap(D_all, os.path.join(out_dir, "dtw_matrix.png"), use_english=args.english)

                if args.mode == "paper":
                    eps_list, avg_list, pow_list, curve_name = _get_density_power_curve_for_plot(
                        D_all,
                        delta_s=delta_s,
                        two_stage=bool(args.poa_two_stage),
                        coarse_steps=int(args.poa_coarse_steps),
                        refine_steps=int(args.poa_refine_steps),
                        refine_window_mult=float(args.poa_refine_window_mult),
                    )
                    if eps_list:
                        print(f"📈 density_power.png EN: {curve_name}")
                        plot_density_power(
                            eps_list,
                            avg_list,
                            pow_list,
                            os.path.join(out_dir, "density_power.png"),
                            mark_eps=eps_all,
                            mark_minpts=minPts_all,
                            use_english=args.english,
                        )
                    else:
                        print("⚠️ EN（strict/engineering EN），EN density_power.png EN")

                plot_clusters_by_label(
                    all_list,
                    labels_all,
                    os.path.join(out_dir, "clusters.png"),
                    use_english=args.english,
                )
                metrics = summarize_and_save_metrics(labels_all)
            else:
                metrics = {}

            plot_trajectories(simplified_trajectories, os.path.join(out_dir, "trajectories_processed.png"))
        except Exception as e:
            print(f"⚠️ EN: {e}")

        print("\n" + "=" * 40)
        print("EN7: EN")
        print("=" * 40)
        
        try:
            export_raw_trajs(raw_trajs, k=len(raw_trajs), out_path=os.path.join(out_dir, "debug_raw_trajs.csv"))
            export_simplified_trajs(simplified_trajectories, k=len(simplified_trajectories), out_path=os.path.join(out_dir, "debug_simplified_trajs.csv"))
        except Exception as e:
            print(f"⚠️ EN: {e}")

        print("\n🎉 AMTBRA EN！EN：", out_dir)
    finally:
        tee.close()
        sys.stdout = tee.stdout
        sys.stderr = sys.__stderr__

if __name__ == "__main__":
    main()