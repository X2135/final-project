#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DRL training starting from teacher-pretrained Actor.
====================================================

EN
----
1. EN `models/pretrained_actor_twostage_v2.pth` (two-stage+mean teacher EN Actor)
2. EN DTW EN PPO EN DRL EN
3. EN DRL EN (Actor+Critic)，EN two-stage / paper EN

EN
----
- EN `drl_dbscan.py`，EN (PPOAgent / reward EN) EN。
- EN，EN two-stage EN run EN DTW .npy EN，EN。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from drl_dbscan import (
    PPOAgent,
    constrain_distance_matrix,
    extract_state_features,
)
from pretrain_drl_teacher import (
    _abs_from_repo,
    _parse_dtw_path_from_summary,
    _resolve_run_dir,
)


def _load_dtw_from_run_dirs(run_dirs: List[str]) -> List[np.ndarray]:
    """
    EN two-stage EN DTW EN (.npy)。

    EN
    ----
    run_dirs : List[str]
        EN:
        - "path/to/two_stage_output_1"
        - "path/to/two_stage_output_2"
        - "path/to/two_stage_output_3"

    EN
    ----
    Ds : List[np.ndarray]
        EN (N, N) EN DTW EN
    """
    Ds: List[np.ndarray] = []
    for rd in run_dirs:
        run_path = _resolve_run_dir(rd)
        summary_path = run_path / "summary.txt"
        dtw_path_str = _parse_dtw_path_from_summary(summary_path)
        if not dtw_path_str:
            print(f"⚠️  EN {summary_path} EN DTW .npy EN，EN run_dir")
            continue
        dtw_path = Path(_abs_from_repo(dtw_path_str))
        if not dtw_path.exists():
            print(f"⚠️  DTW EN: {dtw_path}，EN run_dir")
            continue
        D = np.load(dtw_path)
        if D.ndim != 2 or D.shape[0] != D.shape[1]:
            print(f"⚠️  DTW EN(EN)，EN {dtw_path}，shape={D.shape}")
            continue
        Ds.append(np.asarray(D, dtype=float))
        print(f"✅ Loaded DTW from run_dir={run_path.name}, path={dtw_path}, shape={D.shape}")
    if not Ds:
        raise ValueError("EN DTW EN，EN run_dirs EN。")
    return Ds


def train_drl_multi_from_teacher(
    run_dirs: List[str],
    pretrained_actor_path: str = "models/pretrained_actor_twostage_v2.pth",
    num_episodes: int = 500,
    max_steps_per_episode: int = 1,
    model_save_path: str = "models/drl_multi_from_teacher.pth",
    log_path: Optional[str] = "logs/drl_train_from_teacher.csv",
    seed: int = 0,
    verbose: bool = True,
    dtw_paths: Optional[List[str]] = None,
    update_every: int = 64,
) -> PPOAgent:
    """
    EN DTW EN DRL EN，EN Actor EN two-stage+mean Teacher EN。

    EN
    ----
    run_dirs : List[str]
        two-stage EN (EN summary.txt EN DTW .npy)
        EN:
        - "path/to/two_stage_output_1"
        - "path/to/two_stage_output_2"
        - "path/to/two_stage_output_3"
    pretrained_actor_path : str
        EN Actor EN (EN pretrain_drl_teacher.py)
    num_episodes : int
        EN
    max_steps_per_episode : int
        EN episode EN step EN (EN bandit，EN 1 EN)
    model_save_path : str
        EN DRL EN
    log_path : str | None
        EN None，EN episode EN CSV
    seed : int
        EN
    verbose : bool
        EN
    """
    if dtw_paths:
        Ds: List[np.ndarray] = []
        for p in dtw_paths:
            p_abs = Path(_abs_from_repo(p))
            if not p_abs.exists():
                print(f"⚠️  DTW EN: {p_abs}，EN")
                continue
            D = np.load(p_abs)
            if D.ndim != 2 or D.shape[0] != D.shape[1]:
                print(f"⚠️  DTW EN(EN)，EN {p_abs}，shape={D.shape}")
                continue
            Ds.append(np.asarray(D, dtype=float))
            print(f"✅ Loaded DTW from file={p_abs}, shape={D.shape}")
        if not Ds:
            raise ValueError("dtw_paths EN DTW EN。")
    else:
        Ds = _load_dtw_from_run_dirs(run_dirs)
    rng = np.random.default_rng(int(seed))

    agent = PPOAgent()
    ckpt = torch.load(pretrained_actor_path, map_location=agent.device)
    if "actor_state_dict" not in ckpt:
        raise KeyError(f"EN {pretrained_actor_path} EN 'actor_state_dict'")
    agent.actor.load_state_dict(ckpt["actor_state_dict"])
    if verbose:
        print(f"✅ Loaded pretrained Actor from: {pretrained_actor_path}")

    episode_rewards: List[float] = []
    episode_logs: List[dict] = []

    update_every_i = int(update_every)
    if update_every_i <= 0:
        update_every_i = 1

    for episode in range(int(num_episodes)):
        k = int(rng.integers(0, len(Ds)))
        D_raw = Ds[k]

        D = constrain_distance_matrix(D_raw)

        state = extract_state_features(D)

        valid = D[(D > 0) & np.isfinite(D) & (D < 1e6)]
        N = int(D.shape[0]) if isinstance(D, np.ndarray) and D.ndim == 2 else 0
        eps_min = float(np.quantile(valid, 0.01)) if valid.size else 0.0
        eps_max = float(np.quantile(valid, 0.95)) if valid.size else 1.0
        minpts_min = 1
        minpts_max = max(4, min(50, max(1, N // 2)))

        episode_reward = 0.0
        last_eps = float("nan")
        last_eps_pos = float("nan")
        last_minpts = -1
        last_metrics: dict = {}

        for _ in range(int(max_steps_per_episode)):
            eps, minPts = agent.select_action(
                state,
                eps_min,
                eps_max,
                minpts_min=minpts_min,
                minpts_max=minpts_max,
                deterministic=False,
            )
            last_eps = float(eps)
            denom = float(eps_max - eps_min)
            if np.isfinite(denom) and denom > 1e-12:
                last_eps_pos = float((last_eps - float(eps_min)) / denom)
            else:
                last_eps_pos = 0.0
            last_eps_pos = float(np.clip(last_eps_pos, 0.0, 1.0))
            last_minpts = int(minPts)

            reward = agent.compute_reward(D, eps, minPts)
            episode_reward += float(reward)
            agent.rewards.append(float(reward))
            agent.is_terminals.append(True)

            if hasattr(agent, "last_metrics") and isinstance(agent.last_metrics, dict):
                last_metrics = {
                    "silhouette": float(agent.last_metrics.get("silhouette", 0.0)),
                    "noise_ratio": float(agent.last_metrics.get("noise_ratio", 0.0)),
                    "num_clusters": int(agent.last_metrics.get("num_clusters", 0)),
                    "core_ratio": float(agent.last_metrics.get("core_ratio", 0.0)),
                }

        episode_rewards.append(float(episode_reward))

        if verbose:
            print(
                "[Train-Multi-Teacher] "
                f"ep={episode+1}/{num_episodes} ds={k}/{len(Ds)} "
                f"eps={last_eps:.4f} pos={last_eps_pos:.2%} "
                f"(eps_min={eps_min:.4f}, eps_max={eps_max:.4f}) "
                f"minPts={last_minpts} reward={episode_reward:.4f} "
                f"sil={float(last_metrics.get('silhouette', 0.0)):.4f} "
                f"noise={float(last_metrics.get('noise_ratio', 0.0)):.2%} "
                f"core={float(last_metrics.get('core_ratio', 0.0)):.2%} "
                f"K={int(last_metrics.get('num_clusters', 0))}"
            )

        episode_logs.append(
            {
                "episode": int(episode + 1),
                "dataset_idx": int(k),
                "eps": float(last_eps),
                "eps_min": float(eps_min),
                "eps_max": float(eps_max),
                "eps_pos": float(last_eps_pos),
                "minPts": int(last_minpts),
                "reward": float(episode_reward),
                "silhouette": float(last_metrics.get("silhouette", 0.0)),
                "noise_ratio": float(last_metrics.get("noise_ratio", 0.0)),
                "core_ratio": float(last_metrics.get("core_ratio", 0.0)),
                "num_clusters": int(last_metrics.get("num_clusters", 0)),
            }
        )

        if ((episode + 1) % update_every_i) == 0:
            agent.update()

        if verbose and (episode + 1) % 25 == 0:
            avg_reward = float(np.mean(episode_rewards[-25:]))
            print(
                f"[Train-Multi-Teacher] Episode {episode+1}/{num_episodes}, "
                f"AvgReward(last25)={avg_reward:.4f}"
            )

    try:
        if len(getattr(agent, "states", [])) > 0:
            agent.update()
    except Exception:
        pass

    if model_save_path:
        Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)
        agent.save(model_save_path)
        if verbose:
            print(f"✅ DRL EN: {model_save_path}")

    if log_path:
        import csv

        log_path_p = Path(log_path)
        log_path_p.parent.mkdir(parents=True, exist_ok=True)
        with log_path_p.open("w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "episode",
                "dataset_idx",
                "eps",
                "eps_min",
                "eps_max",
                "eps_pos",
                "minPts",
                "reward",
                "silhouette",
                "noise_ratio",
                "core_ratio",
                "num_clusters",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in episode_logs:
                writer.writerow(row)
        if verbose:
            print(f"✅ DRL EN: {log_path_p}")

    return agent


def main() -> int:
    parser = argparse.ArgumentParser(description="Train DRL from teacher-pretrained Actor on user-provided DTW matrices.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--dtw-npy",
        nargs="+",
        help="One or more DTW .npy matrices used for training.",
    )
    src.add_argument(
        "--run-dirs",
        nargs="+",
        help="One or more pipeline run dirs; DTW paths will be parsed from summary.txt.",
    )
    parser.add_argument("--pretrained-actor", type=str, default="models/pretrained_actor_twostage_v3.pth")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--save", type=str, default="models/drl_multi_from_teacher.pth")
    parser.add_argument("--log", type=str, default="logs/drl_train_from_teacher.csv")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--update-every", type=int, default=64)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    train_drl_multi_from_teacher(
        run_dirs=list(args.run_dirs or []),
        pretrained_actor_path=str(args.pretrained_actor),
        num_episodes=int(args.episodes),
        max_steps_per_episode=1,
        model_save_path=str(args.save),
        log_path=str(args.log),
        seed=int(args.seed),
        verbose=not bool(args.quiet),
        dtw_paths=list(args.dtw_npy or []) if args.dtw_npy else None,
        update_every=int(args.update_every),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

