#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from __future__ import annotations

import argparse
import os
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from drl_dbscan import (
    constrain_distance_matrix,
    extract_state_features,
    ActorNetwork,
)
from Adaptive_DBSCAN import select_eps_minpts_from_D_paper


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _abs_from_repo(p: str) -> str:
    x = (p or "").strip()
    if not x:
        return x
    if os.path.isabs(x):
        return x
    return str((_repo_root() / x).resolve())


def _parse_dtw_path_from_summary(summary_path: Path) -> Optional[str]:
    text = summary_path.read_text(encoding="utf-8", errors="ignore")
    patterns = [
        r"ENDTWEN:\s*([^\s]+\.npy)\s*$",
        r"DTWEN:\s*([^\s]+\.npy)\s*$",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.MULTILINE)
        if m:
            return _abs_from_repo(m.group(1).strip())
    return None


def _resolve_run_dir(p: str) -> Path:
    """Resolve run directory, auto-finding latest run_* if parent dir provided."""
    path = Path(_abs_from_repo(p))
    if not path.exists():
        raise FileNotFoundError(f"run_dir does not exist: {path}")
    if path.is_file():
        if path.name == "summary.txt":
            return path.parent
        raise FileNotFoundError(f"Expected a directory or summary.txt, got file: {path}")

    summ = path / "summary.txt"
    if summ.exists():
        return path

    candidates = [pp for pp in path.glob("run_*/summary.txt") if pp.is_file()]
    if not candidates:
        raise FileNotFoundError(f"summary.txt not found in run_dir or any run_*/ under: {path}")
    candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return candidates[0].parent


def _load_dtw_matrices(
    dtw_paths: Optional[List[str]] = None,
    run_dirs: Optional[List[str]] = None,
) -> List[Tuple[np.ndarray, str]]:
    """
    Load DTW matrices from provided paths or run directories.
    
    Returns
    -------
    List of (D, label) tuples where D is the DTW matrix and label is a descriptive name.
    """
    matrices = []
    
    if dtw_paths:
        for p in dtw_paths:
            path = Path(_abs_from_repo(p))
            if not path.exists():
                print(f"⚠️  Skipping non-existent DTW file: {path}")
                continue
            try:
                D = np.load(path)
                label = path.stem
                matrices.append((D, label))
                print(f"✅ Loaded DTW: {label} (shape={D.shape})")
            except Exception as e:
                print(f"⚠️  Failed to load {path}: {e}")
                continue
    
    if run_dirs:
        for rd in run_dirs:
            try:
                run_path = _resolve_run_dir(rd)
                summary_path = run_path / "summary.txt"
                dtw_path_str = _parse_dtw_path_from_summary(summary_path)
                if not dtw_path_str:
                    print(f"⚠️  Could not find DTW path in {summary_path}")
                    continue
                dtw_path = Path(dtw_path_str)
                if not dtw_path.exists():
                    print(f"⚠️  DTW file not found: {dtw_path}")
                    continue
                D = np.load(dtw_path)
                label = f"{run_path.name}_{dtw_path.stem}"
                matrices.append((D, label))
                print(f"✅ Loaded DTW from run: {label} (shape={D.shape})")
            except Exception as e:
                print(f"⚠️  Failed to process run_dir {rd}: {e}")
                continue
    
    if not matrices:
        raise ValueError("No valid DTW matrices loaded. Please check your input paths.")
    
    return matrices


def _get_action_ranges(D: np.ndarray) -> Tuple[float, float, int, int]:
    """
    Compute action ranges (eps_min, eps_max, minpts_min, minpts_max) for a DTW matrix.
    This matches the logic used in DRL training.
    """
    D_constrained = constrain_distance_matrix(D)
    valid = D_constrained[(D_constrained > 0) & np.isfinite(D_constrained) & (D_constrained < 1e6)]
    N = int(D_constrained.shape[0]) if isinstance(D_constrained, np.ndarray) and D_constrained.ndim == 2 else 0
    
    if valid.size == 0:
        eps_min, eps_max = 0.0, 1.0
    else:
        eps_min = float(np.quantile(valid, 0.01))
        eps_max = float(np.quantile(valid, 0.95))
    
    minpts_min = 1
    minpts_max = max(4, min(50, max(1, N // 2)))
    
    return eps_min, eps_max, minpts_min, minpts_max


def _teacher_to_latent_action(
    eps_teacher: float,
    minPts_teacher: int,
    eps_min: float,
    eps_max: float,
    minpts_min: int,
    minpts_max: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert teacher (eps, minPts) to latent space actions.
    
    Steps:
    1. Normalize to [0,1]: eps_norm = (eps - eps_min) / (eps_max - eps_min)
    2. Inverse sigmoid: z = logit(eps_norm) = log(eps_norm / (1 - eps_norm))
    
    Returns
    -------
    z_eps, z_minpts: latent space actions (torch.Tensor)
    """
    eps_range = eps_max - eps_min
    if eps_range <= 1e-12:
        eps_norm = 0.5
    else:
        eps_norm = (eps_teacher - eps_min) / eps_range
    eps_norm = np.clip(eps_norm, 0.01, 0.99)
    
    minpts_range = minpts_max - minpts_min
    if minpts_range <= 0:
        minpts_norm = 0.5
    else:
        minpts_norm = (minPts_teacher - minpts_min) / minpts_range
    minpts_norm = np.clip(minpts_norm, 0.01, 0.99)
    
    z_eps = np.log(eps_norm / (1.0 - eps_norm))
    z_minpts = np.log(minpts_norm / (1.0 - minpts_norm))
    
    return torch.tensor(z_eps, dtype=torch.float32), torch.tensor(z_minpts, dtype=torch.float32)


def pretrain_actor_with_teacher(
    dtw_matrices: List[Tuple[np.ndarray, str]],
    num_epochs: int = 100,
    learning_rate: float = 1e-3,
    batch_size: Optional[int] = None,
    device: str = "cpu",
    verbose: bool = True,
) -> ActorNetwork:
    """
    Pretrain Actor network using teacher demonstrations.
    
    Parameters
    ----------
    dtw_matrices : List[Tuple[np.ndarray, str]]
        List of (DTW_matrix, label) tuples
    num_epochs : int
        Number of training epochs
    learning_rate : float
        Learning rate for optimizer
    batch_size : int, optional
        Batch size (if None, use all data as one batch)
    device : str
        Device to use ('cpu' or 'cuda')
    verbose : bool
        Whether to print progress
    
    Returns
    -------
    actor : ActorNetwork
        Pretrained Actor network
    """
    device_torch = torch.device(device)
    
    actor = ActorNetwork(state_dim=9, hidden_dim=64).to(device_torch)
    optimizer = optim.Adam(actor.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    states_list = []
    z_eps_targets = []
    z_minpts_targets = []
    labels_list = []
    
    if verbose:
        print("\n" + "=" * 60)
        print("Collecting teacher demonstrations...")
        print("=" * 60)
    
    for D_raw, label in dtw_matrices:
        D = constrain_distance_matrix(D_raw)
        
        try:
            eps_teacher, minPts_teacher = select_eps_minpts_from_D_paper(
                D,
                delta_s=0.07,
                verbose=False,
                minpts_stat="mean",
                two_stage=True,
                coarse_steps=150,
                refine_steps=300,
                refine_window_mult=5.0,
            )
        except Exception as e:
            print(f"⚠️  Failed to get teacher solution for {label}: {e}")
            continue
        
        state = extract_state_features(D)
        
        eps_min, eps_max, minpts_min, minpts_max = _get_action_ranges(D)
        
        z_eps_target, z_minpts_target = _teacher_to_latent_action(
            eps_teacher, minPts_teacher,
            eps_min, eps_max, minpts_min, minpts_max,
        )
        
        states_list.append(state)
        z_eps_targets.append(z_eps_target.item())
        z_minpts_targets.append(z_minpts_target.item())
        labels_list.append(label)
        
        if verbose:
            eps_pos = (eps_teacher - eps_min) / (eps_max - eps_min) if (eps_max - eps_min) > 1e-12 else 0.0
            print(
                f"  {label}: eps={eps_teacher:.4f} (pos={eps_pos:.2%}), "
                f"minPts={minPts_teacher}, "
                f"z_eps={z_eps_target.item():.4f}, z_minpts={z_minpts_target.item():.4f}"
            )
    
    if not states_list:
        raise ValueError("No valid teacher demonstrations collected!")
    
    states_tensor = torch.FloatTensor(np.array(states_list)).to(device_torch)
    z_eps_targets_tensor = torch.FloatTensor(z_eps_targets).unsqueeze(-1).to(device_torch)
    z_minpts_targets_tensor = torch.FloatTensor(z_minpts_targets).unsqueeze(-1).to(device_torch)
    
    n_samples = len(states_list)
    if verbose:
        print(f"\n✅ Collected {n_samples} teacher demonstrations")
        print(f"   State shape: {states_tensor.shape}")
        print(f"   Training for {num_epochs} epochs...")
        print("\n" + "=" * 60)
        print("Training progress:")
        print("=" * 60)
    
    if batch_size is None:
        batch_size = n_samples
    
    for epoch in range(num_epochs):
        indices = torch.randperm(n_samples)
        states_shuffled = states_tensor[indices]
        z_eps_shuffled = z_eps_targets_tensor[indices]
        z_minpts_shuffled = z_minpts_targets_tensor[indices]
        
        epoch_loss = 0.0
        n_batches = 0
        
        for i in range(0, n_samples, batch_size):
            batch_states = states_shuffled[i:i+batch_size]
            batch_z_eps = z_eps_shuffled[i:i+batch_size]
            batch_z_minpts = z_minpts_shuffled[i:i+batch_size]
            
            eps_mean, _, minpts_mean, _ = actor(batch_states)
            
            loss_eps = criterion(eps_mean, batch_z_eps)
            loss_minpts = criterion(minpts_mean, batch_z_minpts)
            loss = loss_eps + loss_minpts
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_loss = epoch_loss / n_batches if n_batches > 0 else 0.0
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:4d}/{num_epochs}: loss = {avg_loss:.6f}")
    
    if verbose:
        print("=" * 60)
        print("✅ Pretraining complete!")
        print("=" * 60)
    
    return actor


def main():
    parser = argparse.ArgumentParser(
        description="Pretrain DRL Actor network using teacher demonstrations (two-stage+mean POA)"
    )
    parser.add_argument(
        "--dtw-npy",
        type=str,
        nargs="+",
        help="Paths to DTW .npy matrices (one or more)",
    )
    parser.add_argument(
        "--run-dirs",
        type=str,
        nargs="+",
        help="Paths to pipeline run directories (will extract DTW from summary.txt)",
    )
    parser.add_argument(
        "--save",
        type=str,
        required=True,
        help="Path to save pretrained Actor model (.pth)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (default: None = use all data as one batch)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use (default: cpu)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print progress (default: True)",
    )
    
    args = parser.parse_args()
    
    if not args.dtw_npy and not args.run_dirs:
        parser.error("Must provide either --dtw-npy or --run-dirs")
    
    dtw_matrices = _load_dtw_matrices(
        dtw_paths=args.dtw_npy,
        run_dirs=args.run_dirs,
    )
    
    actor = pretrain_actor_with_teacher(
        dtw_matrices,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        device=args.device,
        verbose=args.verbose,
    )
    
    save_path = Path(_abs_from_repo(args.save))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'actor_state_dict': actor.state_dict(),
    }, save_path)
    
    print(f"\n✅ Pretrained Actor saved to: {save_path}")
    print(f"   You can now use this model as --model-load-path in train_drl_multi.py")


if __name__ == "__main__":
    main()
