#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trajectory cache for AMTBRA pipeline.

This cache stores:
- preprocessed trajectories (after interpolation / basic cleaning)
- simplified trajectories (after SACSB / DP merge)

It is designed to avoid recomputation across repeated runs, similar to DTW cache.

Disk layout (compatible with existing repo caches):
  {cache_dir}/{dataset_tag}/{params_tag}/
    - traj_cache.pkl.gz
    - traj_cache_meta.json

Meta schema (traj_cache@v1):
  {
    "csv_file": "/abs/path/to.csv",
    "file_hash": "<md5>",
    "params": { ... },
    "cache_version": "traj_cache@v1"
  }
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import pickle
from dataclasses import dataclass
from typing import Any, Dict, Optional


CACHE_VERSION = "traj_cache@v1"


def _abs(p: str) -> str:
    return os.path.abspath(os.path.expanduser(p))


def _dataset_tag_from_csv(csv_file_path: str) -> str:
    base = os.path.basename(csv_file_path)
    name, _ext = os.path.splitext(base)
    return name


def _md5_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    """MD5 of file content (hex). Used for strict cache validation."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def make_params_dict(
    *,
    fixed_dt_s: int,
    min_traj_len: int,
    simp_alpha: float,
    dp_epsilon_scale: float,
    max_interp_points: Optional[int] = None,
    max_simplified_points: Optional[int] = None,
    length_policy: str = "skip_if_length_missing_or_nonpositive",
    version: str = CACHE_VERSION,
) -> Dict[str, Any]:
    """
    Build the parameter dict embedded in meta.json.
    Keep it stable so existing caches remain loadable.
    """
    return {
        "fixed_dt_s": int(fixed_dt_s),
        "min_traj_len": int(min_traj_len),
        "simp_alpha": float(simp_alpha),
        "dp_epsilon_scale": float(dp_epsilon_scale),
        "max_interp_points": int(max_interp_points) if max_interp_points is not None else None,
        "max_simplified_points": int(max_simplified_points) if max_simplified_points is not None else None,
        "length_policy": str(length_policy),
        "version": str(version),
    }


def _params_tag(params: Dict[str, Any]) -> str:
    """
    Convert params -> folder tag.
    Must match existing cache dirs, e.g.:
      dt60_len50_a1.0_dp0.8_minone_msnone
      dt60_len50_a1.0_dp0.8_mi20000_ms5000
    """
    dt = int(params.get("fixed_dt_s"))
    ln = int(params.get("min_traj_len"))
    a = float(params.get("simp_alpha"))
    dp = float(params.get("dp_epsilon_scale"))
    mi = params.get("max_interp_points", None)
    ms = params.get("max_simplified_points", None)
    mi_tag = f"mi{int(mi)}" if mi is not None else "minone"
    ms_tag = f"ms{int(ms)}" if ms is not None else "msnone"
    return f"dt{dt}_len{ln}_a{a}_dp{dp}_{mi_tag}_{ms_tag}"


def _cache_paths(
    *,
    csv_file_path: str,
    cache_dir: Optional[str],
    params: Dict[str, Any],
) -> Dict[str, str]:
    root = _abs(cache_dir or "./traj_cache")
    dataset_tag = _dataset_tag_from_csv(csv_file_path)
    tag = _params_tag(params)
    dir_path = os.path.join(root, dataset_tag, tag)
    return {
        "dir": dir_path,
        "payload": os.path.join(dir_path, "traj_cache.pkl.gz"),
        "meta": os.path.join(dir_path, "traj_cache_meta.json"),
    }


def _safe_json_load(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _safe_payload_load(path: str) -> Optional[Dict[str, Any]]:
    try:
        with gzip.open(path, "rb") as f:
            obj = pickle.load(f)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _is_payload_valid(payload: Dict[str, Any]) -> bool:
    """
    Basic sanity checks to avoid using half-written/old invalid caches.
    """
    processed = payload.get("processed_trajectories", None)
    simplified = payload.get("simplified_trajectories", None)
    if processed is None or simplified is None:
        return False
    try:
        n_p = len(processed)
        n_s = len(simplified)
    except Exception:
        return False
    if n_p > 0 and n_s == 0:
        return False
    if n_p != n_s:
        return False
    return True


def load_traj_cache(
    *,
    csv_file_path: str,
    cache_dir: Optional[str],
    params: Dict[str, Any],
    verbose: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Load cached payload if meta+payload exist and match (csv hash + params).
    Returns payload dict or None.
    """
    csv_file_path = _abs(csv_file_path)
    paths = _cache_paths(csv_file_path=csv_file_path, cache_dir=cache_dir, params=params)
    meta = _safe_json_load(paths["meta"])
    if not meta:
        if verbose:
            print(f"ℹ️  EN: {paths['meta']}")
        return None
    if str(meta.get("cache_version", "")) != CACHE_VERSION:
        if verbose:
            print(f"ℹ️  EN（EN）: {meta.get('cache_version')!r}")
        return None
    if _abs(str(meta.get("csv_file", ""))) != csv_file_path:
        if verbose:
            print("ℹ️  ENCSVEN（EN）")
        return None

    meta_params = meta.get("params", None)
    if not isinstance(meta_params, dict) or meta_params != params:
        if verbose:
            print("ℹ️  EN（EN）")
        return None

    try:
        cur_hash = _md5_file(csv_file_path)
    except Exception as e:
        if verbose:
            print(f"ℹ️  EN file_hash EN（EN）: {e}")
        return None
    if str(meta.get("file_hash", "")) != cur_hash:
        if verbose:
            print("ℹ️  EN file_hash EN（EN）")
        return None

    payload = _safe_payload_load(paths["payload"])
    if payload is None or (not _is_payload_valid(payload)):
        if verbose:
            print("ℹ️  EN payload EN/EN（EN）")
        return None

    if verbose:
        try:
            n = len(payload.get("processed_trajectories", []))
        except Exception:
            n = "?"
        print(f"✅ EN: {paths['payload']}（EN={n}）")
    return payload


def save_traj_cache(
    *,
    csv_file_path: str,
    cache_dir: Optional[str],
    params: Dict[str, Any],
    payload: Dict[str, Any],
    verbose: bool = True,
) -> Dict[str, str]:
    """
    Save payload + meta to disk. Returns paths dict.
    """
    csv_file_path = _abs(csv_file_path)
    paths = _cache_paths(csv_file_path=csv_file_path, cache_dir=cache_dir, params=params)
    os.makedirs(paths["dir"], exist_ok=True)

    file_hash = _md5_file(csv_file_path)

    payload_tmp = paths["payload"] + ".tmp"
    with gzip.open(payload_tmp, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(payload_tmp, paths["payload"])

    meta_obj = {
        "csv_file": csv_file_path,
        "file_hash": file_hash,
        "params": params,
        "cache_version": CACHE_VERSION,
    }
    meta_tmp = paths["meta"] + ".tmp"
    with open(meta_tmp, "w", encoding="utf-8") as f:
        json.dump(meta_obj, f, ensure_ascii=False, indent=2)
    os.replace(meta_tmp, paths["meta"])

    if verbose:
        try:
            n = len(payload.get("processed_trajectories", []))
        except Exception:
            n = "?"
        print(f"💾 EN: {paths['payload']}")
        print(f"   EN: {n}, hash: {file_hash[:8]}..., tag: {_params_tag(params)}")
    return paths


__all__ = [
    "CACHE_VERSION",
    "make_params_dict",
    "load_traj_cache",
    "save_traj_cache",
]
