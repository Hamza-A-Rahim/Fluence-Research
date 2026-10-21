"""Join labels with clip files on disk and compute (cached) features."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from fluence.features.spectral import FeatureConfig, clip_summary_features
from fluence.preprocessing.audio import AudioValidationError, load_wav


def clip_path(clips_dir: Path, show: str, ep_id, clip_id) -> Path:
    """Path layout written by scripts/fetch_sep28k.py."""
    return Path(clips_dir) / show / str(ep_id) / f"{show}_{ep_id}_{clip_id}.wav"


def attach_clip_paths(df: pd.DataFrame, clips_dir: Path) -> pd.DataFrame:
    """Add `path` and `available` columns; does not drop anything."""
    out = df.copy()
    out["path"] = [str(clip_path(clips_dir, s, e, c))
                   for s, e, c in zip(out["Show"], out["EpId"], out["ClipId"])]
    out["available"] = [Path(p).is_file() for p in out["path"]]
    return out


def _cache_key(paths: list[str], cfg: FeatureConfig) -> str:
    h = hashlib.sha1(json.dumps(asdict(cfg), sort_keys=True).encode())
    for p in paths:
        h.update(p.encode())
    return h.hexdigest()[:12]


def compute_features(paths: list[str], cfg: FeatureConfig = FeatureConfig(),
                     cache_dir: Path | None = None, progress: bool = True
                     ) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Summary features for each clip. Returns (X, feature_names, ok_mask).

    Clips that fail validation get a row of NaN and ok_mask False — they are
    reported, never silently dropped. Results are cached by (paths, config).
    """
    cache = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache = cache_dir / f"features_{_cache_key(paths, cfg)}.npz"
        if cache.exists():
            z = np.load(cache, allow_pickle=False)
            return z["X"], list(z["names"]), z["ok"]

    names: list[str] = []
    rows, ok = [], []
    n = len(paths)
    for i, p in enumerate(paths):
        try:
            x, _ = load_wav(p, target_sr=cfg.sr)
            v, names = clip_summary_features(x, cfg)
            rows.append(v)
            ok.append(True)
        except (AudioValidationError, ValueError):
            rows.append(None)
            ok.append(False)
        if progress and (i + 1) % 1000 == 0:
            print(f"  features: {i + 1}/{n}")
    width = len(names) if names else 0
    X = np.stack([r if r is not None else np.full(width, np.nan, np.float32) for r in rows]) \
        if width else np.zeros((n, 0), np.float32)
    ok_arr = np.array(ok, bool)
    if cache is not None:
        np.savez(cache, X=X, names=np.array(names), ok=ok_arr)
    return X, names, ok_arr
