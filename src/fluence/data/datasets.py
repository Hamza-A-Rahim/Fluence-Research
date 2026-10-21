"""Shared loader: usable SEP-28k clips that exist on disk."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from fluence.data.clips import attach_clip_paths
from fluence.data.sep28k import exclude_unusable, load_labels


def load_available_clips(metadata: Path, clips_dir: Path) -> pd.DataFrame:
    """Labels from SEP-28k-Extended, minus unusable clips, keeping only clips on disk.

    Row order follows the labels file, so feature caches built from the same
    rows line up.
    """
    df = exclude_unusable(load_labels(Path(metadata) / "SEP-28k-Extended_clips.csv"))
    df = attach_clip_paths(df, clips_dir)
    return df[df["available"]].copy()


def cv_pool(df: pd.DataFrame) -> pd.DataFrame:
    """Clips for speaker-grouped cross-validation, with the SEP28k-E test set excluded.

    SEP-12k (the TH Nuernberg CV partition) is exactly SEP28k-E dev + test.
    Cross-validating on all of it would train and score on the held-out test
    clips, so we use only the SEP28k-E *dev* part (D-011). The test set stays
    unseen until the single final evaluation.
    """
    keep = (df["SEP12k"].astype(str) == "train") & (df["SEP28k-E"].astype(str) == "dev")
    return df[keep]
