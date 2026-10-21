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
