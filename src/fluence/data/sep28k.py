"""SEP-28k / FluencyBank label handling (annotations only — no audio).

The label CSVs published by Apple (CC BY-NC 4.0) have one row per 3-second
clip. Each label column holds how many of the 3 annotators ticked that label
(0-3). Start/Stop are sample indices at 16 kHz. Values are written with a
space after each comma, so we parse with ``skipinitialspace=True``.

Binarisation follows Lea et al. (2021): a label is "present" when at least
2 of 3 annotators chose it. This threshold is a parameter because it is a
scientific choice that changes class balance; record it in every experiment.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

LABEL_SR = 16_000

# Five stuttering event types studied in the literature (Lea et al. 2021;
# Bayerl et al. 2022) plus the fluent class.
EVENT_LABELS = ["Prolongation", "Block", "SoundRep", "WordRep", "Interjection"]
FLUENT_LABEL = "NoStutteredWords"
QUALITY_LABELS = ["Unsure", "PoorAudioQuality", "DifficultToUnderstand",
                  "Music", "NoSpeech"]
OTHER_LABELS = ["NaturalPause"]
ALL_LABELS = EVENT_LABELS + [FLUENT_LABEL] + QUALITY_LABELS + OTHER_LABELS
ID_COLUMNS = ["Show", "EpId", "ClipId", "Start", "Stop"]

# Lea et al. treat interjections (filled pauses like "um") as a stuttering
# event, but many clinicians do not count them as stuttering-like disfluencies.
# The binary "any stutter" target is therefore configurable.
CORE_STUTTER_LABELS = ["Prolongation", "Block", "SoundRep", "WordRep"]


def load_labels(path: str | Path) -> pd.DataFrame:
    """Read a SEP-28k-style labels CSV and check its schema."""
    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in ID_COLUMNS + EVENT_LABELS + [FLUENT_LABEL] if c not in df.columns]
    if missing:
        raise ValueError(f"Labels file is missing columns: {missing}")
    if "Show" in df.columns:
        df["Show"] = df["Show"].astype(str).str.strip()
    for c in ID_COLUMNS[1:] + [c for c in ALL_LABELS if c in df.columns]:
        df[c] = pd.to_numeric(df[c], errors="raise").astype(int)
    if (df["Stop"] <= df["Start"]).any():
        raise ValueError("Found clips with Stop <= Start")
    counts = df[[c for c in ALL_LABELS if c in df.columns]]
    if (counts < 0).any().any() or (counts > 3).any().any():
        raise ValueError("Annotator counts must be between 0 and 3")
    df["duration_s"] = (df["Stop"] - df["Start"]) / LABEL_SR
    df["clip_key"] = df["Show"] + "_" + df["EpId"].astype(str) + "_" + df["ClipId"].astype(str)
    df["episode_key"] = df["Show"] + "_" + df["EpId"].astype(str)
    return df


def binarize(df: pd.DataFrame, labels: list[str] | None = None, min_votes: int = 2) -> pd.DataFrame:
    """Return a 0/1 frame: label present if >= ``min_votes`` annotators chose it."""
    if not 1 <= min_votes <= 3:
        raise ValueError("min_votes must be 1, 2 or 3")
    labels = labels or EVENT_LABELS
    return (df[labels] >= min_votes).astype(np.int8)


def any_stutter_target(df: pd.DataFrame, min_votes: int = 2,
                       include_interjections: bool = True) -> np.ndarray:
    """Binary target for the Stage 2 baseline: does the clip contain an event?"""
    labels = EVENT_LABELS if include_interjections else CORE_STUTTER_LABELS
    return binarize(df, labels, min_votes).max(axis=1).to_numpy(dtype=np.int8)


def exclude_unusable(df: pd.DataFrame, min_votes: int = 2) -> pd.DataFrame:
    """Drop clips most annotators marked as unusable (music, no speech, poor audio, unclear)."""
    present = [c for c in QUALITY_LABELS if c in df.columns]
    bad = (df[present] >= min_votes).any(axis=1)
    return df.loc[~bad].copy()


def label_summary(df: pd.DataFrame, min_votes: int = 2) -> pd.DataFrame:
    """Count and share of clips per label at a vote threshold, plus co-occurrence size."""
    present = [c for c in ALL_LABELS if c in df.columns]
    b = binarize(df, present, min_votes)
    out = pd.DataFrame({"clips": b.sum(), "share": b.mean().round(4)})
    events_per_clip = binarize(df, EVENT_LABELS, min_votes).sum(axis=1)
    out.attrs["clips_with_multiple_events"] = int((events_per_clip >= 2).sum())
    return out
