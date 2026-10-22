"""Stage 5: from clip classification to finding events in longer recordings.

SEP-28k gives labels for 3-second clips at known positions in each episode,
but no timing *inside* a clip. So we test localisation the way it can be
tested honestly:

1. Offset robustness: slide the 3 s analysis window around each labelled clip
   (e.g. -1.5 s ... +1.5 s) and measure how detection depends on alignment.
   Offset 0 reproduces the clip itself (a built-in sanity check).
2. Sliding-window aggregation: score a region by the max or mean over the
   windows that cover it, as a detector on unsegmented audio would.
3. Timelines over whole episodes, for inspection (not a timing accuracy claim).

Only episodes with no SEP28k-E test clips are used (the test set is spent).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SR = 16_000
CLIP_S = 3.0


def clean_dev_episodes(df: pd.DataFrame, fetch_log: pd.DataFrame | None = None) -> pd.DataFrame:
    """Episodes with >= 1 dev clip and no test clip (and retrieved, if a log is given).

    Returns one row per episode: Show, EpId, n_dev, plus episode_seconds when known.
    """
    part = df["SEP28k-E"].astype(str)
    g = (df.assign(dev=(part == "dev").astype(int), test=(part == "test").astype(int))
           .groupby(["Show", "EpId"], as_index=False)[["dev", "test"]].sum())
    g = g[(g["dev"] > 0) & (g["test"] == 0)].rename(columns={"dev": "n_dev"}).drop(columns="test")
    if fetch_log is not None:
        log = fetch_log.rename(columns={"show": "Show", "ep_id": "EpId"})
        log = log[log["status"].isin(["ok", "length_mismatch"])][["Show", "EpId", "episode_seconds"]]
        g = g.merge(log, on=["Show", "EpId"], how="inner")
    return g.reset_index(drop=True)


def pick_demo_episodes(episodes: pd.DataFrame, n: int = 3, max_seconds: float = 1500.0) -> pd.DataFrame:
    """Short episodes dense in dev clips, at most one per show, for full timelines."""
    e = episodes.copy()
    if "episode_seconds" in e:
        e = e[e["episode_seconds"].astype(float) <= max_seconds]
        e = e.assign(density=e["n_dev"] / e["episode_seconds"].astype(float))
    else:
        e = e.assign(density=e["n_dev"])
    e = e.sort_values("density", ascending=False).drop_duplicates("Show")
    return e.head(n).reset_index(drop=True)


def context_bounds(start: int, stop: int, n_samples: int, pad_s: float) -> tuple[int, int]:
    """Sample range [a, b) of the excerpt kept around a clip, clipped to the episode."""
    pad = int(round(pad_s * SR))
    return max(0, start - pad), min(n_samples, stop + pad)


def offset_windows(excerpt: np.ndarray, clip_start_in_excerpt: int, offsets_s, clip_s: float = CLIP_S
                   ) -> tuple[list[float], list[np.ndarray]]:
    """Cut 3 s windows starting at clip_start + offset; skip offsets that leave the excerpt."""
    n = int(round(clip_s * SR))
    kept, wins = [], []
    for off in offsets_s:
        a = clip_start_in_excerpt + int(round(off * SR))
        if a >= 0 and a + n <= len(excerpt):
            kept.append(float(off))
            wins.append(excerpt[a:a + n])
    return kept, wins


def sliding_windows(signal: np.ndarray, hop_s: float = 1.0, clip_s: float = CLIP_S
                    ) -> tuple[np.ndarray, list[np.ndarray]]:
    """Start times (s) and 3 s windows covering a whole recording."""
    n, hop = int(round(clip_s * SR)), int(round(hop_s * SR))
    starts = np.arange(0, max(1, len(signal) - n + 1), hop)
    return starts / SR, [signal[a:a + n] for a in starts]


def aggregate_over_offsets(scores: pd.DataFrame, how: str = "max", max_abs_offset: float = 1.5) -> pd.Series:
    """Per clip: combine window scores whose offsets lie within +-max_abs_offset.

    ``scores`` has columns clip_key, offset, score.
    """
    s = scores[scores["offset"].abs() <= max_abs_offset + 1e-9]
    g = s.groupby("clip_key")["score"]
    return {"max": g.max, "mean": g.mean}[how]()
