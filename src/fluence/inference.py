"""Stage 7: score a recording of any length with the frozen EXP-004 system (local demo only).

The classifier only ever saw 3 s clips, so a recording is cut into 3 s windows
every ``hop_s`` seconds (the last window is aligned to the end so the tail is
covered). Following EXP-005, overlapping window scores are combined by their
**mean**: the score at time t is the mean over all windows that contain t.

Timing is approximate. SEP-28k has no timing inside clips, so region
boundaries are not validated (D-012); they show where the score is high, not
where an event starts or ends.
"""

from __future__ import annotations

import numpy as np

SR = 16_000
CLIP_S = 3.0


def window_starts(n_samples: int, hop_s: float = 0.5, clip_s: float = CLIP_S, sr: int = SR) -> np.ndarray:
    """Start samples of 3 s windows covering the whole signal (needs n_samples >= one window)."""
    n, hop = int(round(clip_s * sr)), int(round(hop_s * sr))
    if n_samples < n:
        raise ValueError(f"Recording shorter than {clip_s} s")
    starts = np.arange(0, n_samples - n + 1, hop)
    if starts[-1] != n_samples - n:
        starts = np.append(starts, n_samples - n)
    return starts


def timeline(starts_s: np.ndarray, scores: np.ndarray, duration_s: float, step_s: float = 0.1,
             clip_s: float = CLIP_S) -> tuple[np.ndarray, np.ndarray]:
    """Grid times (centre of each step) and the mean score of the windows covering each."""
    t = np.arange(step_s / 2, duration_s, step_s)
    starts_s = np.asarray(starts_s, float)
    cover = (starts_s[None, :] <= t[:, None]) & (t[:, None] < starts_s[None, :] + clip_s)
    counts = cover.sum(axis=1)
    mean = (cover * np.asarray(scores, float)[None, :]).sum(axis=1) / np.maximum(counts, 1)
    return t, mean


def regions(t: np.ndarray, score: np.ndarray, threshold: float, step_s: float = 0.1,
            merge_gap_s: float = 0.3) -> list[dict]:
    """Contiguous spans with score >= threshold; spans closer than ``merge_gap_s`` are joined."""
    above = score >= threshold
    spans: list[list[float]] = []
    i = 0
    while i < len(t):
        if not above[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(t) and above[j + 1]:
            j += 1
        a, b = t[i] - step_s / 2, t[j] + step_s / 2
        if spans and a - spans[-1][1] <= merge_gap_s + 1e-9:
            spans[-1][1] = b
        else:
            spans.append([a, b])
        i = j + 1
    out = []
    for a, b in spans:
        m = (t >= a) & (t <= b)
        out.append({"start": round(float(a), 2), "end": round(float(b), 2), "peak": float(score[m].max())})
    return out
