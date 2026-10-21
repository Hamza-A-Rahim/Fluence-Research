"""Leakage-safe train/validation/test splitting.

Rule: every clip from the same *group* goes to exactly one partition.
The group should be the speaker when speaker identities are known
(SEP-28k-Extended provides them), otherwise the episode or podcast show.
Splitting individual clips at random lets the model memorise a speaker's
voice, which Bayerl et al. (2022) showed inflates and destabilises results.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def group_split(
    groups: pd.Series | np.ndarray,
    *,
    fractions: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 0,
) -> np.ndarray:
    """Assign each row to 'train', 'val' or 'test' with no group in two partitions.

    Groups are shuffled (seeded) and greedily filled so partitions approach the
    requested fractions *of rows*. With few, large groups (e.g. dominant podcast
    hosts) the achieved fractions can differ a lot from the target — always
    report the achieved sizes, not the requested ones.
    """
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError("fractions must sum to 1")
    groups = pd.Series(np.asarray(groups)).astype(str)
    sizes = groups.value_counts()
    order = sizes.index.to_numpy()
    rng = np.random.default_rng(seed)
    rng.shuffle(order)

    total = len(groups)
    targets = {"train": fractions[0] * total, "val": fractions[1] * total,
               "test": fractions[2] * total}
    filled = {"train": 0, "val": 0, "test": 0}
    assignment: dict[str, str] = {}
    for g in order:
        # Put the group where it reduces the remaining deficit most.
        part = max(targets, key=lambda p: targets[p] - filled[p])
        assignment[g] = part
        filled[part] += int(sizes[g])
    return groups.map(assignment).to_numpy()


def check_no_leakage(groups, partition) -> None:
    """Raise if any group appears in more than one partition."""
    df = pd.DataFrame({"g": np.asarray(groups).astype(str), "p": np.asarray(partition)})
    spread = df.groupby("g")["p"].nunique()
    leaked = spread[spread > 1]
    if len(leaked):
        raise AssertionError(f"{len(leaked)} groups appear in several partitions, e.g. "
                             f"{list(leaked.index[:5])}")


def split_report(groups, partition, y: np.ndarray | None = None) -> pd.DataFrame:
    """Rows, groups and (optionally) positive rate per partition."""
    df = pd.DataFrame({"g": np.asarray(groups).astype(str), "p": np.asarray(partition)})
    if y is not None:
        df["y"] = np.asarray(y)
    rep = df.groupby("p").agg(rows=("g", "size"), groups=("g", "nunique"))
    if y is not None:
        rep["positive_rate"] = df.groupby("p")["y"].mean().round(4)
    rep["row_share"] = (rep["rows"] / rep["rows"].sum()).round(4)
    return rep
