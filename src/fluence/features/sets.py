"""Named feature sets, so final comparisons use exactly the representation each experiment used."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fluence.data.clips import compute_features
from fluence.features.embeddings import load_embeddings, model_slug
from fluence.features.spectral import FeatureConfig


def load_feature_set(spec: dict, df: pd.DataFrame, root: Path) -> np.ndarray:
    """spec: {"type": "baseline"} or {"type": "embedding", "model": ..., "layer": int, "pooling": ...}.

    Rows align with ``df``; raises if any clip lacks features.
    """
    if spec["type"] == "baseline":
        X, _, ok = compute_features(df["path"].tolist(), FeatureConfig(), cache_dir=root / "data/processed",
                                    progress=False)
    elif spec["type"] == "embedding":
        X, ok = load_embeddings(root / "data/processed/embeddings" / model_slug(spec["model"]),
                                df["clip_key"], layer=spec["layer"], pooling=spec.get("pooling", "mean+std"))
    else:
        raise ValueError(spec)
    if not ok.all():
        raise RuntimeError(f"{int((~ok).sum())} clips have no features for {spec}")
    return X
