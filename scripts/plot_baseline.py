"""Draw what the Stage 2 baseline models learned (EXP-001 figures).

Usage (after scripts/train_baseline.py has run once, so features are cached):
    python scripts/plot_baseline.py
Writes outputs/EXP-001/figures/{weights,scores,roc,tree}.png
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

from fluence.data.clips import attach_clip_paths, compute_features  # noqa: E402
from fluence.data.sep28k import any_stutter_target, exclude_unusable, load_labels  # noqa: E402
from fluence.features.spectral import FeatureConfig  # noqa: E402
from fluence.visualization.baseline_figures import (  # noqa: E402
    roc_figure, scores_figure, tree_figure, weights_figure,
)
from train_baseline import make_model  # noqa: E402


def main() -> None:
    out = ROOT / "outputs/EXP-001/figures"
    out.mkdir(parents=True, exist_ok=True)
    df = exclude_unusable(load_labels(ROOT / "data/metadata/SEP-28k-Extended_clips.csv"))
    df = attach_clip_paths(df, ROOT / "data/interim/sep28k_clips")
    df = df[df["available"] & df["SEP28k-E"].isin(["train", "dev", "test"])]
    X, names, ok = compute_features(df["path"].tolist(), FeatureConfig(), cache_dir=ROOT / "data/processed")
    df, X = df[ok].reset_index(drop=True), X[ok]
    y = any_stutter_target(df).astype(int)
    part = df["SEP28k-E"].to_numpy()
    tr, dv = np.flatnonzero(part == "train"), np.flatnonzero(part == "dev")

    # Settings selected on dev in EXP-001 (with interjections)
    lr = make_model("logreg", {"C": 0.1}, 0).fit(X[tr], y[tr])
    rf = make_model("random_forest", {"min_samples_leaf": 1}, 0).fit(X[tr], y[tr])
    s_lr, s_rf = lr.predict_proba(X[dv])[:, 1], rf.predict_proba(X[dv])[:, 1]

    weights_figure(lr, names, out / "weights.png")
    scores_figure(s_lr, y[dv], out / "scores.png")
    roc_figure({"Logistic regression": (y[dv], s_lr), "Random forest": (y[dv], s_rf)}, out / "roc.png")
    tree_figure(rf, names, out / "tree.png")
    print(f"Saved 4 figures to {out}")


if __name__ == "__main__":
    main()
