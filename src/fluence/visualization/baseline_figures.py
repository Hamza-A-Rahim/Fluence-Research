"""Figures that show what the Stage 2 baseline models learned (EXP-001).

1. weights.png     — logistic regression: which features push towards "event" vs "fluent"
2. scores.png      — how the model's scores for event and fluent clips overlap
3. roc.png         — ROC curves for both models vs chance
4. tree.png        — the top of one real decision tree from the random forest
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.tree import plot_tree

# Reference palette (light mode)
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#a3a29c"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def readable(name: str) -> str:
    """Turn a feature column name into plain English."""
    fixed = {
        "low_energy_frac": "share of near-silent moments",
        "rms_db_mean": "average loudness", "rms_db_std": "how much loudness varies",
        "rms_db_min": "quietest moment", "rms_db_max": "loudest moment",
    }
    if name in fixed:
        return fixed[name]
    m = re.fullmatch(r"(logmel|mfcc|dmfcc)(\d+)_(mean|std)", name)
    if not m:
        return name
    kind, i, stat = m.group(1), int(m.group(2)), m.group(3)
    what = {"mean": "average", "std": "variation in"}[stat]
    if kind == "logmel":
        band = "low" if i < 13 else "mid" if i < 27 else "high"
        return f"{what} energy, {band} pitch band {i}"
    if kind == "mfcc":
        return f"{what} voice-shape coefficient {i}"
    return f"{what} rate of change, coefficient {i}"


def weights_figure(lr_pipeline, names, out: Path, k: int = 8) -> None:
    coef = lr_pipeline[-1].coef_.ravel()  # on standardised features → comparable sizes
    order = np.argsort(coef)
    pick = np.r_[order[:k], order[-k:]]
    vals, labels = coef[pick], [readable(names[i]) for i in pick]
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    ax.barh(range(len(pick)), vals, color=[ORANGE if v < 0 else BLUE for v in vals],
            height=0.7, edgecolor=SURFACE, linewidth=2)
    ax.set_yticks(range(len(pick)), labels, fontsize=9)
    ax.axvline(0, color=INK2, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Weight (per standard deviation of the feature)")
    ax.set_title("What the logistic regression listens for")
    lim = np.abs(vals).max() * 1.15
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-0.8, len(pick) + 0.6)
    ax.text(lim * 0.97, len(pick) - 0.1, "→ pushes towards \"stutter event\"", color=BLUE,
            ha="right", va="bottom", fontsize=9, fontweight="bold")
    ax.text(-lim * 0.97, len(pick) - 0.1, "pushes towards \"fluent\" ←", color=ORANGE,
            ha="left", va="bottom", fontsize=9, fontweight="bold")
    fig.text(0.01, -0.035, f"The {k} strongest weights each way, out of 137 features. Trained on SEP28k-E "
             "train (3 speakers).\nNeighbouring pitch bands are highly correlated, so pairs like bands 38 (+) "
             "and 39 (−) partly cancel out:\nread the overall pattern, not any single bar.", color=INK2, fontsize=8)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def scores_figure(score, y, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    bins = np.linspace(0, 1, 41)
    ax.hist(score[y == 0], bins=bins, color=ORANGE, alpha=0.55, label="clips labelled fluent",
            edgecolor=SURFACE, linewidth=0.5)
    ax.hist(score[y == 1], bins=bins, color=BLUE, alpha=0.55, label="clips with a stutter event",
            edgecolor=SURFACE, linewidth=0.5)
    ax.axvline(0.5, color=INK2, linewidth=1, linestyle="--")
    ax.text(0.505, ax.get_ylim()[1] * 0.95, "decision line", color=INK2, fontsize=8, va="top")
    ax.set_xlabel("Model's score: how likely it thinks the clip contains an event")
    ax.set_ylabel("Number of clips")
    ax.set_title("The two groups overlap heavily — that's why accuracy is modest")
    ax.legend(frameon=False, loc="upper left")
    auc = roc_auc_score(y, score)
    fig.text(0.01, -0.07, f"Logistic regression on {len(y):,} dev clips from unseen speakers. "
             f"ROC-AUC {auc:.3f}: a random event clip outscores a random fluent clip "
             f"{auc:.0%} of the time.", color=INK2, fontsize=8)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def roc_figure(curves: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    ax.plot([0, 1], [0, 1], color=GRAY, linewidth=1.5, linestyle="--")
    ax.text(0.62, 0.55, "chance (AUC 0.50)", color=INK2, fontsize=8, rotation=40)
    for (label, (yt, sc)), col in zip(curves.items(), [BLUE, ORANGE]):
        fpr, tpr, _ = roc_curve(yt, sc)
        ax.plot(fpr, tpr, color=col, linewidth=2, label=f"{label}  AUC {roc_auc_score(yt, sc):.3f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("Fluent clips wrongly flagged (false-positive rate)")
    ax.set_ylabel("Event clips caught (true-positive rate)")
    ax.set_title("ROC curves, dev set")
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    fig.text(0.01, -0.02, "Further from the dashed line = better. A perfect model hugs the top-left corner.",
             color=INK2, fontsize=8)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def tree_figure(forest, names, out: Path, depth: int = 2) -> None:
    tree = forest.estimators_[0]
    fig, ax = plt.subplots(figsize=(20, 6))
    plot_tree(tree, max_depth=depth, feature_names=[readable(n) for n in names],
              class_names=["fluent", "event"], filled=False, impurity=False,
              proportion=True, rounded=True, fontsize=9, ax=ax)
    ax.set_title(f"One of the forest's 300 trees — first {depth + 1} levels of questions "
                 f"(it continues {tree.get_depth() - depth - 1} levels deeper)")
    fig.text(0.01, 0.0, "Each box asks a yes/no question about one feature; left = yes, right = no. "
             "'value' = [fluent, event] share of training clips reaching that box (after class balancing). "
             "The forest's answer is a vote across all 300 trees.", color=INK2, fontsize=9)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
