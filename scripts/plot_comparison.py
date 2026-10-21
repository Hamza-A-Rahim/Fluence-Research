"""Chart: baseline features (EXP-001/002) vs encoder embeddings (EXP-003), per target.

    python scripts/plot_comparison.py --model facebook/wav2vec2-base-960h
Reads outputs/EXP-001, EXP-002 and EXP-003/<model> metrics; writes
outputs/EXP-003/<model>/comparison.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from fluence.features.embeddings import model_slug  # noqa: E402

PLAIN = {"Interjection": "Interjection", "SoundRep": "Sound repetition", "Prolongation": "Prolongation",
         "WordRep": "Word repetition", "Block": "Block"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="facebook/wav2vec2-base-960h")
    ap.add_argument("--outputs", type=Path, default=ROOT / "outputs")
    args = ap.parse_args()
    d3 = args.outputs / "EXP-003" / model_slug(args.model)
    r = json.loads((d3 / "metrics.json").read_text(encoding="utf-8"))
    b1 = json.loads((args.outputs / "EXP-001/SEP28k-E/metrics.json").read_text(encoding="utf-8"))["targets"]
    b2 = json.loads((args.outputs / "EXP-002/metrics.json").read_text(encoding="utf-8"))["SEP28k-E"]

    rows = [("Any event (incl. interjections)", b1["with_interjections"]["logreg"]["dev"]["roc_auc"],
             r["SEP28k-E"]["any_with_interjections"]),
            ("Any event (excl. interjections)", b1["without_interjections"]["logreg"]["dev"]["roc_auc"],
             r["SEP28k-E"]["any_without_interjections"])]
    rows += [(PLAIN[e], b2[e]["logreg"]["dev"]["roc_auc"], r["SEP28k-E"][e]) for e in PLAIN]

    BLUE, GRAY, INK2, GRID, SURF = "#2a78d6", "#a3a29c", "#52514e", "#e6e5e0", "#fcfcfb"
    fig, ax = plt.subplots(figsize=(8.8, 4.8), facecolor=SURF)
    ax.set_facecolor(SURF)
    for i, (lab, old, new) in enumerate(rows[::-1]):
        ax.plot([old, new["roc_auc"]], [i, i], color=GRID, linewidth=3, zorder=1)
        ax.plot(new["auc_ci"], [i, i], color=BLUE, linewidth=1.5, alpha=0.5, zorder=2)
        ax.plot(old, i, "o", color=GRAY, markersize=9, markeredgecolor=SURF, markeredgewidth=2, zorder=3)
        ax.plot(new["roc_auc"], i, "o", color=BLUE, markersize=9, markeredgecolor=SURF, markeredgewidth=2, zorder=3)
        ax.text(new["auc_ci"][1] + 0.01, i, f"+{new['roc_auc'] - old:.2f}", va="center", fontsize=9, color=INK2)
    ax.set_yticks(range(len(rows)), [lab for lab, _, _ in rows[::-1]], fontsize=9)
    ax.axvline(0.5, color=INK2, linestyle="--", linewidth=1)
    ax.text(0.503, -0.6, "chance", fontsize=8, color=INK2)
    ax.set_xlim(0.45, 1.0)
    ax.set_ylim(-0.8, len(rows) - 0.4)
    ax.set_xlabel("ROC-AUC on SEP28k-E dev (unseen speakers)")
    ax.set_title(f"A better \"ear\": {args.model.split('/')[-1]} vs the hand-made features",
                 loc="left", fontweight="bold")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color=GRID)
    ax.plot([], [], "o", color=GRAY, label="137 hand-made features (EXP-001/002)")
    ax.plot([], [], "o", color=BLUE, label=f"{args.model.split('/')[-1]}, layer {r['layer']} (EXP-003)")
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    fig.text(0.01, -0.03, "Same clips, same speakers, same logistic regression: only the input features changed. "
             "Blue line = 95% speaker-bootstrap CI.", fontsize=8, color=INK2)
    out = d3 / "comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
