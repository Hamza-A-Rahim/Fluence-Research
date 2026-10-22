"""Chart: baseline features (EXP-001/002) vs encoder embeddings (EXP-003), per target.

    python scripts/plot_comparison.py --model facebook/wav2vec2-base-960h --model facebook/wav2vec2-large-xlsr-53
Reads outputs/EXP-001, EXP-002 (optional) and EXP-003/<model> metrics; writes
outputs/EXP-003/comparison.png
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

TARGETS = [("any_with_interjections", "Any event (incl. interjections)"),
           ("any_without_interjections", "Any event (excl. interjections)"),
           ("Interjection", "Interjection"), ("SoundRep", "Sound repetition"),
           ("Prolongation", "Prolongation"), ("WordRep", "Word repetition"), ("Block", "Block")]
COLOURS = ["#2a78d6", "#eb6834", "#1baf7a"]  # reference palette, slots 1-3


def baseline_aucs(outputs: Path) -> dict[str, float]:
    out = {}
    p1 = outputs / "EXP-001/SEP28k-E/metrics.json"
    if p1.exists():
        t = json.loads(p1.read_text(encoding="utf-8"))["targets"]
        out["any_with_interjections"] = t["with_interjections"]["logreg"]["dev"]["roc_auc"]
        out["any_without_interjections"] = t["without_interjections"]["logreg"]["dev"]["roc_auc"]
    p2 = outputs / "EXP-002/metrics.json"
    if p2.exists():
        e = json.loads(p2.read_text(encoding="utf-8"))["SEP28k-E"]
        out.update({k: e[k]["logreg"]["dev"]["roc_auc"] for k, _ in TARGETS if k in e})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append")
    ap.add_argument("--outputs", type=Path, default=ROOT / "outputs")
    args = ap.parse_args()
    models = args.model or ["facebook/wav2vec2-base-960h", "facebook/wav2vec2-large-xlsr-53"]
    runs = []
    for m in models:
        p = args.outputs / "EXP-003" / model_slug(m) / "metrics.json"
        if p.exists():
            runs.append((m, json.loads(p.read_text(encoding="utf-8"))))
        else:
            print(f"(skipping {m}: no {p})")
    if not runs:
        sys.exit("No EXP-003 results found.")
    base = baseline_aucs(args.outputs)

    GRAY, INK2, GRID, SURF = "#a3a29c", "#52514e", "#e6e5e0", "#fcfcfb"
    fig, ax = plt.subplots(figsize=(9, 5.4), facecolor=SURF)
    ax.set_facecolor(SURF)
    n = len(TARGETS)
    offsets = [0.0] if len(runs) == 1 else [0.14 * (i - (len(runs) - 1) / 2) for i in range(len(runs))]
    for row, (key, _) in enumerate(TARGETS[::-1]):
        if key in base:
            ax.plot(base[key], row, "o", color=GRAY, markersize=9, markeredgecolor=SURF, markeredgewidth=2, zorder=3)
        for (m, r), off, col in zip(runs, offsets, COLOURS):
            d = r["SEP28k-E"][key]
            ax.plot(d["auc_ci"], [row - off] * 2, color=col, linewidth=2, alpha=0.55, zorder=2,
                    solid_capstyle="round")
            ax.plot(d["roc_auc"], row - off, "o", color=col, markersize=8, markeredgecolor=SURF,
                    markeredgewidth=2, zorder=3)
    ax.set_yticks(range(n), [lab for _, lab in TARGETS[::-1]], fontsize=9)
    ax.axvline(0.5, color=INK2, linestyle="--", linewidth=1)
    ax.text(0.503, -0.65, "chance", fontsize=8, color=INK2)
    ax.set_xlim(0.45, 1.0)
    ax.set_ylim(-0.8, n - 0.4)
    ax.set_xlabel("ROC-AUC on SEP28k-E dev (136 unseen speakers); line = 95% speaker-bootstrap CI")
    ax.set_title("Which \"ear\" hears stuttering best?", loc="left", fontweight="bold")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color=GRID)
    if base:
        ax.plot([], [], "o", color=GRAY, label="137 hand-made features (EXP-001/002)")
    for (m, r), col in zip(runs, COLOURS):
        ax.plot([], [], "o-", color=col, label=f"{m.split('/')[-1]}, layer {r['layer']}")
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    fig.text(0.01, -0.03, "Same clips, same speakers, same logistic regression: only the input features differ. "
             "Frozen encoders; test set not used.", fontsize=8, color=INK2)
    out = args.outputs / "EXP-003" / "comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
