"""Chart of the EXP-004 test-set ROC-AUCs for all pre-registered systems.

    python scripts/plot_test_results.py
Reads outputs/EXP-004/metrics.json; writes outputs/EXP-004/test_auc.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PLAIN = {"any_with_interjections": "Any event (incl. interjections)",
         "any_without_interjections": "Any event (excl. interjections)", "Interjection": "Interjection",
         "SoundRep": "Sound repetition", "Prolongation": "Prolongation", "WordRep": "Word repetition",
         "Block": "Block"}
LABEL = {"baseline_features": "137 hand-made features", "facebook/wav2vec2-base-960h": "wav2vec2-base-960h (English)",
         "facebook/wav2vec2-large-xlsr-53": "XLSR-53 (multilingual, primary)"}
COLOUR = {"baseline_features": "#a3a29c", "facebook/wav2vec2-base-960h": "#2a78d6",
          "facebook/wav2vec2-large-xlsr-53": "#eb6834"}


def main() -> None:
    p = ROOT / "outputs/EXP-004/metrics.json"
    if not p.exists():
        sys.exit("No EXP-004 results.")
    r = json.loads(p.read_text(encoding="utf-8"))
    order = list(PLAIN)[::-1]
    systems = list(r["systems"])
    INK2, GRID, SURF = "#52514e", "#e6e5e0", "#fcfcfb"
    fig, ax = plt.subplots(figsize=(9, 5.4), facecolor=SURF)
    ax.set_facecolor(SURF)
    offs = [0.2 * (i - (len(systems) - 1) / 2) for i in range(len(systems))]
    for row, t in enumerate(order):
        for s, off in zip(systems, offs):
            d = r["systems"][s][t]
            ax.plot(d["auc_ci"], [row - off] * 2, color=COLOUR[s], linewidth=2, alpha=0.55, solid_capstyle="round")
            ax.plot(d["roc_auc"], row - off, "o", color=COLOUR[s], markersize=7, markeredgecolor=SURF,
                    markeredgewidth=1.5)
    ax.set_yticks(range(len(order)), [PLAIN[t] for t in order], fontsize=9)
    ax.axvline(0.5, color=INK2, linestyle="--", linewidth=1)
    ax.text(0.503, -0.7, "chance", fontsize=8, color=INK2)
    ax.set_xlim(0.45, 1.0)
    ax.set_ylim(-0.85, len(order) - 0.35)
    ax.set_xlabel(f"ROC-AUC on the held-out test set ({r['n_test']} clips, {r['speakers_test']} unseen speakers); "
                  "line = 95% speaker CI")
    ax.set_title("Final test-set results (evaluated once, pre-registered)", loc="left", fontweight="bold")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color=GRID)
    for s in systems:
        ax.plot([], [], "o-", color=COLOUR[s], label=LABEL.get(s, s))
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    out = p.parent / "test_auc.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
