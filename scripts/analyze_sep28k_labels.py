"""Describe the SEP-28k labels and the SEP-28k-Extended partitions (no audio needed).

Usage:
    python scripts/analyze_sep28k_labels.py [--metadata data/metadata] [--out outputs/label_analysis]

Prints and saves: label prevalence at 1/2/3-vote thresholds, multi-label
co-occurrence, per-show composition, and for each published partitioning the
clips, speakers and positive rates per split, plus an automatic speaker
leakage check.
"""

from __future__ import annotations

import argparse
import sys

# Windows consoles and pipes may default to a legacy code page (cp1252) that
# cannot print characters like κ or ±; never crash on output.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from fluence.data.sep28k import (  # noqa: E402
    EVENT_LABELS, QUALITY_LABELS, any_stutter_target, binarize, exclude_unusable,
    label_summary, load_labels,
)
from fluence.data.splits import check_no_leakage, split_report  # noqa: E402

PARTITIONS = ["SEP28k-E", "SEP28k-T", "SEP28k-D"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", type=Path, default=Path("data/metadata"))
    ap.add_argument("--out", type=Path, default=Path("outputs/label_analysis"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []

    def emit(s: str = "") -> None:
        print(s)
        lines.append(s)

    df = load_labels(args.metadata / "SEP-28k-Extended_clips.csv")
    emit(f"Clips: {len(df)}   episodes: {df['episode_key'].nunique()}   shows: {df['Show'].nunique()}")
    emit(f"Speakers (clustered, SEP-28k-Extended): {df['speaker'].nunique()}")

    emit("\n## Label prevalence (share of all clips)")
    tab = pd.DataFrame({f">={k} votes": label_summary(df, k)["share"] for k in (1, 2, 3)})
    emit(tab.to_string())

    usable = exclude_unusable(df)
    emit(f"\nUnusable clips (>=2 votes on any of {QUALITY_LABELS}): {len(df) - len(usable)}"
         f" ({(len(df) - len(usable)) / len(df):.1%}); usable: {len(usable)}")

    ev = binarize(usable, EVENT_LABELS, 2)
    n_ev = ev.sum(axis=1)
    emit("\n## Events per usable clip (>=2 votes)")
    emit(n_ev.value_counts().sort_index().rename("clips").to_string())
    emit("\n## Co-occurrence (clips with both events, >=2 votes)")
    emit((ev.T @ ev).to_string())

    y_with = any_stutter_target(usable, include_interjections=True)
    y_without = any_stutter_target(usable, include_interjections=False)
    emit(f"\n'Any event' positive rate: {y_with.mean():.3f} with interjections, "
         f"{y_without.mean():.3f} without")

    emit("\n## Clips per show")
    emit(usable.groupby("Show").agg(clips=("clip_key", "size"), episodes=("episode_key", "nunique"),
                                    speakers=("speaker", "nunique")).to_string())

    top = usable["speaker"].value_counts()
    emit(f"\nTop 4 speakers hold {top.iloc[:4].sum() / len(usable):.1%} of usable clips; "
         f"largest single speaker {top.iloc[0] / len(usable):.1%}")

    for p in PARTITIONS:
        sub = usable[usable[p].notna() & (usable[p].astype(str).str.len() > 0)]
        emit(f"\n## Partition {p}  ({len(sub)} usable clips assigned)")
        emit(split_report(sub["speaker"], sub[p], y_with[usable.index.get_indexer(sub.index)]).to_string())
        try:
            check_no_leakage(sub["speaker"], sub[p])
            emit("Speaker leakage check: PASS (no speaker appears in two splits)")
        except AssertionError as e:
            emit(f"Speaker leakage check: FAIL - {e}")

    (args.out / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSaved {args.out / 'summary.txt'}")


if __name__ == "__main__":
    main()
