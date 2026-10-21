"""Milestone M1: listen to labelled clips and check them by ear and eye.

Picks example clips for each event type, preferring clear-cut cases
(all 3 annotators agreed, and no other event in the clip), plus a few
"disputed" clips where annotators split 1-2, and clips from episodes whose
downloaded audio was slightly short (to confirm they still line up).

For each clip it copies the WAV and draws a waveform + spectrogram, then
builds one page you open in your browser:

    python scripts/listen_examples.py
    start outputs\\listening\\index.html        (Windows)

Write what you hear in outputs/listening/notes.csv (opens in Excel).
Everything stays in the git-ignored outputs/ folder.
"""

from __future__ import annotations

import argparse
import csv
import html
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fluence.data.datasets import load_available_clips  # noqa: E402
from fluence.data.sep28k import EVENT_LABELS  # noqa: E402
from fluence.preprocessing.audio import load_wav  # noqa: E402
from fluence.visualization.plots import plot_waveform_and_logmel  # noqa: E402

PLAIN = {
    "Prolongation": "Prolongation: a sound stretched out (\"sssssun\")",
    "Block": "Block: speech gets stuck; often silence or tension before a word",
    "SoundRep": "Sound repetition: a sound or syllable repeated (\"b-b-ball\")",
    "WordRep": "Word repetition: a whole word or phrase repeated (\"I-I-I want\")",
    "Interjection": "Interjection: fillers like \"um\", \"uh\", \"like\"",
    "Fluent": "Fluent: all 3 annotators heard no stuttered words",
}


def pick(df: pd.DataFrame, n: int, rng) -> dict[str, pd.DataFrame]:
    ev = df[EVENT_LABELS]
    groups: dict[str, pd.DataFrame] = {}
    for lab in EVENT_LABELS:
        others = [o for o in EVENT_LABELS if o != lab]
        clear = df[(df[lab] == 3) & (ev[others].max(axis=1) == 0)]
        if len(clear) < n:  # fall back to majority + no other majority event
            clear = df[(df[lab] >= 2) & (ev[others].max(axis=1) < 2)]
        groups[lab] = clear.sample(min(n, len(clear)), random_state=rng)
    fluent = df[(df["NoStutteredWords"] == 3) & (ev.max(axis=1) == 0)]
    groups["Fluent"] = fluent.sample(min(n, len(fluent)), random_state=rng)
    split = df[(ev == 1).any(axis=1) & (ev.max(axis=1) < 2)]  # only 1 of 3 heard an event
    groups["Disputed"] = split.sample(min(n, len(split)), random_state=rng)
    return groups


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", type=Path, default=ROOT / "data/metadata")
    ap.add_argument("--clips", type=Path, default=ROOT / "data/interim/sep28k_clips")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/listening")
    ap.add_argument("--per-type", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = load_available_clips(args.metadata, args.clips)
    if df.empty:
        sys.exit("No clips found. Run scripts/fetch_sep28k.py first.")
    groups = pick(df, args.per_type, args.seed)

    log_path = args.metadata / "sep28k_fetch_log.csv"
    if log_path.exists():
        log = pd.read_csv(log_path)
        short = log[log["status"] == "length_mismatch"][["show", "ep_id"]]
        keys = set(zip(short["show"], short["ep_id"].astype(int)))
        in_short = df[[(s, int(e)) in keys for s, e in zip(df["Show"], df["EpId"])]]
        # latest clip in each short episode = closest to the end, most likely to drift
        last = in_short.sort_values("Stop").groupby(["Show", "EpId"]).tail(1)
        groups["Short-episode check"] = last.head(args.per_type)

    if args.out.exists():
        shutil.rmtree(args.out)
    (args.out / "clips").mkdir(parents=True)
    rows, sections = [], []
    for name, sub in groups.items():
        cards = []
        for _, r in sub.iterrows():
            stem = Path(r["path"]).stem
            shutil.copy(r["path"], args.out / "clips" / f"{stem}.wav")
            x, info = load_wav(r["path"])
            plot_waveform_and_logmel(x, info.sr, args.out / "clips" / f"{stem}.png", title=stem)
            votes = ", ".join(f"{lab} {int(r[lab])}/3" for lab in EVENT_LABELS + ["NoStutteredWords"] if r[lab] > 0)
            cards.append(
                f'<div class="card"><h3>{html.escape(stem)}</h3>'
                f'<audio controls preload="none" src="clips/{stem}.wav"></audio>'
                f'<img src="clips/{stem}.png" alt="waveform and spectrogram of {stem}">'
                f'<p class="votes">Annotator votes: {html.escape(votes or "none")}</p></div>')
            rows.append({"group": name, "clip": stem, "annotator_votes": votes,
                         "speaker": r.get("speaker", ""), "split": r.get("SEP28k-E", ""),
                         "i_hear": "", "agree_with_label (y/n/unsure)": "", "notes": ""})
        desc = PLAIN.get(name, {
            "Disputed": "Disputed: exactly one of three annotators heard an event. Do you?",
            "Short-episode check": "Last clip of episodes whose download was slightly short. "
                                   "Does the audio still sound like a natural 3-second window?",
        }.get(name, ""))
        body = "".join(cards) or "<p><em>No clear-cut examples of this type among the downloaded clips.</em></p>"
        sections.append(f"<section><h2>{html.escape(name)}</h2><p>{html.escape(desc)}</p>"
                        f'<div class="grid">{body}</div></section>')

    with open(args.out / "notes.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Fluence listening check</title>
<style>
body{{font-family:system-ui,sans-serif;margin:0 auto;max-width:1100px;padding:24px 16px;color:#0b0b0b;background:#fcfcfb}}
h1{{margin:0 0 4px}} h2{{margin:32px 0 4px}} p{{color:#52514e}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:16px}}
.card{{border:1px solid #e6e5e0;border-radius:8px;padding:12px;background:#fff}}
.card h3{{font-size:13px;margin:0 0 8px;font-family:ui-monospace,monospace}}
audio{{width:100%}} img{{width:100%;margin-top:8px}} .votes{{font-size:12px}}
</style></head><body>
<h1>Fluence: listening check</h1>
<p>{sum(len(g) for g in groups.values())} clips, 3 seconds each. Listen, look at the picture, and write what you hear
in <code>notes.csv</code>. Votes show how many of the 3 annotators chose each label. Not a clinical assessment.</p>
{''.join(sections)}
</body></html>"""
    (args.out / "index.html").write_text(page, encoding="utf-8")
    print(f"Open {args.out / 'index.html'} in your browser; notes go in {args.out / 'notes.csv'}")


if __name__ == "__main__":
    main()
