"""Paired comparison of two encoders on the SEP28k-E dev clips (EXP-003 follow-up).

Both encoders use the layer and per-target C chosen in their EXP-003 runs.
Each model is trained on SEP28k-E train and scored on the same dev clips; the
AUC difference gets a speaker-level paired bootstrap CI.

    python scripts/compare_encoders.py
Output: outputs/EXP-003/paired_comparison.json and .md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

from fluence.data.datasets import load_available_clips  # noqa: E402
from fluence.evaluation.metrics import paired_auc_difference  # noqa: E402
from fluence.features.embeddings import model_slug  # noqa: E402
from fluence.features.sets import load_feature_set  # noqa: E402
from train_baseline import fit_predict  # noqa: E402
from train_embeddings import targets  # noqa: E402

ORDER = ["any_with_interjections", "any_without_interjections", "Prolongation", "Block", "SoundRep",
         "WordRep", "Interjection"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="facebook/wav2vec2-base-960h")
    ap.add_argument("--b", default="facebook/wav2vec2-large-xlsr-53")
    ap.add_argument("--metadata", type=Path, default=ROOT / "data/metadata")
    ap.add_argument("--clips", type=Path, default=ROOT / "data/interim/sep28k_clips")
    ap.add_argument("--data-root", type=Path, default=ROOT, help="folder containing data/processed")
    ap.add_argument("--outputs", type=Path, default=ROOT / "outputs")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    df = load_available_clips(args.metadata, args.clips)
    df = df[df["SEP28k-E"].isin(["train", "dev"])].reset_index(drop=True)  # test never loaded here
    part, spk = df["SEP28k-E"].to_numpy(), df["speaker"].astype(str).to_numpy()
    tr, dv = np.flatnonzero(part == "train"), np.flatnonzero(part == "dev")
    T = targets(df)

    runs = {}
    for name in (args.a, args.b):
        r = json.loads((args.outputs / "EXP-003" / model_slug(name) / "metrics.json").read_text(encoding="utf-8"))
        X = load_feature_set({"type": "embedding", "model": name, "layer": r["layer"]}, df, args.data_root)
        scores = {}
        for t in ORDER:
            _, scores[t] = fit_predict("logreg", {"C": r["SEP28k-E"][t]["C"]}, 0, X[tr], T[t][tr], X[dv])
        runs[name] = {"layer": r["layer"], "scores": scores}
        print(f"scored dev with {name} (layer {r['layer']})", flush=True)

    res = {"a": args.a, "b": args.b, "n_dev": len(dv), "speakers_dev": int(len(np.unique(spk[dv]))), "targets": {}}
    for t in ORDER:
        d = paired_auc_difference(T[t][dv], runs[args.a]["scores"][t], runs[args.b]["scores"][t], spk[dv],
                                  n_boot=args.n_boot)
        res["targets"][t] = d
        print(f"  {t:<26} A {d['auc_a']:.3f}  B {d['auc_b']:.3f}  B-A {d['diff']:+.3f} "
              f"[{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}]  p(B<=A) {d['p_b_not_better']:.3f}", flush=True)

    out = args.outputs / "EXP-003"
    (out / "paired_comparison.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    L = [f"# Paired comparison on SEP28k-E dev: B = {args.b} vs A = {args.a}", "",
         f"{res['n_dev']} dev clips, {res['speakers_dev']} speakers; {args.n_boot} speaker-level paired bootstrap resamples.",
         "", "| Target | AUC A | AUC B | B − A [95 % CI] | p(B ≤ A) |", "|---|---|---|---|---|"]
    for t in ORDER:
        d = res["targets"][t]
        L.append(f"| {t} | {d['auc_a']:.3f} | {d['auc_b']:.3f} | {d['diff']:+.3f} [{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}] | {d['p_b_not_better']:.3f} |")
    L += ["", "Seven targets are tested, so treat any single p near 0.05 with caution (no multiple-comparison correction applied)."]
    (out / "paired_comparison.md").write_text("\n".join(L), encoding="utf-8")
    print(f"Saved {out / 'paired_comparison.md'}")


if __name__ == "__main__":
    main()
