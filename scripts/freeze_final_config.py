"""Freeze every choice for the one-time SEP28k-E test evaluation (pre-registration).

Uses train + dev ONLY. For each system (baseline features, base-960h,
XLSR-53) and each target it records the settings already chosen on dev in
EXP-001/002/003, fits on train, and picks the F1-maximising decision
threshold on dev. Writes configs/final_eval.json, which must be committed
before scripts/final_test_eval.py is run.

    python scripts/freeze_final_config.py --primary facebook/wav2vec2-large-xlsr-53
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
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
from fluence.evaluation.metrics import best_f1_threshold  # noqa: E402
from fluence.features.embeddings import model_slug  # noqa: E402
from fluence.features.sets import load_feature_set  # noqa: E402
from train_baseline import fit_predict, git_commit  # noqa: E402
from train_embeddings import targets  # noqa: E402

ORDER = ["any_with_interjections", "any_without_interjections", "Prolongation", "Block", "SoundRep",
         "WordRep", "Interjection"]
ENCODERS = ["facebook/wav2vec2-base-960h", "facebook/wav2vec2-large-xlsr-53"]


def baseline_C(outputs: Path) -> dict[str, float]:
    r1 = json.loads((outputs / "EXP-001/SEP28k-E/metrics.json").read_text(encoding="utf-8"))["targets"]
    r2 = json.loads((outputs / "EXP-002/metrics.json").read_text(encoding="utf-8"))["SEP28k-E"]
    C = {"any_with_interjections": r1["with_interjections"]["logreg"]["selected_params"]["C"],
         "any_without_interjections": r1["without_interjections"]["logreg"]["selected_params"]["C"]}
    C.update({e: r2[e]["logreg"]["selected_params"]["C"] for e in ORDER[2:]})
    return C


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--primary", required=True)
    ap.add_argument("--rationale", default="")
    ap.add_argument("--metadata", type=Path, default=ROOT / "data/metadata")
    ap.add_argument("--clips", type=Path, default=ROOT / "data/interim/sep28k_clips")
    ap.add_argument("--data-root", type=Path, default=ROOT, help="folder containing data/processed")
    ap.add_argument("--outputs", type=Path, default=ROOT / "outputs")
    ap.add_argument("--out", type=Path, default=ROOT / "configs/final_eval.json")
    args = ap.parse_args()

    df = load_available_clips(args.metadata, args.clips)
    all_e = df[df["SEP28k-E"].isin(["train", "dev", "test"])].reset_index(drop=True)
    keep = all_e["SEP28k-E"].isin(["train", "dev"]).to_numpy()   # test rows are dropped before any use
    sub = all_e[keep].reset_index(drop=True)
    part = sub["SEP28k-E"].to_numpy()
    tr, dv = np.flatnonzero(part == "train"), np.flatnonzero(part == "dev")
    T = targets(sub)

    systems = {"baseline_features": {"features": {"type": "baseline"}, "C": baseline_C(args.outputs)}}
    for m in ENCODERS:
        r = json.loads((args.outputs / "EXP-003" / model_slug(m) / "metrics.json").read_text(encoding="utf-8"))
        systems[m] = {"features": {"type": "embedding", "model": m, "layer": r["layer"], "pooling": r["pooling"],
                                   "revision": r["manifest"].get("revision")},
                      "C": {t: r["SEP28k-E"][t]["C"] for t in ORDER}}
    if args.primary not in systems:
        sys.exit(f"--primary must be one of {list(systems)}")

    for name, s in systems.items():
        # baseline features are cached for the full SEP28k-E clip list, so load that and drop test rows
        X = load_feature_set(s["features"], all_e if s["features"]["type"] == "baseline" else sub, args.data_root)
        if s["features"]["type"] == "baseline":
            X = X[keep]
        s["threshold"] = {}
        for t in ORDER:
            _, score = fit_predict("logreg", {"C": s["C"][t]}, 0, X[tr], T[t][tr], X[dv])
            s["threshold"][t] = best_f1_threshold(T[t][dv], score)
        print(f"{name}: thresholds " + ", ".join(f"{t}={v:.3f}" for t, v in s["threshold"].items()), flush=True)

    cfg = {"purpose": "Pre-registered configuration for the single SEP28k-E test evaluation (EXP-004).",
           "frozen_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "frozen_from_commit": git_commit(),
           "primary_system": args.primary, "primary_rationale": args.rationale,
           "targets": ORDER, "training_data": "SEP28k-E train only (as in development)",
           "model": "logistic regression, class-balanced, standardised, seed 0",
           "label_rule": ">= 2 of 3 annotators", "systems": systems,
           "primary_metric": "ROC-AUC with 95% speaker-bootstrap CI",
           "secondary": ["PR-AUC", "F1 at the frozen dev threshold", "F1 at 0.5",
                         "paired AUC differences: primary vs each other system"]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}. Commit it before running final_test_eval.py.")


if __name__ == "__main__":
    main()
