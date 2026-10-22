"""Stage 7: export the frozen EXP-004 primary system for the local demo (backend/).

Fits the same classifiers as EXP-004/EXP-005 (configs/final_eval.json: encoder,
layer, C; SEP28k-E train clips only, seed 0) and checks them against EXP-005:
on the same dev clips, the dev ROC-AUC must match the logged clip AUC. Test
rows are never loaded and nothing is evaluated on the test set.

Writes a model card with numbers copied from the logged EXP-004/EXP-005
metrics, so the interface only shows results that come from logged runs.

    python scripts/export_demo_model.py
Output: outputs/demo_model/classifiers.joblib, card.json  (git-ignored)
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

import joblib  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from fluence.data.datasets import load_available_clips  # noqa: E402
from fluence.features.embeddings import model_slug  # noqa: E402
from fluence.features.sets import load_feature_set  # noqa: E402
from train_baseline import git_commit, make_model  # noqa: E402
from train_embeddings import targets  # noqa: E402

CARD_FIELDS = ("roc_auc", "auc_ci", "precision", "recall", "f1", "average_precision", "positives", "n", "prevalence")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "configs/final_eval.json")
    ap.add_argument("--metadata", type=Path, default=ROOT / "data/metadata")
    ap.add_argument("--clips", type=Path, default=ROOT / "data/interim/sep28k_clips")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/demo_model")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    sysname = cfg["primary_system"]
    s = cfg["systems"][sysname]
    exp4 = json.loads((ROOT / "outputs/EXP-004/metrics.json").read_text(encoding="utf-8"))
    exp5 = json.loads((ROOT / "outputs/EXP-005/metrics.json").read_text(encoding="utf-8"))

    df = load_available_clips(args.metadata, args.clips)
    df = df[df["SEP28k-E"].isin(["train", "dev"])].reset_index(drop=True)
    X = load_feature_set(s["features"], df, ROOT)
    T = targets(df)
    tr = np.flatnonzero(df["SEP28k-E"].to_numpy() == "train")
    models = {t: make_model("logreg", {"C": s["C"][t]}, 0).fit(X[tr], T[t][tr]) for t in cfg["targets"]}

    # Check: same dev clips as EXP-005 -> same clip AUC.
    win_dir = ROOT / "data/processed/windows" / model_slug(s["features"]["model"])
    keys = set()
    for p in win_dir.glob("offsets_*.npz"):
        keys.update(np.load(p)["clip_key"].tolist())
    dv = np.flatnonzero(df["clip_key"].isin(keys).to_numpy() & (df["SEP28k-E"] == "dev").to_numpy())
    if len(dv) != exp5["n_clips"]:
        sys.exit(f"Expected {exp5['n_clips']} EXP-005 dev clips, found {len(dv)}; cannot verify. Not exported.")
    for t, m in models.items():
        auc = roc_auc_score(T[t][dv], m.predict_proba(X[dv])[:, 1])
        logged = exp5["targets"][t]["clip_auc"]
        ok = abs(auc - logged) < 1e-9
        print(f"  {t:<26} dev AUC {auc:.6f}  EXP-005 {logged:.6f}  {'ok' if ok else 'MISMATCH'}")
        if not ok:
            sys.exit("Exported classifiers do not reproduce EXP-005. Not exported.")

    joblib.dump({"targets": cfg["targets"], "models": models}, args.out / "classifiers.joblib")
    primary = exp4["systems"][sysname]
    card = {
        "exported_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "system": sysname,
        "encoder": s["features"],
        "classifier": cfg["model"],
        "label_rule": cfg["label_rule"],
        "thresholds": s["threshold"],
        "train": {"clips": int(len(tr)), "speakers": int(df.iloc[tr]["speaker"].nunique()),
                  "partition": "SEP28k-E train"},
        "test": {"experiment": "EXP-004", "clips": exp4["n_test"], "speakers": exp4["speakers_test"],
                 "date_utc": exp4["finished_utc"],
                 "targets": {t: {k: primary[t][k] for k in CARD_FIELDS if k in primary[t]} for t in cfg["targets"]},
                 "baselines_roc_auc": {name: {t: exp4["systems"][name][t]["roc_auc"] for t in cfg["targets"]}
                                       for name in exp4["systems"] if name != sysname}},
        "localisation": {"experiment": "EXP-005", "clips": exp5["n_clips"], "speakers": exp5["n_speakers"],
                         "offsets_s": exp5["offsets_s"],
                         "targets": {t: {"clip_auc": r["clip_auc"],
                                         "offset_auc": {o: v["auc"] for o, v in r["offset_auc"].items()},
                                         "sliding_mean_auc": r["agg_mean"]["auc"],
                                         "sliding_mean_ci": r["agg_mean"]["ci"],
                                         "sliding_max_auc": r["agg_max"]["auc"]}
                                     for t, r in exp5["targets"].items()}},
    }
    (args.out / "card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    print(f"Saved {args.out / 'classifiers.joblib'} and card.json")


if __name__ == "__main__":
    main()
