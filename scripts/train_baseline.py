"""Stage 2 baseline: does a 3-second clip contain a stuttering event?

Logistic regression and random forest on clip-summary features (decision D-009),
evaluated per research/methodology/evaluation_protocol.md.

Usage (after scripts/fetch_sep28k.py has produced clips):

    # Development: train on SEP28k-E train, select settings on dev. Test untouched.
    python scripts/train_baseline.py

    # Robustness: speaker-grouped 5-fold CV on SEP-12k (many speakers in training)
    python scripts/train_baseline.py --partition SEP12k

    # Final, once settings are frozen: one evaluation on the SEP28k-E test set
    python scripts/train_baseline.py --evaluate-test

Outputs go to outputs/EXP-001/<partition>/ : report.md, metrics.json,
confusion-matrix figures. Copy the numbers into
research/experiment_logs/EXP-001_*.md only after reading the report.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import sklearn  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from fluence.data.clips import attach_clip_paths, compute_features  # noqa: E402
from fluence.data.sep28k import any_stutter_target, exclude_unusable, load_labels  # noqa: E402
from fluence.data.splits import check_no_leakage, split_report  # noqa: E402
from fluence.evaluation.metrics import (  # noqa: E402
    binary_report, reference_baselines, speaker_bootstrap_ci,
)
from fluence.features.spectral import FeatureConfig  # noqa: E402

TARGETS = {"with_interjections": True, "without_interjections": False}  # decision D-005

GRIDS = {
    "logreg": [{"C": c} for c in (0.01, 0.1, 1.0, 10.0)],
    "random_forest": [{"min_samples_leaf": m} for m in (1, 5, 20)],
}
DEFAULTS = {"logreg": {"C": 1.0}, "random_forest": {"min_samples_leaf": 5}}


def make_model(name: str, params: dict, seed: int):
    if name == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(
            C=params["C"], class_weight="balanced", max_iter=5000, random_state=seed))
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300, min_samples_leaf=params["min_samples_leaf"],
            class_weight="balanced_subsample", n_jobs=-1, random_state=seed)
    raise ValueError(name)


def fit_predict(name, params, seed, Xtr, ytr, Xev):
    m = make_model(name, params, seed).fit(Xtr, ytr)
    score = m.predict_proba(Xev)[:, 1]
    return (score >= 0.5).astype(int), score


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True)
        dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--", "src", "scripts"],
                               capture_output=True, text=True).stdout.strip()
        return out.stdout.strip() + (" (uncommitted changes)" if dirty else "")
    except Exception:
        return "unknown"


def fmt(r: dict) -> str:
    auc = f"{r['roc_auc']:.3f}" if r.get("roc_auc") is not None else "–"
    ap = f"{r['average_precision']:.3f}" if r.get("average_precision") is not None else "–"
    ci = f" [{r['f1_ci'][0]:.3f}, {r['f1_ci'][1]:.3f}]" if r.get("f1_ci") else ""
    return f"| {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f}{ci} | {auc} | {ap} |"


def save_confusion(rep: dict, title: str, path: Path) -> None:
    cm = np.array([[rep["tn"], rep["fp"]], [rep["fn"], rep["tp"]]])
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v}", ha="center", va="center",
                color="white" if v > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["fluent", "event"])
    ax.set_yticks([0, 1], ["fluent", "event"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Annotated (≥2 of 3)")
    ax.set_title(title, fontsize=8)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metadata", type=Path, default=ROOT / "data/metadata")
    ap.add_argument("--clips", type=Path, default=ROOT / "data/interim/sep28k_clips")
    ap.add_argument("--cache", type=Path, default=ROOT / "data/processed")
    ap.add_argument("--partition", default="SEP28k-E", choices=["SEP28k-E", "SEP12k"])
    ap.add_argument("--evaluate-test", action="store_true")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/EXP-001")
    args = ap.parse_args()
    t0 = time.time()
    out = args.out / args.partition
    out.mkdir(parents=True, exist_ok=True)

    # ---- data
    df = load_labels(args.metadata / "SEP-28k-Extended_clips.csv")
    n_all = len(df)
    df = exclude_unusable(df)
    n_usable = len(df)
    df = attach_clip_paths(df, args.clips)
    df = df[df["available"]]
    n_avail = len(df)
    if args.partition == "SEP12k":
        df = df[df["SEP12k"].astype(str) == "train"]
    else:
        df = df[df[args.partition].isin(["train", "dev", "test"])]
    if len(df) == 0:
        sys.exit(f"No clips found under {args.clips}. Run scripts/fetch_sep28k.py first.")
    print(f"Clips: {n_all} labelled, {n_usable} usable, {n_avail} on disk, {len(df)} in {args.partition}")

    print("Features (cached after first run)...")
    X, names, ok = compute_features(df["path"].tolist(), FeatureConfig(), cache_dir=args.cache)
    n_bad = int((~ok).sum())
    df, X = df[ok].reset_index(drop=True), X[ok]
    speakers = df["speaker"].astype(str).to_numpy()

    result = {
        "experiment": "EXP-001", "partition": args.partition,
        "date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(), "seed": args.seed,
        "data": {"labelled": n_all, "usable": n_usable, "on_disk": n_avail,
                 "in_partition": len(df) + n_bad, "failed_validation": n_bad,
                 "n_features": len(names)},
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "sklearn": sklearn.__version__, "platform": platform.platform()},
        "targets": {},
    }

    for tname, incl in TARGETS.items():
        y = any_stutter_target(df, include_interjections=incl).astype(int)
        print(f"\n=== target: any event, {tname.replace('_', ' ')} ===")
        tres: dict = {}

        if args.partition == "SEP12k":
            # ---- speaker-grouped CV with default settings (no tuning inside folds)
            cv = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
            folds = list(cv.split(X, y, groups=speakers))
            for tr, te in folds:
                check_no_leakage(np.r_[speakers[tr], speakers[te]],
                                 ["train"] * len(tr) + ["test"] * len(te))
            for mname in GRIDS:
                per_fold = []
                for k, (tr, te) in enumerate(folds):
                    pred, score = fit_predict(mname, DEFAULTS[mname], args.seed, X[tr], y[tr], X[te])
                    per_fold.append(binary_report(y[te], pred, score).as_dict())
                agg = {m: (float(np.mean([f[m] for f in per_fold])), float(np.std([f[m] for f in per_fold])))
                       for m in ("precision", "recall", "f1", "roc_auc", "average_precision")}
                tres[mname] = {"params": DEFAULTS[mname], "folds": per_fold, "mean_sd": agg}
                print(f"  {mname:<14} F1 {agg['f1'][0]:.3f} ± {agg['f1'][1]:.3f}   "
                      f"AUC {agg['roc_auc'][0]:.3f} ± {agg['roc_auc'][1]:.3f}")
            base = [reference_baselines(y[te], y[tr].mean(), args.seed) for tr, te in folds]
            tres["baselines"] = {b: {"f1_mean": float(np.mean([x[b].f1 for x in base]))} for b in ("majority", "random")}
            result["targets"][tname] = tres
            continue

        # ---- SEP28k-E: select on dev, optionally evaluate once on test
        part = df[args.partition].to_numpy()
        tr, dv, te = (np.flatnonzero(part == p) for p in ("train", "dev", "test"))
        check_no_leakage(speakers, part)
        tres["split"] = split_report(speakers, part, y).reset_index().to_dict("records")

        for mname, grid in GRIDS.items():
            search = []
            for params in grid:
                pred, score = fit_predict(mname, params, args.seed, X[tr], y[tr], X[dv])
                search.append((binary_report(y[dv], pred, score).f1, params, pred, score))
            best_f1, best, pred, score = max(search, key=lambda s: s[0])
            dev = binary_report(y[dv], pred, score).as_dict()
            dev["f1_ci"] = speaker_bootstrap_ci(y[dv], pred, speakers[dv], n_boot=args.n_boot, seed=args.seed)
            tres[mname] = {"selected_params": best, "dev": dev,
                           "search": [{"params": p, "dev_f1": f} for f, p, _, _ in search]}
            print(f"  {mname:<14} dev F1 {dev['f1']:.3f} [{dev['f1_ci'][0]:.3f}, {dev['f1_ci'][1]:.3f}]"
                  f"  AUC {dev['roc_auc']:.3f}  params {best}")
            save_confusion(dev, f"{mname}, dev, {tname}", out / f"cm_{tname}_{mname}_dev.png")

            if args.evaluate_test:
                pred_t, score_t = fit_predict(mname, best, args.seed, X[tr], y[tr], X[te])
                test = binary_report(y[te], pred_t, score_t).as_dict()
                test["f1_ci"] = speaker_bootstrap_ci(y[te], pred_t, speakers[te], n_boot=args.n_boot, seed=args.seed)
                tres[mname]["test"] = test
                print(f"  {mname:<14} TEST F1 {test['f1']:.3f} [{test['f1_ci'][0]:.3f}, {test['f1_ci'][1]:.3f}]"
                      f"  AUC {test['roc_auc']:.3f}")
                save_confusion(test, f"{mname}, test, {tname}", out / f"cm_{tname}_{mname}_test.png")

        tres["baselines_dev"] = {k: v.as_dict() for k, v in reference_baselines(y[dv], y[tr].mean(), args.seed).items()}
        if args.evaluate_test:
            tres["baselines_test"] = {k: v.as_dict() for k, v in reference_baselines(y[te], y[tr].mean(), args.seed).items()}
        result["targets"][tname] = tres

    if args.evaluate_test:
        # Honesty record: every look at the test set is logged permanently.
        with open(args.out / "TEST_SET_EVALUATIONS.log", "a", encoding="utf-8") as f:
            f.write(f"{result['date_utc']}  commit {result['git_commit']}  partition {args.partition}\n")
        n_looks = sum(1 for _ in open(args.out / "TEST_SET_EVALUATIONS.log", encoding="utf-8"))
        if n_looks > 1:
            print(f"\nWARNING: the test set has now been evaluated {n_looks} times. Report this.")
        result["test_evaluations_so_far"] = n_looks

    result["runtime_s"] = round(time.time() - t0, 1)
    (out / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_report(result, out / "report.md")
    print(f"\nSaved {out / 'report.md'} and metrics.json  ({result['runtime_s']} s)")


def write_report(r: dict, path: Path) -> None:
    L = [f"# EXP-001 baseline — {r['partition']}", "",
         f"Generated {r['date_utc']} · commit `{r['git_commit']}` · seed {r['seed']}", "",
         "Data: " + ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in r["data"].items()), ""]
    for tname, t in r["targets"].items():
        L += [f"## Target: any event, {tname.replace('_', ' ')}", ""]
        if r["partition"] == "SEP12k":
            L += ["Speaker-grouped CV, mean ± SD across folds.", "",
                  "| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |", "|---|---|---|---|---|---|"]
            for m in GRIDS:
                a = t[m]["mean_sd"]
                L.append(f"| {m} | " + " | ".join(f"{a[k][0]:.3f} ± {a[k][1]:.3f}" for k in
                                                   ("precision", "recall", "f1", "roc_auc", "average_precision")) + " |")
            for b, v in t["baselines"].items():
                L.append(f"| {b} baseline | | | {v['f1_mean']:.3f} | | |")
            L.append("")
            continue
        L += ["Split (achieved):", "", "| Split | Clips | Speakers | Positive rate |", "|---|---|---|---|"]
        L += [f"| {s['p']} | {s['rows']} | {s['groups']} | {s['positive_rate']:.3f} |" for s in t["split"]]
        for stage in ("dev", "test"):
            if stage == "test" and "baselines_test" not in t:
                continue
            L += ["", f"**{stage}** (F1 with 95 % speaker-bootstrap CI)", "",
                  "| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |", "|---|---|---|---|---|---|"]
            for m in GRIDS:
                L.append(f"| {m} {t[m]['selected_params']} " + fmt(t[m][stage]))
            for b, v in t[f"baselines_{stage}"].items():
                L.append(f"| {b} baseline " + fmt(v))
        L.append("")
    tested = any("baselines_test" in t for t in r["targets"].values())
    if not tested and r["partition"] != "SEP12k":
        L += ["*Test set not evaluated (development run).*"]
    L += ["", f"Environment: {r['environment']}", f"Runtime: {r['runtime_s']} s"]
    path.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
