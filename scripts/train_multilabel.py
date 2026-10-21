"""Stage 3: which stuttering event types can the baseline features detect?

One detector per event type (prolongation, block, sound repetition, word
repetition, interjection). A clip can be positive for several at once, so these
are independent yes/no models, not one pick-a-single-class model.

Same features, models and protocol as EXP-001:
* SEP28k-E: train on train, select settings per event on dev (test untouched)
* SEP-12k: speaker-grouped 5-fold CV with default settings

Usage:
    python scripts/train_multilabel.py
Outputs: outputs/EXP-002/report.md, metrics.json, per_event_auc.png
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Windows consoles and pipes may default to a legacy code page (cp1252) that
# cannot print characters like κ or ±; never crash on output.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import sklearn  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402

from fluence.data.clips import compute_features  # noqa: E402
from fluence.data.datasets import cv_pool, load_available_clips  # noqa: E402
from fluence.data.sep28k import EVENT_LABELS, binarize  # noqa: E402
from fluence.data.splits import check_no_leakage  # noqa: E402
from fluence.evaluation.metrics import binary_report, speaker_bootstrap_auc_ci  # noqa: E402
from fluence.features.spectral import FeatureConfig  # noqa: E402
from train_baseline import DEFAULTS, GRIDS, fit_predict, git_commit  # noqa: E402

# Fleiss' kappa per label from Lea et al. (2021), SEP-28k paper: how much the
# human annotators agreed. A ceiling on how "learnable" each label is.
KAPPA = {"WordRep": 0.62, "Interjection": 0.57, "SoundRep": 0.40, "Block": 0.25, "Prolongation": 0.11}
PLAIN = {"Prolongation": "Prolongation", "Block": "Block", "SoundRep": "Sound repetition",
         "WordRep": "Word repetition", "Interjection": "Interjection"}


def per_event_figure(res: dict, out: Path) -> None:
    BLUE, ORANGE, INK2, GRID, SURF = "#2a78d6", "#eb6834", "#52514e", "#e6e5e0", "#fcfcfb"
    order = sorted(EVENT_LABELS, key=lambda e: res["SEP28k-E"][e]["logreg"]["dev"]["roc_auc"])
    fig, ax = plt.subplots(figsize=(8.5, 4.6), facecolor=SURF)
    ax.set_facecolor(SURF)
    for i, e in enumerate(order):
        d = res["SEP28k-E"][e]["logreg"]["dev"]
        lo, hi = d["auc_ci"]
        ax.plot([lo, hi], [i + 0.13] * 2, color=BLUE, linewidth=2, solid_capstyle="round")
        ax.plot(d["roc_auc"], i + 0.13, "o", color=BLUE, markersize=8, markeredgecolor=SURF, markeredgewidth=2)
        if "SEP12k" in res:
            m, sd = res["SEP12k"][e]["logreg"]["mean_sd"]["roc_auc"]
            ax.plot([m - sd, m + sd], [i - 0.13] * 2, color=ORANGE, linewidth=2, solid_capstyle="round")
            ax.plot(m, i - 0.13, "o", color=ORANGE, markersize=8, markeredgecolor=SURF, markeredgewidth=2)
    ax.axvline(0.5, color=INK2, linestyle="--", linewidth=1)
    ax.text(0.503, -0.42, "chance", color=INK2, fontsize=8)
    ax.set_yticks(range(len(order)), [f"{PLAIN[e]}\nhuman agreement κ = {KAPPA[e]:.2f}" for e in order], fontsize=9)
    ax.set_xlabel("ROC-AUC (logistic regression)")
    ax.set_title("Which stutter types the baseline can detect", loc="left", fontweight="bold")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color=GRID)
    ax.set_xlim(0.45, max(0.85, ax.get_xlim()[1]))
    ax.plot([], [], "o-", color=BLUE, label="SEP28k-E dev (95% speaker CI)")
    if "SEP12k" in res:
        ax.plot([], [], "o-", color=ORANGE, label="SEP-12k CV (mean ± SD)")
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metadata", type=Path, default=ROOT / "data/metadata")
    ap.add_argument("--clips", type=Path, default=ROOT / "data/interim/sep28k_clips")
    ap.add_argument("--cache", type=Path, default=ROOT / "data/processed")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/EXP-002")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=500)
    args = ap.parse_args()
    t0 = time.time()
    args.out.mkdir(parents=True, exist_ok=True)

    all_df = load_available_clips(args.metadata, args.clips)
    if all_df.empty:
        sys.exit("No clips found. Run scripts/fetch_sep28k.py first.")
    res: dict = {"experiment": "EXP-002", "date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "git_commit": git_commit(), "seed": args.seed,
                 "environment": {"python": platform.python_version(), "numpy": np.__version__,
                                 "sklearn": sklearn.__version__, "platform": platform.platform()}}

    # ---------------- SEP28k-E
    df = all_df[all_df["SEP28k-E"].isin(["train", "dev", "test"])]
    print("Features (cached)...")
    X, names, ok = compute_features(df["path"].tolist(), FeatureConfig(), cache_dir=args.cache, progress=True)
    df, X = df[ok].reset_index(drop=True), X[ok]
    Y = binarize(df, EVENT_LABELS).to_numpy()
    part, spk = df["SEP28k-E"].to_numpy(), df["speaker"].astype(str).to_numpy()
    check_no_leakage(spk, part)
    tr, dv = np.flatnonzero(part == "train"), np.flatnonzero(part == "dev")
    res["SEP28k-E"] = {"n_train": len(tr), "n_dev": len(dv)}
    dev_pred = np.zeros((len(dv), len(EVENT_LABELS)), int)
    dev_score = np.zeros((len(dv), len(EVENT_LABELS)))
    print("\n=== SEP28k-E dev (logistic regression | random forest) ===")
    for j, e in enumerate(EVENT_LABELS):
        y = Y[:, j]
        er = {"prevalence_train": float(y[tr].mean()), "prevalence_dev": float(y[dv].mean())}
        for mname, grid in GRIDS.items():
            best = max(((binary_report(y[dv], *fit_predict(mname, p, args.seed, X[tr], y[tr], X[dv])).f1, p)
                        for p in grid), key=lambda t: t[0])[1]
            pred, score = fit_predict(mname, best, args.seed, X[tr], y[tr], X[dv])
            d = binary_report(y[dv], pred, score).as_dict()
            d["auc_ci"] = speaker_bootstrap_auc_ci(y[dv], score, spk[dv], n_boot=args.n_boot, seed=args.seed)
            er[mname] = {"selected_params": best, "dev": d}
            if mname == "logreg":
                dev_pred[:, j], dev_score[:, j] = pred, score
        res["SEP28k-E"][e] = er
        lr, rf = er["logreg"]["dev"], er["random_forest"]["dev"]
        print(f"  {PLAIN[e]:<17} prev {er['prevalence_dev']:.3f}  AUC {lr['roc_auc']:.3f} "
              f"[{lr['auc_ci'][0]:.3f}, {lr['auc_ci'][1]:.3f}]  F1 {lr['f1']:.3f} | "
              f"AUC {rf['roc_auc']:.3f}  F1 {rf['f1']:.3f}   (κ {KAPPA[e]:.2f})")

    Yd = Y[dv]
    res["SEP28k-E"]["multilabel_logreg"] = {
        "macro_f1": float(np.mean([binary_report(Yd[:, j], dev_pred[:, j]).f1 for j in range(len(EVENT_LABELS))])),
        "macro_auc": float(np.mean([res["SEP28k-E"][e]["logreg"]["dev"]["roc_auc"] for e in EVENT_LABELS])),
        "exact_set_match": float((dev_pred == Yd).all(axis=1).mean()),
        "exact_set_match_all_negative_baseline": float((Yd == 0).all(axis=1).mean()),
        "mean_true_labels_per_clip": float(Yd.sum(1).mean()),
        "mean_predicted_labels_per_clip": float(dev_pred.sum(1).mean()),
    }

    # ---------------- SEP-12k CV
    df12 = cv_pool(all_df)  # D-011: test clips excluded
    X12, _, ok12 = compute_features(df12["path"].tolist(), FeatureConfig(), cache_dir=args.cache, progress=True)
    df12, X12 = df12[ok12].reset_index(drop=True), X12[ok12]
    Y12 = binarize(df12, EVENT_LABELS).to_numpy()
    spk12 = df12["speaker"].astype(str).to_numpy()
    res["SEP12k"] = {"n": len(df12), "speakers": int(len(np.unique(spk12)))}
    print(f"\n=== SEP-12k {args.folds}-fold speaker-grouped CV (logistic regression | random forest) ===")
    for j, e in enumerate(EVENT_LABELS):
        y = Y12[:, j]
        folds = list(StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
                     .split(X12, y, groups=spk12))
        er = {"prevalence": float(y.mean())}
        for mname in GRIDS:
            per = [binary_report(y[te], *fit_predict(mname, DEFAULTS[mname], args.seed, X12[tr_], y[tr_], X12[te])).as_dict()
                   for tr_, te in folds]
            er[mname] = {"mean_sd": {m: (float(np.mean([f[m] for f in per])), float(np.std([f[m] for f in per])))
                                     for m in ("precision", "recall", "f1", "roc_auc", "average_precision")}}
        res["SEP12k"][e] = er
        a, b = er["logreg"]["mean_sd"], er["random_forest"]["mean_sd"]
        print(f"  {PLAIN[e]:<17} prev {er['prevalence']:.3f}  AUC {a['roc_auc'][0]:.3f} ± {a['roc_auc'][1]:.3f}  "
              f"F1 {a['f1'][0]:.3f} | AUC {b['roc_auc'][0]:.3f}  F1 {b['f1'][0]:.3f}")

    res["runtime_s"] = round(time.time() - t0, 1)
    (args.out / "metrics.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    per_event_figure(res, args.out / "per_event_auc.png")
    write_report(res, args.out / "report.md")
    print(f"\nSaved {args.out / 'report.md'}, metrics.json, per_event_auc.png  ({res['runtime_s']} s)")


def write_report(r: dict, path: Path) -> None:
    E = r["SEP28k-E"]
    L = ["# EXP-002 per-event detection", "",
         f"Generated {r['date_utc']} · commit `{r['git_commit']}` · seed {r['seed']}", "",
         f"SEP28k-E: train {E['n_train']} clips, dev {E['n_dev']} clips. Test set not evaluated.", "",
         "## SEP28k-E dev", "",
         "PR-AUC of a random guess equals prevalence, so compare PR-AUC to the prevalence column.", "",
         "| Event | κ (humans) | Prevalence | Model | Precision | Recall | F1 | ROC-AUC [95 % CI] | PR-AUC |",
         "|---|---|---|---|---|---|---|---|---|"]
    for e in EVENT_LABELS:
        for m in GRIDS:
            d = E[e][m]["dev"]
            L.append(f"| {PLAIN[e]} | {KAPPA[e]:.2f} | {E[e]['prevalence_dev']:.3f} | {m} {E[e][m]['selected_params']} | "
                     f"{d['precision']:.3f} | {d['recall']:.3f} | {d['f1']:.3f} | "
                     f"{d['roc_auc']:.3f} [{d['auc_ci'][0]:.3f}, {d['auc_ci'][1]:.3f}] | {d['average_precision']:.3f} |")
    ml = E["multilabel_logreg"]
    L += ["", "**Multi-label summary (logistic regression, dev)**", "",
          f"- Macro F1 {ml['macro_f1']:.3f}, macro ROC-AUC {ml['macro_auc']:.3f}",
          f"- Exact label-set match {ml['exact_set_match']:.3f} (predicting 'no events' for every clip: "
          f"{ml['exact_set_match_all_negative_baseline']:.3f})",
          f"- Labels per clip: true {ml['mean_true_labels_per_clip']:.2f}, predicted {ml['mean_predicted_labels_per_clip']:.2f}",
          "", f"## SEP-12k dev-only speaker-grouped CV, test excluded (D-011) ({r['SEP12k']['n']} clips, {r['SEP12k']['speakers']} speakers)", "",
          "| Event | Prevalence | Model | F1 | ROC-AUC | PR-AUC |", "|---|---|---|---|---|---|"]
    for e in EVENT_LABELS:
        for m in GRIDS:
            a = r["SEP12k"][e][m]["mean_sd"]
            L.append(f"| {PLAIN[e]} | {r['SEP12k'][e]['prevalence']:.3f} | {m} | {a['f1'][0]:.3f} ± {a['f1'][1]:.3f} | "
                     f"{a['roc_auc'][0]:.3f} ± {a['roc_auc'][1]:.3f} | {a['average_precision'][0]:.3f} ± {a['average_precision'][1]:.3f} |")
    L += ["", f"Environment: {r['environment']}", f"Runtime: {r['runtime_s']} s"]
    path.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
