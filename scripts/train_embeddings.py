"""Stage 4 (EXP-003): baseline protocol on wav2vec 2.0 embeddings.

1. Layer sweep: for each encoder layer, logistic regression on SEP28k-E train,
   "any event" ROC-AUC on dev. The best layer is chosen on dev only.
2. With that layer: binary (both targets) and per-event detection on SEP28k-E
   dev (speaker-bootstrap CIs) and SEP-12k speaker-grouped CV, with the same
   models, folds, labels and baselines as EXP-001/002.

    python scripts/train_embeddings.py --model facebook/wav2vec2-base-960h
    python scripts/train_embeddings.py --model facebook/wav2vec2-large-xlsr-53

Output: outputs/EXP-003/<model>/report.md, metrics.json, layer_sweep.png
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

from fluence.data.datasets import load_available_clips  # noqa: E402
from fluence.data.sep28k import EVENT_LABELS, any_stutter_target, binarize  # noqa: E402
from fluence.data.splits import check_no_leakage  # noqa: E402
from fluence.evaluation.metrics import (  # noqa: E402
    binary_report, reference_baselines, speaker_bootstrap_auc_ci,
)
from fluence.features.embeddings import load_embeddings, model_slug, read_manifest  # noqa: E402
from train_baseline import fit_predict, git_commit  # noqa: E402

PLAIN = {"Prolongation": "Prolongation", "Block": "Block", "SoundRep": "Sound repetition",
         "WordRep": "Word repetition", "Interjection": "Interjection"}
C_GRID = (0.001, 0.01, 0.1, 1.0)  # embeddings are high-dimensional: favour stronger regularisation


def targets(df) -> dict[str, np.ndarray]:
    t = {"any_with_interjections": any_stutter_target(df, include_interjections=True).astype(int),
         "any_without_interjections": any_stutter_target(df, include_interjections=False).astype(int)}
    Y = binarize(df, EVENT_LABELS).to_numpy()
    t.update({e: Y[:, j] for j, e in enumerate(EVENT_LABELS)})
    return t


def lr(C):
    return {"C": C}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="facebook/wav2vec2-base-960h")
    ap.add_argument("--metadata", type=Path, default=ROOT / "data/metadata")
    ap.add_argument("--clips", type=Path, default=ROOT / "data/interim/sep28k_clips")
    ap.add_argument("--emb", type=Path, default=ROOT / "data/processed/embeddings")
    ap.add_argument("--pooling", default="mean+std", choices=["mean", "std", "mean+std"])
    ap.add_argument("--layer", type=int, default=None, help="skip the sweep and use this layer")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/EXP-003")
    args = ap.parse_args()
    t0 = time.time()
    emb_dir = args.emb / model_slug(args.model)
    out = args.out / model_slug(args.model)
    out.mkdir(parents=True, exist_ok=True)

    all_df = load_available_clips(args.metadata, args.clips)
    X_all, found = load_embeddings(emb_dir, all_df["clip_key"], layer=None, pooling=args.pooling)
    n_missing = int((~found).sum())
    if n_missing:
        print(f"WARNING: {n_missing} clips have no embeddings yet; they are left out.")
    all_df, X_all = all_df[found].reset_index(drop=True), X_all[found]
    n_layers = X_all.shape[1]

    res = {"experiment": "EXP-003", "model": args.model, "manifest": read_manifest(emb_dir),
           "pooling": args.pooling, "date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "git_commit": git_commit(), "seed": args.seed, "clips_missing_embeddings": n_missing,
           "environment": {"python": platform.python_version(), "numpy": np.__version__,
                           "sklearn": sklearn.__version__, "platform": platform.platform()}}

    # ---------- SEP28k-E
    e_mask = all_df["SEP28k-E"].isin(["train", "dev", "test"]).to_numpy()
    df, XE = all_df[e_mask].reset_index(drop=True), X_all[e_mask]
    part, spk = df["SEP28k-E"].to_numpy(), df["speaker"].astype(str).to_numpy()
    check_no_leakage(spk, part)
    tr, dv = np.flatnonzero(part == "train"), np.flatnonzero(part == "dev")
    T = targets(df)

    # 1. layer sweep (dev, any event with interjections, C = 0.01)
    if args.layer is None:
        sweep = []
        y = T["any_with_interjections"]
        for L in range(n_layers):
            _, score = fit_predict("logreg", lr(0.01), args.seed, XE[tr, L], y[tr], XE[dv, L])
            sweep.append(binary_report(y[dv], score >= 0.5, score).roc_auc)
            print(f"  layer {L:>2}: dev AUC {sweep[-1]:.3f}", flush=True)
        best = int(np.argmax(sweep))
        res["layer_sweep_dev_auc"] = sweep
        layer_figure(sweep, best, args.model, out / "layer_sweep.png")
    else:
        best = args.layer
    res["layer"] = best
    print(f"Using layer {best}", flush=True)
    X = XE[:, best]

    # 2. SEP28k-E dev, C chosen per target on dev by F1 (same rule as EXP-001/002)
    res["SEP28k-E"] = {"n_train": len(tr), "n_dev": len(dv), "speakers_train": int(len(np.unique(spk[tr])))}
    print("\n=== SEP28k-E dev ===")
    for name, y in T.items():
        best_C = max(C_GRID, key=lambda C: binary_report(y[dv], *fit_predict("logreg", lr(C), args.seed,
                                                                             X[tr], y[tr], X[dv])).f1)
        pred, score = fit_predict("logreg", lr(best_C), args.seed, X[tr], y[tr], X[dv])
        d = binary_report(y[dv], pred, score).as_dict()
        d["auc_ci"] = speaker_bootstrap_auc_ci(y[dv], score, spk[dv], n_boot=args.n_boot, seed=args.seed)
        d["C"] = best_C
        d["baselines"] = {k: v.as_dict() for k, v in reference_baselines(y[dv], y[tr].mean(), args.seed).items()}
        res["SEP28k-E"][name] = d
        print(f"  {name:<26} AUC {d['roc_auc']:.3f} [{d['auc_ci'][0]:.3f}, {d['auc_ci'][1]:.3f}]  "
              f"F1 {d['f1']:.3f}  PR-AUC {d['average_precision']:.3f}", flush=True)

    # 3. SEP-12k CV (fixed C = 0.01: no tuning inside folds)
    m12 = (all_df["SEP12k"].astype(str) == "train").to_numpy()
    df12, X12 = all_df[m12].reset_index(drop=True), X_all[m12][:, best]
    spk12 = df12["speaker"].astype(str).to_numpy()
    T12 = targets(df12)
    res["SEP12k"] = {"n": len(df12), "speakers": int(len(np.unique(spk12)))}
    print(f"\n=== SEP-12k CV ({len(df12)} clips, {res['SEP12k']['speakers']} speakers) ===")
    for name, y in T12.items():
        folds = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed).split(X12, y, spk12)
        per = [binary_report(y[te], *fit_predict("logreg", lr(0.01), args.seed, X12[a], y[a], X12[te])).as_dict()
               for a, te in folds]
        agg = {m: (float(np.mean([f[m] for f in per])), float(np.std([f[m] for f in per])))
               for m in ("precision", "recall", "f1", "roc_auc", "average_precision")}
        res["SEP12k"][name] = agg
        print(f"  {name:<26} AUC {agg['roc_auc'][0]:.3f} ± {agg['roc_auc'][1]:.3f}  F1 {agg['f1'][0]:.3f}", flush=True)

    res["comparison"] = compare_with_baseline(res)
    res["runtime_s"] = round(time.time() - t0, 1)
    (out / "metrics.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    write_report(res, out / "report.md")
    print(f"\nSaved {out / 'report.md'}  ({res['runtime_s']} s)")


def compare_with_baseline(res: dict) -> dict:
    """Pull matching numbers from EXP-001/002 outputs, if they exist on this machine."""
    cmp = {}
    p1 = ROOT / "outputs/EXP-001/SEP28k-E/metrics.json"
    p2 = ROOT / "outputs/EXP-002/metrics.json"
    if p1.exists():
        r1 = json.loads(p1.read_text(encoding="utf-8"))["targets"]
        cmp["any_with_interjections"] = r1["with_interjections"]["logreg"]["dev"]["roc_auc"]
        cmp["any_without_interjections"] = r1["without_interjections"]["logreg"]["dev"]["roc_auc"]
    if p2.exists():
        r2 = json.loads(p2.read_text(encoding="utf-8"))["SEP28k-E"]
        cmp.update({e: r2[e]["logreg"]["dev"]["roc_auc"] for e in EVENT_LABELS if e in r2})
    return cmp


def layer_figure(sweep, best, model, path: Path) -> None:
    BLUE, INK2, GRID, SURF = "#2a78d6", "#52514e", "#e6e5e0", "#fcfcfb"
    fig, ax = plt.subplots(figsize=(8, 3.8), facecolor=SURF)
    ax.set_facecolor(SURF)
    ax.plot(range(len(sweep)), sweep, "-o", color=BLUE, linewidth=2, markersize=6,
            markeredgecolor=SURF, markeredgewidth=1.5)
    ax.plot(best, sweep[best], "o", color=BLUE, markersize=11, markeredgecolor=SURF, markeredgewidth=2)
    ax.annotate(f"best: layer {best}\nAUC {sweep[best]:.3f}", (best, sweep[best]), textcoords="offset points",
                xytext=(0, -34), ha="center", fontsize=9, color=INK2)
    ax.set_xlabel("Layer (0 = before the transformer)")
    ax.set_ylabel("Dev ROC-AUC, any event")
    ax.set_title(f"Which layer of {model.split('/')[-1]} hears stuttering best", loc="left", fontweight="bold")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(color=GRID)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_report(r: dict, path: Path) -> None:
    E, C = r["SEP28k-E"], r.get("comparison", {})
    L = [f"# EXP-003 — {r['model']} embeddings", "",
         f"Generated {r['date_utc']} · commit `{r['git_commit']}` · layer {r['layer']} (chosen on dev) · "
         f"pooling {r['pooling']} · model revision `{r['manifest'].get('revision')}`", "",
         f"SEP28k-E: train {E['n_train']} clips ({E['speakers_train']} speakers), dev {E['n_dev']}. "
         "Test set not evaluated.", "",
         "## SEP28k-E dev (logistic regression)", "",
         "| Target | ROC-AUC [95 % CI] | Baseline features AUC (EXP-001/002) | F1 | Always-event F1 | PR-AUC | Prevalence |",
         "|---|---|---|---|---|---|---|"]
    for name in ["any_with_interjections", "any_without_interjections"] + EVENT_LABELS:
        d = E[name]
        prev = d["positives"] / d["n"]
        base = f"{C[name]:.3f}" if name in C else "–"
        L.append(f"| {PLAIN.get(name, name.replace('_', ' '))} | {d['roc_auc']:.3f} [{d['auc_ci'][0]:.3f}, {d['auc_ci'][1]:.3f}] | "
                 f"{base} | {d['f1']:.3f} | {d['baselines']['majority']['f1']:.3f} | {d['average_precision']:.3f} | {prev:.3f} |")
    L += ["", f"## SEP-12k speaker-grouped CV ({r['SEP12k']['n']} clips, {r['SEP12k']['speakers']} speakers)", "",
          "| Target | ROC-AUC | F1 | PR-AUC |", "|---|---|---|---|"]
    for name in ["any_with_interjections", "any_without_interjections"] + EVENT_LABELS:
        a = r["SEP12k"][name]
        L.append(f"| {PLAIN.get(name, name.replace('_', ' '))} | {a['roc_auc'][0]:.3f} ± {a['roc_auc'][1]:.3f} | "
                 f"{a['f1'][0]:.3f} ± {a['f1'][1]:.3f} | {a['average_precision'][0]:.3f} ± {a['average_precision'][1]:.3f} |")
    if "layer_sweep_dev_auc" in r:
        L += ["", "Layer sweep (dev ROC-AUC, any event): " +
              ", ".join(f"{i}: {v:.3f}" for i, v in enumerate(r["layer_sweep_dev_auc"]))]
    L += ["", f"Environment: {r['environment']}", f"Runtime: {r['runtime_s']} s"]
    path.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
