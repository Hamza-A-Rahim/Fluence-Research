"""EXP-005: does the detector still work when windows are not aligned to the labelled clips?

Uses the frozen primary system from configs/final_eval.json (same encoder,
layer, C), trained on SEP28k-E train clips, and the windows produced by
scripts/extract_windows.py. Dev clips only, from episodes with no test clips.

    python scripts/localize.py
Output: outputs/EXP-005/report.md, metrics.json, offset_auc.png,
        score_profile.png, timeline_<episode>.png
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

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from fluence.data.datasets import load_available_clips  # noqa: E402
from fluence.evaluation.metrics import speaker_bootstrap_auc_ci  # noqa: E402
from fluence.features.embeddings import model_slug  # noqa: E402
from fluence.features.sets import load_feature_set  # noqa: E402
from fluence.localization import aggregate_over_offsets  # noqa: E402
from train_baseline import git_commit, make_model  # noqa: E402
from train_embeddings import targets  # noqa: E402

PLAIN = {"any_with_interjections": "Any event (incl. interjections)",
         "any_without_interjections": "Any event (excl. interjections)", "Prolongation": "Prolongation",
         "Block": "Block", "SoundRep": "Sound repetition", "WordRep": "Word repetition",
         "Interjection": "Interjection"}
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]


def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(color=GRID)


def load_windows(win_dir: Path):
    parts = [np.load(p) for p in sorted(win_dir.glob("offsets_*.npz"))]
    if not parts:
        sys.exit(f"No offset windows in {win_dir}. Run scripts/extract_windows.py first.")
    keys = np.concatenate([p["clip_key"] for p in parts])
    offs = np.concatenate([p["offset"] for p in parts]).astype(float)
    X = np.concatenate([np.concatenate([p["mean"], p["std"]], axis=1) for p in parts]).astype(np.float32)
    return keys, offs, X


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "configs/final_eval.json")
    ap.add_argument("--metadata", type=Path, default=ROOT / "data/metadata")
    ap.add_argument("--clips", type=Path, default=ROOT / "data/interim/sep28k_clips")
    ap.add_argument("--data-root", type=Path, default=ROOT)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/EXP-005")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    sysname = cfg["primary_system"]
    s = cfg["systems"][sysname]
    win_dir = args.data_root / "data/processed/windows" / model_slug(s["features"]["model"])

    # ---- train the frozen classifiers on SEP28k-E train clips (test rows never loaded)
    df = load_available_clips(args.metadata, args.clips)
    df = df[df["SEP28k-E"].isin(["train", "dev"])].reset_index(drop=True)
    Xc = load_feature_set(s["features"], df, args.data_root)
    T = targets(df)
    tr = np.flatnonzero(df["SEP28k-E"].to_numpy() == "train")
    models = {t: make_model("logreg", {"C": s["C"][t]}, 0).fit(Xc[tr], T[t][tr]) for t in cfg["targets"]}

    keys, offs, Xw = load_windows(win_dir)
    lab = df.set_index("clip_key")
    if not set(keys) <= set(lab.index[lab["SEP28k-E"] == "dev"]):
        sys.exit("Windows include non-dev clips; refusing.")
    W = pd.DataFrame({"clip_key": keys, "offset": offs})
    for t, m in models.items():
        W[t] = m.predict_proba(Xw)[:, 1]
    clip_keys = sorted(set(keys))
    dev = lab.loc[clip_keys]
    spk = dev["speaker"].astype(str)
    Y = {t: pd.Series(targets(dev.reset_index())[t], index=dev.index) for t in cfg["targets"]}

    # sanity: offset-0 windows vs the original clip embeddings
    dv_idx = df.index[df["clip_key"].isin(clip_keys)]
    clip_scores = {t: pd.Series(models[t].predict_proba(Xc[dv_idx])[:, 1], index=df.loc[dv_idx, "clip_key"])
                   for t in cfg["targets"]}
    w0 = W[W["offset"] == 0].set_index("clip_key")
    common = w0.index.intersection(clip_scores["any_with_interjections"].index)
    sanity = {t: float(np.corrcoef(w0.loc[common, t], clip_scores[t].loc[common])[0, 1]) for t in cfg["targets"]}
    print("Sanity, corr(offset-0 window score, clip score): " + ", ".join(f"{t} {v:.4f}" for t, v in sanity.items()),
          flush=True)

    offsets = sorted(W["offset"].unique())
    res = {"experiment": "EXP-005", "date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "git_commit": git_commit(), "system": sysname, "layer": s["features"]["layer"],
           "n_clips": len(clip_keys), "n_windows": len(W), "n_speakers": int(spk.nunique()),
           "n_episodes": int(dev.groupby(["Show", "EpId"]).ngroups), "offsets_s": offsets,
           "sanity_corr_offset0_vs_clip": sanity, "targets": {}}
    for t in cfg["targets"]:
        y = Y[t]
        r = {"prevalence": float(y.mean()), "clip_auc": float(roc_auc_score(y.loc[common], clip_scores[t].loc[common])),
             "offset_auc": {}, "profile_pos": {}, "profile_neg": {}}
        for o in offsets:
            w = W[W["offset"] == o].set_index("clip_key")
            yy = y.loc[w.index]
            r["offset_auc"][str(o)] = {"auc": float(roc_auc_score(yy, w[t])), "n": int(len(w))}
            r["profile_pos"][str(o)] = float(w[t][yy == 1].mean())
            r["profile_neg"][str(o)] = float(w[t][yy == 0].mean())
        for how in ("max", "mean"):
            agg = aggregate_over_offsets(W.rename(columns={t: "score"})[["clip_key", "offset", "score"]], how)
            yy = y.loc[agg.index]
            r[f"agg_{how}"] = {"auc": float(roc_auc_score(yy, agg)),
                               "ci": speaker_bootstrap_auc_ci(yy, agg, spk.loc[agg.index], n_boot=args.n_boot)}
        pos = W[W["clip_key"].isin(y.index[y == 1])]
        best = pos.loc[pos.groupby("clip_key")[t].idxmax(), "offset"]
        r["argmax_offset_share"] = {str(o): float((best == o).mean()) for o in offsets}
        res["targets"][t] = r
        print(f"  {t:<26} clip {r['clip_auc']:.3f} | offset 0 {r['offset_auc']['0.0']['auc']:.3f} | "
              f"±1.5 s {r['offset_auc'][str(offsets[0])]['auc']:.3f}/{r['offset_auc'][str(offsets[-1])]['auc']:.3f} | "
              f"sliding max {r['agg_max']['auc']:.3f} mean {r['agg_mean']['auc']:.3f}", flush=True)

    figures(res, offsets, args.out)
    res["timelines"] = timelines(models, win_dir, args.metadata, df, args.out)
    (args.out / "metrics.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    write_report(res, args.out / "report.md")
    print(f"Saved {args.out / 'report.md'}")


def figures(res, offsets, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.6), facecolor=SURF)
    style(ax)
    for (t, r), col in zip(res["targets"].items(), PALETTE):
        ys = [r["offset_auc"][str(o)]["auc"] for o in offsets]
        ax.plot(offsets, ys, "-o", color=col, linewidth=2, markersize=5, markeredgecolor=SURF, label=PLAIN[t])
    ax.axvline(0, color=INK2, linewidth=1, linestyle=":")
    ax.set_xlabel("Window start relative to the labelled clip (seconds)")
    ax.set_ylabel("ROC-AUC (dev clips)")
    ax.set_title("How much does detection depend on lining up with the clip?", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=7.5, ncol=2, loc="lower center")
    fig.text(0.01, -0.03, "A 3 s window shifted by ±1.5 s still overlaps the labelled clip by 1.5 s.", fontsize=8, color=INK2)
    fig.savefig(out / "offset_auc.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), facecolor=SURF, sharey=True)
    for ax, t in zip(axes, ["any_without_interjections", "WordRep", "Block"]):
        style(ax)
        r = res["targets"][t]
        ax.plot(offsets, [r["profile_pos"][str(o)] for o in offsets], "-o", color=PALETTE[0], label="event clips")
        ax.plot(offsets, [r["profile_neg"][str(o)] for o in offsets], "-o", color="#a3a29c", label="no-event clips")
        ax.axvline(0, color=INK2, linewidth=1, linestyle=":")
        ax.set_title(PLAIN[t], fontsize=10, loc="left")
        ax.set_xlabel("Offset (s)")
    axes[0].set_ylabel("Mean model score")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Mean model score by window offset", x=0.01, ha="left", fontweight="bold")
    fig.savefig(out / "score_profile.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def timelines(models, win_dir: Path, metadata: Path, df: pd.DataFrame, out: Path) -> list:
    made = []
    for p in sorted(win_dir.glob("timeline_*.npz")):
        z = np.load(p)
        X = np.concatenate([z["mean"], z["std"]], axis=1).astype(np.float32)
        t0 = z["start_s"].astype(float) + 1.5  # centre of each window
        show, ep = p.stem.replace("timeline_", "").rsplit("_", 1)
        lab = df[(df["Show"] == show) & (df["EpId"] == int(ep))]
        T = targets(lab.reset_index(drop=True))
        fig, axes = plt.subplots(2, 1, figsize=(12, 5), facecolor=SURF, sharex=True,
                                 gridspec_kw={"height_ratios": [2, 1]})
        ax = axes[0]
        style(ax)
        for t, col in [("any_without_interjections", PALETTE[0]), ("Interjection", PALETTE[1])]:
            ax.plot(t0 / 60, models[t].predict_proba(X)[:, 1], color=col, linewidth=0.9, label=PLAIN[t])
        ax.set_ylabel("Model score")
        ax.set_ylim(0, 1)
        ax.legend(frameon=False, fontsize=8, loc="upper right")
        ax.set_title(f"{show} episode {ep}: sliding 3 s window every {z['start_s'][1] - z['start_s'][0]:.1f} s",
                     loc="left", fontweight="bold")
        ax2 = axes[1]
        style(ax2)
        mid = (lab["Start"] + lab["Stop"]).to_numpy() / 2 / 16000 / 60
        y_any = T["any_without_interjections"]
        y_int = T["Interjection"]
        ax2.scatter(mid[y_any == 1], np.full((y_any == 1).sum(), 2), marker="|", s=120, color=PALETTE[0])
        ax2.scatter(mid[y_int == 1], np.full((y_int == 1).sum(), 1), marker="|", s=120, color=PALETTE[1])
        neither = (y_any == 0) & (y_int == 0)
        ax2.scatter(mid[neither], np.zeros(neither.sum()), marker="|", s=120, color="#a3a29c")
        ax2.set_yticks([0, 1, 2], ["labelled: no event", "labelled: interjection", "labelled: stutter event"], fontsize=8)
        ax2.set_ylim(-0.6, 2.6)
        ax2.set_xlabel("Time in episode (minutes)")
        fig.text(0.01, -0.02, "Bottom: annotated 3 s clips (only a sample of each episode was annotated). "
                 "Includes clips the model was trained on if the episode's host is a training speaker.",
                 fontsize=8, color=INK2)
        f = out / f"timeline_{show}_{ep}.png"
        fig.savefig(f, dpi=140, bbox_inches="tight")
        plt.close(fig)
        made.append({"episode": f"{show} {ep}", "windows": int(len(X)), "minutes": float(t0[-1] / 60),
                     "labelled_clips": int(len(lab)), "figure": f.name})
    return made


def write_report(r, path: Path) -> None:
    off = r["offsets_s"]
    L = [f"# EXP-005: Offset robustness and sliding-window detection", "",
         f"Generated {r['date_utc']} · commit `{r['git_commit']}` · system {r['system']} layer {r['layer']}", "",
         f"{r['n_clips']} dev clips, {r['n_speakers']} speakers, {r['n_episodes']} episodes with no test clips; "
         f"{r['n_windows']} windows. Test set not used.", "",
         "Sanity check, correlation of offset-0 window scores with the original clip scores: "
         + ", ".join(f"{PLAIN[t]} {v:.4f}" for t, v in r["sanity_corr_offset0_vs_clip"].items()), "",
         "## ROC-AUC by window offset", "",
         "| Target | Clip | " + " | ".join(f"{o:+.1f} s" for o in off) + " | Sliding max [95 % CI] | Sliding mean [95 % CI] |",
         "|---|---|" + "---|" * len(off) + "---|---|"]
    for t, x in r["targets"].items():
        L.append(f"| {PLAIN[t]} | {x['clip_auc']:.3f} | " + " | ".join(f"{x['offset_auc'][str(o)]['auc']:.3f}" for o in off)
                 + f" | {x['agg_max']['auc']:.3f} [{x['agg_max']['ci'][0]:.3f}, {x['agg_max']['ci'][1]:.3f}]"
                 + f" | {x['agg_mean']['auc']:.3f} [{x['agg_mean']['ci'][0]:.3f}, {x['agg_mean']['ci'][1]:.3f}] |")
    L += ["", "## Where do event clips score highest? (share of positive clips by best offset)", "",
          "| Target | " + " | ".join(f"{o:+.1f} s" for o in off) + " |", "|---|" + "---|" * len(off)]
    for t, x in r["targets"].items():
        L.append(f"| {PLAIN[t]} | " + " | ".join(f"{x['argmax_offset_share'][str(o)]:.2f}" for o in off) + " |")
    if r.get("timelines"):
        L += ["", "## Episode timelines", ""] + [f"- {d['episode']}: {d['minutes']:.1f} min, {d['windows']} windows, "
                                                f"{d['labelled_clips']} labelled clips → `{d['figure']}`" for d in r["timelines"]]
    path.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
