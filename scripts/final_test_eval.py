"""EXP-004: the single, pre-registered evaluation on the SEP28k-E test set.

Refuses to run unless configs/final_eval.json is committed and unchanged, and
refuses to run a second time (the test set is used once). Every run is
appended to outputs/TEST_SET_EVALUATIONS.log.

    python scripts/final_test_eval.py
Output: outputs/EXP-004/report.md, metrics.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
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
from fluence.data.splits import check_no_leakage  # noqa: E402
from fluence.evaluation.metrics import (  # noqa: E402
    binary_report, paired_auc_difference, reference_baselines, speaker_bootstrap_auc_ci,
)
from fluence.features.sets import load_feature_set  # noqa: E402
from train_baseline import fit_predict, git_commit  # noqa: E402
from train_embeddings import targets  # noqa: E402

PLAIN = {"any_with_interjections": "Any event (incl. interjections)",
         "any_without_interjections": "Any event (excl. interjections)", "Prolongation": "Prolongation",
         "Block": "Block", "SoundRep": "Sound repetition", "WordRep": "Word repetition",
         "Interjection": "Interjection"}


def config_is_committed(path: Path) -> bool:
    path = path.resolve()
    d, name = str(path.parent), path.name
    tracked = subprocess.run(["git", "-C", d, "ls-files", "--error-unmatch", name], capture_output=True)
    dirty = subprocess.run(["git", "-C", d, "status", "--porcelain", "--", name],
                           capture_output=True, text=True)
    return tracked.returncode == 0 and dirty.returncode == 0 and not dirty.stdout.strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "configs/final_eval.json")
    ap.add_argument("--metadata", type=Path, default=ROOT / "data/metadata")
    ap.add_argument("--clips", type=Path, default=ROOT / "data/interim/sep28k_clips")
    ap.add_argument("--data-root", type=Path, default=ROOT, help="folder containing data/processed")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/EXP-004")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--allow-uncommitted-config", action="store_true", help="tests only")
    args = ap.parse_args()

    log = args.out.parent / "TEST_SET_EVALUATIONS.log"
    if (args.out / "metrics.json").exists():
        sys.exit(f"Refusing: {args.out / 'metrics.json'} exists. The test set is evaluated once.")
    if not args.allow_uncommitted_config and not config_is_committed(args.config):
        sys.exit(f"Refusing: {args.config} must be committed and unchanged before the test evaluation.")
    cfg = json.loads(args.config.read_text(encoding="utf-8"))

    df = load_available_clips(args.metadata, args.clips)
    df = df[df["SEP28k-E"].isin(["train", "dev", "test"])].reset_index(drop=True)
    part, spk = df["SEP28k-E"].to_numpy(), df["speaker"].astype(str).to_numpy()
    check_no_leakage(spk, part)
    tr, te = np.flatnonzero(part == "train"), np.flatnonzero(part == "test")
    T = targets(df)
    args.out.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with open(log, "a", encoding="utf-8") as f:
        f.write(f"{started}  EXP-004 final test evaluation  commit {git_commit()}  config {args.config.name}\n")

    res = {"experiment": "EXP-004", "started_utc": started, "git_commit": git_commit(), "config": cfg,
           "n_train": len(tr), "n_test": len(te), "speakers_test": int(len(np.unique(spk[te]))), "systems": {}}
    scores = {}
    for name, s in cfg["systems"].items():
        X = load_feature_set(s["features"], df, args.data_root)
        res["systems"][name] = {}
        scores[name] = {}
        for t in cfg["targets"]:
            _, score = fit_predict("logreg", {"C": s["C"][t]}, 0, X[tr], T[t][tr], X[te])
            scores[name][t] = score
            y = T[t][te]
            thr = s["threshold"][t]
            d = binary_report(y, (score >= thr).astype(int), score).as_dict()
            d["auc_ci"] = speaker_bootstrap_auc_ci(y, score, spk[te], n_boot=args.n_boot)
            d["threshold"] = thr
            d["f1_at_0.5"] = binary_report(y, (score >= 0.5).astype(int)).f1
            d["prevalence"] = float(y.mean())
            d["always_event_f1"] = reference_baselines(y, T[t][tr].mean())["majority"].f1
            res["systems"][name][t] = d
            print(f"  {name.split('/')[-1]:<22} {t:<26} AUC {d['roc_auc']:.3f} [{d['auc_ci'][0]:.3f}, "
                  f"{d['auc_ci'][1]:.3f}]  F1 {d['f1']:.3f}", flush=True)

    primary = cfg["primary_system"]
    res["paired"] = {}
    for other in cfg["systems"]:
        if other == primary:
            continue
        res["paired"][other] = {t: paired_auc_difference(T[t][te], scores[other][t], scores[primary][t], spk[te],
                                                         n_boot=args.n_boot) for t in cfg["targets"]}
    res["finished_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (args.out / "metrics.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    write_report(res, args.out / "report.md")
    print(f"\nSaved {args.out / 'report.md'}")


def write_report(r: dict, path: Path) -> None:
    cfg = r["config"]
    names = list(cfg["systems"])
    short = {n: n.split("/")[-1] for n in names}
    L = ["# EXP-004: SEP28k-E test set (single pre-registered evaluation)", "",
         f"Run {r['started_utc']} · commit `{r['git_commit']}` · config frozen {cfg['frozen_utc']} "
         f"(from `{cfg['frozen_from_commit']}`) · primary system **{short[cfg['primary_system']]}**", "",
         f"Trained on SEP28k-E train ({r['n_train']} clips, 3 speakers); evaluated on test "
         f"({r['n_test']} clips, {r['speakers_test']} unseen speakers).", "",
         "## ROC-AUC [95 % speaker-bootstrap CI]", "",
         "| Target | " + " | ".join(short[n] for n in names) + " |", "|---|" + "---|" * len(names)]
    for t in cfg["targets"]:
        L.append(f"| {PLAIN[t]} | " + " | ".join(
            f"{r['systems'][n][t]['roc_auc']:.3f} [{r['systems'][n][t]['auc_ci'][0]:.3f}, {r['systems'][n][t]['auc_ci'][1]:.3f}]"
            for n in names) + " |")
    p = cfg["primary_system"]
    L += ["", f"## Primary system ({short[p]}): detail", "",
          "| Target | Prevalence | PR-AUC | F1 @ frozen threshold | Precision | Recall | F1 @ 0.5 | Always-event F1 |",
          "|---|---|---|---|---|---|---|---|"]
    for t in cfg["targets"]:
        d = r["systems"][p][t]
        L.append(f"| {PLAIN[t]} | {d['prevalence']:.3f} | {d['average_precision']:.3f} | {d['f1']:.3f} (thr {d['threshold']:.2f}) | "
                 f"{d['precision']:.3f} | {d['recall']:.3f} | {d['f1_at_0.5']:.3f} | {d['always_event_f1']:.3f} |")
    for other, rows in r["paired"].items():
        L += ["", f"## Paired: {short[p]} minus {short[other]} (same test clips)", "",
              "| Target | Δ AUC [95 % CI] | p(primary ≤ other) |", "|---|---|---|"]
        for t in cfg["targets"]:
            d = rows[t]
            L.append(f"| {PLAIN[t]} | {d['diff']:+.3f} [{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}] | {d['p_b_not_better']:.3f} |")
    L += ["", "Not a clinical tool. Labels are majority votes of non-clinician annotators."]
    path.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
