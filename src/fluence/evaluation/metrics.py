"""Evaluation metrics following research/methodology/evaluation_protocol.md.

Key points:
* Metrics are for the *positive* class (the event), never plain accuracy.
* Every model is reported next to two reference baselines: majority class and
  random guessing at the training positive rate.
* Confidence intervals resample **speakers**, not clips, because clips from
  the same speaker are not independent.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


@dataclass
class BinaryReport:
    n: int
    positives: int
    tp: int
    fp: int
    fn: int
    tn: int
    precision: float
    recall: float
    f1: float
    roc_auc: float | None = None
    average_precision: float | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def binary_report(y_true, y_pred, y_score=None) -> BinaryReport:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    p = _safe_div(tp, tp + fp)
    r = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * p * r, p + r)
    auc = ap = None
    if y_score is not None and 0 < y_true.sum() < len(y_true):
        auc = float(roc_auc_score(y_true, y_score))
        ap = float(average_precision_score(y_true, y_score))
    return BinaryReport(len(y_true), int(y_true.sum()), tp, fp, fn, tn, p, r, f1, auc, ap)


def reference_baselines(y_true, train_positive_rate: float, seed: int = 0) -> dict[str, BinaryReport]:
    """Majority-class and random (at the training positive rate) predictions."""
    y_true = np.asarray(y_true).astype(int)
    majority = int(train_positive_rate >= 0.5)
    rng = np.random.default_rng(seed)
    rand_score = rng.random(len(y_true))
    return {
        "majority": binary_report(y_true, np.full(len(y_true), majority)),
        "random": binary_report(y_true, (rand_score < train_positive_rate).astype(int), rand_score),
    }


def speaker_bootstrap_auc_ci(y_true, y_score, speakers, n_boot: int = 1000, seed: int = 0,
                             alpha: float = 0.05) -> tuple[float, float]:
    """Percentile CI for ROC-AUC, resampling speakers; skips resamples with one class."""
    y_true, y_score = np.asarray(y_true).astype(int), np.asarray(y_score, float)
    speakers = np.asarray(speakers).astype(str)
    uniq = np.unique(speakers)
    idx_by_spk = {s: np.flatnonzero(speakers == s) for s in uniq}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([idx_by_spk[s] for s in rng.choice(uniq, size=len(uniq), replace=True)])
        if 0 < y_true[idx].sum() < len(idx):
            vals.append(roc_auc_score(y_true[idx], y_score[idx]))
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def paired_auc_difference(y_true, score_a, score_b, speakers, n_boot: int = 2000, seed: int = 0,
                          alpha: float = 0.05) -> dict:
    """AUC(b) - AUC(a) on the same clips, with a speaker-level paired bootstrap.

    Each resample draws speakers with replacement and recomputes both AUCs on
    the same clips, so shared difficulty cancels out. Returns the observed
    difference, its percentile CI, and the share of resamples where b <= a
    (a one-sided bootstrap p-value for "b is better").
    """
    y = np.asarray(y_true).astype(int)
    a, b = np.asarray(score_a, float), np.asarray(score_b, float)
    speakers = np.asarray(speakers).astype(str)
    uniq = np.unique(speakers)
    idx_by_spk = {s: np.flatnonzero(speakers == s) for s in uniq}
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = np.concatenate([idx_by_spk[s] for s in rng.choice(uniq, size=len(uniq), replace=True)])
        if 0 < y[idx].sum() < len(idx):
            diffs.append(roc_auc_score(y[idx], b[idx]) - roc_auc_score(y[idx], a[idx]))
    diffs = np.array(diffs)
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return {"auc_a": float(roc_auc_score(y, a)), "auc_b": float(roc_auc_score(y, b)),
            "diff": float(roc_auc_score(y, b) - roc_auc_score(y, a)),
            "ci": (float(lo), float(hi)), "p_b_not_better": float(np.mean(diffs <= 0)),
            "n_boot_used": int(len(diffs))}


def best_f1_threshold(y_true, y_score) -> float:
    """Threshold maximising F1 (choose on dev only, then freeze)."""
    from sklearn.metrics import precision_recall_curve
    p, r, t = precision_recall_curve(np.asarray(y_true).astype(int), y_score)
    f1 = 2 * p[:-1] * r[:-1] / np.maximum(p[:-1] + r[:-1], 1e-12)
    return float(t[int(np.argmax(f1))])


def speaker_bootstrap_ci(y_true, y_pred, speakers, metric: str = "f1",
                         n_boot: int = 1000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile CI for a metric, resampling speakers with replacement."""
    y_true, y_pred = np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int)
    speakers = np.asarray(speakers).astype(str)
    uniq = np.unique(speakers)
    idx_by_spk = {s: np.flatnonzero(speakers == s) for s in uniq}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_spk[s] for s in pick])
        vals.append(getattr(binary_report(y_true[idx], y_pred[idx]), metric))
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)
