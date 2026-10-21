# EXP-002 — Per-event detection with baseline features (Stage 3)

| Field | Value |
|---|---|
| Date | 2026-10-21 |
| Status | complete (development); **test set not evaluated** |
| Code | `scripts/train_multilabel.py` (committed with this log) |
| Run on | Cloud environment (Python 3.13, scikit-learn 1.9.1), using the **feature caches computed on the Windows laptop** for EXP-001 (`features_15de61ee2165.npz`, `features_c94ead6845ba.npz`). Row alignment verified by reproducing EXP-001's dev ROC-AUC exactly (0.653). |

## Objective
EXP-001 found weak overall signal (AUC ≈ 0.65). Which event types carry it?

## Hypothesis (written before running)
Repetitions, which have regular acoustic structure, and interjections (a distinctive filled-pause sound) will be easiest. Blocks, which are often silent and have low annotator agreement (κ 0.25), will be hardest. Prolongations, with the lowest agreement (κ 0.11), should also be hard.

## Setup
Same data, features (137 clip summaries), models and selection as EXP-001. One independent detector per event (a clip can have several events). Label = ≥2 of 3 annotators. SEP28k-E: settings chosen per event on dev. CV: 5 speaker-grouped folds over the SEP28k-E dev clips only (D-011; the first run also used the test clips and is superseded — old values shown in italics), defaults (C = 1, leaf = 5).

## Results — logistic regression (the stronger model throughout)

| Event | Human κ | Prevalence (dev) | SEP28k-E dev ROC-AUC [95 % speaker CI] | Dev PR-AUC | Dev-only CV ROC-AUC (D-011) |
|---|---|---|---|---|---|
| Prolongation | 0.11 | 9.9 % | **0.744** [0.714, 0.775] | 0.265 | 0.720 ± 0.033 *(was 0.749)* |
| Interjection | 0.57 | 21.8 % | 0.711 [0.672, 0.743] | 0.448 | 0.696 ± 0.033 *(was 0.716)* |
| Sound repetition | 0.40 | 9.5 % | 0.678 [0.636, 0.719] | 0.208 | 0.636 ± 0.050 *(was 0.646)* |
| Word repetition | 0.62 | 11.9 % | 0.641 [0.608, 0.669] | 0.188 | 0.628 ± 0.042 *(was 0.638)* |
| Block | 0.25 | 12.5 % | **0.608** [0.559, 0.651] | 0.198 | 0.570 ± 0.032 *(was 0.595)* |

(A random guess has ROC-AUC 0.5 and PR-AUC equal to prevalence.) Full tables, including random forest: `outputs/EXP-002/report.md`. Figure: `per_event_auc.png`.

Multi-label summary (logistic regression, dev): macro ROC-AUC 0.677, macro F1 0.310. Exact label-set match is 0.121, versus 0.465 for predicting "no events" everywhere. With class-balanced training and a 0.5 threshold, the models over-predict: 1.80 labels per clip predicted vs 0.66 true.

The random forest's F1 collapses to near zero in CV (0.004–0.076 after the D-011 correction) at the default threshold. It almost never predicts the rare positive class, although its AUC is similar to logistic regression's.

## Interpretation
1. **Hypothesis partly wrong: prolongations are the *easiest* event for these features**, despite the lowest human agreement. A plausible reason: a prolongation is a sustained, spectrally stable sound, and clip-level averages and variances capture "steadiness" well. Caution: it could also reflect a confound (e.g. speaking style or recording of particular speakers), which the speaker-grouped CV reduces but does not rule out. **To check by listening (M1).**
2. **Blocks are the hardest**, as predicted, and as reported for AS-70 and SEP-28k. Often silent, they are hard to tell apart from ordinary pauses using summary statistics.
3. **Word repetitions are surprisingly hard (≈ 0.64)** despite the highest human agreement. Repeating a whole word leaves the clip's average spectrum unchanged, so recognising repetition needs *timing*. This is strong motivation for Stage 4's time-aware models.
4. Results agree closely between SEP28k-E (3 training speakers) and dev-only CV (≈ 109 per fold), again suggesting the representation, not speaker count, limits performance. The event ranking (prolongation and interjection easiest, block hardest) is unchanged after the D-011 correction.
5. Thresholds matter: at 0.5 the balanced logistic regression over-predicts and the random forest under-predicts. Per-event thresholds tuned on dev would change F1 substantially (to do in Stage 4 comparisons).

## Limitations
Same as EXP-001 (3-speaker training set on SEP28k-E; 74 % of SEP-28k; not comparable to published numbers). κ values are from the original paper, not recomputed on our subset.

## Next steps
- ~~M1 listening check~~ done (`M1_listening_check.md`): prolongation labels audibly noisy in both directions; block labels sounded correct, so block weakness is the features.
- Stage 4: time-aware representations. Target to beat: macro ROC-AUC 0.677; word repetition 0.64 and block 0.61 are the most informative tests.
