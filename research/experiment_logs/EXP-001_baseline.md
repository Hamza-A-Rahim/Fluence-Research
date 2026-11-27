# EXP-001 — Clip-level baseline: does a 3-second clip contain a stuttering event?

| Field | Value |
|---|---|
| Date | 2026-10-21 |
| Status | complete (development); **test set not evaluated** |
| Git commit | `a9af275` (see D-013) |
| Run on | Windows 11 laptop, Python 3.12.10, numpy 2.5.4, scikit-learn 1.9.1 |
| Scripts | `scripts/train_baseline.py` (default and `--partition SEP12k`), figures: `scripts/plot_baseline.py` |

## Objective
Set an honest floor for binary event detection using simple, interpretable models (D-009), under the evaluation protocol.

## Hypothesis (written before running)
Clip-summary features carry some signal, so ROC-AUC will be clearly above 0.5. But averaging over 3 s discards the timing structure of repetitions and blocks, so performance will be modest. Training on many speakers (SEP-12k) should generalise better than training on 3 (SEP28k-E).

## Data
- SEP-28k labels @ Apple `fe954e1`, SEP-28k-Extended @ `5f599c5`; audio retrieved 2026-10-21 (EXP-000 addendum: 74 % of clips).
- Usable clips on disk: 19,516. Label = ≥2 of 3 annotators. Both targets reported (D-005).
- **SEP28k-E**: train 11,696 clips / **3 speakers**; dev 3,855 / 136; test 3,965 / 142. Leakage check passed.
- **SEP-12k CV (corrected, D-011)**: SEP-12k restricted to SEP28k-E dev clips — 3,855 clips, 136 speakers, 5 speaker-grouped folds (StratifiedGroupKFold, seed 0). *The first run used all 7,820 SEP-12k clips, which include the SEP28k-E test set; superseded.*

## Features and models
137 clip-summary features (mean/SD of 40 log-mel bands, 13 MFCCs, 13 delta-MFCCs, plus 5 energy/pause descriptors). Logistic regression (standardised, class-balanced; C tuned on dev from 0.01–10) and random forest (300 trees, balanced subsampling; min_samples_leaf tuned from 1/5/20). SEP-12k CV uses fixed defaults (C = 1, leaf = 5), with no tuning inside the folds.

## Results — any event, **including** interjections (positive rate ≈ 53 %)

| | SEP28k-E dev: F1 [95 % CI] | dev ROC-AUC | Dev-only CV: F1 | CV ROC-AUC | *Superseded CV AUC (incl. test)* |
|---|---|---|---|---|---|
| Logistic regression | 0.613 [0.576, 0.643] | **0.653** | 0.607 ± 0.022 | **0.658 ± 0.030** | *0.664 ± 0.027* |
| Random forest | 0.666 [0.634, 0.695] | 0.637 | 0.620 ± 0.047 | 0.635 ± 0.024 | *0.648 ± 0.020* |
| Always "event" | 0.697 | – | – | – | |
| Random at train rate | 0.550 | 0.482 | – | – | |

## Results — any event, **excluding** interjections (positive rate ≈ 38 %)

| | SEP28k-E dev: F1 [95 % CI] | dev ROC-AUC | Dev-only CV: F1 | CV ROC-AUC | *Superseded CV AUC (incl. test)* |
|---|---|---|---|---|---|
| Logistic regression | 0.529 [0.494, 0.562] | **0.632** | 0.515 ± 0.026 | **0.630 ± 0.014** | *0.622 ± 0.011* |
| Random forest | 0.462 [0.404, 0.522] | 0.621 | 0.389 ± 0.096 | 0.622 ± 0.022 | *0.614 ± 0.022* |
| Always "event" | 0.000 | – | – | – | |
| Random at train rate | 0.394 | 0.490 | – | – | |

CIs resample speakers. Reruns of the CV command gave identical numbers (deterministic). Corrected CV re-run 2026-10-21 in a cloud environment on the laptop's cached features.

## Interpretation
1. **There is real but weak signal.** ROC-AUC 0.62–0.66 in every setting, with speaker-bootstrap CIs and fold SDs well clear of 0.5.
2. **F1 alone would have been misleading.** With interjections included, no model beats the trivial always-"event" predictor on F1 (0.697), although both rank clips better than chance. This confirms the protocol's choice of ROC-AUC/PR-AUC as primary metrics for this target.
3. **The features are the bottleneck, not the number of training speakers.** Training on ~109 speakers per fold (dev-only CV) instead of 3 (SEP28k-E) leaves logistic-regression AUC essentially unchanged (0.658 vs 0.653). Hypothesis partly refuted: speaker diversity alone did not help here. (Same conclusion as the superseded run.)
4. **The random forest overfits speaker identity.** Without interjections, its dev F1 at min_samples_leaf = 1 was 0.18 (vs 0.46 at leaf = 20), and it is weakest in CV. A flexible model on summary features learns voices, not events.
5. The score distributions (figure `scores.png`) overlap almost entirely; the model separates the classes only at the tails.

## Limitations
- 3 training speakers on SEP28k-E, and 74 % of SEP-28k retrieved: **not directly comparable with published SEP-28k results.**
- Default 0.5 threshold; no threshold tuning.
- Single binary target; per-event performance is Stage 3.
- Logistic-regression weights on correlated features (adjacent mel bands) should not be interpreted individually.

## Next steps
- Stage 3: per-event (multi-label) detection with the same features, to see which event types carry the signal.
- Stage 4: representations that keep time structure: frame-level log-mel with a small CNN/LSTM, and frozen pretrained-encoder embeddings (e.g. wav2vec 2.0). Both must beat AUC ≈ 0.66 under this protocol.
- M1 listening check, including clips from the 10 length-mismatch episodes.
