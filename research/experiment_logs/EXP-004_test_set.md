# EXP-004: Single pre-registered evaluation on the SEP28k-E test set

| Field | Value |
|---|---|
| Date | 2026-10-22 (run 05:13 UTC) |
| Status | **complete; test set now used** (logged in `outputs/TEST_SET_EVALUATIONS.log`) |
| Pre-registration | [`research/methodology/preregistration_EXP-004.md`](../methodology/preregistration_EXP-004.md) + [`configs/final_eval.json`](../../configs/final_eval.json), committed in `49bfec8` before the run |
| Run | `scripts/final_test_eval.py` at commit `49bfec8` (clean; see D-013), in a cloud environment, on features and embeddings computed on the desktop PC. The script refuses an uncommitted config and a second run. |
| Note | The config records `frozen_from_commit: 9b758ba (uncommitted changes)` (now `9aedf62`, see D-013) because it was generated while the new scripts were not yet committed. It was then committed unchanged, apart from back-filling the base-960h revision and adding two limitation notes, before the evaluation. |

## Setup
All three pre-registered systems: logistic regression trained on SEP28k-E **train only** (11,696 clips, 3 speakers), settings and thresholds frozen from dev. Evaluated once on SEP28k-E **test**: 3,965 clips, **142 speakers never used in any development step**.

## Results: ROC-AUC [95 % speaker-bootstrap CI]

| Target | Baseline features | wav2vec2-base-960h | **XLSR-53 (primary)** | XLSR-53 on dev (EXP-003) |
|---|---|---|---|---|
| Any event (incl. interjections) | 0.643 [0.624, 0.662] | 0.802 [0.784, 0.819] | **0.831** [0.813, 0.849] | 0.839 |
| Any event (excl. interjections) | 0.604 [0.576, 0.637] | 0.766 [0.754, 0.782] | **0.795** [0.777, 0.815] | 0.825 |
| Interjection | 0.676 [0.638, 0.718] | 0.901 [0.881, 0.920] | **0.902** [0.882, 0.919] | 0.906 |
| Sound repetition | 0.623 [0.556, 0.693] | 0.830 [0.799, 0.856] | **0.853** [0.824, 0.879] | 0.878 |
| Prolongation | 0.746 [0.718, 0.776] | 0.817 [0.786, 0.839] | **0.851** [0.824, 0.872] | 0.867 |
| Word repetition | 0.624 [0.596, 0.650] | 0.815 [0.797, 0.835] | **0.843** [0.821, 0.867] | 0.850 |
| Block | 0.609 [0.574, 0.646] | 0.712 [0.688, 0.740] | **0.731** [0.708, 0.756] | 0.745 |

**XLSR-53 detail (test):** "any event incl. interjections" PR-AUC 0.850, F1 0.778 at the frozen threshold (precision 0.731, recall 0.831) vs always-event F1 0.695. Per event, F1 at frozen thresholds: interjection 0.689, word repetition 0.510, prolongation 0.447, sound repetition 0.438, block 0.331. Full tables: `outputs/EXP-004/report.md`; figure `test_auc.png` (`scripts/plot_test_results.py`).

**Paired differences on the same test clips (2,000 speaker resamples):**

| Target | XLSR-53 − baseline features | XLSR-53 − base-960h | p(XLSR ≤ base) |
|---|---|---|---|
| Any event (incl.) | +0.189 [+0.168, +0.208] | +0.030 [+0.018, +0.042] | 0.001 |
| Any event (excl.) | +0.191 [+0.163, +0.218] | +0.028 [+0.015, +0.042] | <0.001 |
| Prolongation | +0.105 [+0.070, +0.134] | +0.034 [+0.017, +0.052] | <0.001 |
| Block | +0.123 [+0.087, +0.156] | +0.019 [−0.005, +0.040] | 0.048 |
| Sound repetition | +0.230 [+0.159, +0.309] | +0.023 [+0.003, +0.044] | 0.013 |
| Word repetition | +0.219 [+0.184, +0.254] | +0.028 [+0.003, +0.052] | 0.017 |
| Interjection | +0.226 [+0.194, +0.259] | +0.001 [−0.012, +0.011] | 0.462 |

## Pre-registered hypotheses
- **H1, supported.** XLSR-53 "any event" test AUC 0.831 [0.813, 0.849], well above chance and above the baseline-features dev value (0.653). On test, the baseline features score 0.643 [0.624, 0.662], so the improvement replicates on unseen speakers (+0.189 paired).
- **H2, mostly supported.** Of the targets where XLSR-53 beat base-960h on dev, the advantage replicates on test with CIs excluding zero for both "any event" targets, sound repetition and word repetition. For **block** the test difference is +0.019 with a CI that just includes zero [−0.005, +0.040], so it is not confirmed. Prolongation, which was *not* significant on dev, is significant on test (+0.034). Interjection: no difference on either.
- **H3, supported.** Block is the hardest event on test for every system (XLSR-53 0.731).
- **Expectation met.** Test AUCs are slightly below dev (by 0.004–0.030 for XLSR-53), consistent with mild selection optimism on dev and no overfitting beyond that.

## Conclusions
1. On 142 completely unseen speakers, frozen multilingual speech representations plus a linear classifier detect stuttering events far better than hand-crafted clip statistics (+0.19 AUC overall), with every event type improving.
2. The multilingual XLSR-53 is reliably a little better than the English ASR model on English stuttered speech (+0.03 AUC overall), most clearly for prolongation and repetitions, and never worse.
3. Blocks remain the open problem (AUC 0.73; F1 0.33).
4. Practical reading of the primary system: at the frozen threshold, about 83 % of clips containing an event are flagged, and about 73 % of flagged clips really contain one, on speakers the model has never heard. Useful for research and screening-style analysis; **not** a clinical tool.

## Limitations
Training on only 3 speakers; 74 % of SEP-28k retrieved; non-clinician majority-vote labels (low agreement for blocks and prolongations; M1 found audible label noise for prolongations); frozen encoders (not comparable to fine-tuned published systems); encoder C at the lower edge of the search grid; English podcast speech only; clip-level, not event-level timing. The test set has now been used: **any further model development must not be evaluated on it again** (a new held-out set would be needed).

## Next steps
- Stage 5: localisation (sliding windows over full episodes).
- Stage 6: cross-language pilot (Boli Hindi/English, licence permitting) with XLSR-53 as the encoder.
