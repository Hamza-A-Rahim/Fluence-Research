# Pre-registration: EXP-004 (single SEP28k-E test evaluation)

Written and committed **before** any test-set prediction was made. The frozen settings are in [`configs/final_eval.json`](../../configs/final_eval.json); `scripts/final_test_eval.py` refuses to run unless that file is committed and unchanged, and refuses to run twice.

## What will be evaluated
Three systems, all logistic regression trained on SEP28k-E **train only** (11,696 clips, 3 speakers), with settings chosen on dev in EXP-001/002/003:

| System | Features | Role |
|---|---|---|
| **XLSR-53** (`facebook/wav2vec2-large-xlsr-53` @ `c3f9d88`) | frozen layer 14, mean+std pooling | **primary** |
| wav2vec2-base-960h (@ `22aad52`) | frozen layer 10, mean+std pooling | comparison |
| Baseline features | 137 clip-summary statistics | comparison |

Targets: any event (incl. / excl. interjections), and each of prolongation, block, sound repetition, word repetition, interjection. Label: ≥2 of 3 annotators.

## Why XLSR-53 is primary (decided on dev data only)
- Highest dev ROC-AUC on 6 of 7 targets (tie on interjection) in EXP-003.
- Paired speaker-bootstrap on the same dev clips (EXP-003 follow-up, `scripts/compare_encoders.py`, 2,000 resamples), A = base-960h, B = XLSR-53:

| Target | AUC A | AUC B | B − A [95 % CI] | p(B ≤ A) |
|---|---|---|---|---|
| any_with_interjections | 0.817 | 0.839 | +0.022 [+0.010, +0.036] | 0.000 |
| any_without_interjections | 0.791 | 0.825 | +0.034 [+0.019, +0.053] | 0.000 |
| Prolongation | 0.855 | 0.867 | +0.012 [-0.007, +0.034] | 0.096 |
| Block | 0.714 | 0.745 | +0.031 [+0.011, +0.053] | 0.002 |
| SoundRep | 0.856 | 0.878 | +0.021 [+0.005, +0.040] | 0.005 |
| WordRep | 0.826 | 0.850 | +0.024 [+0.002, +0.050] | 0.017 |
| Interjection | 0.905 | 0.906 | +0.001 [-0.008, +0.009] | 0.457 |

- Multilingual pretraining is the relevant encoder for the planned Urdu/Hindi transfer work.

## Metrics and hypotheses
- **Primary:** ROC-AUC with 95 % speaker-bootstrap CI, per target.
- Secondary: PR-AUC; F1 at the frozen dev threshold and at 0.5; paired AUC differences, primary minus each other system.
- **H1:** XLSR-53 test AUC for "any event (incl. interjections)" is above 0.5 and its CI excludes the baseline-features dev value (0.653).
- **H2:** XLSR-53 ≥ base-960h on test for the same targets where it was better on dev (paired CI). This could fail; dev advantages of 0.02–0.03 may not replicate.
- **H3:** Block remains the hardest event on test.
- Expectation: test AUCs somewhat below dev, because layer, C and thresholds were all selected on dev.

## Known limitations, fixed in advance
3 training speakers; 74 % of SEP-28k retrieved; non-clinician labels with low agreement for blocks and prolongations; C at the lower edge of the grid; frozen (not fine-tuned) encoders, so results are not comparable to published fine-tuned systems.
