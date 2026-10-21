# EXP-003: Frozen wav2vec 2.0 embeddings (Stage 4)

| Field | Value |
|---|---|
| Date | 2026-10-21 |
| Status | **Part A complete (wav2vec2-base-960h)**; Part B (XLSR-53) extraction running |
| Test set | **not evaluated** |
| Embeddings | Extracted on a desktop PC: `scripts/extract_embeddings.py`, torch 2.14.1+cpu, transformers 5.20.2, Python 3.12.10, Windows 11, 32 threads. Model revision not captured by this transformers version (D-010 note; back-fill with `scripts/record_model_revisions.py`). |
| Training | `scripts/train_embeddings.py`, run in a cloud environment on the PC's embedding shards (Python 3.13, scikit-learn 1.9.1). |

## Objective
EXP-001/002 concluded that clip-summary features, not the number of training speakers, limit performance: averaging over 3 s discards timing. Do contextual representations from a pretrained speech encoder do better under the identical protocol?

## Hypothesis (written before running)
Frozen wav2vec 2.0 embeddings will clearly beat the baseline (dev AUC 0.653). The largest gains will be on **word and sound repetitions**, which need timing and were weak with summary features. Blocks will remain the hardest event. Middle-to-upper layers will work best.

## Setup
- Encoder: `facebook/wav2vec2-base-960h`, frozen; 13 hidden-state layers (CNN output + 12 transformer layers), 768 dims.
- Clip representation: per layer, the mean and the standard deviation over frames, concatenated (1,536 dims).
- Layer selection: logistic regression (C = 0.01) per layer on SEP28k-E train; best "any event" dev AUC → **layer 10** (dev AUC by layer: 0.700, 0.721, 0.736, 0.758, 0.766, 0.788, 0.798, 0.810, 0.807, 0.814, **0.817**, 0.816, 0.807).
- Then the EXP-001/002 protocol: class-balanced, standardised logistic regression; C chosen per target on dev by F1 from {0.001, 0.01, 0.1, 1}; speaker-bootstrap 95 % CIs. Cross-validation: 5 speaker-grouped folds over the SEP28k-E **dev** clips only (D-011), fixed C = 0.01.
- Same 19,516 clips, labels (≥2/3 votes) and splits as EXP-001/002. SEP28k-E train has 3 speakers.

## Results: wav2vec2-base-960h, layer 10

| Target | Dev ROC-AUC [95 % CI] | Baseline features dev AUC | Δ | Dev PR-AUC (prevalence) | Dev F1 | Dev-only CV ROC-AUC |
|---|---|---|---|---|---|---|
| Any event (incl. interjections) | **0.817** [0.799, 0.834] | 0.653 | +0.16 | 0.843 (0.535) | 0.740 | 0.803 ± 0.021 |
| Any event (excl. interjections) | **0.791** [0.767, 0.810] | 0.632 | +0.16 | 0.714 (0.390) | 0.650 | 0.754 ± 0.020 |
| Interjection | **0.905** [0.891, 0.920] | 0.711 | +0.19 | 0.776 (0.218) | 0.693 | 0.892 ± 0.017 |
| Sound repetition | **0.856** [0.818, 0.885] | 0.678 | +0.18 | 0.434 (0.095) | 0.427 | 0.772 ± 0.040 |
| Prolongation | **0.855** [0.837, 0.875] | 0.744 | +0.11 | 0.455 (0.099) | 0.432 | 0.792 ± 0.014 |
| Word repetition | **0.826** [0.807, 0.849] | 0.641 | +0.19 | 0.455 (0.119) | 0.451 | 0.792 ± 0.028 |
| Block | **0.714** [0.677, 0.745] | 0.608 | +0.11 | 0.276 (0.125) | 0.336 | 0.657 ± 0.042 |

Every confidence interval lies entirely above the corresponding baseline value. Figures: `outputs/EXP-003/facebook__wav2vec2-base-960h/{layer_sweep,comparison}.png` (`scripts/plot_comparison.py`).

## Interpretation
1. **Large, consistent improvement**: +0.11 to +0.19 AUC on every target, with only the input representation changed. The "any event" F1 (0.740) now beats the always-event baseline (0.697), which no EXP-001 model did.
2. **Hypothesis largely supported.** The biggest gains are on word repetition (+0.19) and sound repetition (+0.18), the events that need timing; interjection also gains +0.19. This supports the EXP-002 interpretation that clip averages discarded the temporal pattern.
3. **Blocks remain hardest** (0.71), as predicted and as in AS-70 and Haas & Bayerl (2026), and consistent with the M1 listening check finding that block labels themselves sound right.
4. **Upper-middle layers are best** (7–11 within 0.01 of each other), matching the expectation that later transformer layers encode longer-range context.
5. Dev-only CV is a few points below the SEP28k-E dev numbers for sound repetition and prolongation. That is expected with fewer clips per training fold (~3,100 vs 11,696) and noisier per-fold estimates.

## Caveats
- **Mild optimism from selection:** the layer (and C on SEP28k-E) were chosen on the same dev clips that are reported, and the CV pool is those same dev clips. One choice among 13 layers on 3,855 clips inflates estimates only slightly, but the test set is the unbiased check.
- Frozen features and a linear classifier; the published systems fine-tune the encoder, so these numbers are not directly comparable to theirs. Our train split also has 3 speakers and 74 % of SEP-28k (EXP-000).
- wav2vec2-base-960h was fine-tuned for English ASR, so it may encode English-specific phonetics. Part B (XLSR-53, multilingual) tests whether that matters.

## Next steps
- Part B: same run for `facebook/wav2vec2-large-xlsr-53` when extraction finishes.
- Threshold calibration per event on dev (F1 still uses 0.5).
- Then decide the model to take to the single held-out test evaluation.
