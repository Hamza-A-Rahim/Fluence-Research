# EXP-003: Frozen wav2vec 2.0 embeddings (Stage 4)

| Field | Value |
|---|---|
| Date | 2026-10-21 |
| Status | **Complete** (Part A: wav2vec2-base-960h; Part B: XLSR-53) |
| Test set | **not evaluated** |
| Embeddings | Extracted on a desktop PC: `scripts/extract_embeddings.py`, torch 2.14.1+cpu, transformers 5.20.2, Python 3.12.10, Windows 11, 32 threads. Revisions (back-filled from the local HF cache): base-960h `22aad52d435eb6dbaf354bdad9b0da84ce7d6156`, XLSR-53 `c3f9d884181a224a6ac87bf8885c84d1cff3384f`. |
| Training | `scripts/train_embeddings.py`. Part A run in a cloud environment on the PC's embedding shards (Python 3.13, scikit-learn 1.9.1); Part B run on the PC (Python 3.12.10, scikit-learn 1.9.1, commit `24da27e`, 90 s). |

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

## Part B: XLSR-53 (multilingual), layer 14

Same protocol. Encoder `facebook/wav2vec2-large-xlsr-53` (pretrained without labels on 53 languages; not fine-tuned for ASR): 25 hidden-state layers, 1,024 dims, so 2,048 per clip after mean+std pooling. Layer sweep (dev AUC, any event, C = 0.01): rises to a plateau of 0.80–0.82 across layers 9–20, best **layer 14 (0.821)**, then drops sharply to ~0.74 in the last three layers (22–24). That is expected for a model pretrained only on a self-supervised objective, whose top layers specialise in that pretraining task.

| Target | XLSR-53 dev AUC [95 % CI] | base-960h dev AUC | Baseline features | XLSR-53 dev-only CV | base-960h dev-only CV |
|---|---|---|---|---|---|
| Any event (incl. interjections) | **0.839** [0.824, 0.851] | 0.817 | 0.653 | 0.792 ± 0.013 | 0.803 ± 0.021 |
| Any event (excl. interjections) | **0.825** [0.811, 0.840] | 0.791 | 0.632 | 0.780 ± 0.009 | 0.754 ± 0.020 |
| Interjection | 0.906 [0.890, 0.923] | 0.905 | 0.711 | 0.877 ± 0.015 | 0.892 ± 0.017 |
| Sound repetition | **0.878** [0.846, 0.900] | 0.856 | 0.678 | 0.798 ± 0.027 | 0.772 ± 0.040 |
| Prolongation | **0.867** [0.844, 0.893] | 0.855 | 0.744 | 0.832 ± 0.026 | 0.792 ± 0.014 |
| Word repetition | **0.850** [0.826, 0.879] | 0.826 | 0.641 | 0.830 ± 0.026 | 0.792 ± 0.028 |
| Block | **0.745** [0.718, 0.769] | 0.714 | 0.608 | 0.699 ± 0.032 | 0.657 ± 0.042 |

Dev F1 for "any event, incl. interjections": 0.767 (base 0.740; always-event 0.697). Figure: `outputs/EXP-003/comparison.png` (`scripts/plot_comparison.py`).

### Part B interpretation
1. **The multilingual encoder is at least as good as the English ASR model on English stuttered speech, and modestly better on the core stuttering events.** On dev it is higher on every target except interjection (tie); in dev-only CV it is higher on all four core events (prolongation, sound repetition, word repetition, block) and on "any event excl. interjections", but slightly lower on interjections and on "any event incl. interjections".
2. **The differences are small (≈ 0.01–0.04 AUC) and the confidence intervals overlap.** We have *not* shown a statistically reliable difference between the two encoders. A paired comparison on the same clips (bootstrap over speakers of the AUC difference) is needed before claiming one.
3. A plausible reading, still to be tested: the English model was fine-tuned to transcribe *words*, which helps with interjections ("um" is close to a word) but may smooth over sub-word disruptions. XLSR-53 was never trained to transcribe, so it may keep more acoustic detail about blocks, prolongations and part-word repetitions.
4. **Relevance to the research question:** a multilingual encoder loses nothing on English, which supports using XLSR-53 for the Urdu/Hindi transfer experiments, consistent with Haas & Bayerl (2026).

## Next steps
- Paired speaker-bootstrap test of XLSR-53 vs base-960h on dev (same clips) before claiming a difference.
- Threshold calibration per event on dev (F1 still uses 0.5).
- Then decide the model to take to the single held-out test evaluation.
