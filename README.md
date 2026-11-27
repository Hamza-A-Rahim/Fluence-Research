# Fluence

**Research code for detecting and classifying speech disfluencies (stuttering events) in recorded speech.**

Fluence is an independent research project by Hamza Rahim (Aitchison College, Lahore). It builds and rigorously evaluates machine-learning models that flag possible stuttering events — sound and word repetitions, prolongations, blocks and interjections — and will test how well models trained on English speech transfer to Urdu-English bilingual speakers.

> **Status: Stages 1–5 complete; local demo interface working.** Every number in this repository comes from a logged, reproducible experiment in [`research/experiment_logs/`](research/experiment_logs/). The SEP28k-E test set has been used once (EXP-004) and will not be used again.

> **Not a medical tool.** Fluence is experimental research software. It does not diagnose, assess or treat stuttering, and its outputs must not be used for clinical decisions.

---

## Headline result

On 142 speakers never used during development, frozen multilingual speech representations (XLSR-53) with a linear classifier detect whether a 3-second clip contains a stuttering event with ROC-AUC **0.831** [0.813, 0.849], against 0.643 for hand-crafted acoustic features. Every event type improves; blocks remain the hardest (0.731). The evaluation was pre-registered and run once — see [EXP-004](research/experiment_logs/EXP-004_test_set.md).

## What works today

| Component | Status |
|---|---|
| WAV loading & validation (format, rate, NaN, length, clipping, silence, DC offset) | ✅ tested |
| Mono conversion, anti-aliased resampling to 16 kHz, peak normalisation | ✅ tested |
| Clip extraction by seconds or by SEP-28k sample indices | ✅ tested |
| Log-mel filterbank (40 bands, 25 ms / 10 ms — the SEP-28k paper's input), MFCC, deltas | ✅ tested |
| Fixed-length clip summary features for classical baselines (137 values) | ✅ tested |
| SEP-28k label parsing, majority-vote binarisation, multi-label summaries | ✅ tested (synthetic fixture) |
| Speaker/episode-grouped splitting with an automatic leakage check | ✅ tested |
| SEP-28k download + clip extraction (Windows/macOS/Linux, resumable, logged) | ✅ 20,896 of 28,177 clips retrieved (74 %; [EXP-000](research/experiment_logs/EXP-000_sep28k_label_analysis.md)) |
| Label & partition analysis of real SEP-28k metadata | ✅ [EXP-000](research/experiment_logs/EXP-000_sep28k_label_analysis.md) |
| Baseline classifier (logistic regression, random forest), metrics with speaker-bootstrap CIs, reference baselines, test-set access log | ✅ [EXP-001](research/experiment_logs/EXP-001_baseline.md): ROC-AUC ≈ 0.65–0.66 (dev / CV) |
| Per-event (multi-label) detection | ✅ [EXP-002](research/experiment_logs/EXP-002_per_event.md): prolongation easiest (0.74), block hardest (0.60) |
| Listening check of labelled clips | ✅ [M1](research/experiment_logs/M1_listening_check.md): prolongation labels audibly noisy; block labels sound right |
| Pretrained encoders (wav2vec 2.0, frozen) | ✅ [EXP-003](research/experiment_logs/EXP-003_wav2vec2.md): dev ROC-AUC 0.653 → **0.817** (English base) / **0.839** (multilingual XLSR-53); every event improves |
| **Final held-out test (pre-registered, run once)** | ✅ [EXP-004](research/experiment_logs/EXP-004_test_set.md): XLSR-53 test ROC-AUC **0.831** [0.813, 0.849] on 142 unseen speakers vs 0.643 for hand-made features |
| Offset robustness and sliding-window detection | ✅ [EXP-005](research/experiment_logs/EXP-005_localisation.md): averaging overlapping windows keeps clip-level performance (dev ROC-AUC 0.835 vs 0.846, any event) |
| Local demo: FastAPI engine + Next.js interface | ✅ local only — [backend](backend/README.md), [frontend](frontend/README.md) |

## Next steps

- **Cross-language pilot:** an existing Hindi/English dataset (e.g. Boli, licence permitting) with XLSR-53 as the encoder, and an Urdu-English study protocol prepared for ethics review.
- **Event-level timing:** evaluating detected event boundaries needs data with word- or event-level timestamps, which SEP-28k does not provide ([D-012](docs/decisions.md)).
- **Write-up:** a technical report, with a reproducibility check from a fresh clone.

## Repository layout

```text
Fluence-Research/
├── src/fluence/          # importable Python package (pip install -e .)
│   ├── preprocessing/    # audio loading, validation, resampling, clipping
│   ├── features/         # log-mel, MFCC, deltas, clip summaries, encoder embeddings
│   ├── data/             # SEP-28k labels, clip loading, leakage-safe splits
│   ├── evaluation/       # metrics, speaker-bootstrap confidence intervals
│   ├── visualization/    # inspection and results figures
│   ├── localization.py   # offset and sliding-window analysis (EXP-005)
│   └── inference.py      # scoring recordings of any length (used by the demo)
├── tests/                # unit and end-to-end tests (synthetic audio only — no real speech)
├── scripts/              # command-line tools for every experiment
├── configs/              # frozen, pre-registered test configuration (EXP-004)
├── research/
│   ├── literature/       # literature review, dataset assessment
│   ├── methodology/      # evaluation protocol, EXP-004 pre-registration
│   ├── experiment_logs/  # one file per experiment (template provided)
│   └── ethics/           # privacy, consent, participant protection
├── backend/              # local demo API (FastAPI)
├── frontend/             # local demo interface (Next.js)
├── data/                 # local only — git-ignored except README
├── notebooks/            # exploratory analysis (clear outputs before committing)
├── outputs/              # generated figures/reports — git-ignored
├── docs/                 # decision log, setup and reproduction guide
└── pyproject.toml
```

## Quick start

Requires Python 3.10+. **On Windows, follow [docs/setup_windows.md](docs/setup_windows.md)**, which covers installation, downloading the data and the commands that reproduce each experiment.

```bash
git clone https://github.com/Hamza-A-Rahim/Fluence-Research.git
cd Fluence-Research
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev,audio-extras]"
pytest                              # or: python -m unittest discover -s tests
```

The two librosa cross-check tests run only when `librosa` is installed (the `audio-extras` group); they confirm our mel features match the standard library implementation.

Inspect any WAV file (nothing leaves your computer):

```bash
python scripts/inspect_audio.py path/to/recording.wav
# -> outputs/inspect/<name>_inspect.png and <name>_features.csv
```

## Data

No audio or dataset files are stored in this repository. See [`data/README.md`](data/README.md) and the [dataset assessment](research/literature/dataset_assessment.md) for what each dataset contains, how it may be used, and how to obtain it.

## Documentation

- [Decision log](docs/decisions.md) — technical and scientific choices with reasons
- [Setup and reproduction (Windows)](docs/setup_windows.md)
- [Literature review](research/literature/literature_review.md)
- [Dataset assessment](research/literature/dataset_assessment.md)
- [Evaluation protocol](research/methodology/evaluation_protocol.md)
- [EXP-004 pre-registration](research/methodology/preregistration_EXP-004.md)
- [Ethics and privacy](research/ethics/ethics_and_privacy.md)

## Licence

**Not yet decided — all rights reserved for now.** The code licence will be chosen deliberately (see decision log, D-007). Note that the SEP-28k annotations are licensed CC BY-NC 4.0, so anything trained on them is for non-commercial research use.
