# Fluence

**Research code for detecting and classifying speech disfluencies (stuttering events) in recorded speech.**

Fluence is an independent research project by Hamza Rahim (Aitchison College, Lahore). It aims to build and rigorously evaluate machine-learning models that flag possible stuttering events — sound and word repetitions, prolongations, blocks and interjections — and, later, to test how well models trained on English speech transfer to Urdu-English bilingual speakers.

> **Status: Phase 1 (foundations).** The audio-processing pipeline is implemented and tested. **No model has been trained yet, and no performance results exist.** Any number that appears in this repository in future will come from a logged, reproducible experiment.

> **Not a medical tool.** Fluence is experimental research software. It does not diagnose, assess or treat stuttering, and its outputs must not be used for clinical decisions.

---

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
| Waveform + spectrogram inspection figure | ✅ tested |
| SEP-28k download + clip extraction (Windows/macOS/Linux, resumable, logged) | ✅ tested on synthetic episodes; real run pending |
| Label & partition analysis of real SEP-28k metadata | ✅ [EXP-000](research/experiment_logs/EXP-000_sep28k_label_analysis.md) |
| Baseline classifier, evaluation reports | ⏳ Stage 2 |
| Web interface, backend API | ⏳ deliberately deferred until a baseline is evaluated |

## Repository layout

```text
fluence-research/
├── src/fluence/          # importable Python package (pip install -e .)
│   ├── preprocessing/    # audio loading, validation, resampling, clipping
│   ├── features/         # log-mel, MFCC, deltas, clip summaries
│   ├── data/             # dataset label parsers, leakage-safe splits
│   ├── models/           # (Stage 2+) model definitions
│   ├── training/         # (Stage 2+) training loops
│   ├── evaluation/       # (Stage 2+) metrics, confusion matrices, reports
│   └── visualization/    # inspection figures
├── tests/                # unit tests (synthetic audio only — no real speech)
├── scripts/              # command-line tools
├── research/
│   ├── literature/       # literature review, dataset assessment
│   ├── methodology/      # evaluation protocol, splitting strategy
│   ├── experiment_logs/  # one file per experiment (template provided)
│   └── ethics/           # privacy, consent, participant protection
├── data/                 # local only — git-ignored except README
├── notebooks/            # exploratory analysis (clear outputs before committing)
├── outputs/              # generated figures/reports — git-ignored
├── docs/                 # decision log, setup
├── backend/  frontend/   # placeholders; not started by design
└── pyproject.toml
```

## Quick start

Requires Python 3.10+. **On Windows, follow [docs/setup_windows.md](docs/setup_windows.md)** — it covers installation and downloading the data step by step.

```bash
git clone https://github.com/Hamza-A-Rahim/Fluence-Research.git
cd fluence-research
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
- [Literature review](research/literature/literature_review.md)
- [Dataset assessment](research/literature/dataset_assessment.md)
- [Evaluation protocol](research/methodology/evaluation_protocol.md)
- [Ethics and privacy](research/ethics/ethics_and_privacy.md)

## Licence

**Not yet decided — all rights reserved for now.** The code licence will be chosen deliberately (see decision log, D-007). Note that the SEP-28k annotations are licensed CC BY-NC 4.0, so anything trained on them is for non-commercial research use.
