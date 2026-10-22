# Decision log

Short records of choices that shape the project. Once **Decided**, we move on; revisiting needs a new entry that supersedes the old one.

| ID | Decision | Status |
|---|---|---|
| D-001 | Independent repository | Decided |
| D-002 | `src/fluence` package layout instead of top-level `ml/` | Decided |
| D-003 | Standard audio format: mono, float32, 16 kHz | Decided |
| D-004 | Minimal dependencies; PyTorch only from Stage 4 | Decided |
| D-005 | Binary target: include interjections by default, report both | Decided (revisit after clinical reading) |
| D-006 | Download SEP-28k audio for local, non-commercial research | Decided |
| D-007 | Code licence | **Open** |
| D-008 | Primary evaluation partition: SEP-28k-E | Decided |
| D-009 | Stage 2 baseline: logistic regression on clip-summary features | Decided |
| D-010 | Stage 4 encoders: wav2vec2-base-960h and XLSR-53, frozen, mean+std pooled | Decided |
| D-011 | Cross-validation excludes the SEP28k-E test clips (correction) | Decided |
| D-012 | Stage 5 design: offset-robustness on dev-only episodes, not timestamp accuracy | Decided |

---

### D-001 — Independent repository
No code, data, credentials, databases or deployments are shared with any other project. Different purpose, different data-protection needs.

### D-002 — Package layout
The proposed `ml/` folder becomes `src/fluence/` with the same sub-areas (preprocessing, features, models, training, evaluation) plus `data/` and `visualization/`. **Reason:** an installable package (`pip install -e .`) lets tests, scripts, notebooks and the future FastAPI backend all `import fluence` the same way, without path hacks; the `src/` layout ensures tests run against the installed code. A top-level package named `ml` would also clash easily with other libraries.

### D-003 — Audio format
16 kHz is the rate of SEP-28k label indices, KSoF and AS-70, and the input rate of wav2vec 2.0, HuBERT and Whisper. Mono because disfluency cues don't depend on stereo. Resampling uses polyphase filtering (anti-aliased).

### D-004 — Dependencies
Core needs only NumPy, SciPy, pandas, scikit-learn and matplotlib. Feature extraction is implemented on SciPy and written to match librosa's defaults (verified by optional tests when librosa is installed). PyTorch and Hugging Face are added at Stage 4, when a model needs them.

### D-005 — Interjections in the "any stutter" target
Lea et al. count interjections as stuttering events; many clinical frameworks treat filled pauses ("um") as typical disfluencies rather than stuttering-like. Default follows Lea et al. for comparability (`include_interjections=True`); every Stage 2 result is also reported with interjections excluded.

### D-006 — SEP-28k audio (decided 2026-10-21)
Audio is fetched from the podcasts' own public hosting (the route Apple's scripts use) for **local, non-commercial research use only**; audio and clips are never committed, uploaded or shared. By default only the 3-second labelled clips (~3 GB) are kept; full episodes only with `--keep-episodes`. Retrieval runs locally with `scripts/fetch_sep28k.py`, which records the status of every episode.

### D-007 — Licence (open)
Options: keep all rights reserved while private; MIT or Apache-2.0 (permissive) when public; or GPL (copyleft). Data-derived artefacts (e.g. model weights trained on SEP-28k) are non-commercial regardless of the code licence. No LICENSE file until decided.

### D-008 — Evaluation partition
SEP-28k-E from SEP-28k-Extended as the fixed primary split, SEP-12k 5-fold CV as robustness check. EXP-000 confirmed SEP-28k-E trains on only **4 speakers** (14,092 clips) and evaluates on 245 unseen speakers each in dev and test — a strict generalisation test, so SEP-12k CV is required alongside it, not optional. See `research/methodology/evaluation_protocol.md`.

### D-009 — Baseline model
Logistic regression (class-balanced, standardised features) on 137 clip-summary features (log-mel, MFCC and delta statistics, energy/pause descriptors). **Why:** trains in seconds on a laptop, coefficients are interpretable, and it sets an honest floor. A random forest is run alongside as a non-linear check. Neural and pretrained-encoder models come in Stage 4 and must beat this under the same protocol.

### D-010 — Stage 4 encoders (2026-10-21)
Extract frozen embeddings from two wav2vec 2.0 models: **facebook/wav2vec2-base-960h** (English, ASR fine-tuned; the setup of Bayerl et al. 2022, for comparability) and **facebook/wav2vec2-large-xlsr-53** (pretrained on 53 languages; best in Haas & Bayerl 2026, and the natural candidate for the later Urdu-English work). Per layer, mean and standard deviation over frames; layer chosen on SEP28k-E dev only; logistic regression with the EXP-001/002 protocol. **Frozen, not fine-tuned**: it runs on a laptop CPU and isolates the effect of the representation. Fine-tuning (as in the papers) is a later option if a GPU becomes available. Both models are Apache-2.0 licensed on Hugging Face; the exact revision is recorded in each manifest.

### D-011 — Cross-validation must not touch the test set (correction, 2026-10-21)
**Error found:** the SEP-12k partition is exactly SEP28k-E dev ∪ test (verified: 3,855 dev + 3,965 test clips). The SEP-12k cross-validation in EXP-001, EXP-002 and the first EXP-003 run therefore trained and scored on held-out test clips. No hyper-parameters were tuned on those runs, but design conclusions (e.g. "features, not speaker count, are the bottleneck") partly drew on them. **Fix:** CV now uses only the SEP28k-E dev clips (`fluence.data.datasets.cv_pool`): 3,855 clips, 136 speakers. All CV results were re-run and the logs corrected; the superseded numbers are kept in each log, marked as such. The SEP28k-E test set has never been used for model selection and remains unevaluated.

### D-012 — Stage 5 design (2026-10-22)
SEP-28k labels 3 s clips at known positions but gives no timing *inside* a clip, so precise event timestamps cannot be validated with this data. Stage 5 therefore asks the question the data can answer: **does the detector still work on unsegmented audio, when its 3 s window is not aligned to a labelled clip?** For each dev clip, 3 s windows are taken at offsets −1.5 … +1.5 s (0.5 s steps) from the clip start, plus sliding-max/mean aggregation, and whole-episode timelines for 3 demo episodes. Only the 108 episodes containing dev clips and **no test clips** are used (the test set is spent). The model is the frozen EXP-004 primary system (XLSR-53 layer 14, same C), trained on train clips only. Re-downloading these episodes keeps only ±1.5 s context around each dev clip (~0.6 GB) to avoid storing 85 h of audio. True timestamp evaluation needs data with word-level timing (e.g. Boli, if licensed) and is deferred.

