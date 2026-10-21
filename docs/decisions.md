# Decision log

Short records of choices that shape the project. Once **Decided**, we move on; revisiting needs a new entry that supersedes the old one.

| ID | Decision | Status |
|---|---|---|
| D-001 | Independent repository | Decided |
| D-002 | `src/fluence` package layout instead of top-level `ml/` | Decided |
| D-003 | Standard audio format: mono, float32, 16 kHz | Decided |
| D-004 | Minimal dependencies; PyTorch only from Stage 4 | Decided |
| D-005 | Binary target: include interjections by default, report both | Decided (revisit after clinical reading) |
| D-006 | Whether to download SEP-28k podcast audio | **Open** |
| D-007 | Code licence | **Open** |
| D-008 | Primary evaluation partition: SEP-28k-E | Decided |
| D-009 | Stage 2 baseline: logistic regression on clip-summary features | Decided |

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

### D-006 — SEP-28k audio (open)
The annotations are CC BY-NC 4.0, but the audio is the podcast owners' copyright and is fetched from their public hosting by Apple's download script — the intended route for research use. Proposal: (1) run the HEAD-only availability check; (2) if acceptable, download for **local, non-commercial research use only**, never redistributing audio or clips. Needs a decision before any audio is downloaded (~32 GB raw).

### D-007 — Licence (open)
Options: keep all rights reserved while private; MIT or Apache-2.0 (permissive) when public; or GPL (copyleft). Data-derived artefacts (e.g. model weights trained on SEP-28k) are non-commercial regardless of the code licence. No LICENSE file until decided.

### D-008 — Evaluation partition
SEP-28k-E from SEP-28k-Extended as the fixed primary split, SEP-12k 5-fold CV as robustness check. See `research/methodology/evaluation_protocol.md`.

### D-009 — Baseline model
Logistic regression (class-balanced, standardised features) on 137 clip-summary features (log-mel, MFCC and delta statistics, energy/pause descriptors). **Why:** trains in seconds on a laptop, coefficients are interpretable, and it sets an honest floor. A random forest is run alongside as a non-linear check. Neural and pretrained-encoder models come in Stage 4 and must beat this under the same protocol.
