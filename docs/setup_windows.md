# Setting up Fluence on Windows

One-time setup, then one command to download the data. Run everything in **PowerShell** (Start menu → type "PowerShell").

## 1. Install the tools (skip any you already have)

- **Python 3.12** — <https://www.python.org/downloads/>. In the installer, tick **"Add python.exe to PATH"**.
- **Git** — <https://git-scm.com/download/win>. Defaults are fine.

Close and reopen PowerShell, then check:

```powershell
python --version
git --version
```

## 2. Get the code

```powershell
cd $HOME\Documents
git clone https://github.com/Hamza-A-Rahim/Fluence-Research.git
cd Fluence-Research
```

## 3. Create the Python environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev,audio-extras,fetch]"
```

If `Activate.ps1` is blocked ("running scripts is disabled"), run this once, then try again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Every time you open a new PowerShell window for Fluence, `cd` into the folder and run `.\.venv\Scripts\Activate.ps1` again.

## 4. Check everything works

```powershell
pytest
```

Expected: all tests pass, **including the two librosa cross-checks** (they need `librosa`, from the `audio-extras` group).

## 5. Download SEP-28k

```powershell
python scripts/fetch_sep28k.py --check-only     # ~1 minute: which podcast links still work
python scripts/fetch_sep28k.py --limit 3        # try 3 episodes first
python scripts/fetch_sep28k.py                  # everything (a few hours; needs ~4 GB free)
```

- Safe to stop with **Ctrl+C** and re-run — it resumes where it left off.
- Results: clips in `data\interim\sep28k_clips\`, a per-episode log in `data\metadata\sep28k_fetch_log.csv`.
- Nothing in `data\` is ever uploaded to GitHub (it's git-ignored).

## 6. Train the baseline (once clips exist)

```powershell
python scripts/train_baseline.py                       # train + dev results (test untouched)
python scripts/train_baseline.py --partition SEP12k    # speaker-grouped cross-validation
```

The first run computes features for every clip (roughly 10–30 minutes); later runs reuse a cache and take a minute or two. Reports appear in `outputs\EXP-001\`. **Don't run `--evaluate-test` yet** — that's done once, at the end, and every use is logged.

## Stage 4: wav2vec 2.0 embeddings (one-time, ~3 GB download)

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[deep]"
pytest                                             # includes a tiny-model check of the Stage 4 code
python scripts/extract_embeddings.py --limit 16    # quick real check, ~1–3 min (downloads both models)
python scripts/extract_embeddings.py               # all clips, both models: several hours, leave it running
```

Resumable: if it stops (Ctrl+C, sleep, restart), run the same command again. Windows sleep pauses it, so set **Settings → System → Power → Screen and sleep → "When plugged in, put my device to sleep after: Never"** for the night.

Afterwards (minutes each):
```powershell
python scripts/train_embeddings.py --model facebook/wav2vec2-base-960h
python scripts/train_embeddings.py --model facebook/wav2vec2-large-xlsr-53
```

## Stage 5: localisation (EXP-005)

```powershell
git pull
pytest
python scripts/fetch_context.py             # re-downloads 108 dev-only episodes, keeps ±1.5 s around each dev clip (~30–60 min)
python scripts/extract_windows.py           # XLSR-53 windows: ~19,500 shifted windows + 3 episode timelines (~1–2 h)
python scripts/localize.py                  # analysis + figures (minutes)
```
All three are resumable. Results: `outputs\EXP-005\`.


## Stage 7: local demo interface

Needs Node.js 20+ (`node --version`). Two terminals, both from the repo root:
```powershell
# terminal 1: engine
pip install -e ".[web]"
python scripts/export_demo_model.py         # once; verifies the classifiers against EXP-005
python backend/run.py                       # http://127.0.0.1:8000, model loads in a few seconds

# terminal 2: interface
cd frontend
npm install                                 # once
npm run dev                                 # open http://127.0.0.1:3000
```
Both bind to 127.0.0.1 only. A 30 s recording takes about 20 s on the i9 CPU.
