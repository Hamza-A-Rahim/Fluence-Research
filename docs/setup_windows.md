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
