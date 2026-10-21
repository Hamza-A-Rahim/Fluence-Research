"""Download SEP-28k metadata and audio, and cut the labelled 3-second clips.

Works on Windows, macOS and Linux. Replaces Apple's download_audio.py and
extract_clips.py, which call `wget` through a Unix shell.

What it does
------------
1. Downloads the label/metadata CSVs (pinned to fixed commits, for
   reproducibility) into data/metadata/.
2. For each of the 385 podcast episodes:
   download the MP3/M4A from the podcast's own host -> decode to 16 kHz mono
   with ffmpeg -> cut every labelled clip -> save clips -> delete the MP3.
3. Writes data/metadata/sep28k_fetch_log.csv: one row per episode with its
   status, so you can see exactly what was retrieved. Re-running the script
   skips episodes that are already done (safe to stop with Ctrl+C and resume).

Usage (from the repository root)
--------------------------------
    python scripts/fetch_sep28k.py --metadata-only     # just the CSVs (seconds)
    python scripts/fetch_sep28k.py --check-only        # which links still work (no audio)
    python scripts/fetch_sep28k.py                     # full run: clips only (~3 GB)
    python scripts/fetch_sep28k.py --keep-episodes     # also keep full 16 kHz episodes (~32 GB more)
    python scripts/fetch_sep28k.py --limit 5           # try just 5 episodes first

Licence and use
---------------
Annotations: CC BY-NC 4.0 (Apple; TH Nuernberg). Audio copyright remains with
the podcast owners. Everything stays in the git-ignored data/ folder, on this
computer only, for non-commercial research. Never commit or share the audio.

Requires ffmpeg: either `pip install imageio-ffmpeg` (bundles it; easiest on
Windows, included in the project's `fetch` extra) or ffmpeg on your PATH.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.io import wavfile

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "data" / "metadata"
CLIPS = ROOT / "data" / "interim" / "sep28k_clips"
EPISODES = ROOT / "data" / "raw" / "sep28k_episodes"
TMP = ROOT / "data" / "raw" / "_download_tmp"
LOG = META / "sep28k_fetch_log.csv"

SR = 16_000

# Pinned commits (checked 2026-10-21). Record these in experiment logs.
APPLE_COMMIT = "fe954e110c8cb2216a0a99eb2bf77e8e6b8208ea"
EXT_COMMIT = "5f599c5549a42ca11ee332a4445ee881f6fc238f"
APPLE_RAW = f"https://raw.githubusercontent.com/apple-aiml-research/ml-stuttering-events-dataset/{APPLE_COMMIT}/"
EXT_RAW = f"https://raw.githubusercontent.com/th-nuernberg/ml-stuttering-events-dataset-extended/{EXT_COMMIT}/"
METADATA_FILES = {
    "SEP-28k_labels.csv": APPLE_RAW,
    "SEP-28k_episodes.csv": APPLE_RAW,
    "fluencybank_labels.csv": APPLE_RAW,
    "fluencybank_episodes.csv": APPLE_RAW,
    "SEP-28k-Extended_clips.csv": EXT_RAW,
    "SEP-28k-Extended_episodes.csv": EXT_RAW,
}

UA = "Mozilla/5.0 (fluence-research; non-commercial stuttering research)"
LOG_FIELDS = ["show", "ep_id", "url", "status", "detail", "n_clips_expected",
              "n_clips_written", "episode_seconds", "last_label_end_seconds", "finished_utc"]


# ---------------------------------------------------------------- helpers

def find_ffmpeg() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    sys.exit("ffmpeg not found. Run:  pip install imageio-ffmpeg   (or install ffmpeg and add it to PATH)")


MAX_DOWNLOAD_S = 20 * 60   # give up on one episode after 20 minutes
MIN_RATE_BPS = 20_000      # ...or if it averages under 20 KB/s after the first minute


def _copy_with_deadline(r, f, label: str) -> None:
    """Copy a response to disk, printing progress and aborting stalled transfers."""
    start = last_print = time.time()
    got = 0
    total = int(r.headers.get("Content-Length") or 0)
    while True:
        chunk = r.read(1 << 20)
        if not chunk:
            return
        f.write(chunk)
        got += len(chunk)
        now = time.time()
        elapsed = now - start
        if elapsed > MAX_DOWNLOAD_S:
            raise TimeoutError(f"download exceeded {MAX_DOWNLOAD_S // 60} min")
        if elapsed > 60 and got / elapsed < MIN_RATE_BPS:
            raise TimeoutError(f"too slow ({got / elapsed / 1000:.0f} KB/s)")
        if now - last_print > 30:
            of = f" of {total / 1e6:.0f}" if total else ""
            print(f"      ... {label}: {got / 1e6:.0f}{of} MB", flush=True)
            last_print = now


def http_get(url: str, dest: Path, timeout: float = 60.0, tries: int = 3, label: str = "") -> None:
    last = None
    for attempt in range(1, tries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as r, open(dest, "wb") as f:
                _copy_with_deadline(r, f, label or dest.name)
            if dest.stat().st_size < 10_000:
                raise IOError(f"suspiciously small download ({dest.stat().st_size} bytes)")
            return
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (403, 404, 410):  # permanent: don't retry
                break
        except Exception as e:  # timeouts, resets
            last = e
        time.sleep(2 * attempt)
    raise RuntimeError(f"{type(last).__name__}: {last}")


def http_head(url: str, timeout: float = 20.0) -> str:
    for method in ("HEAD", "GET"):  # some hosts reject HEAD
        try:
            req = urllib.request.Request(url, method=method, headers={"User-Agent": UA, "Range": "bytes=0-0"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return str(r.status)
        except urllib.error.HTTPError as e:
            if method == "GET" or e.code not in (403, 405):
                return str(e.code)
        except Exception as e:
            if method == "GET":
                return type(e).__name__
    return "unknown"


def decode_16k_mono(ffmpeg: str, path: Path) -> np.ndarray:
    """Decode any audio file to int16 mono 16 kHz with ffmpeg (same conversion as Apple's script)."""
    cmd = [ffmpeg, "-nostdin", "-loglevel", "error", "-i", str(path),
           "-ac", "1", "-ar", str(SR), "-f", "s16le", "-acodec", "pcm_s16le", "-"]
    out = subprocess.run(cmd, capture_output=True)
    if out.returncode != 0 or not out.stdout:
        raise RuntimeError("ffmpeg failed: " + out.stderr.decode(errors="replace")[:300])
    return np.frombuffer(out.stdout, dtype=np.int16)


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, skipinitialspace=True))
    return [{k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in r.items()} for r in rows]


def load_log() -> dict[tuple[str, str], dict]:
    if not LOG.exists():
        return {}
    return {(r["show"], r["ep_id"]): r for r in read_csv(LOG)}


def save_log(log: dict) -> None:
    rows = sorted(log.values(), key=lambda r: (r["show"], int(r["ep_id"])))
    tmp = LOG.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        w.writeheader()
        w.writerows(rows)
    tmp.replace(LOG)


# ---------------------------------------------------------------- steps

def fetch_metadata(force: bool = False) -> None:
    META.mkdir(parents=True, exist_ok=True)
    for name, base in METADATA_FILES.items():
        dest = META / name
        if dest.exists() and not force:
            print(f"  have  {name}")
            continue
        req = urllib.request.Request(base + name, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=60) as r:
            dest.write_bytes(r.read())
        print(f"  got   {name}  ({dest.stat().st_size // 1024} KB)")


def load_episode_plan() -> tuple[list[dict], dict[tuple[str, str], list[dict]]]:
    episodes = read_csv(META / "SEP-28k-Extended_episodes.csv")
    clips = read_csv(META / "SEP-28k_labels.csv")
    by_ep: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for c in clips:
        by_ep[(c["Show"], c["EpId"])].append(c)
    return episodes, by_ep


def process_episode(ep: dict, clips: list[dict], ffmpeg: str, keep_episode: bool) -> dict:
    show, ep_id, url = ep["Show"], ep["EpId"], ep["url"]
    row = {"show": show, "ep_id": ep_id, "url": url, "n_clips_expected": len(clips),
           "n_clips_written": 0, "episode_seconds": "", "detail": "",
           "last_label_end_seconds": max((int(c["Stop"]) for c in clips), default=0) / SR}
    ext = next((e for e in (".mp3", ".m4a", ".mp4") if e in url.lower()), ".mp3")
    TMP.mkdir(parents=True, exist_ok=True)
    tmp = TMP / f"{show}_{ep_id}{ext}"
    print(f"    start  {show} ep {ep_id}", flush=True)
    try:
        http_get(url, tmp, label=f"{show} ep {ep_id}")
        audio = decode_16k_mono(ffmpeg, tmp)
    except Exception as e:
        row.update(status="unavailable", detail=str(e)[:200])
        return row
    finally:
        tmp.unlink(missing_ok=True)

    row["episode_seconds"] = round(len(audio) / SR, 2)
    if keep_episode:
        (EPISODES / show).mkdir(parents=True, exist_ok=True)
        wavfile.write(EPISODES / show / f"{ep_id}.wav", SR, audio)

    out_dir = CLIPS / show / ep_id
    out_dir.mkdir(parents=True, exist_ok=True)
    beyond_end = 0
    for c in clips:
        start, stop = int(c["Start"]), int(c["Stop"])
        if stop > len(audio):
            beyond_end += 1
            continue
        wavfile.write(out_dir / f"{show}_{ep_id}_{c['ClipId']}.wav", SR, audio[start:stop])
        row["n_clips_written"] += 1

    if beyond_end:
        # The hosted file is shorter than when it was labelled: it may have been
        # re-edited, so even the clips we could cut may be misaligned. Flag it.
        row.update(status="length_mismatch",
                   detail=f"{beyond_end} clips beyond end of audio; check alignment by listening")
    else:
        row["status"] = "ok"
    return row


def check_only(episodes: list[dict]) -> None:
    results = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = {pool.submit(http_head, e["url"]): e for e in episodes}
        for f in as_completed(futs):
            e = futs[f]
            results.append((e["Show"], int(e["EpId"]), e["url"], f.result()))
    results.sort()
    out = META / "sep28k_url_status.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["show", "ep_id", "url", "status"])
        w.writerows(results)
    summarise([(s, st.startswith(("2", "3"))) for s, _, _, st in results], "reachable")
    print(f"\nPer-episode results: {out}")


def summarise(pairs: list[tuple[str, bool]], word: str) -> None:
    tally = defaultdict(lambda: [0, 0])
    for show, ok in pairs:
        tally[show][0] += ok
        tally[show][1] += 1
    print(f"\n{'Show':<14} {word}")
    for show in sorted(tally):
        ok, n = tally[show]
        print(f"  {show:<12} {ok:>4} / {n}")
    ok = sum(v[0] for v in tally.values())
    n = sum(v[1] for v in tally.values())
    print(f"  {'TOTAL':<12} {ok:>4} / {n}  ({ok / max(n, 1):.0%})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metadata-only", action="store_true")
    ap.add_argument("--check-only", action="store_true", help="test links; download no audio")
    ap.add_argument("--keep-episodes", action="store_true", help="also save full 16 kHz episodes")
    ap.add_argument("--limit", type=int, default=0, help="process at most N new episodes")
    ap.add_argument("--workers", type=int, default=3, help="parallel episode downloads")
    ap.add_argument("--retry-failed", action="store_true", help="retry episodes logged as unavailable")
    args = ap.parse_args()

    print("Metadata:")
    fetch_metadata()
    if args.metadata_only:
        return

    episodes, by_ep = load_episode_plan()
    if args.check_only:
        check_only(episodes)
        return

    ffmpeg = find_ffmpeg()
    free_gb = shutil.disk_usage(ROOT).free / 1e9
    need_gb = 3.5 + (32 if args.keep_episodes else 0)
    print(f"\nffmpeg: {ffmpeg}\nFree disk: {free_gb:.1f} GB (need about {need_gb:.0f} GB)")
    if free_gb < need_gb:
        sys.exit("Not enough free disk space.")

    log = load_log()
    skip = {"ok", "length_mismatch"} | (set() if args.retry_failed else {"unavailable"})
    todo = [e for e in episodes if log.get((e["Show"], e["EpId"]), {}).get("status") not in skip]
    if args.limit:
        todo = todo[: args.limit]
    print(f"Episodes: {len(episodes)} total, {len(todo)} to process\n")

    done = 0
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(process_episode, e, by_ep[(e["Show"], e["EpId"])], ffmpeg,
                                args.keep_episodes): e for e in todo}
            for f in as_completed(futs):
                row = f.result()
                row["finished_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
                log[(row["show"], row["ep_id"])] = row
                save_log(log)  # save after every episode so progress survives interruption
                done += 1
                print(f"[{done}/{len(todo)}] {row['show']:<12} ep {row['ep_id']:>3}: {row['status']:<16}"
                      f" {row['n_clips_written']}/{row['n_clips_expected']} clips {row['detail'][:60]}", flush=True)
    except KeyboardInterrupt:
        print("\nStopped. Progress is saved; run the same command again to resume.")
        return
    finally:
        shutil.rmtree(TMP, ignore_errors=True)

    rows = list(log.values())
    summarise([(r["show"], r["status"] == "ok") for r in rows], "episodes ok")
    clips_ok = sum(int(r["n_clips_written"]) for r in rows)
    print(f"\nClips written: {clips_ok} of {sum(len(v) for v in by_ep.values())}")
    print(f"Clips folder:  {CLIPS}\nLog:           {LOG}")


if __name__ == "__main__":
    main()
