"""Stage 5 data: audio context around dev clips, plus a few whole episodes.

Re-downloads only the episodes that contain dev clips and NO test clips
(the test set is spent), and keeps:
  * for every dev clip: the clip plus PAD seconds either side (default 1.5 s)
    -> data/interim/sep28k_context/<Show>/<EpId>/<clip_key>.wav
       + data/metadata/sep28k_context_index.csv (where the clip sits inside)
  * 3 short "demo" episodes in full -> data/raw/sep28k_episodes/<Show>/<EpId>.wav

    python scripts/fetch_context.py --limit 2      # quick check
    python scripts/fetch_context.py                # all (~108 episodes, ~0.6 GB kept)

Resumable. Same licence rules as fetch_sep28k.py (local, non-commercial, never shared).
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd  # noqa: E402
from scipy.io import wavfile  # noqa: E402

from fetch_sep28k import decode_16k_mono, find_ffmpeg, http_get, read_csv  # noqa: E402
from fluence.data.sep28k import exclude_unusable, load_labels  # noqa: E402
from fluence.localization import SR, clean_dev_episodes, context_bounds, pick_demo_episodes  # noqa: E402

INDEX_FIELDS = ["clip_key", "Show", "EpId", "ClipId", "ctx_path", "ctx_start", "clip_start_in_ctx", "ctx_len"]


def process(ep, clips: pd.DataFrame, url: str, ffmpeg: str, ctx_root: Path, ep_root: Path, tmp: Path,
            pad_s: float, keep_full: bool) -> tuple[list[dict], str]:
    show, ep_id = ep["Show"], int(ep["EpId"])
    out_dir = ctx_root / show / str(ep_id)
    done_marker = out_dir / "_done"
    if done_marker.exists() and (not keep_full or (ep_root / show / f"{ep_id}.wav").exists()):
        return [], "skip"
    ext = next((e for e in (".mp3", ".m4a", ".mp4") if e in url.lower()), ".mp3")
    tmp.mkdir(parents=True, exist_ok=True)
    f = tmp / f"ctx_{show}_{ep_id}{ext}"
    try:
        http_get(url, f, label=f"{show} ep {ep_id}")
        audio = decode_16k_mono(ffmpeg, f)
    except Exception as e:
        return [], f"unavailable: {str(e)[:100]}"
    finally:
        f.unlink(missing_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for _, c in clips.iterrows():
        start, stop = int(c["Start"]), int(c["Stop"])
        if stop > len(audio):
            continue
        a, b = context_bounds(start, stop, len(audio), pad_s)
        p = out_dir / f"{c['clip_key']}.wav"
        wavfile.write(p, SR, audio[a:b])
        rows.append({"clip_key": c["clip_key"], "Show": show, "EpId": ep_id, "ClipId": int(c["ClipId"]),
                     "ctx_path": p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else str(p),
                     "ctx_start": a, "clip_start_in_ctx": start - a, "ctx_len": b - a})
    if keep_full:
        (ep_root / show).mkdir(parents=True, exist_ok=True)
        wavfile.write(ep_root / show / f"{ep_id}.wav", SR, audio)
    pd.DataFrame(rows, columns=INDEX_FIELDS).to_csv(out_dir / "_index.csv", index=False)
    done_marker.write_text("ok")
    return rows, f"ok ({len(rows)} clips{', full episode kept' if keep_full else ''})"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metadata", type=Path, default=ROOT / "data/metadata")
    ap.add_argument("--ctx-root", type=Path, default=ROOT / "data/interim/sep28k_context")
    ap.add_argument("--episodes-root", type=Path, default=ROOT / "data/raw/sep28k_episodes")
    ap.add_argument("--pad", type=float, default=1.5)
    ap.add_argument("--demo", type=int, default=3, help="number of whole episodes to keep")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()

    df = exclude_unusable(load_labels(args.metadata / "SEP-28k-Extended_clips.csv"))
    log = pd.read_csv(args.metadata / "sep28k_fetch_log.csv")
    eps = clean_dev_episodes(df, log)
    demo = pick_demo_episodes(eps, n=args.demo)
    demo_keys = set(zip(demo["Show"], demo["EpId"].astype(int)))
    urls = {(r["Show"], int(r["EpId"])): r["url"] for r in read_csv(args.metadata / "SEP-28k-Extended_episodes.csv")}
    dev = df[df["SEP28k-E"].astype(str) == "dev"]
    todo = eps.head(args.limit) if args.limit else eps
    print(f"{len(eps)} clean episodes ({int(eps['n_dev'].sum())} dev clips); demo episodes: "
          + ", ".join(f"{s} {e}" for s, e in sorted(demo_keys)), flush=True)

    ffmpeg = find_ffmpeg()
    tmp = args.ctx_root.parent / "_ctx_tmp"
    jobs = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for _, ep in todo.iterrows():
            key = (ep["Show"], int(ep["EpId"]))
            clips = dev[(dev["Show"] == key[0]) & (dev["EpId"] == key[1])]
            jobs.append(pool.submit(process, ep, clips, urls[key], ffmpeg, args.ctx_root, args.episodes_root,
                                    tmp, args.pad, key in demo_keys))
        for i, j in enumerate(as_completed(jobs), 1):
            _, status = j.result()
            print(f"[{i}/{len(jobs)}] {status}", flush=True)
    shutil.rmtree(tmp, ignore_errors=True)

    parts = [pd.read_csv(p) for p in sorted(args.ctx_root.glob("*/*/_index.csv"))]
    index = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=INDEX_FIELDS)
    index.to_csv(args.metadata / "sep28k_context_index.csv", index=False)
    demo.to_csv(args.metadata / "sep28k_demo_episodes.csv", index=False)
    print(f"\nContext excerpts: {len(index)} dev clips -> {args.metadata / 'sep28k_context_index.csv'}")


if __name__ == "__main__":
    main()
