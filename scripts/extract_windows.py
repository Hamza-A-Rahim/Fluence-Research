"""Stage 5: encoder embeddings for shifted windows and whole-episode timelines.

For every dev clip with saved context (scripts/fetch_context.py): 3 s windows
starting at clip_start + offset, offsets -1.5 ... +1.5 s in 0.5 s steps.
For each demo episode: 3 s windows every --hop seconds over the whole episode.

Only the layer used by the frozen classifier is kept (from configs/final_eval.json).

    python scripts/extract_windows.py --limit 20     # quick check
    python scripts/extract_windows.py                # all (~1-2 h on a fast CPU)

Resumable. Output: data/processed/windows/<model>/  (git-ignored)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fluence.features.embeddings import model_slug  # noqa: E402
from fluence.localization import offset_windows, sliding_windows  # noqa: E402
from fluence.preprocessing.audio import load_wav  # noqa: E402

OFFSETS = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
SHARD = 250  # clips per shard (x 7 windows; ~350 MB of audio in memory)


def embed_layer(fe, model, waves, device, layer, batch):
    from extract_embeddings import embed_batch
    ms, ss = [], []
    for i in range(0, len(waves), batch):
        m, s = embed_batch(fe, model, waves[i:i + batch], device)
        ms.append(m[:, layer])
        ss.append(s[:, layer])
    return np.concatenate(ms), np.concatenate(ss)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=ROOT / "configs/final_eval.json")
    ap.add_argument("--metadata", type=Path, default=ROOT / "data/metadata")
    ap.add_argument("--episodes-root", type=Path, default=ROOT / "data/raw/sep28k_episodes")
    ap.add_argument("--out", type=Path, default=ROOT / "data/processed/windows")
    ap.add_argument("--hop", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="only N new clips (testing)")
    ap.add_argument("--skip-demo", action="store_true")
    args = ap.parse_args()

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    spec = cfg["systems"][cfg["primary_system"]]["features"]
    name, layer = spec["model"], int(spec["layer"])
    out = args.out / model_slug(name)
    out.mkdir(parents=True, exist_ok=True)

    import torch
    from extract_embeddings import load_model, model_revision
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(os.cpu_count() or 4)
    print(f"Model {name}, layer {layer}, device {device}", flush=True)
    fe, model = load_model(name, device)
    (out / "manifest.json").write_text(json.dumps({"model": name, "layer": layer, "offsets_s": OFFSETS,
                                                   "hop_s": args.hop, "revision": model_revision(name),
                                                   "torch": torch.__version__}, indent=2), encoding="utf-8")

    # ---- offset windows around dev clips
    index = pd.read_csv(args.metadata / "sep28k_context_index.csv")
    done = set()
    for p in out.glob("offsets_*.npz"):
        done.update(np.load(p)["clip_key"].tolist())
    todo = index[~index["clip_key"].isin(done)]
    if args.limit:
        todo = todo.head(args.limit)
    print(f"Offset windows: {len(done)} clips done, {len(todo)} to go", flush=True)
    shard_i = len(list(out.glob("offsets_*.npz")))
    t0 = time.time()
    for s in range(0, len(todo), SHARD):
        chunk = todo.iloc[s:s + SHARD]
        keys, offs, waves = [], [], []
        for _, r in chunk.iterrows():
            x, _ = load_wav(ROOT / r["ctx_path"] if not Path(r["ctx_path"]).is_absolute() else r["ctx_path"])
            kept, wins = offset_windows(x, int(r["clip_start_in_ctx"]), OFFSETS)
            keys += [r["clip_key"]] * len(kept)
            offs += kept
            waves += wins
        m, sd = embed_layer(fe, model, waves, device, layer, args.batch_size)
        np.savez(out / f"offsets_{shard_i:04d}.npz", clip_key=np.array(keys), offset=np.array(offs, np.float32),
                 mean=m.astype(np.float16), std=sd.astype(np.float16))
        shard_i += 1
        n = s + len(chunk)
        rate = n / (time.time() - t0)
        print(f"  {n}/{len(todo)} clips ({len(waves)} windows in this shard)  ~{(len(todo) - n) / rate / 60:.0f} min left",
              flush=True)

    # ---- whole-episode timelines
    if args.skip_demo:
        return
    demo_csv = args.metadata / "sep28k_demo_episodes.csv"
    for _, e in (pd.read_csv(demo_csv).iterrows() if demo_csv.exists() else []):
        target = out / f"timeline_{e['Show']}_{int(e['EpId'])}.npz"
        wav = args.episodes_root / e["Show"] / f"{int(e['EpId'])}.wav"
        if target.exists() or not wav.exists():
            continue
        x, _ = load_wav(wav, min_duration_s=3.0)
        starts, wins = sliding_windows(x, hop_s=args.hop)
        print(f"Timeline {e['Show']} {int(e['EpId'])}: {len(wins)} windows", flush=True)
        m, sd = embed_layer(fe, model, wins, device, layer, args.batch_size)
        np.savez(target, start_s=starts.astype(np.float32), mean=m.astype(np.float16), std=sd.astype(np.float16))
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
