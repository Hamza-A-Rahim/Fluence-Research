"""Stage 4: turn every clip into wav2vec 2.0 embeddings (run once; resumable).

Needs PyTorch + transformers (see docs/setup_windows.md, "Stage 4"). The first
run downloads each model from Hugging Face (~0.4 GB base, ~1.3 GB XLSR-53).

    python scripts/extract_embeddings.py --limit 16          # quick check (~1 min)
    python scripts/extract_embeddings.py                     # both models, all clips (hours)

Safe to stop with Ctrl+C; re-running continues from the last finished shard.
Output: data/processed/embeddings/<model>/  (git-ignored)
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Windows consoles and pipes may default to a legacy code page (cp1252) that
# cannot print characters like κ or ±; never crash on output.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np  # noqa: E402

from fluence.data.datasets import load_available_clips  # noqa: E402
from fluence.features.embeddings import (  # noqa: E402
    SHARD_SIZE, done_keys, model_slug, pool_hidden_states, write_shard,
)
from fluence.preprocessing.audio import TARGET_SR, load_wav  # noqa: E402

# D-010: English model used in prior SEP-28k work, and the multilingual model
# that worked best in Haas & Bayerl (2026) and is relevant to Urdu-English.
DEFAULT_MODELS = ["facebook/wav2vec2-base-960h", "facebook/wav2vec2-large-xlsr-53"]
BYTES_PER_CLIP = {"base": 13 * 768 * 2 * 2, "large": 25 * 1024 * 2 * 2}  # layers x dim x (mean,std) x fp16


def load_model(name: str, device: str):
    import torch
    from transformers import AutoFeatureExtractor, Wav2Vec2FeatureExtractor, Wav2Vec2Model
    try:
        fe = AutoFeatureExtractor.from_pretrained(name)
    except Exception:  # some checkpoints ship without a preprocessor config
        fe = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=TARGET_SR, padding_value=0.0,
                                      do_normalize=True, return_attention_mask=False)
    model = Wav2Vec2Model.from_pretrained(name).to(device).eval()
    torch.set_grad_enabled(False)
    return fe, model


def model_revision(name: str) -> str | None:
    """Exact commit of the downloaded checkpoint, read from the local Hugging Face cache."""
    try:
        from huggingface_hub import try_to_load_from_cache
        path = try_to_load_from_cache(name, "config.json")
        if isinstance(path, str):
            return Path(path).parent.name  # .../snapshots/<commit-sha>/config.json
    except Exception:
        pass
    return None


def embed_batch(fe, model, waves: list[np.ndarray], device: str):
    import torch
    inputs = fe(waves, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.inference_mode():
        out = model(**inputs, output_hidden_states=True)
    return pool_hidden_states([h.float().cpu().numpy() for h in out.hidden_states])


def run_model(name: str, df, out_root: Path, batch: int, device: str, limit: int) -> None:
    import torch
    import transformers
    out_dir = out_root / model_slug(name)
    out_dir.mkdir(parents=True, exist_ok=True)
    have = done_keys(out_dir)
    todo = df[~df["clip_key"].isin(have)]
    if limit:
        todo = todo.head(limit)
    print(f"\n=== {name}: {len(have)} clips done, {len(todo)} to go ===", flush=True)
    if todo.empty:
        return

    need = len(todo) * BYTES_PER_CLIP["large" if "large" in name else "base"] / 1e9
    free = shutil.disk_usage(out_root).free / 1e9
    print(f"Disk: need ~{need:.1f} GB, free {free:.1f} GB", flush=True)
    if free < need + 1:
        sys.exit("Not enough free disk space.")

    print("Loading model (first time downloads it)...", flush=True)
    fe, model = load_model(name, device)
    (out_dir / "manifest.json").write_text(json.dumps({
        "model": name, "revision": getattr(model.config, "_commit_hash", None) or model_revision(name),
        "n_layers_including_cnn_output": model.config.num_hidden_layers + 1,
        "hidden_size": model.config.hidden_size, "pooling": "mean and std over frames, per layer",
        "sample_rate": TARGET_SR, "device": device, "torch": torch.__version__,
        "transformers": transformers.__version__, "python": platform.python_version(),
        "platform": platform.platform(), "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, indent=2), encoding="utf-8")

    next_index = len(list(out_dir.glob("shard_[0-9]*.npz")))
    keys, means, stds = [], [], []
    t0, n_done = time.time(), 0
    rows = list(todo[["clip_key", "path"]].itertuples(index=False))
    try:
        for i in range(0, len(rows), batch):
            chunk = rows[i:i + batch]
            waves = [load_wav(p)[0] for _, p in chunk]
            m, s = embed_batch(fe, model, waves, device)
            keys += [k for k, _ in chunk]
            means.append(m)
            stds.append(s)
            n_done += len(chunk)
            if len(keys) >= SHARD_SIZE:
                write_shard(out_dir, next_index, keys, np.concatenate(means), np.concatenate(stds))
                next_index += 1
                keys, means, stds = [], [], []
            if n_done == len(chunk) or n_done % (batch * 25) < batch:
                rate = n_done / (time.time() - t0)
                eta = (len(rows) - n_done) / rate / 60
                print(f"  {n_done}/{len(rows)} clips  {rate:.1f} clips/s  ~{eta:.0f} min left", flush=True)
    except KeyboardInterrupt:
        print("\nStopping; saving what's finished...", flush=True)
    finally:
        if keys:
            write_shard(out_dir, next_index, keys, np.concatenate(means), np.concatenate(stds))
    print(f"Saved to {out_dir}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", help="Hugging Face model id (repeatable)")
    ap.add_argument("--metadata", type=Path, default=ROOT / "data/metadata")
    ap.add_argument("--clips", type=Path, default=ROOT / "data/interim/sep28k_clips")
    ap.add_argument("--out", type=Path, default=ROOT / "data/processed/embeddings")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="only N new clips per model (for testing)")
    args = ap.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(os.cpu_count() or 4)
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else
                                 f", {torch.get_num_threads()} threads"))

    df = load_available_clips(args.metadata, args.clips)
    if df.empty:
        sys.exit("No clips found. Run scripts/fetch_sep28k.py first.")
    for name in args.model or DEFAULT_MODELS:
        run_model(name, df, args.out, args.batch_size, device, args.limit)


if __name__ == "__main__":
    main()
