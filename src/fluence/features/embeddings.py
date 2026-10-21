"""Pretrained speech-encoder embeddings (Stage 4).

Each 3 s clip goes through a frozen wav2vec 2.0 model. For every layer we keep
the mean and the standard deviation of its frame vectors over time. Because the
transformer layers see the whole clip, each frame vector already reflects its
context (what came before and after), which is the timing information the
EXP-001/002 summary features lacked.

Stored as shards so long runs can be stopped and resumed:
    data/processed/embeddings/<model-slug>/shard_00000.npz  (keys, mean, std)
    data/processed/embeddings/<model-slug>/manifest.json
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

SHARD_SIZE = 1000


def model_slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "__", name)


def pool_hidden_states(hidden_states) -> tuple[np.ndarray, np.ndarray]:
    """hidden_states: sequence of L arrays/tensors [B, T, H] -> (mean, std) each [B, L, H]."""
    stack = np.stack([np.asarray(h, dtype=np.float32) for h in hidden_states], axis=1)  # B, L, T, H
    return stack.mean(axis=2), stack.std(axis=2)


def write_shard(out_dir: Path, index: int, keys: list[str], mean: np.ndarray, std: np.ndarray) -> Path:
    path = Path(out_dir) / f"shard_{index:05d}.npz"
    tmp = path.with_name(path.stem + "_tmp.npz")  # write-then-rename: no half-written shards
    np.savez(tmp, keys=np.array(keys), mean=mean.astype(np.float16), std=std.astype(np.float16))
    tmp.replace(path)
    return path


def done_keys(out_dir: Path) -> set[str]:
    keys: set[str] = set()
    for p in sorted(Path(out_dir).glob("shard_[0-9]*.npz")):
        if p.stem.endswith("_tmp"):
            continue
        keys.update(np.load(p)["keys"].tolist())
    return keys


def load_embeddings(out_dir: Path, keys, layer: int | None = None, pooling: str = "mean+std"
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, found_mask) aligned to ``keys``.

    layer=None returns all layers: shape [n, L, D]; otherwise [n, D].
    pooling: "mean", "std" or "mean+std" (concatenated).
    """
    keys = list(keys)
    pos = {k: i for i, k in enumerate(keys)}
    X = None
    found = np.zeros(len(keys), bool)
    for p in sorted(Path(out_dir).glob("shard_[0-9]*.npz")):
        if p.stem.endswith("_tmp"):
            continue
        z = np.load(p)
        parts = {"mean": [z["mean"]], "std": [z["std"]], "mean+std": [z["mean"], z["std"]]}[pooling]
        block = np.concatenate(parts, axis=-1)  # [n, L, D]
        if layer is not None:
            block = block[:, layer, :]
        if X is None:
            X = np.zeros((len(keys),) + block.shape[1:], np.float32)
        for k, row in zip(z["keys"].tolist(), block):
            if k in pos:
                X[pos[k]] = row
                found[pos[k]] = True
    if X is None:
        raise FileNotFoundError(f"No embedding shards in {out_dir}")
    return X, found


def read_manifest(out_dir: Path) -> dict:
    return json.loads((Path(out_dir) / "manifest.json").read_text(encoding="utf-8"))
