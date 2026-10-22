"""The frozen EXP-004 primary system, wrapped for the local demo.

Loads the classifiers exported by scripts/export_demo_model.py and the
XLSR-53 encoder at the revision recorded in configs/final_eval.json (refuses a
different revision). Embeddings are pooled exactly as in extract_embeddings.py
and rounded through float16, as the stored training embeddings were.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from fluence.inference import CLIP_S, SR, regions, timeline, window_starts  # noqa: E402

MODEL_DIR = ROOT / "outputs/demo_model"
HOP_S = 0.5      # same step as the EXP-005 offsets
STEP_S = 0.1     # timeline resolution
BATCH = 8


class Engine:
    def __init__(self, model_dir: Path = MODEL_DIR):
        import joblib
        if not (model_dir / "classifiers.joblib").exists():
            raise FileNotFoundError(f"No exported model in {model_dir}. Run: python scripts/export_demo_model.py")
        bundle = joblib.load(model_dir / "classifiers.joblib")
        self.targets: list[str] = bundle["targets"]
        self.models = bundle["models"]
        self.card = json.loads((model_dir / "card.json").read_text(encoding="utf-8"))
        self.thresholds: dict[str, float] = self.card["thresholds"]
        self.layer = int(self.card["encoder"]["layer"])
        self.status, self.error = "loading", None
        self._fe = self._encoder = None
        self._device = "cpu"
        self._lock = threading.Lock()

    def load_encoder(self) -> None:
        try:
            import torch
            from extract_embeddings import load_model, model_revision
            name, want = self.card["encoder"]["model"], self.card["encoder"].get("revision")
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            if self._device == "cpu":
                torch.set_num_threads(os.cpu_count() or 4)
            self._fe, self._encoder = load_model(name, self._device)
            have = model_revision(name)
            if want and have != want:
                raise RuntimeError(f"Encoder revision {have} differs from the frozen {want}")
            self.status = "ready"
        except Exception as exc:  # reported through /api/health
            self.status, self.error = "error", str(exc)

    def embed(self, waves: list[np.ndarray]) -> np.ndarray:
        from extract_embeddings import embed_batch
        m, s = embed_batch(self._fe, self._encoder, waves, self._device)
        X = np.concatenate([m[:, self.layer], s[:, self.layer]], axis=1)
        return X.astype(np.float16).astype(np.float32)

    def analyze(self, x: np.ndarray, progress: Callable[[float], None] = lambda p: None) -> dict:
        if self.status != "ready":
            raise RuntimeError("Model is not ready")
        t0 = time.time()
        n = int(CLIP_S * SR)
        starts = window_starts(len(x), HOP_S)
        feats = []
        with self._lock:
            for i in range(0, len(starts), BATCH):
                feats.append(self.embed([x[a:a + n] for a in starts[i:i + BATCH]]))
                progress(min(1.0, (i + BATCH) / len(starts)))
        X = np.concatenate(feats)
        duration = len(x) / SR
        tracks = {}
        times = None
        for t in self.targets:
            p = self.models[t].predict_proba(X)[:, 1]
            times, score = timeline(starts / SR, p, duration, STEP_S)
            thr = float(self.thresholds[t])
            reg = regions(times, score, thr, STEP_S)
            flagged = sum(r["end"] - r["start"] for r in reg)
            tracks[t] = {"threshold": thr, "scores": np.round(score, 3).tolist(), "regions": reg,
                         "flagged_s": round(flagged, 2), "share": round(flagged / duration, 4),
                         "max": round(float(score.max()), 3), "mean": round(float(score.mean()), 3)}
        return {"duration_s": round(duration, 2), "hop_s": HOP_S, "step_s": STEP_S, "windows": int(len(starts)),
                "times": np.round(times, 2).tolist(), "tracks": tracks, "elapsed_s": round(time.time() - t0, 1)}
