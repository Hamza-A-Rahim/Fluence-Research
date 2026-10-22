"""Stage 5 tests: window maths, context download (file:// episodes), and the EXP-005 analysis
with stand-in window embeddings (PyTorch not needed)."""

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import wavfile

from fluence.features.embeddings import model_slug, write_shard
from fluence.localization import (
    SR, aggregate_over_offsets, clean_dev_episodes, context_bounds, offset_windows, pick_demo_episodes,
    sliding_windows,
)
from test_baseline_pipeline import ROOT, build_synthetic


class TestWindowMaths(unittest.TestCase):
    def test_context_bounds_clip_to_episode(self):
        self.assertEqual(context_bounds(48000, 96000, 200000, 1.5), (24000, 120000))
        self.assertEqual(context_bounds(1000, 49000, 60000, 1.5), (0, 60000))

    def test_offset_windows_positions_and_edges(self):
        x = np.arange(6 * SR, dtype=np.float32)  # value = sample index
        kept, wins = offset_windows(x, 24000, [-1.5, 0.0, 1.5, 2.0])
        self.assertEqual(kept, [-1.5, 0.0, 1.5])  # +2.0 would run past the end
        self.assertEqual([int(w[0]) for w in wins], [0, 24000, 48000])
        self.assertTrue(all(len(w) == 3 * SR for w in wins))

    def test_sliding_windows(self):
        starts, wins = sliding_windows(np.zeros(10 * SR, np.float32), hop_s=1.0)
        self.assertEqual(starts.tolist(), [0, 1, 2, 3, 4, 5, 6, 7])
        self.assertEqual(len(wins[-1]), 3 * SR)

    def test_aggregate(self):
        s = pd.DataFrame({"clip_key": ["a"] * 3 + ["b"] * 2, "offset": [-2.0, 0.0, 1.0, 0.0, 0.5],
                          "score": [0.99, 0.2, 0.4, 0.1, 0.3]})
        self.assertAlmostEqual(aggregate_over_offsets(s, "max")["a"], 0.4)  # -2.0 is outside +-1.5
        self.assertAlmostEqual(aggregate_over_offsets(s, "mean")["b"], 0.2)

    def test_clean_episodes_exclude_any_test(self):
        df = pd.DataFrame({"Show": ["A"] * 4 + ["B"] * 2, "EpId": [0, 0, 1, 1, 0, 0],
                           "SEP28k-E": ["dev", "train", "dev", "test", "train", "train"]})
        e = clean_dev_episodes(df)
        self.assertEqual(list(zip(e["Show"], e["EpId"])), [("A", 0)])
        self.assertEqual(len(pick_demo_episodes(e.assign(episode_seconds=100.0), n=3)), 1)


class TestStage5EndToEnd(unittest.TestCase):
    def test_fetch_context_and_localize(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            meta, clips = build_synthetic(base)
            labels = pd.read_csv(meta / "SEP-28k-Extended_clips.csv")
            # rebuild each synthetic episode by concatenating its clips (Start = ClipId * 48000)
            media = base / "media"
            media.mkdir()
            eps, logrows = [], []
            for s in range(12):
                audio = np.concatenate([wavfile.read(clips / "Synth" / str(s) / f"Synth_{s}_{c}.wav")[1]
                                        for c in range(16)])
                p = media / f"ep{s}.wav"
                wavfile.write(p, SR, audio)
                eps.append({"url": p.as_uri(), "Show": "Synth", "EpId": s})
                logrows.append({"show": "Synth", "ep_id": s, "status": "ok", "episode_seconds": len(audio) / SR})
            pd.DataFrame(eps).to_csv(meta / "SEP-28k-Extended_episodes.csv", index=False)
            pd.DataFrame(logrows).to_csv(meta / "sep28k_fetch_log.csv", index=False)

            ctx_root, ep_root = base / "ctx", base / "episodes"
            res = subprocess.run([sys.executable, str(ROOT / "scripts/fetch_context.py"), "--metadata", str(meta),
                                  "--ctx-root", str(ctx_root), "--episodes-root", str(ep_root), "--demo", "1"],
                                 capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, res.stdout + res.stderr)
            index = pd.read_csv(meta / "sep28k_context_index.csv")
            dev_keys = set(labels[labels["SEP28k-E"] == "dev"].apply(lambda r: f"Synth_{r.EpId}_{r.ClipId}", axis=1))
            self.assertEqual(set(index["clip_key"]), dev_keys)  # only dev clips, all of them
            self.assertEqual(len(list(ep_root.rglob("*.wav"))), 1)  # one demo episode kept
            # the clip inside its context excerpt is sample-identical to the original clip
            r = index.iloc[5]
            ctx = wavfile.read(r["ctx_path"])[1]
            orig = wavfile.read(clips / "Synth" / str(r["EpId"]) / f"{r['clip_key']}.wav")[1]
            a = int(r["clip_start_in_ctx"])
            # synthetic clips are float WAVs, excerpts are 16-bit PCM: compare on the same scale
            np.testing.assert_allclose(ctx[a:a + len(orig)] / 32768.0, orig, atol=2e-4)

            # ---- stand-in encoder: "embedding" = clip summary features of the window
            sys.path.insert(0, str(ROOT / "src"))
            from fluence.data.datasets import load_available_clips
            from fluence.features.spectral import clip_summary_features
            from fluence.localization import offset_windows
            from fluence.preprocessing.audio import load_wav
            model = "fake/encoder"
            df = load_available_clips(meta, clips)
            feats = np.stack([clip_summary_features(load_wav(p)[0])[0] for p in df["path"]])
            emb = base / "data/processed/embeddings" / model_slug(model)
            emb.mkdir(parents=True)
            write_shard(emb, 0, df["clip_key"].tolist(), feats[:, None, :], np.zeros_like(feats)[:, None, :])
            (emb / "manifest.json").write_text("{}")
            wdir = base / "data/processed/windows" / model_slug(model)
            wdir.mkdir(parents=True)
            keys, offs, ms = [], [], []
            for _, r in index.iterrows():
                x, _ = load_wav(r["ctx_path"])
                kept, wins = offset_windows(x, int(r["clip_start_in_ctx"]), [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5])
                for o, w in zip(kept, wins):
                    keys.append(r["clip_key"])
                    offs.append(o)
                    ms.append(clip_summary_features(w)[0])
            ms = np.array(ms)
            np.savez(wdir / "offsets_0000.npz", clip_key=np.array(keys), offset=np.array(offs, np.float32),
                     mean=ms.astype(np.float16), std=np.zeros_like(ms, np.float16))
            demo = pd.read_csv(meta / "sep28k_demo_episodes.csv").iloc[0]
            from fluence.localization import sliding_windows
            x, _ = load_wav(ep_root / demo["Show"] / f"{int(demo['EpId'])}.wav")
            starts, wins = sliding_windows(x, 1.0)
            tm = np.array([clip_summary_features(w)[0] for w in wins])
            np.savez(wdir / f"timeline_{demo['Show']}_{int(demo['EpId'])}.npz", start_s=starts.astype(np.float32),
                     mean=tm.astype(np.float16), std=np.zeros_like(tm, np.float16))

            tg = ["any_with_interjections", "any_without_interjections", "Prolongation", "Block", "SoundRep",
                  "WordRep", "Interjection"]
            cfg = {"primary_system": model, "targets": tg,
                   "systems": {model: {"features": {"type": "embedding", "model": model, "layer": 0, "pooling": "mean+std"},
                                       "C": {t: 1.0 for t in tg}}}}
            cfg_path = base / "cfg.json"
            cfg_path.write_text(json.dumps(cfg))
            out = base / "out"
            res = subprocess.run([sys.executable, str(ROOT / "scripts/localize.py"), "--config", str(cfg_path),
                                  "--metadata", str(meta), "--clips", str(clips), "--data-root", str(base),
                                  "--out", str(out), "--n-boot", "50"], capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, res.stdout + res.stderr)
            r = json.loads((out / "metrics.json").read_text())
            self.assertGreater(r["sanity_corr_offset0_vs_clip"]["Block"], 0.99)  # offset 0 == the clip
            self.assertEqual(r["n_speakers"], 4)
            self.assertGreater(r["targets"]["Block"]["offset_auc"]["0.0"]["auc"], 0.9)
            for f in ("offset_auc.png", "score_profile.png", "report.md"):
                self.assertTrue((out / f).is_file())
            self.assertEqual(len(r["timelines"]), 1)


if __name__ == "__main__":
    unittest.main()
