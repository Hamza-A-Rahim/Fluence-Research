"""Tests for embedding storage/loading and the EXP-003 training script.

PyTorch is not needed: stand-in embeddings are written in the same shard format.
The real model call is checked on the laptop with `extract_embeddings.py --limit 16`.
"""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from fluence.features.embeddings import (
    done_keys, load_embeddings, model_slug, pool_hidden_states, write_shard,
)
from test_baseline_pipeline import ROOT, build_synthetic


class TestShards(unittest.TestCase):
    def test_pooling_shapes_and_values(self):
        h = [np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4) + 10 * L for L in range(5)]
        mean, std = pool_hidden_states(h)
        self.assertEqual(mean.shape, (2, 5, 4))
        np.testing.assert_allclose(mean[0, 0], h[0][0].mean(axis=0))
        np.testing.assert_allclose(std[1, 4], h[4][1].std(axis=0))

    def test_roundtrip_alignment_and_resume(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            rng = np.random.default_rng(0)
            m1, s1 = rng.random((3, 2, 4)), rng.random((3, 2, 4))
            write_shard(d, 0, ["a", "b", "c"], m1, s1)
            write_shard(d, 1, ["d"], rng.random((1, 2, 4)), rng.random((1, 2, 4)))
            self.assertEqual(done_keys(d), {"a", "b", "c", "d"})
            X, found = load_embeddings(d, ["c", "zz", "a"], layer=1, pooling="mean+std")
            self.assertEqual(X.shape, (3, 8))
            self.assertEqual(found.tolist(), [True, False, True])
            np.testing.assert_allclose(X[0, :4], m1[2, 1], atol=1e-3)   # fp16 storage
            np.testing.assert_allclose(X[2, 4:], s1[0, 1], atol=1e-3)
            X_all, _ = load_embeddings(d, ["a"], layer=None, pooling="mean")
            self.assertEqual(X_all.shape, (1, 2, 4))

    def test_slug(self):
        self.assertEqual(model_slug("facebook/wav2vec2-base-960h"), "facebook__wav2vec2-base-960h")


class TestTrainEmbeddings(unittest.TestCase):
    def test_end_to_end(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            meta, clips = build_synthetic(base)
            sys.path.insert(0, str(ROOT / "src"))
            from fluence.data.datasets import load_available_clips
            from fluence.features.spectral import clip_summary_features
            from fluence.preprocessing.audio import load_wav
            df = load_available_clips(meta, clips)
            # stand-in "encoder": 3 layers; only layer 2 carries the clip's summary features
            feats = np.stack([clip_summary_features(load_wav(p)[0])[0] for p in df["path"]])
            rng = np.random.default_rng(0)
            noise = rng.standard_normal((len(df), 3, feats.shape[1]))
            mean = noise.copy()
            mean[:, 2] = feats
            emb = base / "emb" / model_slug("fake/encoder")
            emb.mkdir(parents=True)
            write_shard(emb, 0, df["clip_key"].tolist(), mean, np.abs(noise))
            (emb / "manifest.json").write_text(json.dumps({"model": "fake/encoder", "revision": "x"}))

            out = base / "out"
            res = subprocess.run([sys.executable, str(ROOT / "scripts/train_embeddings.py"),
                                  "--model", "fake/encoder", "--metadata", str(meta), "--clips", str(clips),
                                  "--emb", str(base / "emb"), "--out", str(out), "--pooling", "mean",
                                  "--folds", "3", "--n-boot", "50"], capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, res.stdout + res.stderr)
            r = json.loads((out / model_slug("fake/encoder") / "metrics.json").read_text())
            self.assertEqual(r["layer"], 2)  # the sweep finds the informative layer
            self.assertGreater(r["SEP28k-E"]["any_without_interjections"]["roc_auc"], 0.9)
            for e in ("Block", "SoundRep", "Prolongation"):
                self.assertIn(e, r["SEP12k"])
            self.assertTrue((out / model_slug("fake/encoder") / "layer_sweep.png").is_file())
            self.assertIn("Test set not evaluated", (out / model_slug("fake/encoder") / "report.md").read_text())


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(__import__("importlib").util.find_spec("torch")
                     and __import__("importlib").util.find_spec("transformers"),
                     "PyTorch/transformers not installed")
class TestRealModelCodePath(unittest.TestCase):
    """Runs the actual transformers call on a tiny, randomly initialised wav2vec 2.0 (no download)."""

    def test_embed_batch(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        from transformers import Wav2Vec2Config, Wav2Vec2FeatureExtractor, Wav2Vec2Model
        from extract_embeddings import embed_batch
        cfg = Wav2Vec2Config(hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
                             conv_dim=(16, 16), conv_kernel=(10, 3), conv_stride=(5, 2),
                             num_conv_pos_embeddings=16, num_conv_pos_embedding_groups=2)
        model = Wav2Vec2Model(cfg).eval()
        fe = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True)
        waves = [np.random.default_rng(i).standard_normal(48000).astype(np.float32) * 0.1 for i in range(3)]
        mean, std = embed_batch(fe, model, waves, "cpu")
        self.assertEqual(mean.shape, (3, 3, 32))   # 3 clips, 2 layers + CNN output, 32 dims
        self.assertEqual(std.shape, (3, 3, 32))
        self.assertTrue(np.isfinite(mean).all())


@unittest.skipUnless(__import__("importlib").util.find_spec("torch")
                     and __import__("importlib").util.find_spec("transformers"),
                     "PyTorch/transformers not installed")
class TestWindowExtractorCodePath(unittest.TestCase):
    def test_embed_layer_keeps_one_layer(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        from transformers import Wav2Vec2Config, Wav2Vec2FeatureExtractor, Wav2Vec2Model
        from extract_windows import embed_layer
        cfg = Wav2Vec2Config(hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
                             conv_dim=(16, 16), conv_kernel=(10, 3), conv_stride=(5, 2),
                             num_conv_pos_embeddings=16, num_conv_pos_embedding_groups=2)
        model = Wav2Vec2Model(cfg).eval()
        fe = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True)
        waves = [np.random.default_rng(i).standard_normal(48000).astype(np.float32) * 0.1 for i in range(5)]
        m, s = embed_layer(fe, model, waves, "cpu", layer=2, batch=2)  # batches of 2, 2, 1
        self.assertEqual(m.shape, (5, 32))
        self.assertEqual(s.shape, (5, 32))
