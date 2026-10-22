"""End-to-end test of the final stage on synthetic data:
EXP-001/002/003 outputs -> paired comparison -> frozen config -> single test evaluation (and refusal to repeat).
"""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from fluence.features.embeddings import model_slug, write_shard
from test_baseline_pipeline import ROOT, build_synthetic

ENCODERS = ["facebook/wav2vec2-base-960h", "facebook/wav2vec2-large-xlsr-53"]


class TestFinalStage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        b = cls.base = Path(cls.tmp.name)
        cls.meta, cls.clips = build_synthetic(b)
        cls.outputs = b / "outputs"
        cls.data_root = b / "root"
        common = ["--metadata", str(cls.meta), "--clips", str(cls.clips)]
        cls.common = common

        def go(script, *extra):
            res = subprocess.run([sys.executable, str(ROOT / "scripts" / script), *common, *extra],
                                 capture_output=True, text=True)
            assert res.returncode == 0, res.stdout + res.stderr
            return res

        cache = cls.data_root / "data/processed"
        go("train_baseline.py", "--cache", str(cache), "--out", str(cls.outputs / "EXP-001"), "--n-boot", "20")
        go("train_multilabel.py", "--cache", str(cache), "--out", str(cls.outputs / "EXP-002"),
            "--folds", "3", "--n-boot", "20")

        sys.path.insert(0, str(ROOT / "src"))
        from fluence.data.datasets import load_available_clips
        from fluence.features.spectral import clip_summary_features
        from fluence.preprocessing.audio import load_wav
        df = load_available_clips(cls.meta, cls.clips)
        feats = np.stack([clip_summary_features(load_wav(p)[0])[0] for p in df["path"]])
        rng = np.random.default_rng(1)
        for k, m in enumerate(ENCODERS):
            d = cls.data_root / "data/processed/embeddings" / model_slug(m)
            d.mkdir(parents=True)
            mean = rng.standard_normal((len(df), 3, feats.shape[1]))
            mean[:, 2] = feats + (0.5 - k * 0.4) * rng.standard_normal(feats.shape)  # second encoder less noisy
            write_shard(d, 0, df["clip_key"].tolist(), mean, np.abs(rng.standard_normal(mean.shape)))
            (d / "manifest.json").write_text(json.dumps({"model": m, "revision": f"rev{k}"}))
            go("train_embeddings.py", "--model", m, "--emb", str(cls.data_root / "data/processed/embeddings"),
                "--out", str(cls.outputs / "EXP-003"), "--folds", "3", "--n-boot", "20")
        cls.go = staticmethod(go)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_pipeline(self):
        self.go("compare_encoders.py", "--outputs", str(self.outputs), "--data-root", str(self.data_root),
                 "--n-boot", "50")
        pc = json.loads((self.outputs / "EXP-003/paired_comparison.json").read_text())
        self.assertEqual(len(pc["targets"]), 7)

        cfg_path = self.base / "configs/final_eval.json"
        self.go("freeze_final_config.py", "--primary", ENCODERS[1], "--outputs", str(self.outputs),
                 "--data-root", str(self.data_root), "--out", str(cfg_path))
        cfg = json.loads(cfg_path.read_text())
        self.assertEqual(set(cfg["systems"]), {"baseline_features", *ENCODERS})
        self.assertEqual(cfg["systems"][ENCODERS[1]]["features"]["revision"], "rev1")
        for s in cfg["systems"].values():
            self.assertEqual(len(s["threshold"]), 7)

        out = self.outputs / "EXP-004"
        args = ["--config", str(cfg_path), "--data-root", str(self.data_root), "--out", str(out), "--n-boot", "50"]
        # an uncommitted config is refused
        res = subprocess.run([sys.executable, str(ROOT / "scripts/final_test_eval.py"), *self.common, *args],
                             capture_output=True, text=True)
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("must be committed", res.stdout + res.stderr)

        self.go("final_test_eval.py", *args, "--allow-uncommitted-config")
        r = json.loads((out / "metrics.json").read_text())
        self.assertEqual(r["speakers_test"], 4)
        self.assertGreater(r["systems"][ENCODERS[1]]["any_without_interjections"]["roc_auc"], 0.9)
        self.assertIn("baseline_features", r["paired"])
        self.assertIn("EXP-004", (self.outputs / "TEST_SET_EVALUATIONS.log").read_text())

        # a second evaluation is refused
        res = subprocess.run([sys.executable, str(ROOT / "scripts/final_test_eval.py"), *self.common, *args,
                              "--allow-uncommitted-config"], capture_output=True, text=True)
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("evaluated once", res.stdout + res.stderr)


if __name__ == "__main__":
    unittest.main()
