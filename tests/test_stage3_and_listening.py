"""End-to-end tests for scripts/train_multilabel.py and scripts/listen_examples.py."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from test_baseline_pipeline import ROOT, build_synthetic


class TestStage3AndListening(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.base = Path(cls.tmp.name)
        cls.meta, cls.clips = build_synthetic(cls.base)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_script(self, name, *extra):
        cmd = [sys.executable, str(ROOT / "scripts" / name), "--metadata", str(self.meta),
               "--clips", str(self.clips), *extra]
        res = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, res.stdout + res.stderr)
        return res

    def test_multilabel(self):
        out = self.base / "exp2"
        self.run_script("train_multilabel.py", "--cache", str(self.base / "cache"), "--out", str(out),
                        "--folds", "3", "--n-boot", "50")
        r = json.loads((out / "metrics.json").read_text())
        e = r["SEP28k-E"]
        for lab in ("Prolongation", "Block", "SoundRep", "WordRep", "Interjection"):
            self.assertIn(lab, e)
            self.assertIn(lab, r["SEP12k"])
        self.assertGreater(e["Block"]["logreg"]["dev"]["roc_auc"], 0.9)  # Block = the audible gap
        self.assertGreater(e["multilabel_logreg"]["mean_true_labels_per_clip"], 0.9)  # clips carry >1 label
        self.assertTrue((out / "per_event_auc.png").is_file())
        self.assertIn("Test set not evaluated", (out / "report.md").read_text())

    def test_listening_page(self):
        out = self.base / "listen"
        self.run_script("listen_examples.py", "--out", str(out), "--per-type", "2")
        page = (out / "index.html").read_text()
        self.assertIn("<audio", page)
        wavs = list((out / "clips").glob("*.wav"))
        pngs = list((out / "clips").glob("*.png"))
        self.assertGreaterEqual(len(wavs), 4)
        self.assertIn("No clear-cut examples", page)  # synthetic events always co-occur with Block
        self.assertEqual(len(wavs), len(pngs))
        self.assertTrue((out / "notes.csv").read_text(encoding="utf-8-sig").startswith("group,clip"))


if __name__ == "__main__":
    unittest.main()
