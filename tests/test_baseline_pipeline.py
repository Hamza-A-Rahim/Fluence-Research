"""End-to-end test of scripts/train_baseline.py on a tiny synthetic dataset.

"Event" clips contain a silent gap and a repeated burst; "fluent" clips are
continuous. This checks the plumbing (labels -> clips -> features -> models ->
metrics -> report), not that anything is learned about real stuttering.
"""

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import wavfile

ROOT = Path(__file__).resolve().parents[1]
SR = 16000
HEADER = ["Show", "EpId", "ClipId", "Start", "Stop", "is_probably_host", "speaker",
          "clip_silhouette_score", "SEP12k", "SEP28k-E", "SEP28k-T", "SEP28k-D",
          "Unsure", "PoorAudioQuality", "Prolongation", "Block", "SoundRep", "WordRep",
          "DifficultToUnderstand", "Interjection", "NoStutteredWords", "NaturalPause",
          "Music", "NoSpeech"]


def make_clip(event: bool, rng) -> np.ndarray:
    t = np.arange(3 * SR) / SR
    f0 = rng.uniform(100, 220)
    x = 0.3 * np.sin(2 * np.pi * f0 * t) * (1 + 0.4 * np.sin(2 * np.pi * rng.uniform(3, 6) * t))
    x += 0.01 * rng.standard_normal(len(t))
    if event:
        g = int(rng.uniform(0.6, 1.6) * SR)
        x[g:g + int(0.7 * SR)] = 0.0005 * rng.standard_normal(int(0.7 * SR))
    return x.astype(np.float32)


def build_synthetic(base: Path) -> tuple[Path, Path]:
    """12 speakers x 16 clips; returns (metadata_dir, clips_dir)."""
    meta, clips = base / "meta", base / "clips"
    meta.mkdir()
    rng = np.random.default_rng(0)
    rows = []
    # 12 speakers: 4 train, 4 dev, 4 test (mirrors SEP28k-E structure)
    for s in range(12):
        part = ["train", "dev", "test"][s // 4]
        for c in range(16):
            event = c % 2 == 0
            d = clips / "Synth" / str(s)
            d.mkdir(parents=True, exist_ok=True)
            wavfile.write(d / f"Synth_{s}_{c}.wav", SR, make_clip(event, rng))
            labels = dict(Unsure=0, PoorAudioQuality=0,
                          Prolongation=2 if c % 8 == 4 else 0,
                          Block=3 if event else 0,
                          SoundRep=3 if c % 8 == 0 else 0,      # co-occurs with Block: multi-label
                          WordRep=2 if c % 4 == 2 else 0,
                          DifficultToUnderstand=0,
                          Interjection=2 if (c % 4 == 1) else 0,
                          NoStutteredWords=0 if event else 3, NaturalPause=0, Music=0, NoSpeech=0)
            rows.append(dict(Show="Synth", EpId=s, ClipId=c, Start=c * 48000, Stop=(c + 1) * 48000,
                             is_probably_host=False, speaker=f"spk{s}", clip_silhouette_score=0.5,
                             SEP12k="train", **{"SEP28k-E": part, "SEP28k-T": part, "SEP28k-D": part},
                             **labels))
    rows[0]["Music"] = 3  # one unusable clip that must be excluded
    with open(meta / "SEP-28k-Extended_clips.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HEADER)
        w.writeheader()
        w.writerows(rows)
    return meta, clips


class TestBaselinePipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.base = Path(cls.tmp.name)
        cls.meta, cls.clips = build_synthetic(cls.base)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_script(self, *extra):
        cmd = [sys.executable, str(ROOT / "scripts/train_baseline.py"), "--metadata", str(self.meta),
               "--clips", str(self.clips), "--cache", str(self.base / "cache"),
               "--out", str(self.base / "out"), "--n-boot", "50", *extra]
        res = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, res.stdout + res.stderr)
        return res

    def test_dev_run_learns_and_reports(self):
        self.run_script()
        r = json.loads((self.base / "out/SEP28k-E/metrics.json").read_text())
        self.assertEqual(r["data"]["usable"], 191)
        t = r["targets"]["without_interjections"]
        self.assertGreater(t["logreg"]["dev"]["f1"], 0.9)
        self.assertEqual(t["baselines_dev"]["majority"]["f1"], 0.0)
        self.assertNotIn("test", t["logreg"])  # test untouched by default
        report = (self.base / "out/SEP28k-E/report.md").read_text()
        self.assertIn("Test set not evaluated", report)
        self.assertTrue((self.base / "out/SEP28k-E/cm_with_interjections_logreg_dev.png").is_file())

    def test_test_evaluation_is_logged(self):
        self.run_script("--evaluate-test", "--out", str(self.base / "out_t"))
        r = json.loads((self.base / "out_t/SEP28k-E/metrics.json").read_text())
        self.assertIn("test", r["targets"]["with_interjections"]["logreg"])
        self.assertEqual(r["test_evaluations_so_far"], 1)
        res = self.run_script("--evaluate-test", "--out", str(self.base / "out_t"))
        self.assertIn("WARNING", res.stdout)

    def test_cv_mode(self):
        self.run_script("--partition", "SEP12k", "--folds", "3")
        r = json.loads((self.base / "out/SEP12k/metrics.json").read_text())
        self.assertEqual(len(r["targets"]["with_interjections"]["logreg"]["folds"]), 3)


if __name__ == "__main__":
    unittest.main()
