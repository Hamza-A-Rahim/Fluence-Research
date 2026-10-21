"""Tests for evaluation metrics."""

import unittest

import numpy as np

from fluence.evaluation.metrics import binary_report, reference_baselines, speaker_bootstrap_ci


class TestBinaryReport(unittest.TestCase):
    def test_hand_computed(self):
        y = [1, 1, 1, 0, 0, 0, 0, 1]
        p = [1, 1, 0, 1, 0, 0, 0, 0]
        r = binary_report(y, p)
        self.assertEqual((r.tp, r.fp, r.fn, r.tn), (2, 1, 2, 3))
        self.assertAlmostEqual(r.precision, 2 / 3)
        self.assertAlmostEqual(r.recall, 0.5)
        self.assertAlmostEqual(r.f1, 2 * (2 / 3) * 0.5 / (2 / 3 + 0.5))

    def test_no_positive_predictions_gives_zero_not_nan(self):
        r = binary_report([1, 0, 1], [0, 0, 0])
        self.assertEqual((r.precision, r.recall, r.f1), (0.0, 0.0, 0.0))

    def test_auc_needs_both_classes(self):
        self.assertIsNone(binary_report([1, 1], [1, 1], [0.9, 0.8]).roc_auc)
        self.assertAlmostEqual(binary_report([0, 1], [0, 1], [0.1, 0.9]).roc_auc, 1.0)


class TestBaselines(unittest.TestCase):
    def test_majority_f1_is_zero_when_majority_is_fluent(self):
        b = reference_baselines([0, 0, 0, 1], train_positive_rate=0.3)
        self.assertEqual(b["majority"].f1, 0.0)

    def test_random_auc_near_half(self):
        y = np.random.default_rng(1).integers(0, 2, 20000)
        self.assertAlmostEqual(reference_baselines(y, 0.5)["random"].roc_auc, 0.5, delta=0.02)


class TestBootstrap(unittest.TestCase):
    def test_perfect_predictions_ci_is_one(self):
        y = np.array([0, 1] * 50)
        lo, hi = speaker_bootstrap_ci(y, y, np.repeat(np.arange(10), 10), n_boot=50)
        self.assertEqual((lo, hi), (1.0, 1.0))

    def test_ci_contains_point_estimate(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 600)
        p = np.where(rng.random(600) < 0.8, y, 1 - y)
        spk = np.repeat(np.arange(30), 20)
        lo, hi = speaker_bootstrap_ci(y, p, spk, n_boot=300)
        self.assertLessEqual(lo, binary_report(y, p).f1)
        self.assertGreaterEqual(hi, binary_report(y, p).f1)
        self.assertLess(lo, hi)


if __name__ == "__main__":
    unittest.main()
