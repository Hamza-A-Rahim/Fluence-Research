"""Tests for SEP-28k label parsing and leakage-safe splitting.

The fixture mimics the real file's format, including the space after each
comma, but every row is invented — no licensed annotations are stored here.
"""

import unittest
from pathlib import Path

import numpy as np

from fluence.data.sep28k import (
    any_stutter_target, binarize, exclude_unusable, label_summary, load_labels,
)
from fluence.data.splits import check_no_leakage, group_split, split_report

FIXTURE = Path(__file__).parent / "fixtures" / "sep28k_labels_sample.csv"


class TestLabels(unittest.TestCase):
    def setUp(self):
        self.df = load_labels(FIXTURE)

    def test_parses_spaced_csv(self):
        self.assertEqual(len(self.df), 8)
        self.assertEqual(self.df["Show"].iloc[0], "ShowA")  # whitespace stripped
        self.assertTrue(np.allclose(self.df["duration_s"], 3.0))

    def test_majority_binarisation(self):
        b = binarize(self.df, ["SoundRep", "Interjection"], min_votes=2)
        self.assertEqual(b["SoundRep"].tolist(), [0, 1, 0, 0, 1, 0, 0, 0])
        self.assertEqual(binarize(self.df, ["SoundRep"], min_votes=1)["SoundRep"].sum(), 3)

    def test_any_stutter_with_and_without_interjections(self):
        with_int = any_stutter_target(self.df, include_interjections=True)
        without = any_stutter_target(self.df, include_interjections=False)
        self.assertEqual(with_int.tolist(), [0, 1, 1, 0, 1, 1, 0, 0])
        self.assertEqual(without.tolist(), [0, 1, 0, 0, 1, 1, 0, 0])

    def test_multilabel_clips_are_counted(self):
        self.assertEqual(label_summary(self.df).attrs["clips_with_multiple_events"], 1)

    def test_exclude_unusable(self):
        self.assertEqual(len(exclude_unusable(self.df)), 7)

    def test_bad_schema_rejected(self):
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
            f.write("Show,EpId\nA,0\n")
        with self.assertRaises(ValueError):
            load_labels(f.name)
        Path(f.name).unlink()


class TestSplits(unittest.TestCase):
    def test_groups_never_cross_partitions(self):
        rng = np.random.default_rng(1)
        groups = rng.integers(0, 60, size=3000)
        part = group_split(groups, seed=3)
        check_no_leakage(groups, part)
        self.assertEqual(set(part), {"train", "val", "test"})

    def test_fractions_roughly_respected_with_many_groups(self):
        groups = np.repeat(np.arange(200), 10)
        rep = split_report(groups, group_split(groups, seed=0))
        self.assertAlmostEqual(rep.loc["train", "row_share"], 0.70, delta=0.02)
        self.assertAlmostEqual(rep.loc["test", "row_share"], 0.15, delta=0.02)

    def test_deterministic_given_seed(self):
        g = np.repeat(np.arange(30), 5)
        np.testing.assert_array_equal(group_split(g, seed=7), group_split(g, seed=7))

    def test_leak_detector_catches_leak(self):
        with self.assertRaises(AssertionError):
            check_no_leakage(["a", "a", "b"], ["train", "test", "train"])

    def test_bad_fractions(self):
        with self.assertRaises(ValueError):
            group_split(["a"], fractions=(0.5, 0.5, 0.5))


if __name__ == "__main__":
    unittest.main()
