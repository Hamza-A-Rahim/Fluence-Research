import numpy as np
import pytest

from fluence.inference import SR, regions, timeline, window_starts


def test_window_starts_cover_the_tail():
    s = window_starts(int(4.2 * SR), hop_s=0.5)
    assert s[0] == 0
    assert s[-1] == int(4.2 * SR) - 3 * SR  # last window ends exactly at the end
    assert np.all(np.diff(s) > 0)


def test_window_starts_exact_fit_has_no_duplicate():
    s = window_starts(int(4.0 * SR), hop_s=0.5)
    assert list(s / SR) == [0.0, 0.5, 1.0]


def test_window_starts_rejects_short_audio():
    with pytest.raises(ValueError):
        window_starts(SR * 2)


def test_timeline_is_mean_of_covering_windows():
    starts = np.array([0.0, 1.0, 2.0])
    scores = np.array([0.0, 1.0, 0.5])
    t, m = timeline(starts, scores, duration_s=5.0, step_s=0.5)
    at = dict(zip(np.round(t, 2), m))
    assert at[0.25] == pytest.approx(0.0)    # only window 0
    assert at[1.25] == pytest.approx(0.5)    # windows 0, 1
    assert at[2.25] == pytest.approx(0.5)    # windows 0, 1, 2
    assert at[4.75] == pytest.approx(0.5)    # only window 2


def test_regions_merge_small_gaps_and_report_peak():
    t = np.arange(0.05, 3.0, 0.1)
    score = np.zeros_like(t)
    score[(t > 0.5) & (t < 1.0)] = 0.8
    score[(t > 1.2) & (t < 1.5)] = 0.9     # 0.2 s gap -> merged
    score[(t > 2.4) & (t < 2.8)] = 0.7     # far -> separate
    r = regions(t, score, threshold=0.6, step_s=0.1, merge_gap_s=0.3)
    assert len(r) == 2
    assert r[0]["start"] == pytest.approx(0.5) and r[0]["end"] == pytest.approx(1.5)
    assert r[0]["peak"] == pytest.approx(0.9)
    assert r[1]["peak"] == pytest.approx(0.7)


def test_regions_empty_when_below_threshold():
    t = np.arange(0.05, 1.0, 0.1)
    assert regions(t, np.full_like(t, 0.2), threshold=0.5) == []
