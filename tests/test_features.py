"""Tests for spectral features."""

import importlib.util
import unittest

import numpy as np

from fluence.features.spectral import (
    FeatureConfig, clip_summary_features, delta, frame_rms, hz_to_mel,
    log_mel_spectrogram, mel_filterbank, mel_to_hz, mfcc, power_spectrogram,
)

CFG = FeatureConfig()


def sine(freq, sr=16000, dur=3.0, amp=0.5):
    t = np.arange(int(sr * dur)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


class TestMelScale(unittest.TestCase):
    def test_roundtrip(self):
        f = np.array([0, 100, 999, 1000, 1001, 4000, 8000.0])
        np.testing.assert_allclose(mel_to_hz(hz_to_mel(f)), f, rtol=1e-9, atol=1e-6)

    def test_linear_below_1khz(self):
        self.assertAlmostEqual(float(hz_to_mel(1000.0)), 15.0)
        self.assertAlmostEqual(float(hz_to_mel(500.0)), 7.5)

    def test_filterbank_shape_and_coverage(self):
        fb = mel_filterbank(CFG)
        self.assertEqual(fb.shape, (40, 257))
        self.assertTrue(np.all(fb >= 0))
        self.assertTrue(np.all(fb.sum(axis=1) > 0))  # no empty filters
        peaks = fb.argmax(axis=1)
        self.assertTrue(np.all(np.diff(peaks) >= 0))  # centres increase


class TestSpectrogram(unittest.TestCase):
    def test_frame_count_100_per_second(self):
        # 3 s clip, centred frames: 1 + 48000 // 160 = 301
        self.assertEqual(power_spectrogram(sine(440), CFG).shape, (257, 301))
        self.assertEqual(log_mel_spectrogram(sine(440), CFG).shape, (40, 301))

    def test_sine_peaks_in_right_fft_bin(self):
        spec = power_spectrogram(sine(1000), CFG)
        peak_hz = spec[:, 150].argmax() * CFG.sr / CFG.n_fft
        self.assertLess(abs(peak_hz - 1000), CFG.sr / CFG.n_fft)

    def test_higher_tone_activates_higher_mel_band(self):
        low = log_mel_spectrogram(sine(300))[:, 150].argmax()
        high = log_mel_spectrogram(sine(3000))[:, 150].argmax()
        self.assertGreater(high, low)

    def test_spectrum_finite_and_scales_with_amplitude(self):
        quiet = power_spectrogram(sine(440, amp=0.1), CFG)[:, 150].sum()
        loud = power_spectrogram(sine(440, amp=0.2), CFG)[:, 150].sum()
        self.assertTrue(np.isfinite(loud))
        self.assertAlmostEqual(loud / quiet, 4.0, places=3)  # power ∝ amplitude²

    def test_silence_hits_floor_not_minus_inf(self):
        lm = log_mel_spectrogram(np.zeros(16000, np.float32))
        self.assertTrue(np.all(np.isfinite(lm)))
        self.assertAlmostEqual(float(lm.min()), -100.0, places=3)

    def test_rejects_stereo(self):
        with self.assertRaises(ValueError):
            power_spectrogram(np.zeros((100, 2), np.float32))


class TestMfccAndDeltas(unittest.TestCase):
    def test_mfcc_shape(self):
        self.assertEqual(mfcc(sine(440)).shape, (13, 301))

    def test_delta_of_ramp_is_slope(self):
        ramp = np.tile(np.arange(50, dtype=np.float32), (2, 1)) * 2.0
        d = delta(ramp, 9)
        np.testing.assert_allclose(d[:, 10:40], 2.0, rtol=1e-5)

    def test_delta_of_constant_is_zero(self):
        np.testing.assert_allclose(delta(np.ones((3, 20), np.float32)), 0.0, atol=1e-7)

    def test_delta_width_validation(self):
        with self.assertRaises(ValueError):
            delta(np.ones((1, 10)), 4)


class TestSummary(unittest.TestCase):
    def test_vector_length_and_names(self):
        v, names = clip_summary_features(sine(440))
        self.assertEqual(len(v), len(names))
        self.assertEqual(len(v), 40 * 2 + 13 * 2 + 13 * 2 + 5)
        self.assertEqual(len(set(names)), len(names))
        self.assertTrue(np.all(np.isfinite(v)))

    def test_pause_fraction_detects_silence_gap(self):
        x = sine(200)
        gapped = x.copy()
        gapped[16000:32000] = 0  # a 1 s silent "block-like" gap
        _, names = clip_summary_features(x)
        i = names.index("low_energy_frac")
        self.assertLess(clip_summary_features(x)[0][i], 0.05)
        self.assertGreater(clip_summary_features(gapped)[0][i], 0.25)

    def test_rms_of_sine(self):
        r = frame_rms(sine(440, amp=0.5))
        self.assertAlmostEqual(float(np.median(r)), 0.5 / np.sqrt(2), places=2)


@unittest.skipUnless(importlib.util.find_spec("librosa"), "librosa not installed")
class TestMatchesLibrosa(unittest.TestCase):
    """Cross-check against librosa when available (pip install '.[audio-extras]')."""

    def test_filterbank(self):
        import librosa
        ref = librosa.filters.mel(sr=16000, n_fft=512, n_mels=40, fmin=0, fmax=8000)
        np.testing.assert_allclose(mel_filterbank(CFG), ref, rtol=1e-4, atol=1e-7)

    def test_log_mel(self):
        import librosa
        x = sine(440) + 0.01 * np.random.default_rng(0).standard_normal(48000).astype(np.float32)
        S = librosa.feature.melspectrogram(y=x, sr=16000, n_fft=512, win_length=400,
                                           hop_length=160, n_mels=40, fmin=0, fmax=8000,
                                           center=True, pad_mode="constant", power=2.0)
        ref = librosa.power_to_db(S, ref=1.0, amin=1e-10, top_db=None)
        np.testing.assert_allclose(log_mel_spectrogram(x), ref, atol=0.05)


if __name__ == "__main__":
    unittest.main()
