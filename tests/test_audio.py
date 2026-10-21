"""Tests for loading, validating and normalising audio.

All test audio is synthesised here, so the suite needs no private recordings.
Written as unittest.TestCase classes: pytest runs them, and so does
``python -m unittest`` when pytest is unavailable.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from fluence.preprocessing.audio import (
    TARGET_SR, AudioValidationError, check_signal, extract_clip,
    extract_clip_by_samples, load_wav, peak_normalize, resample, save_wav, to_mono,
)


def sine(freq, sr, dur=1.0, amp=0.5):
    t = np.arange(int(sr * dur)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def dominant_freq(x, sr):
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    return np.fft.rfftfreq(len(x), 1 / sr)[np.argmax(spec)]


class TempDirCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()


class TestLoading(TempDirCase):
    def test_int16_mono_at_target_rate(self):
        p = self.tmp / "a.wav"
        wavfile.write(p, 16000, (sine(440, 16000) * 32767).astype(np.int16))
        x, info = load_wav(p)
        self.assertEqual(x.dtype, np.float32)
        self.assertEqual(x.ndim, 1)
        self.assertEqual(info.sr, TARGET_SR)
        self.assertAlmostEqual(info.duration_s, 1.0, places=3)
        self.assertAlmostEqual(float(np.max(np.abs(x))), 0.5, places=2)
        self.assertEqual(info.warnings, [])

    def test_stereo_44k1_becomes_mono_16k_and_keeps_pitch(self):
        sr = 44100
        left, right = sine(440, sr), sine(440, sr)
        p = self.tmp / "stereo.wav"
        wavfile.write(p, sr, (np.stack([left, right], 1) * 32767).astype(np.int16))
        x, info = load_wav(p)
        self.assertEqual(info.original_channels, 2)
        self.assertEqual(info.original_sr, 44100)
        self.assertEqual(x.ndim, 1)
        self.assertEqual(len(x), 16000)
        self.assertLess(abs(dominant_freq(x, 16000) - 440), 2.0)

    def test_keep_channels_when_mono_false(self):
        p = self.tmp / "stereo.wav"
        wavfile.write(p, 16000, np.stack([sine(300, 16000), sine(600, 16000)], 1))
        x, _ = load_wav(p, mono=False)
        self.assertEqual(x.shape, (16000, 2))

    def test_float32_and_uint8_and_int32_formats(self):
        s = sine(500, 16000)
        cases = {
            "f32.wav": s,
            "u8.wav": (s * 127 + 128).astype(np.uint8),
            "i32.wav": (s * 2147483647).astype(np.int32),
        }
        for name, data in cases.items():
            with self.subTest(fmt=name):
                p = self.tmp / name
                wavfile.write(p, 16000, data)
                x, _ = load_wav(p)
                self.assertAlmostEqual(float(np.max(np.abs(x))), 0.5, delta=0.02)

    def test_upsampling_8k(self):
        p = self.tmp / "tel.wav"
        wavfile.write(p, 8000, sine(1000, 8000))
        x, info = load_wav(p)
        self.assertEqual(len(x), 16000)
        self.assertLess(abs(dominant_freq(x, 16000) - 1000), 2.0)


class TestValidation(TempDirCase):
    def test_missing_file(self):
        with self.assertRaises(AudioValidationError):
            load_wav(self.tmp / "nope.wav")

    def test_empty_file(self):
        p = self.tmp / "empty.wav"
        p.write_bytes(b"")
        with self.assertRaises(AudioValidationError):
            load_wav(p)

    def test_not_a_wav(self):
        p = self.tmp / "fake.wav"
        p.write_bytes(b"this is not audio" * 100)
        with self.assertRaises(AudioValidationError):
            load_wav(p)

    def test_too_short(self):
        p = self.tmp / "short.wav"
        wavfile.write(p, 16000, sine(440, 16000, dur=0.05))
        with self.assertRaises(AudioValidationError):
            load_wav(p)

    def test_nan_rejected(self):
        x = sine(440, 16000)
        x[100] = np.nan
        with self.assertRaises(AudioValidationError):
            check_signal(x, 16000)

    def test_unsupported_rate(self):
        p = self.tmp / "low.wav"
        wavfile.write(p, 4000, sine(200, 4000))
        with self.assertRaises(AudioValidationError):
            load_wav(p)

    def test_soft_warnings(self):
        self.assertTrue(any("silent" in w for w in check_signal(np.zeros(16000), 16000)))
        self.assertTrue(any("quiet" in w for w in check_signal(sine(440, 16000, amp=1e-4), 16000)))
        clipped = np.clip(sine(440, 16000, amp=3.0), -1, 1)
        self.assertTrue(any("clipping" in w for w in check_signal(clipped, 16000)))
        self.assertTrue(any("DC" in w for w in check_signal(sine(440, 16000) + 0.2, 16000)))


class TestTransforms(TempDirCase):
    def test_to_mono_averages(self):
        st = np.stack([np.ones(10), -np.ones(10)], 1).astype(np.float32)
        np.testing.assert_allclose(to_mono(st), np.zeros(10))

    def test_resample_length(self):
        self.assertEqual(len(resample(np.zeros(48000, np.float32), 48000, 16000)), 16000)
        self.assertEqual(len(resample(np.zeros(22050, np.float32), 22050, 16000)), 16000)

    def test_peak_normalize(self):
        y = peak_normalize(sine(440, 16000, amp=0.1), 0.9)
        self.assertAlmostEqual(float(np.max(np.abs(y))), 0.9, places=4)
        np.testing.assert_array_equal(peak_normalize(np.zeros(5, np.float32)), np.zeros(5))

    def test_save_load_roundtrip(self):
        p = self.tmp / "rt.wav"
        s = sine(440, 16000)
        save_wav(p, s)
        x, _ = load_wav(p)
        np.testing.assert_allclose(x, s, atol=1e-4)


class TestClips(unittest.TestCase):
    def setUp(self):
        self.x = np.arange(16000 * 5, dtype=np.float32)  # 5 s, value == sample index

    def test_clip_by_seconds(self):
        c = extract_clip(self.x, 16000, 1.0, 3.0)
        self.assertEqual(len(c), 48000)
        self.assertEqual(c[0], 16000)

    def test_clip_pads_at_end(self):
        c = extract_clip(self.x, 16000, 4.0, 3.0)
        self.assertEqual(len(c), 48000)
        self.assertTrue(np.all(c[16000:] == 0))

    def test_clip_no_pad(self):
        self.assertEqual(len(extract_clip(self.x, 16000, 4.0, 3.0, pad=False)), 16000)

    def test_clip_errors(self):
        with self.assertRaises(ValueError):
            extract_clip(self.x, 16000, 10.0, 3.0)
        with self.assertRaises(ValueError):
            extract_clip(self.x, 16000, -1.0, 3.0)

    def test_clip_by_samples_sep28k_style(self):
        c = extract_clip_by_samples(self.x, 16000, 64000)
        self.assertEqual(len(c), 48000)
        with self.assertRaises(ValueError):
            extract_clip_by_samples(self.x, 70000, 118000)  # runs past the end


if __name__ == "__main__":
    unittest.main()
