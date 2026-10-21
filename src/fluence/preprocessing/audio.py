"""Audio loading, validation and normalisation.

Design decisions (see docs/decisions.md, D-003):

* Everything downstream works on **mono, float32, 16 kHz** signals in [-1, 1].
  16 kHz is the rate used by SEP-28k clips (Start/Stop are 16 kHz sample
  indices), KSoF, FluencyBank as clipped by Lea et al. (2021), and by most
  pretrained speech encoders (wav2vec 2.0, HuBERT, Whisper).
* The core depends only on NumPy and SciPy so it is easy to install and
  deterministic. librosa is optional and only used in cross-check tests.
* Validation never silently "fixes" a broken file: it raises
  ``AudioValidationError`` with a reason, or returns warnings the caller can log.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from math import gcd
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

TARGET_SR = 16_000

# Sensible bounds for speech research audio. Outside these we refuse to load.
MIN_SR = 8_000
MAX_SR = 192_000


class AudioValidationError(ValueError):
    """Raised when an audio file cannot be used safely."""


@dataclass
class AudioInfo:
    """What we learned about a file while loading it."""

    path: str
    original_sr: int
    original_channels: int
    original_dtype: str
    duration_s: float
    sr: int
    warnings: list[str] = field(default_factory=list)


def _to_float32(data: np.ndarray) -> np.ndarray:
    """Convert integer or float PCM into float32 in [-1, 1]."""
    dtype = data.dtype
    if dtype == np.uint8:  # 8-bit WAV is unsigned, centred on 128
        return (data.astype(np.float32) - 128.0) / 128.0
    if dtype == np.int16:
        return data.astype(np.float32) / 32768.0
    if dtype == np.int32:  # also how SciPy returns 24-bit audio (left-aligned)
        return data.astype(np.float32) / 2147483648.0
    if dtype in (np.float32, np.float64):
        return data.astype(np.float32)
    raise AudioValidationError(f"Unsupported sample format: {dtype}")


def to_mono(signal: np.ndarray) -> np.ndarray:
    """Average channels. Accepts shape (n,) or (n, channels)."""
    if signal.ndim == 1:
        return signal
    if signal.ndim == 2:
        return signal.mean(axis=1).astype(signal.dtype, copy=False)
    raise AudioValidationError(f"Unexpected audio array shape {signal.shape}")


def resample(signal: np.ndarray, orig_sr: int, target_sr: int = TARGET_SR) -> np.ndarray:
    """Polyphase resampling (anti-aliased), exact for integer rate ratios."""
    if orig_sr == target_sr:
        return signal.astype(np.float32, copy=False)
    g = gcd(int(orig_sr), int(target_sr))
    up, down = target_sr // g, orig_sr // g
    return resample_poly(signal, up, down).astype(np.float32)


def peak_normalize(signal: np.ndarray, peak: float = 0.95) -> np.ndarray:
    """Scale so max |x| == peak. Silent input is returned unchanged."""
    m = float(np.max(np.abs(signal))) if signal.size else 0.0
    if m == 0.0:
        return signal
    return (signal * (peak / m)).astype(np.float32)


def check_signal(signal: np.ndarray, sr: int, *, min_duration_s: float = 0.1) -> list[str]:
    """Hard checks raise; soft problems are returned as warning strings."""
    if signal.size == 0:
        raise AudioValidationError("Audio contains no samples")
    if not np.all(np.isfinite(signal)):
        raise AudioValidationError("Audio contains NaN or infinite values")
    duration = signal.shape[0] / sr
    if duration < min_duration_s:
        raise AudioValidationError(
            f"Audio is {duration:.3f}s long; minimum is {min_duration_s}s"
        )

    found: list[str] = []
    peak = float(np.max(np.abs(signal)))
    if peak == 0.0:
        found.append("silent: every sample is zero")
    elif peak < 1e-3:
        found.append(f"very quiet: peak amplitude {peak:.2e}")
    if peak > 1.0:
        found.append(f"out of range: peak amplitude {peak:.3f} > 1.0")
    clipped = float(np.mean(np.abs(signal) >= 0.999))
    if clipped > 0.001:
        found.append(f"possible clipping: {clipped:.2%} of samples at full scale")
    dc = float(np.mean(signal))
    if abs(dc) > 0.05:
        found.append(f"DC offset {dc:+.3f}")
    return found


def load_wav(
    path: str | Path,
    *,
    target_sr: int = TARGET_SR,
    mono: bool = True,
    min_duration_s: float = 0.1,
) -> tuple[np.ndarray, AudioInfo]:
    """Load a WAV file, validate it, convert to float32 mono at ``target_sr``.

    Returns ``(signal, info)``. ``signal`` has shape (n,) when ``mono`` is True,
    otherwise (n, channels).
    """
    path = Path(path)
    if not path.is_file():
        raise AudioValidationError(f"File not found: {path}")
    if path.stat().st_size == 0:
        raise AudioValidationError(f"File is empty: {path}")

    try:
        with warnings.catch_warnings():
            # SciPy warns on unknown chunks (e.g. LIST metadata) — harmless.
            warnings.simplefilter("ignore", wavfile.WavFileWarning)
            sr, data = wavfile.read(path)
    except Exception as exc:  # SciPy raises several types for bad headers
        raise AudioValidationError(f"Not a readable WAV file ({exc}): {path}") from exc

    if not (MIN_SR <= sr <= MAX_SR):
        raise AudioValidationError(f"Sample rate {sr} Hz outside [{MIN_SR}, {MAX_SR}]")

    channels = 1 if data.ndim == 1 else data.shape[1]
    original_dtype = str(data.dtype)
    signal = _to_float32(data)
    if mono:
        signal = resample(to_mono(signal), sr, target_sr)
    else:
        if signal.ndim == 1:
            signal = signal[:, None]
        signal = np.stack(
            [resample(signal[:, c], sr, target_sr) for c in range(signal.shape[1])],
            axis=1,
        )

    found = check_signal(signal if mono else to_mono(signal), target_sr,
                         min_duration_s=min_duration_s)
    info = AudioInfo(
        path=str(path),
        original_sr=int(sr),
        original_channels=int(channels),
        original_dtype=original_dtype,
        duration_s=signal.shape[0] / target_sr,
        sr=target_sr,
        warnings=found,
    )
    return signal, info


def save_wav(path: str | Path, signal: np.ndarray, sr: int = TARGET_SR) -> None:
    """Write float audio as 16-bit PCM (the format most tools expect)."""
    clipped = np.clip(signal, -1.0, 1.0)
    wavfile.write(path, sr, (clipped * 32767.0).astype(np.int16))


def extract_clip(
    signal: np.ndarray,
    sr: int,
    start_s: float,
    duration_s: float,
    *,
    pad: bool = True,
) -> np.ndarray:
    """Cut ``duration_s`` seconds starting at ``start_s``.

    If the window runs past the end and ``pad`` is True the clip is zero-padded
    so every clip has identical length (needed for batching). If ``pad`` is
    False a short clip is returned instead.
    """
    if start_s < 0 or duration_s <= 0:
        raise ValueError("start_s must be >= 0 and duration_s > 0")
    start = int(round(start_s * sr))
    length = int(round(duration_s * sr))
    if start >= signal.shape[0]:
        raise ValueError(
            f"Clip start {start_s:.3f}s is beyond the end of the audio "
            f"({signal.shape[0] / sr:.3f}s)"
        )
    clip = signal[start:start + length]
    if pad and clip.shape[0] < length:
        clip = np.pad(clip, (0, length - clip.shape[0]))
    return clip.astype(np.float32, copy=False)


def extract_clip_by_samples(signal: np.ndarray, start: int, stop: int) -> np.ndarray:
    """Cut by sample indices, as SEP-28k labels specify (16 kHz Start/Stop)."""
    if not (0 <= start < stop):
        raise ValueError(f"Invalid sample range [{start}, {stop})")
    if stop > signal.shape[0]:
        raise ValueError(
            f"Clip end {stop} beyond audio length {signal.shape[0]} samples"
        )
    return signal[start:stop].astype(np.float32, copy=False)
