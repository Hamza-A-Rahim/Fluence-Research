"""Spectral features: STFT, log-mel filterbank energies, MFCCs, deltas.

Defaults reproduce the input representation of the SEP-28k paper
(Lea et al., ICASSP 2021): 40 mel filterbank energies, 0-8000 Hz,
25 ms window, 10 ms hop (100 frames/s).

The mel filterbank follows the "Slaney" construction that librosa uses by
default (``htk=False, norm='slaney'``), so results can later be cross-checked
against librosa or swapped for it without changing feature values. The
optional test ``test_matches_librosa`` verifies this when librosa is installed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.fft import dct, rfft
from scipy.signal import get_window


@dataclass(frozen=True)
class FeatureConfig:
    sr: int = 16_000
    n_fft: int = 512          # FFT size (32 ms at 16 kHz), zero-pads the window
    win_length: int = 400     # 25 ms analysis window
    hop_length: int = 160     # 10 ms hop -> 100 frames per second
    n_mels: int = 40
    fmin: float = 0.0
    fmax: float = 8_000.0
    n_mfcc: int = 13
    delta_width: int = 9      # frames used for delta regression (must be odd)
    amin: float = 1e-10       # floor before taking log


# --- mel scale (Slaney / Auditory Toolbox) -------------------------------

_F_SP = 200.0 / 3.0                 # Hz per mel below the break frequency
_MIN_LOG_HZ = 1000.0
_MIN_LOG_MEL = _MIN_LOG_HZ / _F_SP  # = 15
_LOGSTEP = np.log(6.4) / 27.0


def hz_to_mel(f):
    f = np.asanyarray(f, dtype=np.float64)
    mel = f / _F_SP
    log_region = f >= _MIN_LOG_HZ
    mel = np.where(log_region, _MIN_LOG_MEL + np.log(np.maximum(f, 1e-12) / _MIN_LOG_HZ) / _LOGSTEP, mel)
    return mel


def mel_to_hz(m):
    m = np.asanyarray(m, dtype=np.float64)
    f = m * _F_SP
    log_region = m >= _MIN_LOG_MEL
    f = np.where(log_region, _MIN_LOG_HZ * np.exp(_LOGSTEP * (m - _MIN_LOG_MEL)), f)
    return f


def mel_filterbank(cfg: FeatureConfig = FeatureConfig()) -> np.ndarray:
    """Triangular mel filters, shape (n_mels, 1 + n_fft // 2), Slaney-normalised."""
    fft_freqs = np.linspace(0.0, cfg.sr / 2.0, 1 + cfg.n_fft // 2)
    mel_pts = np.linspace(hz_to_mel(cfg.fmin), hz_to_mel(cfg.fmax), cfg.n_mels + 2)
    hz_pts = mel_to_hz(mel_pts)
    fdiff = np.diff(hz_pts)
    ramps = hz_pts[:, None] - fft_freqs[None, :]

    weights = np.zeros((cfg.n_mels, fft_freqs.size))
    for i in range(cfg.n_mels):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        weights[i] = np.maximum(0.0, np.minimum(lower, upper))
    enorm = 2.0 / (hz_pts[2:cfg.n_mels + 2] - hz_pts[:cfg.n_mels])
    weights *= enorm[:, None]
    return weights.astype(np.float32)


# --- framing / STFT ------------------------------------------------------

def power_spectrogram(signal: np.ndarray, cfg: FeatureConfig = FeatureConfig()) -> np.ndarray:
    """|STFT|^2, shape (1 + n_fft//2, n_frames). Frames are centred (zero-padded)."""
    if signal.ndim != 1:
        raise ValueError("power_spectrogram expects a mono 1-D signal")
    pad = cfg.n_fft // 2
    x = np.pad(signal.astype(np.float64), (pad, pad))

    # Window of win_length, centred inside an n_fft frame.
    win = get_window("hann", cfg.win_length, fftbins=True)
    lpad = (cfg.n_fft - cfg.win_length) // 2
    window = np.zeros(cfg.n_fft)
    window[lpad:lpad + cfg.win_length] = win

    n_frames = 1 + (x.shape[0] - cfg.n_fft) // cfg.hop_length
    idx = np.arange(cfg.n_fft)[None, :] + cfg.hop_length * np.arange(n_frames)[:, None]
    frames = x[idx] * window[None, :]
    spec = np.abs(rfft(frames, n=cfg.n_fft, axis=1)) ** 2
    return spec.T.astype(np.float32)


def log_mel_spectrogram(signal: np.ndarray, cfg: FeatureConfig = FeatureConfig()) -> np.ndarray:
    """Log mel energies in dB, shape (n_mels, n_frames)."""
    mel = mel_filterbank(cfg) @ power_spectrogram(signal, cfg)
    return (10.0 * np.log10(np.maximum(mel, cfg.amin))).astype(np.float32)


def mfcc(signal: np.ndarray, cfg: FeatureConfig = FeatureConfig(),
         log_mel: np.ndarray | None = None) -> np.ndarray:
    """MFCCs (orthonormal DCT-II of dB log-mel), shape (n_mfcc, n_frames)."""
    if log_mel is None:
        log_mel = log_mel_spectrogram(signal, cfg)
    return dct(log_mel, type=2, axis=0, norm="ortho")[: cfg.n_mfcc].astype(np.float32)


def delta(features: np.ndarray, width: int = 9) -> np.ndarray:
    """First-order regression deltas along time (axis 1), edge-padded."""
    if width < 3 or width % 2 == 0:
        raise ValueError("width must be an odd integer >= 3")
    n = width // 2
    padded = np.pad(features, ((0, 0), (n, n)), mode="edge")
    denom = 2.0 * sum(k * k for k in range(1, n + 1))
    T = features.shape[1]
    out = np.zeros_like(features, dtype=np.float32)
    for k in range(1, n + 1):
        out += k * (padded[:, n + k:n + k + T] - padded[:, n - k:n - k + T])
    return (out / denom).astype(np.float32)


def frame_rms(signal: np.ndarray, cfg: FeatureConfig = FeatureConfig()) -> np.ndarray:
    """RMS energy per frame (same framing as the spectrogram), shape (n_frames,)."""
    pad = cfg.n_fft // 2
    x = np.pad(signal.astype(np.float64), (pad, pad))
    n_frames = 1 + (x.shape[0] - cfg.n_fft) // cfg.hop_length
    lpad = (cfg.n_fft - cfg.win_length) // 2
    idx = (lpad + np.arange(cfg.win_length))[None, :] + cfg.hop_length * np.arange(n_frames)[:, None]
    return np.sqrt(np.mean(x[idx] ** 2, axis=1)).astype(np.float32)


# --- fixed-length clip summary for classical baselines ---------------------

def clip_summary_features(signal: np.ndarray, cfg: FeatureConfig = FeatureConfig()
                          ) -> tuple[np.ndarray, list[str]]:
    """Summarise a clip as one fixed-length vector (for logistic regression etc).

    Returns (vector, names). Statistics over time of: log-mel (mean, std),
    MFCC (mean, std), delta-MFCC (mean, std), and frame energy descriptors
    including the fraction of low-energy frames (a crude proxy for pauses,
    which matter for blocks).
    """
    lm = log_mel_spectrogram(signal, cfg)
    cc = mfcc(signal, cfg, log_mel=lm)
    dcc = delta(cc, cfg.delta_width)
    rms_db = 20.0 * np.log10(np.maximum(frame_rms(signal, cfg), 1e-5))

    parts, names = [], []

    def add(block: np.ndarray, prefix: str) -> None:
        parts.extend([block.mean(axis=1), block.std(axis=1)])
        names.extend([f"{prefix}{i}_mean" for i in range(block.shape[0])])
        names.extend([f"{prefix}{i}_std" for i in range(block.shape[0])])

    add(lm, "logmel")
    add(cc, "mfcc")
    add(dcc, "dmfcc")

    low = float(np.mean(rms_db < rms_db.max() - 30.0))  # >30 dB below the loudest frame
    energy = np.array([rms_db.mean(), rms_db.std(), rms_db.min(), rms_db.max(), low],
                      dtype=np.float32)
    parts.append(energy)
    names.extend(["rms_db_mean", "rms_db_std", "rms_db_min", "rms_db_max", "low_energy_frac"])

    return np.concatenate(parts).astype(np.float32), names
