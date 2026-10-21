"""Waveform and spectrogram figures for inspecting recordings."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write files, never open windows
import matplotlib.pyplot as plt
import numpy as np

from fluence.features.spectral import FeatureConfig, log_mel_spectrogram


def plot_waveform_and_logmel(
    signal: np.ndarray,
    sr: int,
    out_path: str | Path,
    *,
    title: str = "",
    cfg: FeatureConfig = FeatureConfig(),
    spans: list[tuple[float, float, str]] | None = None,
) -> Path:
    """Save a two-panel figure: waveform (top) and log-mel spectrogram (bottom).

    ``spans`` optionally shades (start_s, end_s, label) regions, e.g. annotated
    clips, so predictions and labels can be checked by eye.
    """
    out_path = Path(out_path)
    fig = make_inspection_figure(signal, sr, title=title, cfg=cfg, spans=spans)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def make_inspection_figure(signal, sr, *, title="", cfg=FeatureConfig(), spans=None):
    """Build (but do not save) the waveform + log-mel figure. Axes: [wave, spec, cbar]."""
    t = np.arange(signal.shape[0]) / sr
    lm = log_mel_spectrogram(signal, cfg)
    duration = signal.shape[0] / sr

    # A dedicated colour-bar column keeps both panels exactly the same width,
    # so time positions line up vertically (a colour bar attached to one axis
    # would shrink only that panel and misalign the two time axes).
    fig = plt.figure(figsize=(10, 5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.4], width_ratios=[60, 1], wspace=0.03)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    cax = fig.add_subplot(gs[1, 1])
    ax1.set_xlim(0, duration)
    ax1.tick_params(labelbottom=False)
    ax1.plot(t, signal, linewidth=0.5, color="#1f3b57")
    ax1.set_ylabel("Amplitude")
    ax1.set_ylim(-1.05, 1.05)
    if title:
        ax1.set_title(title, fontsize=10)

    im = ax2.imshow(lm, origin="lower", aspect="auto", cmap="magma",
                    extent=[0, duration, 0, cfg.n_mels])
    ax2.set_ylabel("Mel band")
    ax2.set_xlabel("Time (s)")
    fig.colorbar(im, cax=cax, label="dB")

    for start, end, label in spans or []:
        for ax in (ax1, ax2):
            ax.axvspan(start, end, alpha=0.18, color="#2a9d8f")
        ax1.text(start, 0.92, label, fontsize=7, va="top")
    return fig
