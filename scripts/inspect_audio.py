"""Validate a WAV file and produce an inspection figure plus a feature summary.

Usage:
    python scripts/inspect_audio.py path/to/recording.wav [--out outputs/inspect]

Writes <name>_inspect.png (waveform + log-mel spectrogram) and
<name>_features.csv (one row of clip-summary features). Nothing leaves your
computer. Outputs go to the git-ignored ``outputs/`` folder by default.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fluence.features.spectral import FeatureConfig, clip_summary_features  # noqa: E402
from fluence.preprocessing.audio import AudioValidationError, load_wav  # noqa: E402
from fluence.visualization.plots import plot_waveform_and_logmel  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("wav", type=Path)
    ap.add_argument("--out", type=Path, default=Path("outputs/inspect"))
    args = ap.parse_args()

    try:
        signal, info = load_wav(args.wav)
    except AudioValidationError as exc:
        print(f"INVALID: {exc}")
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    stem = args.wav.stem
    print(f"File:      {info.path}")
    print(f"Original:  {info.original_sr} Hz, {info.original_channels} ch, {info.original_dtype}")
    print(f"Processed: {info.sr} Hz mono float32, {info.duration_s:.2f} s")
    print("Warnings:  " + ("; ".join(info.warnings) if info.warnings else "none"))

    fig = plot_waveform_and_logmel(signal, info.sr, args.out / f"{stem}_inspect.png", title=stem)
    vec, names = clip_summary_features(signal, FeatureConfig(sr=info.sr))
    with open(args.out / f"{stem}_features.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(names)
        w.writerow([f"{v:.6g}" for v in vec])
    print(f"Figure:    {fig}")
    print(f"Features:  {args.out / f'{stem}_features.csv'} ({len(names)} values)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
