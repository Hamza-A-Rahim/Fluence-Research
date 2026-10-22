"""Decode an uploaded recording in a temporary folder that is deleted straight away.

WAV goes through ``fluence.preprocessing.audio.load_wav`` (the research loader).
Other formats (e.g. browser recordings in WebM/Opus, MP3, M4A) are first
converted to 16-bit WAV by ffmpeg *at their own sample rate and channel count*,
so mono mixing and resampling are still done by the research code.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np

from fluence.preprocessing.audio import AudioValidationError, load_wav

ALLOWED_SUFFIXES = {".wav", ".mp3", ".m4a", ".ogg", ".opus", ".webm", ".flac", ".aac", ".mp4"}


def _ffmpeg() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        raise AudioValidationError("Only WAV is supported without ffmpeg (pip install imageio-ffmpeg)") from exc


def decode(data: bytes, filename: str) -> tuple[np.ndarray, list[str]]:
    """Bytes of an audio file -> (mono float32 16 kHz signal, warnings)."""
    suffix = Path(filename or "").suffix.lower() or ".wav"
    if suffix not in ALLOWED_SUFFIXES:
        raise AudioValidationError(f"Unsupported file type {suffix}")
    with tempfile.TemporaryDirectory(prefix="fluence_") as d:
        src = Path(d) / f"input{suffix}"
        src.write_bytes(data)
        if suffix == ".wav":
            try:
                x, info = load_wav(src)
                return x, info.warnings
            except AudioValidationError:
                pass  # e.g. a compressed WAV that SciPy cannot read: let ffmpeg convert it
        wav = Path(d) / "converted.wav"
        r = subprocess.run([_ffmpeg(), "-nostdin", "-v", "error", "-i", str(src), "-vn", "-c:a", "pcm_s16le", str(wav)],
                           capture_output=True, timeout=180)
        if r.returncode != 0 or not wav.exists():
            raise AudioValidationError("Could not decode this audio file")
        x, info = load_wav(wav)
        return x, info.warnings
