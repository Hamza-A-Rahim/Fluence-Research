"""Local demo API, with a fake engine and synthetic audio (no model, no real speech)."""

import io
import sys
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from app.main import create_app  # noqa: E402
from fluence.preprocessing.audio import save_wav  # noqa: E402


class FakeEngine:
    status, error = "ready", None
    card = {"system": "fake", "thresholds": {"x": 0.5}}

    def __init__(self):
        self.seen = None

    def analyze(self, x, progress):
        self.seen = x
        progress(1.0)
        return {"duration_s": len(x) / 16000, "tracks": {}}


def wav_bytes(seconds, sr=22050, tmp=None):
    t = np.arange(int(seconds * sr)) / sr
    p = tmp / "tone.wav"
    save_wav(p, (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), sr)
    return p.read_bytes()


def post(client, data, consent=True, name="a.wav"):
    return client.post("/api/analyze", files={"file": (name, io.BytesIO(data), "audio/wav")},
                       data={"consent": str(consent).lower()})


def wait(client, job_id):
    for _ in range(100):
        j = client.get(f"/api/jobs/{job_id}").json()
        if j["status"] in ("done", "error"):
            return j
        time.sleep(0.02)
    raise AssertionError("job did not finish")


def test_analyze_resamples_and_returns_result(tmp_path):
    eng = FakeEngine()
    c = TestClient(create_app(eng))
    r = post(c, wav_bytes(4.0, tmp=tmp_path))
    assert r.status_code == 202
    j = wait(c, r.json()["job_id"])
    assert j["status"] == "done" and j["progress"] == 1.0
    assert len(eng.seen) == 4 * 16000  # mono, 16 kHz via the research loader


def test_requires_consent(tmp_path):
    c = TestClient(create_app(FakeEngine()))
    assert post(c, wav_bytes(4.0, tmp=tmp_path), consent=False).status_code == 400


def test_rejects_short_unsupported_and_garbage(tmp_path):
    c = TestClient(create_app(FakeEngine()))
    assert post(c, wav_bytes(1.0, tmp=tmp_path)).status_code == 422
    assert post(c, b"hello", name="notes.txt").status_code == 422


def test_not_ready_engine_refuses(tmp_path):
    eng = FakeEngine()
    eng.status = "loading"
    c = TestClient(create_app(eng))
    assert post(c, wav_bytes(4.0, tmp=tmp_path)).status_code == 503
    assert c.get("/api/health").json()["status"] == "loading"


def test_job_can_be_deleted(tmp_path):
    c = TestClient(create_app(FakeEngine()))
    jid = post(c, wav_bytes(4.0, tmp=tmp_path)).json()["job_id"]
    wait(c, jid)
    assert c.delete(f"/api/jobs/{jid}").status_code == 204
    assert c.get(f"/api/jobs/{jid}").status_code == 404
