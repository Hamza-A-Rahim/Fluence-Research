"""Fluence local demo API. Runs on 127.0.0.1 only (see backend/run.py); no public uploads.

Privacy: uploaded audio is decoded in a temporary folder that is deleted at
once, analysed in memory, and never written to disk or logged. Results hold
only scores and times, and are dropped RESULT_TTL_S after they are finished.
"""

from __future__ import annotations

import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from fluence.inference import CLIP_S, SR
from fluence.preprocessing.audio import AudioValidationError

from .audio_io import decode

MAX_BYTES = 30 * 1024 * 1024
MAX_SECONDS = 300.0
RESULT_TTL_S = 600.0
ORIGIN_RE = r"http://(localhost|127\.0\.0\.1)(:\d+)?"  # the local frontend, on whatever port it got


def create_app(engine=None) -> FastAPI:
    state = {"engine": engine}
    jobs: dict[str, dict] = {}
    lock = threading.Lock()
    pool = ThreadPoolExecutor(max_workers=1)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if state["engine"] is None:
            from .engine import Engine
            state["engine"] = Engine()
            threading.Thread(target=state["engine"].load_encoder, daemon=True).start()
        yield
        pool.shutdown(wait=False, cancel_futures=True)

    app = FastAPI(title="Fluence local demo", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origin_regex=ORIGIN_RE, allow_methods=["GET", "POST", "DELETE"],
                       allow_headers=["*"])

    def purge() -> None:
        now = time.time()
        with lock:
            for k in [k for k, j in jobs.items() if j.get("finished") and now - j["finished"] > RESULT_TTL_S]:
                del jobs[k]

    @app.get("/api/health")
    def health():
        e = state["engine"]
        return {"status": e.status if e else "loading", "error": e.error if e else None}

    @app.get("/api/model")
    def model():
        e = state["engine"]
        if e is None:
            raise HTTPException(503, "Model is loading")
        return {**e.card, "limits": {"max_seconds": MAX_SECONDS, "min_seconds": CLIP_S, "max_bytes": MAX_BYTES}}

    @app.post("/api/analyze", status_code=202)
    async def analyze(file: UploadFile = File(...), consent: bool = Form(False)):
        e = state["engine"]
        if not consent:
            raise HTTPException(400, "Consent is required before analysis")
        if e is None or e.status != "ready":
            raise HTTPException(503, "Model is still loading" if not e or e.status == "loading" else e.error)
        data = await file.read(MAX_BYTES + 1)
        if len(data) > MAX_BYTES:
            raise HTTPException(413, f"File larger than {MAX_BYTES // 2**20} MB")
        try:
            x, warnings = decode(data, file.filename or "")
        except AudioValidationError as exc:
            raise HTTPException(422, str(exc))
        finally:
            del data
        dur = len(x) / SR
        if dur < CLIP_S:
            raise HTTPException(422, f"Recording is {dur:.1f} s; at least {CLIP_S:.0f} s is needed")
        if dur > MAX_SECONDS:
            raise HTTPException(422, f"Recording is {dur / 60:.1f} min; the local demo accepts up to "
                                     f"{MAX_SECONDS / 60:.0f} min")
        purge()
        job_id = uuid.uuid4().hex
        jobs[job_id] = {"status": "queued", "progress": 0.0, "warnings": warnings}

        def run(signal):
            job = jobs[job_id]
            job["status"] = "running"
            try:
                job["result"] = e.analyze(signal, lambda p: job.__setitem__("progress", p))
                job["status"] = "done"
            except Exception as exc:
                job["status"], job["error"] = "error", str(exc)
            finally:
                job["finished"] = time.time()

        pool.submit(run, x)
        return {"job_id": job_id, "duration_s": round(dur, 2)}

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str):
        j = jobs.get(job_id)
        if j is None:
            raise HTTPException(404, "Unknown or expired job")
        return {k: v for k, v in j.items() if k != "finished"}

    @app.delete("/api/jobs/{job_id}", status_code=204)
    def forget(job_id: str):
        with lock:
            jobs.pop(job_id, None)

    return app


app = create_app()
