# Backend: local demo API (Stage 7)

FastAPI service that runs the frozen EXP-004 primary system (XLSR-53 layer 14 + logistic heads) on a recording of any length. **Local only:** it binds to `127.0.0.1` and accepts browser requests only from a localhost origin. It is not built for public use (see `research/ethics/ethics_and_privacy.md`).

```powershell
pip install -e ".[web]"                 # fastapi, uvicorn, python-multipart (+ imageio-ffmpeg for non-WAV audio)
python scripts/export_demo_model.py     # once: fits + verifies the classifiers -> outputs/demo_model/
python backend/run.py                   # http://127.0.0.1:8000
```

`export_demo_model.py` refits the classifiers exactly as EXP-004/005 did (train clips only) and refuses to export unless they reproduce the logged EXP-005 dev AUCs. The model card it writes copies its numbers from `outputs/EXP-004` and `outputs/EXP-005` metrics, so the interface shows only logged results.

**How a recording is scored** (`src/fluence/inference.py`): 3 s windows every 0.5 s; each event's score at time t is the **mean** over the windows covering t (EXP-005: mean keeps clip-level AUC, max does not). Regions are spans above the detector's frozen dev threshold. Region edges are not validated (D-012).

**Privacy:** uploads are decoded in a temporary folder that is deleted at once (WAV via the research loader; other formats converted by ffmpeg at their native rate first), analysed in memory, never logged. Results contain only times and scores and expire after 10 minutes. Analysis requires an explicit consent flag.

| Endpoint | |
|---|---|
| `GET /api/health` | `loading` / `ready` / `error` |
| `GET /api/model` | model card + limits |
| `POST /api/analyze` | multipart `file` + `consent=true` → `job_id` (3 s – 5 min, ≤ 30 MB) |
| `GET /api/jobs/{id}` | progress and result |
| `DELETE /api/jobs/{id}` | forget a result |
