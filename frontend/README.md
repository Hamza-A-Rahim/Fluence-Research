# Frontend: local demo interface (Stage 7)

Next.js 16 + TypeScript + Tailwind 4. Pages: **Home**, **Analyze** (upload or record → waveform, per-event timeline, region table with playback) and **Research** (EXP-004 test results and EXP-005 offset robustness).

```powershell
cd frontend
npm install
npm run dev        # http://127.0.0.1:3000 (bound to localhost only)
```

Needs the backend running (`python backend/run.py`). API address: `NEXT_PUBLIC_FLUENCE_API` (default `http://127.0.0.1:8000`).

Rules this interface follows:
- Every number is read from the backend's model card, which is built from logged experiment metrics. Nothing is hard-coded.
- Wording is "possible event" and "model score", never diagnosis, severity or error. A "not a diagnostic tool" notice sits beside every result.
- Consent checkbox before any analysis; audio never leaves the computer.
- Timeline colours are the validated dark categorical palette. Each event type has its own labelled lane, and a table view lists every region.
