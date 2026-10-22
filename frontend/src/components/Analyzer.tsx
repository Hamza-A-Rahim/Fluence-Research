"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { deleteJob, getJob, submit, type Result } from "@/lib/api";
import { fmtTime } from "@/lib/events";
import { Results } from "./Results";
import { useCard, useHealth } from "./useApi";

const MAX_BYTES = 30 * 1024 * 1024;
const MAX_REC_S = 300;
const ACCEPT = ".wav,.mp3,.m4a,.ogg,.opus,.webm,.flac,.aac,.mp4,audio/*";

type Audio = { blob: Blob; url: string; name: string; size: number };
type Phase =
  | { k: "idle" }
  | { k: "recording"; started: number }
  | { k: "ready" }
  | { k: "uploading" }
  | { k: "processing"; jobId: string; progress: number; duration: number; started: number }
  | { k: "done"; jobId: string; result: Result; warnings: string[] }
  | { k: "error"; message: string };

const fmtSize = (b: number) => (b > 1e6 ? `${(b / 1e6).toFixed(1)} MB` : `${Math.max(1, Math.round(b / 1e3))} KB`);

export function Analyzer() {
  const health = useHealth();
  const { card } = useCard();
  const ready = health !== null && health !== "offline" && health.status === "ready";
  const [audio, setAudio] = useState<Audio | null>(null);
  const [phase, setPhase] = useState<Phase>({ k: "idle" });
  const [consent, setConsent] = useState(false);
  const [drag, setDrag] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [level, setLevel] = useState(0);
  const input = useRef<HTMLInputElement>(null);
  const rec = useRef<{ mr: MediaRecorder; stream: MediaStream; ctx: AudioContext; raf: number } | null>(null);

  const setFile = useCallback((blob: Blob, name: string) => {
    if (blob.size > MAX_BYTES) {
      setPhase({ k: "error", message: `That file is ${fmtSize(blob.size)}; the limit is 30 MB.` });
      return;
    }
    setAudio((old) => {
      if (old) URL.revokeObjectURL(old.url);
      return { blob, url: URL.createObjectURL(blob), name, size: blob.size };
    });
    setPhase({ k: "ready" });
  }, []);

  const reset = useCallback(() => {
    setPhase((p) => {
      if (p.k === "done" || p.k === "processing") void deleteJob(p.jobId).catch(() => {});
      return { k: "idle" };
    });
    setAudio((old) => {
      if (old) URL.revokeObjectURL(old.url);
      return null;
    });
  }, []);

  // recording timer
  useEffect(() => {
    if (phase.k !== "recording" && phase.k !== "processing") return;
    const id = setInterval(() => setElapsed((Date.now() - phase.started) / 1000), 200);
    return () => clearInterval(id);
  }, [phase]);

  const stopRecording = useCallback(() => {
    const r = rec.current;
    if (!r) return;
    cancelAnimationFrame(r.raf);
    if (r.mr.state !== "inactive") r.mr.stop();
    r.stream.getTracks().forEach((t) => t.stop());
    void r.ctx.close();
    rec.current = null;
  }, []);

  useEffect(() => {
    if (phase.k === "recording" && elapsed >= MAX_REC_S) stopRecording();
  }, [phase.k, elapsed, stopRecording]);

  useEffect(() => () => stopRecording(), [stopRecording]);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: false } });
      const mime = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4"].find((m) => MediaRecorder.isTypeSupported(m));
      const mr = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined);
      const chunks: Blob[] = [];
      mr.ondataavailable = (e) => e.data.size && chunks.push(e.data);
      mr.onstop = () => {
        const type = mr.mimeType || "audio/webm";
        setFile(new Blob(chunks, { type }), `Recording.${type.includes("mp4") ? "mp4" : "webm"}`);
      };
      const ctx = new AudioContext();
      const an = ctx.createAnalyser();
      an.fftSize = 512;
      ctx.createMediaStreamSource(stream).connect(an);
      const buf = new Uint8Array(an.fftSize);
      const tick = () => {
        an.getByteTimeDomainData(buf);
        let m = 0;
        for (const v of buf) m = Math.max(m, Math.abs(v - 128) / 128);
        setLevel(m);
        if (rec.current) rec.current.raf = requestAnimationFrame(tick);
      };
      rec.current = { mr, stream, ctx, raf: 0 };
      rec.current.raf = requestAnimationFrame(tick);
      mr.start(250);
      setElapsed(0);
      setPhase({ k: "recording", started: Date.now() });
    } catch {
      setPhase({ k: "error", message: "Microphone access was blocked. Allow it in the browser, or upload a file instead." });
    }
  };

  const analyze = async () => {
    if (!audio || !consent) return;
    setPhase({ k: "uploading" });
    try {
      const { job_id, duration_s } = await submit(audio.blob, audio.name, true);
      setElapsed(0);
      setPhase({ k: "processing", jobId: job_id, progress: 0, duration: duration_s, started: Date.now() });
    } catch (e) {
      setPhase({ k: "error", message: (e as Error).message });
    }
  };

  // poll the job
  const jobId = phase.k === "processing" ? phase.jobId : null;
  useEffect(() => {
    if (!jobId) return;
    let alive = true;
    const poll = async () => {
      try {
        const j = await getJob(jobId);
        if (!alive) return;
        if (j.status === "done" && j.result) setPhase({ k: "done", jobId, result: j.result, warnings: j.warnings });
        else if (j.status === "error") setPhase({ k: "error", message: j.error ?? "Analysis failed" });
        else {
          setPhase((p) => (p.k === "processing" ? { ...p, progress: j.progress } : p));
          setTimeout(poll, 400);
        }
      } catch (e) {
        if (alive) setPhase({ k: "error", message: (e as Error).message });
      }
    };
    poll();
    return () => {
      alive = false;
    };
  }, [jobId]);

  if (phase.k === "done" && audio) {
    return <Results result={phase.result} audio={audio} name={audio.name} warnings={phase.warnings} onReset={reset} />;
  }

  const busy = phase.k === "uploading" || phase.k === "processing";
  const maxMin = card ? card.limits.max_seconds / 60 : 5;

  return (
    <div className="mx-auto max-w-3xl space-y-5">
      {/* engine status */}
      {health === "offline" && (
        <div className="rounded-xl border border-line bg-surface-2 px-4 py-3 text-sm text-ink-2">
          <span className="font-medium text-ink">The local engine isn&apos;t running.</span> Start it with{" "}
          <code className="rounded bg-surface-3 px-1.5 py-0.5 font-mono text-xs">python backend/run.py</code>
        </div>
      )}
      {health !== null && health !== "offline" && health.status === "loading" && (
        <div className="rounded-xl border border-line bg-surface-2 px-4 py-3 text-sm text-ink-2">
          Loading the speech model (about 1.2 GB) — this takes a few seconds the first time.
        </div>
      )}

      {/* drop zone / recorder / processing */}
      <div
        className={`ring-gradient relative overflow-hidden rounded-3xl bg-surface transition ${drag ? "bg-surface-2" : ""}`}
        onDragOver={(e) => {
          e.preventDefault();
          if (!busy && phase.k !== "recording") setDrag(true);
        }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDrag(false);
          const f = e.dataTransfer.files?.[0];
          if (f && !busy && phase.k !== "recording") setFile(f, f.name);
        }}
      >
        <div className="pointer-events-none absolute left-1/2 top-[-160px] h-[300px] w-[520px] -translate-x-1/2 rounded-full bg-[radial-gradient(closest-side,rgb(91_108_255_/_0.16),transparent)]" />

        {phase.k === "recording" ? (
          <div className="relative flex flex-col items-center px-6 py-14 text-center">
            <div className="flex h-16 items-center gap-1">
              {Array.from({ length: 28 }, (_, i) => (
                <span key={i} className="w-1 rounded-full bg-critical transition-[height] duration-75"
                      style={{ height: `${8 + level * 56 * (0.4 + 0.6 * Math.abs(Math.sin(i * 0.9 + elapsed * 6)))}px` }} />
              ))}
            </div>
            <p className="tabular mt-6 font-mono text-3xl">{fmtTime(elapsed).replace(/\.\d$/, "")}</p>
            <p className="mt-1 text-sm text-muted">Recording · stops automatically at {MAX_REC_S / 60} min</p>
            <button onClick={stopRecording}
                    className="mt-8 flex items-center gap-2 rounded-xl bg-ink px-5 py-3 text-sm font-medium text-bg hover:bg-white">
              <span className="h-2.5 w-2.5 rounded-sm bg-critical" /> Stop recording
            </button>
          </div>
        ) : busy ? (
          <div className="relative px-6 py-14 sm:px-12">
            <p className="text-center text-sm text-muted">{phase.k === "uploading" ? "Decoding audio…" : "Analysing"}</p>
            <p className="mt-1 truncate text-center text-lg font-medium">{audio?.name}</p>
            <div className="mx-auto mt-8 h-1.5 max-w-md overflow-hidden rounded-full bg-surface-3">
              <div className="h-full rounded-full bg-gradient-to-r from-[#5b6cff] to-[#c7cfff] transition-[width] duration-300"
                   style={{ width: `${phase.k === "processing" ? Math.max(3, phase.progress * 100) : 3}%` }} />
            </div>
            {phase.k === "processing" && (
              <div className="tabular mx-auto mt-3 flex max-w-md justify-between text-xs text-muted">
                <span>{Math.round(phase.progress * 100)}% · {fmtTime(phase.duration)} of audio</span>
                <span>
                  {phase.progress > 0.05
                    ? `~${Math.max(1, Math.round((elapsed / phase.progress) * (1 - phase.progress)))} s left`
                    : "estimating…"}
                </span>
              </div>
            )}
            <ol className="mx-auto mt-10 grid max-w-md gap-3 text-sm">
              {[
                ["Decode & resample to 16 kHz mono", phase.k === "processing"],
                ["Encode 3 s windows every 0.5 s (XLSR-53)", phase.k === "processing" && phase.progress >= 1],
                ["Score five event detectors & build timeline", false],
              ].map(([label, done], i) => (
                <li key={i} className="flex items-center gap-3">
                  <span className={`flex h-5 w-5 items-center justify-center rounded-full border text-[10px] ${
                    done ? "border-transparent bg-ink text-bg" : "border-line-strong text-muted"}`}>
                    {done ? "✓" : i + 1}
                  </span>
                  <span className={done ? "text-ink" : "text-muted"}>{label as string}</span>
                </li>
              ))}
            </ol>
          </div>
        ) : audio && phase.k !== "error" ? (
          <div className="relative flex flex-col items-center px-6 py-12 text-center">
            <div className="flex h-12 w-12 items-center justify-center rounded-2xl border border-line-strong bg-surface-3">
              <svg viewBox="0 0 24 24" className="h-5 w-5 text-ink" fill="none" stroke="currentColor" strokeWidth="1.6">
                <path d="M9 18V5l12-2v13" /><circle cx="6" cy="18" r="3" /><circle cx="18" cy="16" r="3" />
              </svg>
            </div>
            <p className="mt-4 max-w-full truncate text-lg font-medium">{audio.name}</p>
            <p className="text-sm text-muted">{fmtSize(audio.size)}</p>
            <audio src={audio.url} controls className="mt-5 h-10 w-full max-w-sm opacity-90" />
            <button onClick={reset} className="mt-4 text-sm text-muted underline-offset-4 hover:text-ink hover:underline">
              Choose a different recording
            </button>
          </div>
        ) : (
          <div className="relative flex flex-col items-center px-6 py-16 text-center">
            <div className="flex h-14 w-14 items-center justify-center rounded-2xl border border-line-strong bg-surface-3">
              <svg viewBox="0 0 24 24" className="h-6 w-6 text-ink" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round">
                <path d="M12 16V4m0 0-4 4m4-4 4 4" /><path d="M4 16v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2" />
              </svg>
            </div>
            <p className="mt-5 text-xl font-medium tracking-tight">Drop a recording here</p>
            <p className="mt-1 text-sm text-muted">
              WAV, MP3, M4A, FLAC, OGG or WebM · 3 s to {maxMin} min · up to 30 MB
            </p>
            <div className="mt-8 flex flex-wrap justify-center gap-3">
              <button onClick={() => input.current?.click()}
                      className="rounded-xl bg-ink px-5 py-3 text-sm font-medium text-bg transition hover:bg-white">
                Choose file
              </button>
              <button onClick={startRecording}
                      className="flex items-center gap-2 rounded-xl border border-line-strong bg-surface-2 px-5 py-3 text-sm font-medium transition hover:bg-surface-3">
                <span className="h-2 w-2 rounded-full bg-critical" /> Record
              </button>
            </div>
            {phase.k === "error" && (
              <p className="mt-6 max-w-md rounded-lg border border-critical/30 bg-critical/10 px-3 py-2 text-sm text-ink-2">
                {phase.message}
              </p>
            )}
          </div>
        )}
        <input ref={input} type="file" accept={ACCEPT} className="hidden"
               onChange={(e) => {
                 const f = e.target.files?.[0];
                 if (f) setFile(f, f.name);
                 e.target.value = "";
               }} />
      </div>

      {/* consent + action */}
      {!busy && phase.k !== "recording" && (
        <div className="space-y-4 rounded-2xl border border-line bg-surface p-5">
          <label className="flex cursor-pointer items-start gap-3 text-sm leading-relaxed text-ink-2">
            <input type="checkbox" checked={consent} onChange={(e) => setConsent(e.target.checked)}
                   className="mt-1 h-4 w-4 shrink-0 accent-[#8b9bff]" />
            <span>
              This is my own recording, or everyone in it has agreed to it being analysed. I understand Fluence is
              experimental research software and <span className="text-ink">not a diagnostic or assessment tool</span>.
            </span>
          </label>
          <div className="flex flex-wrap items-center justify-between gap-3 border-t border-line pt-4">
            <p className="flex items-center gap-2 text-xs text-muted">
              <svg viewBox="0 0 16 16" className="h-3.5 w-3.5" fill="none" stroke="currentColor" strokeWidth="1.5">
                <rect x="3" y="7" width="10" height="7" rx="1.5" /><path d="M5.5 7V5a2.5 2.5 0 0 1 5 0v2" />
              </svg>
              Processed in memory on this computer, then discarded.
            </p>
            <button
              onClick={analyze}
              disabled={!audio || !consent || !ready || phase.k !== "ready"}
              className="rounded-xl bg-ink px-5 py-2.5 text-sm font-medium text-bg transition hover:bg-white disabled:cursor-not-allowed disabled:opacity-30"
            >
              Analyze recording
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
