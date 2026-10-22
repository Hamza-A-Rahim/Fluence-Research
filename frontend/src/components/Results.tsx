"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import type { Result } from "@/lib/api";
import { EVENTS, OVERALL, OVERALL_EXCL, fmtTime, pct } from "@/lib/events";
import { Timeline } from "./Timeline";

async function computePeaks(blob: Blob, bins = 2400): Promise<Float32Array | null> {
  try {
    const ctx = new AudioContext();
    const buf = await ctx.decodeAudioData(await blob.arrayBuffer());
    void ctx.close();
    const chans = Array.from({ length: buf.numberOfChannels }, (_, c) => buf.getChannelData(c));
    const n = buf.length;
    const out = new Float32Array(bins);
    let max = 0;
    for (let i = 0; i < bins; i++) {
      const a = Math.floor((i / bins) * n);
      const b = Math.floor(((i + 1) / bins) * n);
      let m = 0;
      for (let j = a; j < b; j += 4) {
        let v = 0;
        for (const ch of chans) v += ch[j];
        m = Math.max(m, Math.abs(v / chans.length));
      }
      out[i] = m;
      max = Math.max(max, m);
    }
    if (max > 0) for (let i = 0; i < bins; i++) out[i] /= max;
    return out;
  } catch {
    return null; // waveform is cosmetic; the analysis still shows
  }
}

export function Results({
  result,
  audio,
  name,
  warnings,
  onReset,
}: {
  result: Result;
  audio: { blob: Blob; url: string };
  name: string;
  warnings: string[];
  onReset: () => void;
}) {
  const el = useRef<HTMLAudioElement>(null);
  const [peaks, setPeaks] = useState<Float32Array | null>(null);
  const [t, setT] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [showAll, setShowAll] = useState(false);

  useEffect(() => {
    let alive = true;
    computePeaks(audio.blob).then((p) => alive && setPeaks(p));
    return () => {
      alive = false;
    };
  }, [audio.blob]);

  useEffect(() => {
    if (!playing) return;
    let raf = 0;
    const loop = () => {
      if (el.current) setT(el.current.currentTime);
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [playing]);

  const seek = (s: number, play = false) => {
    if (!el.current) return;
    el.current.currentTime = s;
    setT(s);
    if (play) void el.current.play();
  };
  const toggle = () => {
    if (!el.current) return;
    if (el.current.paused) void el.current.play();
    else el.current.pause();
  };

  const rows = useMemo(
    () =>
      EVENTS.flatMap((e) => result.tracks[e.key].regions.map((r) => ({ ...r, ev: e }))).sort(
        (a, b) => a.start - b.start,
      ),
    [result],
  );
  const overall = result.tracks[OVERALL];
  const excl = result.tracks[OVERALL_EXCL];

  return (
    <div className="space-y-6">
      <audio
        ref={el}
        src={audio.url}
        preload="auto"
        onPlay={() => setPlaying(true)}
        onPause={() => setPlaying(false)}
        onEnded={() => setPlaying(false)}
        onSeeked={() => el.current && setT(el.current.currentTime)}
      />

      {/* header */}
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <p className="text-sm text-muted">Analysis complete</p>
          <h2 className="mt-1 max-w-xl truncate text-2xl font-semibold tracking-tight">{name}</h2>
          <p className="tabular mt-1 text-sm text-muted">
            {fmtTime(result.duration_s)} · {result.windows} windows · analysed in {result.elapsed_s.toFixed(1)} s
          </p>
        </div>
        <button
          onClick={onReset}
          className="rounded-lg border border-line-strong px-3.5 py-2 text-sm text-ink-2 transition hover:bg-surface-3 hover:text-ink"
        >
          Discard &amp; start over
        </button>
      </div>

      {warnings.length > 0 && (
        <div className="rounded-xl border border-line bg-surface-2 px-4 py-3 text-sm text-ink-2">
          <span className="font-medium text-ink">Audio note:</span> {warnings.join(" · ")}
        </div>
      )}

      {/* summary */}
      <div className="grid grid-cols-2 gap-px overflow-hidden rounded-2xl border border-line bg-line sm:grid-cols-3 lg:grid-cols-6">
        <div className="col-span-2 bg-surface p-5 sm:col-span-3 lg:col-span-1">
          <p className="tabular text-3xl font-semibold tracking-tight">{overall.regions.length}</p>
          <p className="mt-1 text-sm text-ink-2">regions with a possible event</p>
          <p className="tabular mt-1 text-xs text-muted">
            {pct(overall.share)} of the recording · {pct(excl.share)} excl. interjections
          </p>
        </div>
        {EVENTS.map((e) => {
          const tr = result.tracks[e.key];
          return (
            <div key={e.key} className="bg-surface p-5">
              <div className="flex items-center gap-2">
                <span className="h-2 w-2 rounded-full" style={{ background: e.color }} />
                <p className="text-sm text-ink-2">{e.name}</p>
              </div>
              <p className="tabular mt-3 text-2xl font-semibold">{tr.regions.length}</p>
              <p className="tabular text-xs text-muted">
                {tr.regions.length === 1 ? "region" : "regions"} · {tr.flagged_s.toFixed(1)} s
              </p>
            </div>
          );
        })}
      </div>

      {/* player + timeline */}
      <div className="ring-gradient rounded-2xl bg-surface p-5 sm:p-6">
        <div className="mb-5 flex items-center gap-4">
          <button
            onClick={toggle}
            aria-label={playing ? "Pause" : "Play"}
            className="flex h-11 w-11 items-center justify-center rounded-full bg-ink text-bg transition hover:scale-105"
          >
            {playing ? (
              <svg viewBox="0 0 16 16" className="h-4 w-4" fill="currentColor"><rect x="3" y="2" width="3.5" height="12" rx="1" /><rect x="9.5" y="2" width="3.5" height="12" rx="1" /></svg>
            ) : (
              <svg viewBox="0 0 16 16" className="ml-0.5 h-4 w-4" fill="currentColor"><path d="M4 2.5v11a.8.8 0 0 0 1.2.7l9-5.5a.8.8 0 0 0 0-1.4l-9-5.5A.8.8 0 0 0 4 2.5Z" /></svg>
            )}
          </button>
          <p className="tabular font-mono text-sm">
            {fmtTime(t)} <span className="text-muted">/ {fmtTime(result.duration_s)}</span>
          </p>
          <p className="ml-auto hidden text-xs text-muted sm:block">Click anywhere on the timeline to jump · dashed line = threshold</p>
        </div>
        <Timeline result={result} peaks={peaks} current={t} onSeek={(s) => seek(s)} />
      </div>

      <div className="rounded-2xl border border-line bg-surface-2/40 px-5 py-4 text-sm leading-relaxed text-muted">
        <span className="font-medium text-ink-2">How to read this.</span> Each lane is an independent detector&apos;s score
        (0–1), averaged over the overlapping 3-second windows that cover each moment. Shaded regions are where the score
        is above that detector&apos;s threshold, chosen on development data. Timing is approximate: the training data has
        no timing inside its 3-second clips, so region edges are not validated. This is research software, not a
        diagnostic or assessment tool.
      </div>

      {/* table view */}
      <div className="overflow-hidden rounded-2xl border border-line">
        <div className="flex items-center justify-between border-b border-line bg-surface px-5 py-3">
          <p className="font-medium">Regions</p>
          <p className="tabular text-xs text-muted">{rows.length} across all event types</p>
        </div>
        {rows.length === 0 ? (
          <p className="bg-surface px-5 py-8 text-center text-sm text-muted">No region is above any event detector&apos;s threshold.</p>
        ) : (
          <table className="w-full text-sm">
            <thead className="bg-surface text-left text-xs text-muted">
              <tr>
                <th className="px-5 py-2 font-normal">Event type</th>
                <th className="px-5 py-2 font-normal">Time</th>
                <th className="hidden px-5 py-2 font-normal sm:table-cell">Length</th>
                <th className="px-5 py-2 text-right font-normal">Peak score</th>
                <th className="w-12" />
              </tr>
            </thead>
            <tbody className="tabular divide-y divide-line">
              {(showAll ? rows : rows.slice(0, 12)).map((r, i) => (
                <tr key={i} className="bg-bg transition-colors hover:bg-surface">
                  <td className="px-5 py-2.5">
                    <span className="flex items-center gap-2">
                      <span className="h-2 w-2 rounded-full" style={{ background: r.ev.color }} />
                      {r.ev.name}
                    </span>
                  </td>
                  <td className="whitespace-nowrap px-5 py-2.5 font-mono text-xs text-ink-2 sm:text-sm">
                    {fmtTime(r.start)}–{fmtTime(r.end)}
                  </td>
                  <td className="hidden px-5 py-2.5 text-muted sm:table-cell">{(r.end - r.start).toFixed(1)} s</td>
                  <td className="px-5 py-2.5 text-right">{r.peak.toFixed(2)}</td>
                  <td className="pr-4 text-right">
                    <button
                      onClick={() => seek(Math.max(0, r.start - 0.5), true)}
                      aria-label={`Play from ${fmtTime(r.start)}`}
                      className="rounded-md p-1.5 text-muted transition hover:bg-surface-3 hover:text-ink"
                    >
                      <svg viewBox="0 0 16 16" className="h-3.5 w-3.5" fill="currentColor"><path d="M4 2.5v11a.8.8 0 0 0 1.2.7l9-5.5a.8.8 0 0 0 0-1.4l-9-5.5A.8.8 0 0 0 4 2.5Z" /></svg>
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
        {rows.length > 12 && (
          <button onClick={() => setShowAll((s) => !s)}
                  className="w-full border-t border-line bg-surface py-2.5 text-sm text-ink-2 hover:text-ink">
            {showAll ? "Show fewer" : `Show all ${rows.length}`}
          </button>
        )}
      </div>
    </div>
  );
}
