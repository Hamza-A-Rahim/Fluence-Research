"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import type { Result, Track } from "@/lib/api";
import { EVENTS, OVERALL, fmtTime } from "@/lib/events";

const LABEL_W = 176;
const WAVE_H = 72;
const OVERALL_H = 60;
const LANE_H = 40;
const GAP = 8;
const AXIS_H = 26;

type Lane = { key: string; name: string; color: string; h: number; y: number; track: Track };

function useWidth<T extends HTMLElement>() {
  const ref = useRef<T>(null);
  const [w, setW] = useState(0);
  useEffect(() => {
    if (!ref.current) return;
    const ro = new ResizeObserver(([e]) => setW(e.contentRect.width));
    ro.observe(ref.current);
    return () => ro.disconnect();
  }, []);
  return [ref, w] as const;
}

function niceStep(duration: number, px: number) {
  const target = duration / Math.max(1, px / 90);
  return [0.5, 1, 2, 5, 10, 15, 30, 60, 120].find((s) => s >= target) ?? 300;
}

export function Timeline({
  result,
  peaks,
  current,
  onSeek,
}: {
  result: Result;
  peaks: Float32Array | null;
  current: number;
  onSeek: (t: number) => void;
}) {
  const [wrapRef, wrapW] = useWidth<HTMLDivElement>();
  const narrow = wrapW < 640;
  const plotW = Math.max(0, narrow ? wrapW : wrapW - LABEL_W);
  const [hover, setHover] = useState<{ t: number; x: number } | null>(null);
  const D = result.duration_s;
  const x = (t: number) => (t / D) * plotW;

  const lanes: Lane[] = useMemo(() => {
    let y = WAVE_H + GAP * 2;
    const out: Lane[] = [];
    const add = (key: string, name: string, color: string, h: number) => {
      out.push({ key, name, color, h, y, track: result.tracks[key] });
      y += h + GAP;
    };
    add(OVERALL, "Any event", "var(--color-ink)", OVERALL_H);
    for (const e of EVENTS) add(e.key, e.name, e.color, LANE_H);
    return out;
  }, [result]);
  const plotH = lanes[lanes.length - 1].y + LANE_H + GAP + AXIS_H;

  const idx = (t: number) =>
    Math.min(result.times.length - 1, Math.max(0, Math.round((t - result.step_s / 2) / result.step_s)));

  const path = (lane: Lane, close: boolean) => {
    const { times } = result;
    const s = lane.track.scores;
    const top = lane.y + 4;
    const h = lane.h - 8;
    let d = "";
    for (let i = 0; i < times.length; i++) {
      d += `${i ? "L" : "M"}${x(times[i]).toFixed(1)},${(top + h - s[i] * h).toFixed(1)}`;
    }
    if (close) d += `L${x(times[times.length - 1]).toFixed(1)},${top + h}L${x(times[0]).toFixed(1)},${top + h}Z`;
    return d;
  };

  const waveBars = useMemo(() => {
    if (!peaks || plotW <= 0) return [];
    const n = Math.floor(plotW / 3);
    const bars: { x: number; h: number; t: number }[] = [];
    for (let i = 0; i < n; i++) {
      const a = Math.floor((i / n) * peaks.length);
      const b = Math.max(a + 1, Math.floor(((i + 1) / n) * peaks.length));
      let m = 0;
      for (let j = a; j < b; j++) m = Math.max(m, peaks[j]);
      bars.push({ x: i * 3, h: Math.max(1.5, m * (WAVE_H - 8)), t: ((i + 0.5) / n) * D });
    }
    return bars;
  }, [peaks, plotW, D]);

  const step = niceStep(D, plotW);
  const ticks = Array.from({ length: Math.floor(D / step) + 1 }, (_, i) => i * step);

  const onMove = (e: React.PointerEvent<SVGSVGElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    const px = Math.min(Math.max(0, e.clientX - r.left), plotW);
    setHover({ t: (px / plotW) * D, x: px });
  };

  const hi = hover ? idx(hover.t) : 0;

  return (
    <div ref={wrapRef} className="relative w-full select-none">
      <div className="flex">
        {!narrow && (
          <div className="shrink-0" style={{ width: LABEL_W }}>
            <div style={{ height: WAVE_H + GAP * 2 }} className="flex items-center text-xs text-muted">Waveform</div>
            {lanes.map((l) => (
              <div key={l.key} style={{ height: l.h + GAP }} className="flex flex-col justify-center pr-4">
                <div className="flex items-center gap-2">
                  <span className="h-[3px] w-3 rounded-full" style={{ background: l.color }} />
                  <span className={l.key === OVERALL ? "text-sm font-medium" : "text-[13px] text-ink-2"}>{l.name}</span>
                </div>
                <span className="tabular pl-5 text-[11px] text-muted">
                  {l.track.regions.length} {l.track.regions.length === 1 ? "region" : "regions"}
                </span>
              </div>
            ))}
          </div>
        )}
        <div className="relative min-w-0 flex-1">
          {plotW > 0 && (
            <svg
              width={plotW}
              height={plotH}
              className="block cursor-crosshair touch-none"
              onPointerMove={onMove}
              onPointerLeave={() => setHover(null)}
              onClick={(e) => {
                const r = e.currentTarget.getBoundingClientRect();
                onSeek(Math.min(Math.max(0, ((e.clientX - r.left) / plotW) * D), D));
              }}
              role="img"
              aria-label="Waveform and model scores over time. The table below lists every region."
            >
              {/* waveform */}
              <g>
                {waveBars.map((b, i) => (
                  <rect key={i} x={b.x} y={WAVE_H / 2 + GAP - b.h / 2} width={2} height={b.h} rx={1}
                        fill={b.t <= current ? "var(--color-ink)" : "rgb(255 255 255 / 0.22)"} />
                ))}
                {!peaks && <rect x={0} y={WAVE_H / 2 + GAP - 0.5} width={plotW} height={1} fill="var(--color-line-strong)" />}
              </g>

              {lanes.map((l) => {
                const top = l.y + 4;
                const h = l.h - 8;
                const thrY = top + h - l.track.threshold * h;
                return (
                  <g key={l.key}>
                    <rect x={0} y={l.y} width={plotW} height={l.h} rx={8} fill="var(--color-surface-2)" />
                    {narrow && (
                      <text x={8} y={l.y + 14} fontSize={11} fill="var(--color-ink-2)">{l.name}</text>
                    )}
                    {l.track.regions.map((r, i) => (
                      <g key={i}>
                        <rect x={x(r.start)} y={l.y} width={Math.max(2, x(r.end) - x(r.start))} height={l.h}
                              fill={l.color} fillOpacity={0.1} />
                        <rect x={x(r.start)} y={l.y + l.h - 3} width={Math.max(2, x(r.end) - x(r.start))} height={3}
                              rx={1.5} fill={l.color} />
                      </g>
                    ))}
                    <path d={path(l, true)} fill={l.color} fillOpacity={0.16} />
                    <path d={path(l, false)} fill="none" stroke={l.color} strokeWidth={1.5} strokeLinejoin="round" />
                    <line x1={0} x2={plotW} y1={thrY} y2={thrY} stroke="var(--color-muted)" strokeOpacity={0.55}
                          strokeDasharray="3 4" strokeWidth={1} />
                  </g>
                );
              })}

              {/* time axis */}
              <g>
                {ticks.map((t) => (
                  <text key={t} x={Math.min(x(t), plotW - 2)} y={plotH - 8} fontSize={11} fill="var(--color-muted)"
                        textAnchor={t === 0 ? "start" : x(t) > plotW - 20 ? "end" : "middle"} className="tabular">
                    {fmtTime(t).replace(/\.0$/, "")}
                  </text>
                ))}
              </g>

              {/* playhead */}
              <line x1={x(current)} x2={x(current)} y1={0} y2={plotH - AXIS_H} stroke="var(--color-ink)" strokeWidth={1.5} />
              <circle cx={x(current)} cy={3} r={3.5} fill="var(--color-ink)" />

              {/* crosshair */}
              {hover && (
                <line x1={hover.x} x2={hover.x} y1={0} y2={plotH - AXIS_H} stroke="var(--color-accent)"
                      strokeOpacity={0.7} strokeWidth={1} />
              )}
            </svg>
          )}

          {hover && (
            <div
              className="pointer-events-none absolute top-2 z-10 w-56 rounded-xl border border-line-strong bg-surface-3/95 p-3 text-xs shadow-2xl backdrop-blur"
              style={{ left: Math.min(Math.max(0, hover.x + 14), Math.max(0, plotW - 224)) }}
            >
              <p className="tabular mb-2 font-mono text-muted">{fmtTime(hover.t)}</p>
              {lanes.map((l) => {
                const v = l.track.scores[hi];
                const above = v >= l.track.threshold;
                return (
                  <div key={l.key} className="flex items-center gap-2 py-0.5">
                    <span className="h-[2px] w-3 rounded-full" style={{ background: l.color }} />
                    <span className="tabular w-10 font-semibold text-ink">{v.toFixed(2)}</span>
                    <span className="flex-1 truncate text-ink-2">{l.name}</span>
                    {above && <span className="text-[10px] text-ink">▲</span>}
                  </div>
                );
              })}
              <p className="mt-2 border-t border-line pt-2 text-[10px] leading-snug text-muted">
                ▲ above the detector&apos;s threshold. Scores are model outputs, not measurements.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
