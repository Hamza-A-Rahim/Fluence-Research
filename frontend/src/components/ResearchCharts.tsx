"use client";

import { useState } from "react";
import type { Card } from "@/lib/api";
import { EVENTS, OVERALL, OVERALL_EXCL, TARGET_NAMES } from "@/lib/events";

const ROWS = [OVERALL, OVERALL_EXCL, ...EVENTS.map((e) => e.key)];
const SERIES = {
  xlsr: { name: "XLSR-53 (this model)", color: "var(--color-block)" },
  base: { name: "wav2vec2-base-960h", color: "var(--color-ink-2)" },
  feat: { name: "Hand-crafted features", color: "var(--color-muted)" },
};

/** Test ROC-AUC per target: dot + 95 % CI for the primary system, hollow marks for the comparisons. */
export function AucChart({ card }: { card: Card }) {
  const [hover, setHover] = useState<string | null>(null);
  const lo = 0.55, hi = 0.95;
  const W = 640, LW = 210, RH = 40, top = 8;
  const x = (v: number) => LW + ((v - lo) / (hi - lo)) * (W - LW - 16);
  const H = top + ROWS.length * RH + 28;
  const b = card.test.baselines_roc_auc;
  const ticks = [0.6, 0.7, 0.8, 0.9];
  return (
    <div>
      <div className="mb-4 flex flex-wrap gap-x-5 gap-y-2 text-xs text-ink-2">
        <span className="flex items-center gap-2">
          <svg width="22" height="10"><line x1="1" x2="21" y1="5" y2="5" stroke={SERIES.xlsr.color} strokeWidth="2" /><circle cx="11" cy="5" r="4" fill={SERIES.xlsr.color} /></svg>
          {SERIES.xlsr.name}, 95% CI
        </span>
        <span className="flex items-center gap-2">
          <svg width="10" height="10"><circle cx="5" cy="5" r="3.5" fill="none" stroke={SERIES.base.color} strokeWidth="1.5" /></svg>
          {SERIES.base.name}
        </span>
        <span className="flex items-center gap-2">
          <svg width="10" height="10"><rect x="1.5" y="1.5" width="7" height="7" fill="none" stroke={SERIES.feat.color} strokeWidth="1.5" /></svg>
          {SERIES.feat.name}
        </span>
      </div>
      <div className="relative overflow-x-auto">
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full min-w-[520px]" role="img"
             aria-label="Test ROC-AUC per event type; values are listed in the table below.">
          {ticks.map((t) => (
            <g key={t}>
              <line x1={x(t)} x2={x(t)} y1={top} y2={H - 24} stroke="var(--color-line)" />
              <text x={x(t)} y={H - 6} fontSize={11} textAnchor="middle" fill="var(--color-muted)" className="tabular">{t.toFixed(1)}</text>
            </g>
          ))}
          {ROWS.map((k, i) => {
            const m = card.test.targets[k];
            const cy = top + i * RH + RH / 2;
            const on = hover === k;
            return (
              <g key={k} onPointerEnter={() => setHover(k)} onPointerLeave={() => setHover(null)}>
                <rect x={0} y={cy - RH / 2} width={W} height={RH} fill={on ? "rgb(255 255 255 / 0.03)" : "transparent"} rx={6} />
                <text x={8} y={cy + 4} fontSize={13} fill={i < 2 ? "var(--color-ink)" : "var(--color-ink-2)"}
                      fontWeight={i === 0 ? 600 : 400}>{TARGET_NAMES[k]}</text>
                <line x1={x(m.auc_ci[0])} x2={x(m.auc_ci[1])} y1={cy} y2={cy} stroke={SERIES.xlsr.color} strokeWidth={2} strokeLinecap="round" />
                <rect x={x(b.baseline_features[k]) - 3.5} y={cy - 3.5} width={7} height={7} fill="var(--color-surface)"
                      stroke={SERIES.feat.color} strokeWidth={1.5} />
                <circle cx={x(b["facebook/wav2vec2-base-960h"][k])} cy={cy} r={3.5} fill="var(--color-surface)"
                        stroke={SERIES.base.color} strokeWidth={1.5} />
                <circle cx={x(m.roc_auc)} cy={cy} r={on ? 6 : 5} fill={SERIES.xlsr.color} stroke="var(--color-surface)" strokeWidth={2} />
                {on && (
                  <text x={x(m.auc_ci[1]) + 10} y={cy + 4} fontSize={12} fill="var(--color-ink)" fontWeight={600} className="tabular">
                    {m.roc_auc.toFixed(3)}
                  </text>
                )}
              </g>
            );
          })}
        </svg>
        {hover && (
          <div className="pointer-events-none absolute right-2 top-2 rounded-xl border border-line-strong bg-surface-3/95 p-3 text-xs shadow-2xl">
            <p className="mb-1.5 text-muted">{TARGET_NAMES[hover]}</p>
            {([
              ["xlsr", card.test.targets[hover].roc_auc, `[${card.test.targets[hover].auc_ci.map((v) => v.toFixed(3)).join(", ")}]`],
              ["base", b["facebook/wav2vec2-base-960h"][hover], ""],
              ["feat", b.baseline_features[hover], ""],
            ] as const).map(([s, v, ci]) => (
              <p key={s} className="tabular flex items-center gap-2 py-0.5">
                <span className="h-[2px] w-3 rounded-full" style={{ background: SERIES[s].color }} />
                <span className="font-semibold text-ink">{v.toFixed(3)}</span>
                <span className="text-muted">{ci}</span>
                <span className="text-ink-2">{SERIES[s].name}</span>
              </p>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

/** EXP-005: AUC when the 3 s window is shifted away from the labelled clip. */
export function OffsetChart({ card }: { card: Card }) {
  const [target, setTarget] = useState<string>(OVERALL);
  const [hi, setHi] = useState<number | null>(null);
  const L = card.localisation.targets[target];
  const offs = card.localisation.offsets_s;
  const vals = offs.map((o) => L.offset_auc[o.toFixed(1)]);
  const W = 640, H = 260, pl = 44, pr = 150, pt = 16, pb = 32;
  const lo = 0.6, hi_ = 0.95;
  const x = (o: number) => pl + ((o - offs[0]) / (offs[offs.length - 1] - offs[0])) * (W - pl - pr);
  const y = (v: number) => pt + (1 - (v - lo) / (hi_ - lo)) * (H - pt - pb);
  const d = offs.map((o, i) => `${i ? "L" : "M"}${x(o)},${y(vals[i])}`).join("");
  const opts = [OVERALL, OVERALL_EXCL, ...EVENTS.map((e) => e.key)];
  return (
    <div>
      <div className="mb-4 flex flex-wrap gap-1.5">
        {opts.map((k) => (
          <button key={k} onClick={() => setTarget(k)}
                  className={`rounded-full border px-3 py-1 text-xs transition ${
                    target === k ? "border-ink bg-ink text-bg" : "border-line-strong text-ink-2 hover:text-ink"}`}>
            {TARGET_NAMES[k]}
          </button>
        ))}
      </div>
      <div className="mb-3 flex flex-wrap gap-x-5 gap-y-2 text-xs text-ink-2">
        <span className="flex items-center gap-2">
          <svg width="22" height="10"><line x1="1" x2="21" y1="5" y2="5" stroke="var(--color-block)" strokeWidth="2" /><circle cx="11" cy="5" r="3.5" fill="var(--color-block)" /></svg>
          One shifted window
        </span>
        <span className="flex items-center gap-2">
          <svg width="22" height="10"><line x1="1" x2="21" y1="5" y2="5" stroke="var(--color-soundrep)" strokeWidth="2" strokeDasharray="4 3" /></svg>
          Mean of 7 overlapping windows (used by the analyzer)
        </span>
      </div>
      <div className="overflow-x-auto">
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full min-w-[520px]" role="img"
             aria-label="ROC-AUC by window offset; values in the experiment log EXP-005.">
          {[0.6, 0.7, 0.8, 0.9].map((v) => (
            <g key={v}>
              <line x1={pl} x2={W - pr} y1={y(v)} y2={y(v)} stroke="var(--color-line)" />
              <text x={pl - 8} y={y(v) + 4} fontSize={11} textAnchor="end" fill="var(--color-muted)" className="tabular">{v.toFixed(1)}</text>
            </g>
          ))}
          {offs.map((o) => (
            <text key={o} x={x(o)} y={H - 10} fontSize={11} textAnchor="middle" fill="var(--color-muted)" className="tabular">
              {o > 0 ? "+" : ""}{o.toFixed(1)} s
            </text>
          ))}
          <line x1={pl} x2={W - pr} y1={y(L.sliding_mean_auc)} y2={y(L.sliding_mean_auc)} stroke="var(--color-soundrep)"
                strokeWidth={2} strokeDasharray="5 4" />
          <text x={W - pr + 8} y={y(L.sliding_mean_auc) + 4} fontSize={11} fill="var(--color-ink-2)" className="tabular">
            mean {L.sliding_mean_auc.toFixed(3)}
          </text>
          <path d={d} fill="none" stroke="var(--color-block)" strokeWidth={2} strokeLinejoin="round" />
          {offs.map((o, i) => (
            <g key={o} onPointerEnter={() => setHi(i)} onPointerLeave={() => setHi(null)}>
              <rect x={x(o) - 20} y={pt} width={40} height={H - pt - pb} fill="transparent" />
              {hi === i && <line x1={x(o)} x2={x(o)} y1={pt} y2={H - pb} stroke="var(--color-accent)" strokeOpacity={0.6} />}
              <circle cx={x(o)} cy={y(vals[i])} r={hi === i ? 6 : 4.5} fill="var(--color-block)" stroke="var(--color-surface)" strokeWidth={2} />
              {(hi === i || o === 0) && (
                <text x={x(o)} y={y(vals[i]) - 12} fontSize={12} textAnchor="middle" fill="var(--color-ink)" fontWeight={600} className="tabular">
                  {vals[i].toFixed(3)}
                </text>
              )}
            </g>
          ))}
        </svg>
      </div>
      <p className="mt-2 text-xs text-muted">
        Window start relative to the labelled clip. {card.localisation.clips.toLocaleString()} development clips,{" "}
        {card.localisation.speakers} speakers; the test set was not used.
      </p>
    </div>
  );
}
