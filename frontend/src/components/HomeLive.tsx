"use client";

import { EVENTS, OVERALL } from "@/lib/events";
import { useCard } from "./useApi";

/** Headline numbers, read from the model card (EXP-004 test set). Nothing is hard-coded. */
export function HomeStats() {
  const { card } = useCard();
  const t = card?.test.targets[OVERALL];
  const items = [
    { value: t ? t.roc_auc.toFixed(3) : "—", label: "ROC-AUC, any event", sub: card ? `${card.test.speakers} unseen speakers` : "held-out test" },
    { value: t ? `${Math.round(t.recall * 100)}%` : "—", label: "of event clips flagged", sub: t ? `${Math.round(t.precision * 100)}% of flags correct` : "at frozen threshold" },
    { value: "5", label: "event types", sub: "independent detectors" },
    { value: "0 B", label: "leave this computer", sub: "processed in memory" },
  ];
  return (
    <div className="grid grid-cols-2 overflow-hidden rounded-2xl border border-line bg-line md:grid-cols-4 [&>*]:bg-bg">
      {items.map((i) => (
        <div key={i.label} className="space-y-1 p-6">
          <p className="tabular text-3xl font-semibold tracking-tight">{i.value}</p>
          <p className="text-sm text-ink-2">{i.label}</p>
          <p className="text-xs text-muted">{i.sub}</p>
        </div>
      ))}
    </div>
  );
}

export function EventGrid() {
  const { card } = useCard();
  return (
    <div className="grid gap-px overflow-hidden rounded-2xl border border-line bg-line sm:grid-cols-2 lg:grid-cols-5">
      {EVENTS.map((e) => {
        const m = card?.test.targets[e.key];
        return (
          <div key={e.key} className="group flex flex-col gap-4 bg-surface p-6 transition-colors hover:bg-surface-2">
            <div className="flex items-center gap-2">
              <span className="h-2.5 w-2.5 rounded-full" style={{ background: e.color }} />
              <p className="font-medium">{e.name}</p>
            </div>
            <p className="font-mono text-sm text-ink-2">{e.example}</p>
            <p className="flex-1 text-sm leading-relaxed text-muted">{e.description}</p>
            <div className="border-t border-line pt-3">
              <p className="tabular text-lg font-semibold">{m ? m.roc_auc.toFixed(3) : "—"}</p>
              <p className="text-xs text-muted">test ROC-AUC</p>
            </div>
          </div>
        );
      })}
    </div>
  );
}
