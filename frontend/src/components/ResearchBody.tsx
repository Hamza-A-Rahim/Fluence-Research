"use client";

import type { Card } from "@/lib/api";
import { EVENTS, OVERALL, OVERALL_EXCL, TARGET_NAMES, pct } from "@/lib/events";
import { AucChart, OffsetChart } from "./ResearchCharts";
import { useCard } from "./useApi";

const limitations = (c: Card) => [
  `Trained on only ${c.train.speakers} speakers (SEP-28k-E train); tested on ${c.test.speakers} others.`,
  "Labels are majority votes of non-clinician annotators; agreement is low for blocks and prolongations.",
  "English podcast speech only. Other languages, accents and recording conditions are untested.",
  "The speech encoder is frozen, not fine-tuned, so results are not directly comparable with fine-tuned published systems.",
  "Timing inside a 3-second window is not validated: region edges in the analyzer are approximate.",
  "About 74 % of SEP-28k audio could still be retrieved; the rest is offline.",
];

function Section({ eyebrow, title, children, note }: { eyebrow: string; title: string; children: React.ReactNode; note?: string }) {
  return (
    <section className="mt-20">
      <p className="text-sm font-medium text-accent">{eyebrow}</p>
      <h2 className="mt-2 text-2xl font-semibold tracking-tight sm:text-3xl">{title}</h2>
      {note && <p className="mt-3 max-w-2xl text-sm leading-relaxed text-muted">{note}</p>}
      <div className="mt-8">{children}</div>
    </section>
  );
}

export function ResearchBody() {
  const { card, error } = useCard();

  return (
    <div className="mx-auto max-w-5xl px-4 pt-14 sm:px-6 sm:pt-20">
      <p className="text-sm font-medium text-accent">Research</p>
      <h1 className="mt-2 max-w-3xl text-4xl font-semibold tracking-[-0.03em] sm:text-5xl">
        Honest numbers, from logged experiments.
      </h1>
      <p className="mt-5 max-w-2xl text-lg leading-relaxed text-ink-2">
        The model behind the analyzer was chosen on development data only, frozen in a pre-registered plan, and then
        evaluated once on speakers it had never heard. These are those results.
      </p>

      {!card ? (
        <div className="mt-16 rounded-2xl border border-line bg-surface p-8 text-sm text-muted">
          {error ? (
            <>Results are read from the local engine&apos;s model card. Start it with{" "}
              <code className="rounded bg-surface-3 px-1.5 py-0.5 font-mono text-xs">python backend/run.py</code>.</>
          ) : "Loading results…"}
        </div>
      ) : (
        <>
          <div className="mt-12 grid gap-px overflow-hidden rounded-2xl border border-line bg-line sm:grid-cols-3">
            {[
              [card.test.targets[OVERALL].roc_auc.toFixed(3), "Test ROC-AUC, any event",
               `95% CI ${card.test.targets[OVERALL].auc_ci.map((v) => v.toFixed(3)).join("–")}`],
              [`+${(card.test.targets[OVERALL].roc_auc - card.test.baselines_roc_auc.baseline_features[OVERALL]).toFixed(3)}`,
               "over hand-crafted features", "same clips, same speakers"],
              [card.test.speakers.toString(), "unseen test speakers", `${card.test.clips.toLocaleString()} clips · evaluated once`],
            ].map(([v, l, s]) => (
              <div key={l} className="bg-surface p-6">
                <p className="tabular text-3xl font-semibold tracking-tight">{v}</p>
                <p className="mt-1 text-sm text-ink-2">{l}</p>
                <p className="tabular mt-1 text-xs text-muted">{s}</p>
              </div>
            ))}
          </div>

          <Section eyebrow="EXP-004 · held-out test" title="Detection by event type"
                   note="ROC-AUC measures how well scores rank clips with an event above clips without one: 0.5 is chance, 1.0 is perfect. All three systems use the same classifier and training clips; only the speech representation differs.">
            <div className="ring-gradient rounded-2xl bg-surface p-6">
              <AucChart card={card} />
            </div>
            <div className="mt-6 overflow-x-auto rounded-2xl border border-line">
              <table className="w-full min-w-[640px] text-sm">
                <thead className="bg-surface text-left text-xs text-muted">
                  <tr>
                    <th className="px-5 py-2.5 font-normal">Target</th>
                    <th className="px-5 py-2.5 font-normal">ROC-AUC [95% CI]</th>
                    <th className="px-5 py-2.5 text-right font-normal">Precision</th>
                    <th className="px-5 py-2.5 text-right font-normal">Recall</th>
                    <th className="px-5 py-2.5 text-right font-normal">F1</th>
                    <th className="px-5 py-2.5 text-right font-normal">Event clips</th>
                  </tr>
                </thead>
                <tbody className="tabular divide-y divide-line">
                  {[OVERALL, OVERALL_EXCL, ...EVENTS.map((e) => e.key)].map((k) => {
                    const m = card.test.targets[k];
                    return (
                      <tr key={k} className="bg-bg">
                        <td className="px-5 py-2.5">{TARGET_NAMES[k]}</td>
                        <td className="px-5 py-2.5">
                          <span className="font-medium">{m.roc_auc.toFixed(3)}</span>{" "}
                          <span className="text-muted">[{m.auc_ci[0].toFixed(3)}, {m.auc_ci[1].toFixed(3)}]</span>
                        </td>
                        <td className="px-5 py-2.5 text-right">{pct(m.precision)}</td>
                        <td className="px-5 py-2.5 text-right">{pct(m.recall)}</td>
                        <td className="px-5 py-2.5 text-right">{m.f1.toFixed(2)}</td>
                        <td className="px-5 py-2.5 text-right text-muted">{m.positives.toLocaleString()} ({pct(m.prevalence)})</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            <p className="mt-3 text-xs text-muted">
              Precision, recall and F1 at thresholds frozen on development data. Speaker-bootstrap confidence intervals.
            </p>
          </Section>

          <Section eyebrow="EXP-005 · unaligned audio" title="What happens off the clip boundaries"
                   note="Real recordings aren't cut into labelled clips. Shifting the 3-second window lowers accuracy, but averaging overlapping windows (as the analyzer does) recovers almost all of it. Part of the drop is label mismatch: a shifted window hears unlabelled neighbouring speech.">
            <div className="ring-gradient rounded-2xl bg-surface p-6">
              <OffsetChart card={card} />
            </div>
          </Section>

          <Section eyebrow="Method" title="How the model is built">
            <div className="grid gap-4 sm:grid-cols-2">
              {[
                ["Data", `SEP-28k with SEP-28k-Extended speaker partitions. Trained on ${card.train.clips.toLocaleString()} clips from ${card.train.speakers} speakers; an event counts when ${card.label_rule.replace(">=", "≥")} agree.`],
                ["Representation", `Frozen ${card.encoder.model.replace("facebook/", "")}, layer ${card.encoder.layer}, ${card.encoder.pooling} pooled over each 3-second window. Pretrained on 53 languages.`],
                ["Detectors", "One class-balanced logistic regression per event type, so a moment can carry several events at once."],
                ["Discipline", `Model choice and thresholds fixed on development data, pre-registered, then a single test evaluation (${card.test.date_utc.slice(0, 10)}).`],
              ].map(([t, b]) => (
                <div key={t} className="rounded-2xl border border-line bg-surface p-6">
                  <p className="font-medium">{t}</p>
                  <p className="mt-2 text-sm leading-relaxed text-muted">{b}</p>
                </div>
              ))}
            </div>
          </Section>

          <Section eyebrow="Limitations" title="What this model can't tell you">
            <ul className="divide-y divide-line overflow-hidden rounded-2xl border border-line">
              {limitations(card).map((l) => (
                <li key={l} className="flex gap-3 bg-surface px-5 py-3.5 text-sm text-ink-2">
                  <span className="mt-2 h-1 w-1 shrink-0 rounded-full bg-muted" />
                  {l}
                </li>
              ))}
            </ul>
            <div className="mt-6 rounded-2xl border border-line-strong bg-surface-2 p-6">
              <p className="font-medium">Not a diagnostic tool</p>
              <p className="mt-2 text-sm leading-relaxed text-muted">
                Fluence marks where a model trained on podcast speech sees patterns resembling labelled stuttering events.
                It does not diagnose, measure severity, or recommend treatment. For questions about your own speech, talk
                to a qualified speech-language pathologist.
              </p>
            </div>
            <p className="tabular mt-6 font-mono text-[11px] text-muted">
              model card exported {card.exported_utc} · commit {card.git_commit} · encoder rev{" "}
              {card.encoder.revision?.slice(0, 7)}
            </p>
          </Section>
        </>
      )}
    </div>
  );
}
