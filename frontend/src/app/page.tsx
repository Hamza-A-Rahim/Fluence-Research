import Link from "next/link";
import { EventGrid, HomeStats } from "@/components/HomeLive";
import { StatusPill } from "@/components/Nav";

const BARS = Array.from({ length: 64 }, (_, i) => {
  const x = i / 63;
  const env = Math.sin(Math.PI * x) ** 0.7;
  return 0.18 + 0.82 * env * (0.55 + 0.45 * Math.abs(Math.sin(i * 1.7) * Math.cos(i * 0.45)));
});

function HeroWave() {
  // Decorative only: not model output.
  return (
    <div aria-hidden className="flex h-28 items-center justify-center gap-[5px]">
      {BARS.map((h, i) => (
        <span
          key={i}
          className="wave-bar w-[3px] rounded-full"
          style={{
            height: `${h * 100}%`,
            animationDelay: `${(i % 16) * -0.1}s`,
            background:
              i > 18 && i < 25 ? "var(--color-block)"
              : i > 36 && i < 42 ? "var(--color-soundrep)"
              : i > 48 && i < 52 ? "var(--color-interjection)"
              : "rgb(255 255 255 / 0.22)",
          }}
        />
      ))}
    </div>
  );
}

const STEPS = [
  {
    n: "01",
    title: "Record or drop audio",
    body: "WAV, MP3, M4A or a recording from your microphone, up to five minutes. Decoded on this machine.",
  },
  {
    n: "02",
    title: "Encode every half second",
    body: "3-second windows every 0.5 s pass through a frozen multilingual speech model (XLSR-53, layer 14).",
  },
  {
    n: "03",
    title: "Map possible events",
    body: "Five independent detectors score each window; overlapping scores are averaged into a timeline.",
  },
];

const PRINCIPLES = [
  {
    title: "Private by design",
    body: "The engine runs on 127.0.0.1. Audio is decoded in a temporary folder deleted at once, analysed in memory and never stored, logged or uploaded.",
  },
  {
    title: "Only logged numbers",
    body: "Every figure in this interface is read from a logged experiment. The test set was evaluated once, under a pre-registered plan.",
  },
  {
    title: "Not a diagnosis",
    body: "Fluence marks where a model trained on podcast speech sees patterns like stuttering events. It does not assess, measure severity or advise.",
  },
  {
    title: "Respectful by default",
    body: "Stuttering is a difference in speech, not an error. Fluence speaks of possible events, never mistakes.",
  },
];

export default function Home() {
  return (
    <>
      <section className="relative overflow-hidden">
        <div className="grid-bg pointer-events-none absolute inset-0" />
        <div className="pointer-events-none absolute left-1/2 top-[-280px] h-[560px] w-[900px] -translate-x-1/2 rounded-full bg-[radial-gradient(closest-side,rgb(91_108_255_/_0.22),transparent)]" />
        <div className="relative mx-auto max-w-6xl px-4 pb-20 pt-20 text-center sm:px-6 sm:pt-28">
          <div className="fade-up mb-8 flex justify-center">
            <StatusPill />
          </div>
          <h1 className="fade-up mx-auto max-w-3xl text-balance text-5xl font-semibold tracking-[-0.035em] sm:text-7xl"
              style={{ animationDelay: "60ms" }}>
            Speech, mapped
            <span className="bg-gradient-to-b from-ink to-ink/40 bg-clip-text text-transparent"> moment by moment.</span>
          </h1>
          <p className="fade-up mx-auto mt-6 max-w-2xl text-pretty text-lg leading-relaxed text-ink-2"
             style={{ animationDelay: "120ms" }}>
            Fluence is a research model that marks possible stuttering events — blocks, prolongations, repetitions
            and interjections — across a recording. It runs entirely on this computer.
          </p>
          <div className="fade-up mt-10 flex flex-wrap justify-center gap-3" style={{ animationDelay: "180ms" }}>
            <Link href="/analyze"
                  className="rounded-xl bg-ink px-5 py-3 text-sm font-medium text-bg shadow-[0_0_0_1px_rgb(255_255_255_/_0.2),0_8px_30px_rgb(139_155_255_/_0.25)] transition hover:bg-white">
              Analyze a recording
            </Link>
            <Link href="/research"
                  className="rounded-xl border border-line-strong bg-surface-2/50 px-5 py-3 text-sm font-medium text-ink transition hover:bg-surface-3">
              Read the research →
            </Link>
          </div>
          <div className="fade-up ring-gradient mx-auto mt-16 max-w-4xl rounded-3xl bg-surface/80 p-8 shadow-2xl shadow-black/50"
               style={{ animationDelay: "240ms" }}>
            <HeroWave />
          </div>
        </div>
      </section>

      <section className="mx-auto max-w-6xl px-4 sm:px-6">
        <HomeStats />
        <p className="mt-3 text-xs text-muted">
          Held-out SEP-28k-E test set (EXP-004), evaluated once. Clip-level results on English podcast speech.
        </p>
      </section>

      <section className="mx-auto mt-28 max-w-6xl px-4 sm:px-6">
        <p className="text-sm font-medium text-accent">How it works</p>
        <h2 className="mt-2 max-w-xl text-3xl font-semibold tracking-tight sm:text-4xl">
          From waveform to timeline in three steps.
        </h2>
        <div className="mt-10 grid gap-4 md:grid-cols-3">
          {STEPS.map((s) => (
            <div key={s.n} className="ring-gradient rounded-2xl bg-surface p-6">
              <p className="font-mono text-xs text-muted">{s.n}</p>
              <p className="mt-6 text-lg font-medium">{s.title}</p>
              <p className="mt-2 text-sm leading-relaxed text-muted">{s.body}</p>
            </div>
          ))}
        </div>
      </section>

      <section className="mx-auto mt-28 max-w-6xl px-4 sm:px-6">
        <p className="text-sm font-medium text-accent">What it listens for</p>
        <h2 className="mt-2 max-w-xl text-3xl font-semibold tracking-tight sm:text-4xl">Five event types, five detectors.</h2>
        <div className="mt-10">
          <EventGrid />
        </div>
      </section>

      <section id="principles" className="mx-auto mt-28 max-w-6xl scroll-mt-24 px-4 sm:px-6">
        <p className="text-sm font-medium text-accent">Principles</p>
        <h2 className="mt-2 max-w-xl text-3xl font-semibold tracking-tight sm:text-4xl">Built with care, not hype.</h2>
        <div className="mt-10 grid gap-4 sm:grid-cols-2">
          {PRINCIPLES.map((p) => (
            <div key={p.title} className="rounded-2xl border border-line bg-surface p-6">
              <p className="font-medium">{p.title}</p>
              <p className="mt-2 text-sm leading-relaxed text-muted">{p.body}</p>
            </div>
          ))}
        </div>
      </section>

      <section className="mx-auto mt-28 max-w-6xl px-4 sm:px-6">
        <div className="ring-gradient relative overflow-hidden rounded-3xl bg-surface px-8 py-14 text-center">
          <div className="pointer-events-none absolute inset-x-0 bottom-[-200px] mx-auto h-[300px] w-[600px] rounded-full bg-[radial-gradient(closest-side,rgb(91_108_255_/_0.18),transparent)]" />
          <h2 className="relative text-3xl font-semibold tracking-tight sm:text-4xl">Try it on a recording.</h2>
          <p className="relative mx-auto mt-3 max-w-md text-ink-2">Nothing leaves this computer.</p>
          <Link href="/analyze"
                className="relative mt-8 inline-block rounded-xl bg-ink px-5 py-3 text-sm font-medium text-bg transition hover:bg-white">
            Open the analyzer
          </Link>
        </div>
      </section>
    </>
  );
}
