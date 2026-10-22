import type { Metadata } from "next";
import { Analyzer } from "@/components/Analyzer";

export const metadata: Metadata = { title: "Analyze — Fluence" };

export default function AnalyzePage() {
  return (
    <div className="mx-auto max-w-6xl px-4 pt-14 sm:px-6 sm:pt-20">
      <div className="mx-auto mb-10 max-w-3xl text-center">
        <p className="text-sm font-medium text-accent">Analyzer</p>
        <h1 className="mt-2 text-4xl font-semibold tracking-[-0.03em] sm:text-5xl">Map a recording.</h1>
        <p className="mx-auto mt-4 max-w-xl text-ink-2">
          Upload or record speech. Fluence scores every half second for five types of possible stuttering event.
        </p>
      </div>
      <Analyzer />
    </div>
  );
}
