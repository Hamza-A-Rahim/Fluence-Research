import Link from "next/link";
import { Logo } from "./Logo";

export function Footer() {
  return (
    <footer className="mt-24 border-t border-line">
      <div className="mx-auto grid max-w-6xl gap-10 px-4 py-12 sm:px-6 md:grid-cols-[1.4fr_1fr_1fr]">
        <div className="space-y-3">
          <div className="flex items-center gap-2.5">
            <Logo className="h-5 w-5" />
            <span className="font-semibold tracking-tight">Fluence</span>
          </div>
          <p className="max-w-sm text-sm leading-relaxed text-muted">
            Independent research on detecting stuttering events in speech. Experimental research software —
            <span className="text-ink-2"> not a diagnostic or assessment tool.</span>
          </p>
        </div>
        <div className="space-y-2 text-sm">
          <p className="font-medium text-ink-2">Product</p>
          <Link href="/analyze" className="block text-muted hover:text-ink">Analyze a recording</Link>
          <Link href="/research" className="block text-muted hover:text-ink">Research &amp; results</Link>
        </div>
        <div className="space-y-2 text-sm">
          <p className="font-medium text-ink-2">Privacy</p>
          <p className="text-muted">Runs on this computer only. Audio is analysed in memory and never stored or sent anywhere.</p>
        </div>
      </div>
      <div className="border-t border-line">
        <p className="mx-auto max-w-6xl px-4 py-5 text-xs text-muted sm:px-6">
          Trained on SEP-28k (Apple, CC BY-NC 4.0) with SEP-28k-Extended partitions. Non-commercial research use.
        </p>
      </div>
    </footer>
  );
}
