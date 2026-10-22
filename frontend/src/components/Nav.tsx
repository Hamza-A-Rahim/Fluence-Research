"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Logo } from "./Logo";
import { useHealth } from "./useApi";

const LINKS = [
  { href: "/analyze", label: "Analyze" },
  { href: "/research", label: "Research" },
  { href: "/#principles", label: "Principles" },
];

export function StatusPill() {
  const h = useHealth();
  const state =
    h === null ? { label: "Connecting", color: "var(--color-muted)", pulse: true }
    : h === "offline" ? { label: "Engine offline", color: "var(--color-critical)", pulse: false }
    : h.status === "ready" ? { label: "Engine ready · local", color: "var(--color-good)", pulse: false }
    : h.status === "loading" ? { label: "Loading model", color: "var(--color-wordrep)", pulse: true }
    : { label: "Engine error", color: "var(--color-critical)", pulse: false };
  return (
    <span
      className="inline-flex items-center gap-2 rounded-full border border-line bg-surface-2/60 px-3 py-1 text-xs text-ink-2"
      title={h && h !== "offline" && h.error ? h.error : undefined}
    >
      <span className="relative flex h-2 w-2">
        {state.pulse && (
          <span className="absolute inline-flex h-full w-full animate-ping rounded-full opacity-60"
                style={{ background: state.color }} />
        )}
        <span className="relative inline-flex h-2 w-2 rounded-full" style={{ background: state.color }} />
      </span>
      {state.label}
    </span>
  );
}

export function Nav() {
  const path = usePathname();
  return (
    <header className="sticky top-0 z-40 border-b border-line bg-bg/70 backdrop-blur-xl">
      <div className="mx-auto flex h-16 max-w-6xl items-center justify-between gap-4 px-4 sm:px-6">
        <Link href="/" className="flex items-center gap-2.5">
          <Logo />
          <span className="text-[15px] font-semibold tracking-tight">Fluence</span>
          <span className="hidden rounded-md border border-line px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wider text-muted sm:inline">
            Research preview
          </span>
        </Link>
        <nav className="flex items-center gap-1 text-sm">
          {LINKS.map((l) => (
            <Link
              key={l.href}
              href={l.href}
              className={`hidden rounded-lg px-3 py-1.5 transition-colors hover:bg-surface-3 hover:text-ink sm:block ${
                path === l.href ? "text-ink" : "text-ink-2"
              }`}
            >
              {l.label}
            </Link>
          ))}
          <span className="ml-2 hidden md:inline-flex">
            <StatusPill />
          </span>
          <Link
            href="/analyze"
            className="ml-2 rounded-lg bg-ink px-3.5 py-1.5 font-medium text-bg transition hover:bg-white sm:hidden"
          >
            Analyze
          </Link>
        </nav>
      </div>
    </header>
  );
}
