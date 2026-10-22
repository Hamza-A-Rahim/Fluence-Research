"use client";

import { useEffect, useState } from "react";
import { getCard, getHealth, type Card, type Health } from "@/lib/api";

/** Polls the local API until the model is ready (or reports an error). */
export function useHealth(intervalMs = 3000) {
  const [health, setHealth] = useState<Health | "offline" | null>(null);
  useEffect(() => {
    let alive = true;
    let timer: ReturnType<typeof setTimeout>;
    const tick = async () => {
      try {
        const h = await getHealth();
        if (!alive) return;
        setHealth(h);
        if (h.status === "loading") timer = setTimeout(tick, intervalMs);
      } catch {
        if (!alive) return;
        setHealth("offline");
        timer = setTimeout(tick, intervalMs);
      }
    };
    tick();
    return () => {
      alive = false;
      clearTimeout(timer);
    };
  }, [intervalMs]);
  return health;
}

/** The model card: every number shown in the interface comes from here (logged runs only). */
export function useCard() {
  const [card, setCard] = useState<Card | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    let alive = true;
    let timer: ReturnType<typeof setTimeout>;
    const load = () =>
      getCard()
        .then((c) => alive && (setCard(c), setError(null)))
        .catch((e: Error) => {
          if (!alive) return;
          setError(e.message);
          timer = setTimeout(load, 4000);
        });
    load();
    return () => {
      alive = false;
      clearTimeout(timer);
    };
  }, []);
  return { card, error };
}
