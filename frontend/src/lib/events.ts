// Event types, in lane order. Colours are fixed per event (never by rank).
export type EventKey = "Block" | "Prolongation" | "SoundRep" | "WordRep" | "Interjection";

export const EVENTS: { key: EventKey; name: string; example: string; description: string; color: string }[] = [
  {
    key: "Block",
    name: "Block",
    example: "“…  ball”",
    description: "A stoppage of sound or airflow before or within a word.",
    color: "var(--color-block)",
  },
  {
    key: "Prolongation",
    name: "Prolongation",
    example: "“sssun”",
    description: "A sound held longer than usual.",
    color: "var(--color-prolongation)",
  },
  {
    key: "SoundRep",
    name: "Sound repetition",
    example: "“b-b-ball”",
    description: "A sound or syllable repeated.",
    color: "var(--color-soundrep)",
  },
  {
    key: "WordRep",
    name: "Word repetition",
    example: "“I-I-I want”",
    description: "A whole word repeated.",
    color: "var(--color-wordrep)",
  },
  {
    key: "Interjection",
    name: "Interjection",
    example: "“um”, “uh”",
    description: "A filler sound or word. Counted as an event by SEP-28k; many clinicians treat it as typical.",
    color: "var(--color-interjection)",
  },
];

export const OVERALL = "any_with_interjections";
export const OVERALL_EXCL = "any_without_interjections";

export const TARGET_NAMES: Record<string, string> = {
  any_with_interjections: "Any event",
  any_without_interjections: "Any event (excl. interjections)",
  ...Object.fromEntries(EVENTS.map((e) => [e.key, e.name])),
};

export const fmtTime = (s: number) => {
  const m = Math.floor(s / 60);
  const r = s - m * 60;
  return `${m}:${r < 10 ? "0" : ""}${r.toFixed(1)}`;
};
export const pct = (x: number, d = 0) => `${(x * 100).toFixed(d)}%`;
