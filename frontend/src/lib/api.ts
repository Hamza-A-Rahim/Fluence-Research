// Client for the local demo API (backend/run.py, 127.0.0.1 only).
export const API = process.env.NEXT_PUBLIC_FLUENCE_API ?? "http://127.0.0.1:8000";

export type Region = { start: number; end: number; peak: number };
export type Track = {
  threshold: number;
  scores: number[];
  regions: Region[];
  flagged_s: number;
  share: number;
  max: number;
  mean: number;
};
export type Result = {
  duration_s: number;
  hop_s: number;
  step_s: number;
  windows: number;
  times: number[];
  tracks: Record<string, Track>;
  elapsed_s: number;
};
export type Job = {
  status: "queued" | "running" | "done" | "error";
  progress: number;
  warnings: string[];
  result?: Result;
  error?: string;
};
export type TestMetrics = {
  roc_auc: number;
  auc_ci: [number, number];
  precision: number;
  recall: number;
  f1: number;
  average_precision: number;
  positives: number;
  n: number;
  prevalence: number;
};
export type Card = {
  exported_utc: string;
  git_commit: string;
  system: string;
  encoder: { model: string; layer: number; pooling: string; revision?: string };
  classifier: string;
  label_rule: string;
  thresholds: Record<string, number>;
  train: { clips: number; speakers: number; partition: string };
  test: {
    experiment: string;
    clips: number;
    speakers: number;
    date_utc: string;
    targets: Record<string, TestMetrics>;
    baselines_roc_auc: Record<string, Record<string, number>>;
  };
  localisation: {
    experiment: string;
    clips: number;
    speakers: number;
    offsets_s: number[];
    targets: Record<
      string,
      {
        clip_auc: number;
        offset_auc: Record<string, number>;
        sliding_mean_auc: number;
        sliding_mean_ci: [number, number];
        sliding_max_auc: number;
      }
    >;
  };
  limits: { max_seconds: number; min_seconds: number; max_bytes: number };
};
export type Health = { status: "loading" | "ready" | "error"; error: string | null };

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body?.detail) detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {}
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

export const getHealth = () => fetch(`${API}/api/health`, { cache: "no-store" }).then((r) => json<Health>(r));
export const getCard = () => fetch(`${API}/api/model`, { cache: "no-store" }).then((r) => json<Card>(r));
export const getJob = (id: string) => fetch(`${API}/api/jobs/${id}`, { cache: "no-store" }).then((r) => json<Job>(r));
export const deleteJob = (id: string) => fetch(`${API}/api/jobs/${id}`, { method: "DELETE" });

export function submit(file: Blob, name: string, consent: boolean) {
  const fd = new FormData();
  fd.append("file", file, name);
  fd.append("consent", String(consent));
  return fetch(`${API}/api/analyze`, { method: "POST", body: fd }).then((r) =>
    json<{ job_id: string; duration_s: number }>(r),
  );
}
