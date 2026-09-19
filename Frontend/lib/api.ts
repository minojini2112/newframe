import type {
  BatchResult,
  FlowResult,
  FrameDownloadKey,
  FrameDownloadResult,
  GapInfo,
  HealthStatus,
  LocalFolder,
  PredictRequest,
  PredictResult,
  ReportDownloadResult,
  ReportPayload,
  SatelliteSource,
  ScanRow,
  TrainingStatus,
  TripletLoadResult,
  TripletRequest,
  UploadPredictOptions,
} from "./types";

/** Call Python API directly — Next.js proxy times out on slow S3 triplet loads. */
const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

function uploadFormData(path: string, form: FormData) {
  return fetch(`${API_BASE}${path}`, { method: "POST", body: form });
}

async function uploadRequest<T>(path: string, form: FormData): Promise<T> {
  const res = await uploadFormData(path, form);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ?? JSON.stringify(body);
    } catch {
      /* ignore */
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

function buildUploadForm(opts: UploadPredictOptions, extra?: Record<string, string>) {
  const form = new FormData();
  form.append("t0", opts.t0);
  form.append("t2", opts.t2);
  if (opts.t1) form.append("t1", opts.t1);
  form.append("size", String(opts.size));
  form.append("scale", String(opts.scale));
  form.append("gap_min", String(opts.gap_min));
  if (extra) {
    for (const [key, value] of Object.entries(extra)) {
      form.append(key, value);
    }
  }
  return form;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...init?.headers,
    },
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ?? JSON.stringify(body);
    } catch {
      /* ignore */
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

export function imgSrc(b64: string) {
  return `data:image/png;base64,${b64}`;
}

export function downloadB64(b64: string, filename: string, mime = "image/png") {
  const link = document.createElement("a");
  link.href = `data:${mime};base64,${b64}`;
  link.download = filename;
  link.click();
}

export function gifSrc(b64: string) {
  return `data:image/gif;base64,${b64}`;
}

export const api = {
  health: () => request<HealthStatus>("/api/health"),

  sources: () => request<SatelliteSource[]>("/api/sources"),

  localFolders: () => request<LocalFolder[]>("/api/local-folders"),

  scans: (source_id: string, day: string, hour?: number | null) => {
    const params = new URLSearchParams({ source_id, day });
    if (hour != null) params.set("hour", String(hour));
    return request<{ total: number; scans: ScanRow[] }>(`/api/scans?${params}`);
  },

  gapInfo: (opts: {
    source_id?: string | null;
    day?: string | null;
    hour?: number | null;
    gap_min: number;
    local_folder?: string | null;
  }) => {
    const params = new URLSearchParams({ gap_min: String(opts.gap_min) });
    if (opts.source_id) params.set("source_id", opts.source_id);
    if (opts.day) params.set("day", opts.day);
    if (opts.hour != null) params.set("hour", String(opts.hour));
    if (opts.local_folder) params.set("local_folder", opts.local_folder);
    return request<GapInfo>(`/api/gap-info?${params}`);
  },

  loadTriplet: (body: TripletRequest) =>
    request<TripletLoadResult>("/api/triplet/load", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  uploadTriplet: (opts: UploadPredictOptions) =>
    uploadRequest<TripletLoadResult>(
      "/api/upload/triplet",
      buildUploadForm(opts)
    ),

  predict: (body: PredictRequest) =>
    request<PredictResult>("/api/predict", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  predictAndMotion: (body: PredictRequest) =>
    request<{ predict: PredictResult; flow: FlowResult }>("/api/predict-and-motion", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  uploadPredictAndMotion: (opts: UploadPredictOptions) =>
    uploadRequest<{ predict: PredictResult; flow: FlowResult }>(
      "/api/upload/predict-and-motion",
      buildUploadForm(opts, {
        quiver_step: "24",
        quiver_window: "12",
        timelapse_ms: "700",
        min_mag_px: "0.001",
      })
    ),

  batchValidate: (body: PredictRequest & { max_triplets?: number }) =>
    request<BatchResult>("/api/batch/validate", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  flowExtract: (body: PredictRequest) =>
    request<FlowResult>("/api/flow/extract", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  downloadFrame: (body: PredictRequest & { frame: FrameDownloadKey }) =>
    request<FrameDownloadResult>("/api/download/frame", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  downloadUploadFrame: (uploadId: string, frame: FrameDownloadKey) =>
    request<FrameDownloadResult>(
      `/api/upload/download/frame?upload_id=${encodeURIComponent(uploadId)}&frame=${encodeURIComponent(frame)}`
    ),

  downloadReport: (body: ReportPayload) =>
    request<ReportDownloadResult>("/api/download/report", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  trainingStatus: () => request<TrainingStatus>("/api/training/status"),
};
