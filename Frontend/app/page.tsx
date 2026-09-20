"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { IBM_Plex_Sans, IBM_Plex_Mono } from "next/font/google";
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";
import { api, downloadB64, gifSrc, imgSrc } from "../lib/api";
import type {
  BatchResult,
  FlowResult,
  FrameDownloadKey,
  GapInfo,
  HealthStatus,
  LocalFolder,
  Metrics,
  PredictRequest,
  PredictResult,
  SatelliteSource,
  ScanRow,
  TrainingStatus,
  TripletLoadResult,
  TripletMeta,
} from "../lib/types";

const display = IBM_Plex_Sans({
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
  variable: "--font-display",
});
const mono = IBM_Plex_Mono({
  subsets: ["latin"],
  weight: ["400", "500", "600"],
  variable: "--font-mono",
});

type TabId = "overview" | "interpolation" | "batch" | "motion" | "training";
type SourceMode = "catalog" | "local" | "upload";

const SIZES = [256, 384, 512, 768] as const;
const SCALES = [1.0, 0.5, 0.25] as const;
const TABS: { id: TabId; label: string }[] = [
  { id: "overview", label: "Overview" },
  { id: "interpolation", label: "Interpolation" },
  { id: "motion", label: "Motion / AMV" },
  { id: "batch", label: "Batch Validate" },
  { id: "training", label: "Training" },
];

const DEFAULT_DAY = "2026-06-12";
const HIMAWARI8_DEFAULT_DAY = "2022-06-30";
const HIMAWARI8_LAST_DAY = "2022-12-31";

/** Demo assets (video + screenshots). Override with NEXT_PUBLIC_DEMO_VIDEO_URL on Vercel if needed. */
const DEMO_VIDEO_URL =
  process.env.NEXT_PUBLIC_DEMO_VIDEO_URL ??
  "https://drive.google.com/drive/folders/1RjTIGuIH572-jUZwdtExbAzGAXTq56DV?usp=sharing";

/** Hidden from the mission console dropdown (still available via API / Streamlit). */
const HIDDEN_SOURCE_IDS = new Set(["himawari9_b13", "gk2a_ir105"]);

function fmtTime(iso: string | null | undefined) {
  if (!iso) return "—";
  return new Date(iso).toISOString().slice(11, 19) + " UTC";
}

function frameB64(value: string | null | undefined): string | null {
  return value && value.length > 0 ? value : null;
}

function hasGroundTruthData(
  predict: PredictResult | null,
  triplet: TripletLoadResult | null,
  flow: FlowResult | null
): boolean {
  if (predict?.has_ground_truth === false || triplet?.has_ground_truth === false) {
    return false;
  }
  return !!(
    frameB64(predict?.frames?.ground_truth) ||
    frameB64(triplet?.frames?.t1_gt) ||
    frameB64(flow?.frames?.ground_truth) ||
    frameB64(flow?.overlay_frames?.t1_gt)
  );
}

/** Show model id only — never expose local machine paths in the dashboard. */
function formatCheckpointLabel(path: string | null | undefined) {
  if (!path) return "—";
  const normalized = path.replace(/\\/g, "/");
  const marker = "checkpoints/";
  const idx = normalized.indexOf(marker);
  if (idx >= 0) return normalized.slice(idx + marker.length);
  const parts = normalized.split("/").filter(Boolean);
  if (parts.length >= 2) return `${parts[parts.length - 2]}/${parts[parts.length - 1]}`;
  return parts[parts.length - 1] ?? "Fine-tuned RIFE weights";
}

function StatusPill({ ok, label }: { ok: boolean; label: string }) {
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[10px] uppercase tracking-wide font-[family-name:var(--font-mono)] ${
        ok ? "bg-[var(--good)]/15 text-[var(--good)]" : "bg-[var(--bad)]/15 text-[var(--bad)]"
      }`}
    >
      <span className={`h-1.5 w-1.5 rounded-full ${ok ? "bg-[var(--good)]" : "bg-[var(--bad)]"}`} />
      {label}
    </span>
  );
}

function MetricRing({
  label,
  value,
  display: displayValue,
  max = 1,
  accent = "var(--warm)",
}: {
  label: string;
  value: number;
  display: string;
  max?: number;
  accent?: string;
}) {
  const pct = Math.min(1, value / max);
  const deg = pct * 360;
  return (
    <div className="flex flex-col items-center gap-2">
      <div
        className="relative h-20 w-20 rounded-full flex items-center justify-center"
        style={{
          background: `conic-gradient(${accent} ${deg}deg, rgba(255,255,255,0.06) ${deg}deg)`,
        }}
      >
        <div className="h-16 w-16 rounded-full bg-[var(--panel)] flex items-center justify-center">
          <span className="font-[family-name:var(--font-mono)] text-xs text-[var(--text)]">
            {displayValue}
          </span>
        </div>
      </div>
      <span className="text-[10px] uppercase tracking-[0.12em] text-[var(--muted)]">{label}</span>
    </div>
  );
}

function MetricsGrid({ metrics, prefix }: { metrics: Metrics; prefix: string }) {
  return (
    <div className="grid grid-cols-4 gap-3">
      <MetricRing label={`${prefix} SSIM`} value={metrics.ssim} display={metrics.ssim.toFixed(4)} accent="var(--warm)" />
      <MetricRing label={`${prefix} FSIM`} value={metrics.fsim} display={metrics.fsim.toFixed(4)} accent="var(--cool)" />
      <MetricRing label={`${prefix} PSNR`} value={metrics.psnr} max={45} display={metrics.psnr.toFixed(1)} accent="var(--good)" />
      <MetricRing label={`${prefix} MSE`} value={1 - metrics.mse * 100} display={metrics.mse.toFixed(5)} accent="var(--warm)" />
    </div>
  );
}

function ScanTimeline({ meta }: { meta: TripletMeta | null }) {
  if (!meta?.t0 || !meta?.t2) {
    return (
      <div className="panel p-6 text-sm text-[var(--muted)]">
        Load a triplet to see scan cadence timeline.
      </div>
    );
  }
  return (
    <div className="panel p-6 md:p-8">
      <div className="flex flex-wrap items-baseline justify-between gap-2 mb-8">
        <span className="panel-header">Scan cadence · {meta.satellite}</span>
        <span className="text-[11px] font-[family-name:var(--font-mono)] text-[var(--cool)]">
          {meta.cadence_from} min → {meta.cadence_to} min
        </span>
      </div>
      <div className="relative flex items-center justify-between">
        <div className="absolute left-0 right-0 h-px bg-white/[0.08] top-1/2 -translate-y-1/2" />
        <div className="relative z-10 flex flex-col items-center gap-3">
          <div className="h-3 w-3 rounded-full bg-white/70" />
          <div className="text-center">
            <div className="font-[family-name:var(--font-mono)] text-sm">{fmtTime(meta.t0)}</div>
            <div className="text-[10px] uppercase tracking-wide text-[var(--muted)] mt-1">t0 · real scan</div>
          </div>
        </div>
        <div className="relative z-10 flex flex-col items-center gap-3">
          <span className="absolute -top-7 text-[10px] font-[family-name:var(--font-mono)] text-[var(--warm)] whitespace-nowrap">
            synthesized
          </span>
          <div className="relative h-4 w-4">
            <div className="absolute inset-0 rounded-full bg-[var(--warm)] animate-ping opacity-40" />
            <div className="relative h-4 w-4 rounded-full bg-[var(--warm)]" />
          </div>
          <div className="text-center">
            <div className="font-[family-name:var(--font-mono)] text-sm text-[var(--warm)]">
              {fmtTime(meta.t1_pred)}
            </div>
            <div className="text-[10px] uppercase tracking-wide text-[var(--muted)] mt-1">t1 · RIFE predicted</div>
          </div>
        </div>
        <div className="relative z-10 flex flex-col items-center gap-3">
          <div className="h-3 w-3 rounded-full bg-white/70" />
          <div className="text-center">
            <div className="font-[family-name:var(--font-mono)] text-sm">{fmtTime(meta.t2)}</div>
            <div className="text-[10px] uppercase tracking-wide text-[var(--muted)] mt-1">t2 · real scan</div>
          </div>
        </div>
      </div>
    </div>
  );
}

function CompareSlider({
  predictedB64,
  groundTruthB64,
  metrics,
}: {
  predictedB64: string | null;
  groundTruthB64: string | null;
  metrics: Metrics | null;
}) {
  const [pos, setPos] = useState(50);
  if (!predictedB64 || !groundTruthB64) {
    return (
      <div className="panel p-8 text-center text-[var(--muted)] text-sm">
        Run <strong className="text-[var(--warm)]">Predict t1</strong> to compare RIFE output
        {groundTruthB64 ? "" : " vs ground truth (upload optional t1 for comparison)"}.
      </div>
    );
  }
  return (
    <div className="panel p-6 md:p-8">
      <div className="flex items-center justify-between mb-5">
        <span className="panel-header">Predicted vs ground truth · t1</span>
        <span className="text-[11px] font-[family-name:var(--font-mono)] text-[var(--muted)]">drag to compare</span>
      </div>
      <div className="relative aspect-video w-full overflow-hidden rounded-xl select-none bg-black">
        <img src={imgSrc(groundTruthB64)} alt="Ground truth t1" className="absolute inset-0 h-full w-full object-contain" />
        <div className="absolute inset-0 overflow-hidden" style={{ clipPath: `inset(0 ${100 - pos}% 0 0)` }}>
          <img src={imgSrc(predictedB64)} alt="RIFE predicted t1" className="h-full w-full object-contain" />
        </div>
        <div className="absolute top-0 bottom-0 w-[2px] bg-[var(--warm)]" style={{ left: `${pos}%` }} />
        <input
          type="range"
          min={0}
          max={100}
          value={pos}
          onChange={(e) => setPos(Number(e.target.value))}
          className="absolute inset-0 h-full w-full cursor-ew-resize opacity-0"
          aria-label="Compare predicted and ground-truth frame"
        />
        <div className="absolute bottom-3 left-3 text-[10px] font-[family-name:var(--font-mono)] text-white/70 bg-black/50 px-2 py-1 rounded">
          RIFE ← → GT
        </div>
      </div>
      {metrics && (
        <div className="mt-6">
          <MetricsGrid metrics={metrics} prefix="RIFE" />
        </div>
      )}
    </div>
  );
}

function DownloadBtn({
  label,
  b64,
  filename,
  mime = "image/png",
}: {
  label: string;
  b64: string;
  filename: string;
  mime?: string;
}) {
  return (
    <button
      type="button"
      onClick={() => downloadB64(b64, filename, mime)}
      className="text-[10px] font-[family-name:var(--font-mono)] text-[var(--cool)] hover:text-[var(--text)] transition-colors"
    >
      ↓ {label}
    </button>
  );
}

function FrameDownloadMenu({
  pngB64,
  pngFile,
  frame,
  downloadReq,
  onNcDownload,
  uploadId,
  ncEnabled = true,
}: {
  pngB64: string;
  pngFile: string;
  frame: FrameDownloadKey;
  downloadReq: PredictRequest;
  onNcDownload: (message: string | null) => void;
  uploadId?: string | null;
  ncEnabled?: boolean;
}) {
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);

  const downloadNc = async () => {
    setOpen(false);
    setBusy(true);
    onNcDownload(`Preparing NetCDF (${frame})…`);
    try {
      const res = uploadId
        ? await api.downloadUploadFrame(uploadId, frame)
        : await api.downloadFrame({ ...downloadReq, frame });
      if (!res.cached) {
        onNcDownload(`Fetching radiance from S3 for ${frame}…`);
      } else {
        onNcDownload(`Downloading ${res.filename}…`);
      }
      downloadB64(res.data_b64, res.filename, res.mime);
    } catch (e) {
      alert(e instanceof Error ? e.message : "NetCDF download failed");
    } finally {
      setBusy(false);
      onNcDownload(null);
    }
  };

  return (
    <div className="relative">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        disabled={busy}
        className="text-[10px] font-[family-name:var(--font-mono)] text-[var(--cool)] hover:text-[var(--text)] transition-colors disabled:opacity-50"
      >
        ↓ {busy ? "Preparing…" : "Download"}
      </button>
      {open && (
        <>
          <button
            type="button"
            className="fixed inset-0 z-30 cursor-default"
            aria-label="Close download menu"
            onClick={() => setOpen(false)}
          />
          <div className="absolute right-0 bottom-full mb-1 z-40 min-w-[128px] rounded-lg border border-[var(--border)] bg-[var(--panel-elevated)] py-1 shadow-lg">
            <button
              type="button"
              onClick={() => {
                downloadB64(pngB64, pngFile);
                setOpen(false);
              }}
              className="block w-full text-left px-3 py-1.5 text-[10px] font-[family-name:var(--font-mono)] text-[var(--text)] hover:bg-white/5"
            >
              Image (PNG)
            </button>
            <button
              type="button"
              onClick={downloadNc}
              disabled={busy || !ncEnabled}
              className="block w-full text-left px-3 py-1.5 text-[10px] font-[family-name:var(--font-mono)] text-[var(--text)] hover:bg-white/5 disabled:opacity-50"
            >
              NetCDF (.nc){busy ? " …" : ncEnabled ? "" : " — upload .nc first"}
            </button>
          </div>
        </>
      )}
    </div>
  );
}

function MotionOverlayFrames({
  flow,
  downloadReq,
  onNcDownload,
  uploadId,
  ncEnabled = true,
  showGroundTruth = true,
}: {
  flow: FlowResult;
  downloadReq: PredictRequest;
  onNcDownload: (message: string | null) => void;
  uploadId?: string | null;
  ncEnabled?: boolean;
  showGroundTruth?: boolean;
}) {
  const specs: { key: FrameDownloadKey; label: string; file: string }[] = [
    { key: "t0", label: "t0 · input", file: "amv_t0.png" },
    ...(showGroundTruth
      ? [{ key: "t1_gt" as const, label: "t1 · ground truth", file: "amv_t1_ground_truth.png" }]
      : []),
    { key: "t2", label: "t2 · input", file: "amv_t2.png" },
    { key: "t1_pred", label: "t1 · RIFE predicted", file: "amv_t1_predicted.png" },
  ];
  const overlays = flow.overlay_frames;
  if (!overlays?.t0) {
    return (
      <div className="panel p-6 text-sm text-[var(--muted)]">
        No arrow overlays — try a smaller grid stride or a day with more cloud texture.
      </div>
    );
  }
  return (
    <div className="panel p-6">
      <div className="flex flex-wrap items-baseline justify-between gap-2 mb-4">
        <span className="panel-header">Four frames with AMV arrows</span>
        <span className="text-[10px] font-[family-name:var(--font-mono)] text-[var(--muted)]">
          {flow.arrow_stats.n_shown}/{flow.arrow_stats.n_grid} vectors · speed-colored · NC pre-built after Predict
        </span>
      </div>
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {specs.map(({ key, label, file }) => {
          const b64 = overlays[key];
          if (!b64) return null;
          return (
            <div key={key} className="space-y-2">
              <img
                src={imgSrc(b64)}
                alt={label}
                className="rounded-xl w-full aspect-square object-contain bg-black border border-[var(--border)]"
              />
              <div className="flex items-center justify-between gap-2">
                <span className="text-[10px] uppercase tracking-wide text-[var(--muted)]">{label}</span>
                <FrameDownloadMenu
                  pngB64={b64}
                  pngFile={file}
                  frame={key}
                  downloadReq={downloadReq}
                  onNcDownload={onNcDownload}
                  uploadId={uploadId}
                  ncEnabled={ncEnabled}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function MotionTimelapses({ flow, showGroundTruth = true }: { flow: FlowResult; showGroundTruth?: boolean }) {
  const { gifs } = flow;
  if (!gifs?.predicted) return null;
  const items = [
    {
      key: "predicted" as const,
      title: "Predicted timelapse",
      subtitle: "t0 → RIFE t1 → t2",
      file: "amv_timelapse_predicted.gif",
    },
    ...(showGroundTruth
      ? [
          {
            key: "ground_truth" as const,
            title: "Ground-truth timelapse",
            subtitle: "t0 → real t1 → t2",
            file: "amv_timelapse_ground_truth.gif",
          },
        ]
      : []),
  ];
  return (
    <div className="panel p-6">
      <span className="panel-header">Timelapse animations</span>
      <p className="text-xs text-[var(--muted)] mt-2 mb-4">
        Patch-averaged AMV grid (no quality mask) — same arrow density as Streamlit. One arrow per
        grid cell; slow motion included.
      </p>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {items.map(({ key, title, subtitle, file }) => {
          const b64 = gifs[key];
          if (!b64) return null;
          return (
            <div key={key} className="space-y-3">
              <div>
                <div className="text-sm font-medium text-[var(--text)]">{title}</div>
                <div className="text-[10px] font-[family-name:var(--font-mono)] text-[var(--muted)]">{subtitle}</div>
              </div>
              <div className="rounded-xl overflow-hidden bg-black border border-[var(--border)] aspect-video flex items-center justify-center">
                <img
                  src={gifSrc(b64)}
                  alt={title}
                  className="max-h-full max-w-full object-contain"
                />
              </div>
              <DownloadBtn label="Download GIF" b64={b64} filename={file} mime="image/gif" />
            </div>
          );
        })}
      </div>
    </div>
  );
}

function ChartTooltipStyle() {
  return {
    background: "#12161F",
    border: "1px solid rgba(255,255,255,0.1)",
    borderRadius: 8,
    fontFamily: "var(--font-mono)",
    fontSize: 12,
  };
}

export default function PS12Dashboard() {
  const [tab, setTab] = useState<TabId>("overview");
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [sources, setSources] = useState<SatelliteSource[]>([]);
  const [localFolders, setLocalFolders] = useState<LocalFolder[]>([]);
  const [sourceMode, setSourceMode] = useState<SourceMode>("catalog");
  const [sourceId, setSourceId] = useState("goes19_c13");
  const [localFolder, setLocalFolder] = useState("");
  const [uploadT0, setUploadT0] = useState<File | null>(null);
  const [uploadT2, setUploadT2] = useState<File | null>(null);
  const [uploadT1, setUploadT1] = useState<File | null>(null);
  const [uploadId, setUploadId] = useState<string | null>(null);
  const [uploadHasNc, setUploadHasNc] = useState(false);
  const [day, setDay] = useState(DEFAULT_DAY);
  const [hour, setHour] = useState<number | "">("");
  const [gapMin, setGapMin] = useState(20);
  const [tripletIndex, setTripletIndex] = useState(0);
  const [size, setSize] = useState<number>(512);
  const [scale, setScale] = useState(1.0);
  const [maxBatch, setMaxBatch] = useState(12);
  const [gapInfo, setGapInfo] = useState<GapInfo | null>(null);
  const [scans, setScans] = useState<ScanRow[]>([]);
  const [scanTotal, setScanTotal] = useState(0);
  const [triplet, setTriplet] = useState<TripletLoadResult | null>(null);
  const [predict, setPredict] = useState<PredictResult | null>(null);
  const [batch, setBatch] = useState<BatchResult | null>(null);
  const [flow, setFlow] = useState<FlowResult | null>(null);
  const [training, setTraining] = useState<TrainingStatus | null>(null);
  const [loading, setLoading] = useState<string | null>(null);
  const [reportLoading, setReportLoading] = useState(false);
  const [ncDownload, setNcDownload] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showHostedNotice, setShowHostedNotice] = useState(false);

  useEffect(() => {
    if (typeof window === "undefined") return;
    if (!sessionStorage.getItem("fillframe_hosted_notice_dismissed")) {
      setShowHostedNotice(true);
    }
  }, []);

  const dismissHostedNotice = useCallback(() => {
    setShowHostedNotice(false);
    try {
      sessionStorage.setItem("fillframe_hosted_notice_dismissed", "1");
    } catch {
      /* ignore */
    }
  }, []);

  const openDemoVideo = useCallback(() => {
    window.open(DEMO_VIDEO_URL, "_blank", "noopener,noreferrer");
  }, []);

  const reqBase = useMemo(
    () => ({
      source_id: sourceMode === "catalog" ? sourceId : null,
      day: sourceMode === "catalog" ? day : null,
      hour: sourceMode === "catalog" && hour !== "" ? hour : null,
      gap_min: gapMin,
      triplet_index: tripletIndex,
      size,
      local_folder: sourceMode === "local" ? localFolder : null,
    }),
    [sourceMode, sourceId, day, hour, gapMin, tripletIndex, size, localFolder]
  );

  const predictReq = useMemo(() => ({ ...reqBase, scale }), [reqBase, scale]);

  const uploadOpts = useMemo(() => {
    if (!uploadT0 || !uploadT2) return null;
    return {
      t0: uploadT0,
      t2: uploadT2,
      t1: uploadT1,
      size,
      scale,
      gap_min: gapMin,
    };
  }, [uploadT0, uploadT2, uploadT1, size, scale, gapMin]);

  const isNcUpload = useMemo(() => {
    const names = [uploadT0?.name, uploadT2?.name, uploadT1?.name].filter(Boolean) as string[];
    return names.some((n) => /\.(nc|h5|hdf|hdf5)$/i.test(n));
  }, [uploadT0, uploadT2, uploadT1]);

  useEffect(() => {
    (async () => {
      try {
        const [h, s, f, t] = await Promise.all([
          api.health(),
          api.sources(),
          api.localFolders(),
          api.trainingStatus(),
        ]);
        const visibleSources = s.filter((src) => !HIDDEN_SOURCE_IDS.has(src.id));
        setHealth(h);
        setSources(visibleSources);
        setLocalFolders(f);
        setTraining(t);
        if (f.length && !localFolder) setLocalFolder(f[0].path);
        if (f.length && sourceMode === "catalog" && !visibleSources.find((x) => x.id === sourceId)) {
          setSourceId(visibleSources[0]?.id ?? "goes19_c13");
        }
      } catch (e) {
        setError(e instanceof Error ? e.message : "Failed to connect to backend");
      }
    })();
  }, []);

  const refreshGapInfo = useCallback(async () => {
    try {
      const info = await api.gapInfo({
        source_id: reqBase.source_id,
        day: reqBase.day,
        hour: reqBase.hour,
        gap_min: gapMin,
        local_folder: reqBase.local_folder,
      });
      setGapInfo(info);
      if (tripletIndex >= info.n_triplets) setTripletIndex(0);
    } catch {
      setGapInfo(null);
    }
  }, [reqBase, gapMin, tripletIndex]);

  const listScans = useCallback(async () => {
    if (sourceMode !== "catalog") return;
    setLoading("Listing scans…");
    setError(null);
    try {
      const res = await api.scans(sourceId, day, hour === "" ? null : hour);
      setScans(res.scans);
      setScanTotal(res.total);
      if (res.total === 0) {
        const hint =
          sourceId === "himawari8_b13"
            ? "No Himawari-8 scans for this date — archive is 2015–2022 only. Try 2022-06-30."
            : "No scans found for this date/source. Try another UTC date or hour filter.";
        setError(hint);
        setGapInfo(null);
        return;
      }
      await refreshGapInfo();
    } catch (e) {
      setError(e instanceof Error ? e.message : "List scans failed");
      setScans([]);
    } finally {
      setLoading(null);
    }
  }, [sourceMode, sourceId, day, hour, refreshGapInfo]);

  const loadTriplet = useCallback(async () => {
    setLoading(sourceMode === "upload" ? "Reading uploaded frames…" : "Loading triplet from S3 (can take 1–2 min)…");
    setError(null);
    setPredict(null);
    setFlow(null);
    setUploadId(null);
    try {
      if (sourceMode === "upload") {
        if (!uploadOpts) {
          throw new Error("Upload input t0 and input t2 are required.");
        }
        const res = await api.uploadTriplet(uploadOpts);
        setTriplet(res);
        setTripletIndex(0);
        setGapInfo(res.gap_info);
        setUploadHasNc(isNcUpload);
      } else {
        const res = await api.loadTriplet(reqBase);
        setTriplet(res);
        setTripletIndex(res.triplet_index);
        setGapInfo(res.gap_info);
      }
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Load triplet failed";
      setError(
        sourceMode === "upload"
          ? msg
          : msg.includes("fetch") || msg.includes("Failed") || msg.includes("Internal")
            ? `${msg} — S3 downloads are slow; try image size 256 or use a local .nc folder.`
            : msg
      );
    } finally {
      setLoading(null);
    }
  }, [reqBase, sourceMode, uploadOpts, isNcUpload]);

  const runPredict = useCallback(async () => {
    setError(null);
    setFlow(null);
    try {
      setLoading(
        sourceMode === "upload"
          ? "Predicting midpoint from uploaded frames…"
          : "Predicting midpoint frame + AMV timelapses (2–4 min)…"
      );
      if (sourceMode === "upload") {
        if (!uploadOpts) {
          throw new Error("Upload input t0 and input t2 before predicting.");
        }
        const res = await api.uploadPredictAndMotion(uploadOpts);
        setPredict(res.predict);
        setFlow(res.flow);
        setUploadId(res.predict.upload_id ?? null);
        setUploadHasNc(isNcUpload);
        if (!triplet && res.predict.frames.t0) {
          setTriplet({
            triplet_index: 0,
            n_triplets: 1,
            indices: { i0: 0, i1: 0, i2: 0 },
            names: res.predict.names,
            gap_info: gapInfo ?? {
              cadence_min: gapMin,
              skip: 1,
              actual_step_min: gapMin,
              total_span_min: gapMin * 2,
              n_triplets: 1,
              min_files_needed: 2,
            },
            timing: [],
            meta: {
              t0: null,
              t1_pred: null,
              t1_gt: null,
              t2: null,
              cadence_from: gapMin,
              cadence_to: gapMin / 2,
              satellite: "User upload",
            },
            frames: {
              t0: res.predict.frames.t0,
              t1_gt: res.predict.frames.ground_truth,
              t2: res.predict.frames.t2,
            },
            has_ground_truth: res.predict.has_ground_truth,
          });
        }
      } else {
        const res = await api.predictAndMotion(predictReq);
        setPredict(res.predict);
        setFlow(res.flow);
      }
      setTab("motion");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Predict or motion pipeline failed");
    } finally {
      setLoading(null);
    }
  }, [predictReq, sourceMode, uploadOpts, isNcUpload, triplet, gapInfo, gapMin]);

  const runBatch = useCallback(async () => {
    setLoading("Batch validation (may take several minutes)…");
    setError(null);
    try {
      const res = await api.batchValidate({ ...predictReq, max_triplets: maxBatch });
      setBatch(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Batch validation failed");
    } finally {
      setLoading(null);
    }
  }, [predictReq, maxBatch]);

  const isInsatSource = sourceId.includes("insat");
  const sourceLabel =
    sourceMode === "upload"
      ? "User upload"
      : sources.find((s) => s.id === sourceId)?.label ?? sourceId;

  const showGroundTruth = useMemo(
    () => hasGroundTruthData(predict, triplet, flow),
    [predict, triplet, flow]
  );
  const groundTruthB64 = useMemo(() => {
    if (!showGroundTruth) return null;
    return (
      frameB64(predict?.frames?.ground_truth) ||
      frameB64(triplet?.frames?.t1_gt) ||
      null
    );
  }, [showGroundTruth, predict, triplet]);

  const scorecard = useMemo(
    () => [
      {
        area: "Frame interpolation",
        criterion: "MSE, PSNR, SSIM, FSIM vs ground-truth t1",
        status: predict ? "Validated" : triplet ? "Ready" : "Pending",
        note: predict?.metrics
          ? `RIFE SSIM ${predict.metrics.rife.ssim.toFixed(4)} · PSNR ${predict.metrics.rife.psnr.toFixed(2)} dB · Δ SSIM +${(predict.metrics.rife.ssim - predict.metrics.linear.ssim).toFixed(4)} vs linear`
          : predict
            ? "Predicted t1 available — upload ground-truth t1 for SSIM/PSNR/MSE/FSIM"
            : "Load triplet and run Predict t1",
      },
      {
        area: "Optical flow / AMV",
        criterion: "Motion vectors between consecutive frames (RIFE bidirectional flow)",
        status: flow ? "Validated" : "Pending",
        note: flow
          ? `${flow.arrow_stats.n_shown} vectors · mean speed ${flow.speed_stats.mean_ms?.toFixed(1) ?? "—"} m/s · ${flow.geo_note}`
          : "Computed automatically after Predict t1",
      },
      {
        area: "Temporal resolution",
        criterion: "Synthetic intermediate frames (e.g. 30→15→7.5 min cadence)",
        status: predict && triplet?.meta ? "Validated" : triplet ? "Ready" : "Pending",
        note: triplet?.meta
          ? `${triplet.meta.cadence_from} min input span → ${triplet.meta.cadence_to} min effective step · predicted t1 at ${fmtTime(triplet.meta.t1_pred)}`
          : "Configure gap and load triplet",
      },
      {
        area: "Batch validation",
        criterion: "SSIM / FSIM across gap-aligned triplets",
        status: batch ? "Validated" : "Pending",
        note: batch
          ? `${batch.summary.n} triplets · mean RIFE SSIM ${batch.summary.rife_ssim_mean.toFixed(4)} · Δ +${batch.summary.delta_ssim.toFixed(4)}`
          : "Run batch validation on catalog or local folder",
      },
      {
        area: "Visualization",
        criterion: showGroundTruth
          ? "Web dashboard timelapse animations (ground truth vs predicted)"
          : "Web dashboard timelapse animations (predicted)",
        status: flow?.gifs?.predicted ? "Validated" : predict ? "Partial" : "Pending",
        note: flow?.gifs?.predicted
          ? showGroundTruth
            ? "GT and predicted AMV overlay GIFs in Motion / AMV tab · .nc frame export available"
            : "Predicted AMV overlay GIF in Motion / AMV tab · .nc frame export available"
          : "Run Predict t1 for timelapse GIFs",
      },
      {
        area: "INSAT-3DS / 3DR",
        criterion: "15 min intermediate frames on TIR1 channel (.nc I/O)",
        status: isInsatSource && predict ? "Validated" : isInsatSource ? "Ready" : predict ? "Transfer ready" : "Pending",
        note: isInsatSource
          ? predict
            ? "Model applied on INSAT TIR1 — animations at enhanced temporal resolution"
            : "Select INSAT source, load triplet, run Predict t1"
          : "Validate on GOES-19 / Himawari; switch satellite to INSAT-3DS for Step 4 deployment",
      },
      {
        area: "Comparative report",
        criterion: "PDF with metrics, ablation, frames, and PS12 scorecard",
        status: predict ? "Available" : "Pending",
        note: predict ? "Download PDF Report from Overview for evaluation deliverable" : "Run Predict t1 first",
      },
      {
        area: "Model checkpoint",
        criterion: "GOES fine-tuned RIFE weights (Practical-RIFE)",
        status: health?.checkpoint_exists ? "Live" : "Missing",
        note: health?.checkpoint_exists
          ? `GOES fine-tuned RIFE · ${formatCheckpointLabel(health.checkpoint_path)}`
          : "Start api_server with checkpoint in place",
      },
    ],
    [predict, triplet, batch, flow, health, isInsatSource, showGroundTruth]
  );

  const downloadReport = useCallback(async () => {
    if (!predict) return;
    setReportLoading(true);
    setError(null);
    try {
      const res = await api.downloadReport({
        config: {
          source_id: sourceId,
          source_label: sourceLabel,
          source_mode: sourceMode,
          day,
          gap_min: gapMin,
          size,
          scale,
          triplet_index: tripletIndex,
          hour: hour === "" ? null : hour,
        },
        triplet: triplet
          ? { meta: triplet.meta, timing: triplet.timing, names: triplet.names }
          : null,
        predict: predict
          ? {
              metrics: predict.metrics ?? { rife: { mse: 0, psnr: 0, ssim: 0, fsim: 0 }, linear: { mse: 0, psnr: 0, ssim: 0, fsim: 0 } },
              checkpoint: predict.checkpoint,
              names: predict.names,
            }
          : null,
        flow: flow
          ? {
              arrow_stats: flow.arrow_stats,
              speed_stats: flow.speed_stats,
              ablation_chart: flow.ablation_chart,
              geo_note: flow.geo_note,
              n_arrows: flow.n_arrows,
            }
          : null,
        batch: batch ?? null,
        health: health ?? null,
        training: training ?? null,
        scorecard,
        images: {
          t0: predict.frames.t0,
          t2: predict.frames.t2,
          predicted: predict.frames.predicted,
          ...(showGroundTruth && frameB64(predict.frames.ground_truth)
            ? { ground_truth: predict.frames.ground_truth }
            : {}),
          ...(flow?.overlay_frames?.t1_pred
            ? { overlay_pred: flow.overlay_frames.t1_pred }
            : {}),
        },
      });
      downloadB64(res.data_b64, res.filename, res.mime);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Report download failed");
    } finally {
      setReportLoading(false);
    }
  }, [
    predict,
    flow,
    batch,
    health,
    training,
    scorecard,
    triplet,
    sourceId,
    sourceLabel,
    sourceMode,
    day,
    gapMin,
    size,
    scale,
    tripletIndex,
    hour,
    showGroundTruth,
  ]);

  const ablationData = flow?.ablation_chart ?? [];
  const batchRows = batch?.rows ?? [];

  return (
    <div className={`${display.variable} ${mono.variable} h-screen flex overflow-hidden`} style={{ fontFamily: "var(--font-display)" }}>
      {showHostedNotice && (
        <div
          className="fixed inset-0 z-[100] flex items-center justify-center bg-black/70 px-4 backdrop-blur-sm"
          role="dialog"
          aria-modal="true"
          aria-labelledby="hosted-notice-title"
        >
          <div className="w-full max-w-md rounded-2xl border border-white/[0.1] bg-[var(--panel-elevated)] p-6 shadow-2xl">
            <p className="text-[10px] uppercase tracking-[0.18em] text-[var(--warm)] font-[family-name:var(--font-mono)]">
              Live demo notice
            </p>
            <h2 id="hosted-notice-title" className="mt-3 text-lg font-semibold leading-snug text-[var(--text)]">
              Backend fetches heavy satellite data
            </h2>
            <p className="mt-3 text-sm leading-relaxed text-[var(--muted)]">
              Full Predict / Load triplet needs a strong always-on server. The free hosting tier is limited, so
              S3 downloads and RIFE inference may time out. Would you like to watch the demo video instead?
            </p>
            <div className="mt-6 flex flex-col gap-2 sm:flex-row sm:justify-end">
              <button
                type="button"
                onClick={dismissHostedNotice}
                className="rounded-lg border border-white/[0.12] px-4 py-2.5 text-sm text-[var(--text)] hover:bg-white/[0.04]"
              >
                Close
              </button>
              <button
                type="button"
                onClick={openDemoVideo}
                className="rounded-lg bg-[var(--warm)] px-4 py-2.5 text-sm font-medium text-[#1a120e] hover:brightness-110"
              >
                Watch demo video
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Sidebar */}
      <aside className="w-72 shrink-0 border-r border-[var(--border)] bg-[var(--panel)] flex flex-col">
        <div className="p-5 border-b border-[var(--border)]">
          <div className="text-[10px] uppercase tracking-[0.2em] text-[var(--warm)] font-[family-name:var(--font-mono)]">
            ISRO PS12
          </div>
          <div className="text-sm font-semibold mt-1">Mission Console</div>
        </div>

        <div className="p-4 space-y-4 flex-1 overflow-y-auto text-sm">
          <div>
            <label className="panel-header block mb-2">Data source</label>
            <select
              value={sourceMode}
              onChange={(e) => {
                setSourceMode(e.target.value as SourceMode);
                setTriplet(null);
                setPredict(null);
                setFlow(null);
                setUploadId(null);
              }}
              className="w-full rounded-lg bg-[var(--bg)] border border-[var(--border)] px-3 py-2 text-sm"
            >
              <option value="catalog">AWS / S3 catalog</option>
              <option value="local">Local .nc folder</option>
              <option value="upload">Upload frames</option>
            </select>
          </div>

          {sourceMode === "upload" ? (
            <div className="space-y-3">
              <p className="text-[10px] leading-relaxed text-[var(--muted)]">
                Upload <strong className="text-[var(--text)]">t0</strong> and{" "}
                <strong className="text-[var(--text)]">t2</strong> (required). Ground-truth{" "}
                <strong className="text-[var(--text)]">t1</strong> is optional — PNG/JPG or NetCDF/HDF5.
              </p>
              <p className="text-[10px] leading-relaxed text-[var(--muted)]">
                Works with satellite exports from{" "}
                <strong className="text-[var(--text)]">GOES</strong>,{" "}
                <strong className="text-[var(--text)]">Himawari</strong>, or{" "}
                <strong className="text-[var(--text)]">INSAT</strong> (image or native .nc / .h5).
              </p>
              <div>
                <label className="panel-header block mb-2">Input t0</label>
                <input
                  type="file"
                  accept=".png,.jpg,.jpeg,.webp,.nc,.h5,.hdf,.hdf5,image/*"
                  onChange={(e) => setUploadT0(e.target.files?.[0] ?? null)}
                  className="w-full text-xs file:mr-2 file:rounded file:border-0 file:bg-[var(--warm)]/20 file:px-2 file:py-1 file:text-[var(--warm)]"
                />
              </div>
              <div>
                <label className="panel-header block mb-2">Ground-truth t1 (optional)</label>
                <input
                  type="file"
                  accept=".png,.jpg,.jpeg,.webp,.nc,.h5,.hdf,.hdf5,image/*"
                  onChange={(e) => setUploadT1(e.target.files?.[0] ?? null)}
                  className="w-full text-xs file:mr-2 file:rounded file:border-0 file:bg-white/10 file:px-2 file:py-1"
                />
              </div>
              <div>
                <label className="panel-header block mb-2">Input t2</label>
                <input
                  type="file"
                  accept=".png,.jpg,.jpeg,.webp,.nc,.h5,.hdf,.hdf5,image/*"
                  onChange={(e) => setUploadT2(e.target.files?.[0] ?? null)}
                  className="w-full text-xs file:mr-2 file:rounded file:border-0 file:bg-[var(--warm)]/20 file:px-2 file:py-1 file:text-[var(--warm)]"
                />
              </div>
              {(uploadT0 || uploadT2) && (
                <p className="text-[10px] font-[family-name:var(--font-mono)] text-[var(--muted)]">
                  {uploadT0?.name ?? "—"} → {uploadT1?.name ?? "(no t1)"} → {uploadT2?.name ?? "—"}
                </p>
              )}
            </div>
          ) : sourceMode === "catalog" ? (
            <>
              <div>
                <label className="panel-header block mb-2">Satellite</label>
                <select
                  value={sourceId}
                  onChange={(e) => {
                    const next = e.target.value;
                    setSourceId(next);
                    if (next === "himawari8_b13" && day > HIMAWARI8_LAST_DAY) {
                      setDay(HIMAWARI8_DEFAULT_DAY);
                    }
                  }}
                  className="w-full rounded-lg bg-[var(--bg)] border border-[var(--border)] px-3 py-2 text-sm"
                >
                  {sources.map((s) => (
                    <option key={s.id} value={s.id}>{s.label}</option>
                  ))}
                </select>
              </div>
              <div>
                <p className="text-[10px] leading-relaxed text-[var(--muted)] mb-2">
                  This day is not a calm day — the model predicts accurately and works best on such days.
                </p>
                <label className="panel-header block mb-2">UTC date</label>
                {sourceId === "himawari8_b13" && (
                  <p className="text-[10px] leading-relaxed text-[var(--muted)] mb-2">
                    Himawari-8 archive: use dates between <strong className="text-[var(--text)]">2015</strong> and{" "}
                    <strong className="text-[var(--text)]">2022</strong> (e.g. 2022-06-30).
                  </p>
                )}
                <input
                  type="date"
                  value={day}
                  onChange={(e) => setDay(e.target.value)}
                  className="w-full rounded-lg bg-[var(--bg)] border border-[var(--border)] px-3 py-2 text-sm font-[family-name:var(--font-mono)]"
                />
              </div>
              <div>
                <label className="panel-header block mb-2">Hour filter</label>
                <select
                  value={hour}
                  onChange={(e) => setHour(e.target.value === "" ? "" : Number(e.target.value))}
                  className="w-full rounded-lg bg-[var(--bg)] border border-[var(--border)] px-3 py-2 text-sm"
                >
                  <option value="">All hours</option>
                  {Array.from({ length: 24 }, (_, h) => (
                    <option key={h} value={h}>{String(h).padStart(2, "0")}:00 UTC</option>
                  ))}
                </select>
              </div>
              <button
                onClick={listScans}
                disabled={!!loading}
                className="w-full rounded-lg border border-[var(--border)] py-2 text-xs font-[family-name:var(--font-mono)] hover:bg-white/5 disabled:opacity-50"
              >
                List scans
              </button>
            </>
          ) : (
            <div>
              <label className="panel-header block mb-2">Folder</label>
              <select
                value={localFolder}
                onChange={(e) => setLocalFolder(e.target.value)}
                className="w-full rounded-lg bg-[var(--bg)] border border-[var(--border)] px-3 py-2 text-sm"
              >
                {localFolders.map((f) => (
                  <option key={f.path} value={f.path}>{f.label}</option>
                ))}
              </select>
            </div>
          )}

          <div>
            <label className="panel-header block mb-2">Gap (minutes)</label>
            <input
              type="number"
              min={5}
              max={60}
              value={gapMin}
              onChange={(e) => setGapMin(Number(e.target.value))}
              className="w-full rounded-lg bg-[var(--bg)] border border-[var(--border)] px-3 py-2 text-sm font-[family-name:var(--font-mono)]"
            />
            {gapInfo && (
              <p className="text-[10px] text-[var(--muted)] mt-1 font-[family-name:var(--font-mono)]">
                {gapInfo.n_triplets} triplets · step {gapInfo.actual_step_min.toFixed(0)} min
              </p>
            )}
          </div>

          {sourceMode !== "upload" && (
          <div>
            <label className="panel-header block mb-2">Triplet index</label>
            <input
              type="range"
              min={0}
              max={Math.max(0, (gapInfo?.n_triplets ?? 1) - 1)}
              value={tripletIndex}
              onChange={(e) => setTripletIndex(Number(e.target.value))}
              className="w-full"
            />
            <span className="text-[10px] font-[family-name:var(--font-mono)] text-[var(--muted)]">{tripletIndex}</span>
          </div>
          )}

          <div className="grid grid-cols-2 gap-2">
            <div>
              <label className="panel-header block mb-2">Size</label>
              <select
                value={size}
                onChange={(e) => setSize(Number(e.target.value))}
                className="w-full rounded-lg bg-[var(--bg)] border border-[var(--border)] px-2 py-2 text-xs"
              >
                {SIZES.map((s) => (
                  <option key={s} value={s}>{s}</option>
                ))}
              </select>
            </div>
            <div>
              <label className="panel-header block mb-2">Scale</label>
              <select
                value={scale}
                onChange={(e) => setScale(Number(e.target.value))}
                className="w-full rounded-lg bg-[var(--bg)] border border-[var(--border)] px-2 py-2 text-xs"
              >
                {SCALES.map((s) => (
                  <option key={s} value={s}>{s}</option>
                ))}
              </select>
            </div>
          </div>

          <div className="space-y-2 pt-2">
            <button
              onClick={loadTriplet}
              disabled={!!loading || (sourceMode === "upload" && (!uploadT0 || !uploadT2))}
              className="w-full rounded-lg bg-[var(--cool)]/20 text-[var(--cool)] border border-[var(--cool)]/30 py-2 text-xs font-medium hover:bg-[var(--cool)]/30 disabled:opacity-50"
            >
              {sourceMode === "upload" ? "Preview uploads" : "Load triplet"}
            </button>
            <button
              onClick={runPredict}
              disabled={!!loading || (sourceMode === "upload" && (!uploadT0 || !uploadT2))}
              className="w-full rounded-lg bg-[var(--warm)] text-[#1a0f08] py-2 text-xs font-semibold hover:opacity-90 disabled:opacity-50"
            >
              Predict t1
            </button>
          </div>
        </div>

        <div className="p-4 border-t border-[var(--border)] text-[10px] font-[family-name:var(--font-mono)] text-[var(--muted)] space-y-1">
          {health && (
            <>
              <div>Device: {health.device?.toUpperCase() ?? "—"}</div>
              <div className="flex flex-wrap gap-1 mt-1">
                <StatusPill ok={health.rife_ready} label="RIFE" />
                <StatusPill ok={health.checkpoint_exists} label="CKPT" />
              </div>
            </>
          )}
        </div>
      </aside>

      {/* Main */}
      <div className="flex-1 flex flex-col min-w-0 min-h-0 overflow-hidden">
        <div className="sticky top-0 z-20 shrink-0 border-b border-[var(--border)] bg-[var(--panel-elevated)]">
          <header className="px-6 py-4 flex flex-wrap items-center justify-between gap-4">
            <div>
              <h1 className="text-lg font-semibold">Filling the gap between scans</h1>
              <p className="text-xs text-[var(--muted)] mt-0.5">
                RIFE frame interpolation · GOES-19 / Himawari / INSAT-3DS
              </p>
            </div>
            <div className="flex items-center gap-2">
              {(loading || ncDownload || reportLoading) && (
                <span className="text-[10px] font-[family-name:var(--font-mono)] text-[var(--warm)] animate-pulse">
                  {loading || ncDownload || (reportLoading ? "Generating PDF report…" : null)}
                </span>
              )}
              <button
                type="button"
                onClick={downloadReport}
                disabled={!predict || !!loading || reportLoading}
                className="rounded-lg border border-[var(--warm)]/40 bg-[var(--warm)]/10 px-3 py-1.5 text-[10px] font-semibold uppercase tracking-wide text-[var(--warm)] hover:bg-[var(--warm)]/20 disabled:opacity-40 disabled:cursor-not-allowed"
                title={predict ? "Download PS12 evaluation PDF" : "Run Predict t1 first"}
              >
                Download Report
              </button>
              {health?.aws_mode && (
                <StatusPill ok label="AWS" />
              )}
            </div>
          </header>

          <nav className="flex gap-1 px-6 py-2 border-t border-[var(--border)] bg-[var(--panel)] overflow-x-auto">
            {TABS.map((t) => (
              <button
                key={t.id}
                onClick={() => setTab(t.id)}
                className={`px-4 py-2 rounded-lg text-xs font-medium whitespace-nowrap transition-colors ${
                  tab === t.id
                    ? "bg-[var(--warm)]/15 text-[var(--warm)]"
                    : "text-[var(--muted)] hover:text-[var(--text)] hover:bg-white/5"
                }`}
              >
                {t.label}
              </button>
            ))}
          </nav>
        </div>

        <main className="flex-1 overflow-y-auto min-h-0 p-6 space-y-6">
          {error && (
            <div className="rounded-xl border border-[var(--bad)]/40 bg-[var(--bad)]/10 px-4 py-3 text-sm text-[var(--bad)]">
              {error}
              <span className="block text-xs mt-1 text-[var(--muted)]">
                Backend: <code className="text-[var(--text)]">uvicorn api_server:app --port 8000</code>
                {" · "}First S3 load is slow — keep size at 256 and wait up to 2 minutes.
              </span>
            </div>
          )}

          {tab === "overview" && (
            <>
              <div className="panel p-6 flex flex-col lg:flex-row lg:items-start justify-between gap-4">
                <div className="max-w-3xl">
                  <h2 className="text-base font-semibold">Fill in the Frames Seamlessly</h2>
                  <p className="text-xs text-[var(--muted)] mt-2 leading-relaxed">
                    AI/ML optical-flow frame interpolation for geostationary satellite thermal infrared
                    imagery. Estimates motion between consecutive scans, synthesizes intermediate frames with
                    RIFE deep learning, and validates against higher-cadence ground truth (SSIM, PSNR, MSE, FSIM).
                    Supports GOES-19, Himawari-8, and INSAT-3DS/3DR TIR1 with .nc I/O.
                  </p>
                </div>
                <button
                  type="button"
                  onClick={downloadReport}
                  disabled={!predict || !!loading || reportLoading}
                  className="shrink-0 rounded-lg bg-[var(--warm)] text-[#1a0f08] px-5 py-2.5 text-xs font-semibold hover:opacity-90 disabled:opacity-40 disabled:cursor-not-allowed"
                >
                  {reportLoading ? "Generating PDF…" : "Download PDF Report"}
                </button>
              </div>

              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {[
                  { label: "RIFE SSIM", value: predict?.metrics?.rife?.ssim?.toFixed(4) ?? "—", accent: "var(--warm)" },
                  { label: "RIFE PSNR (dB)", value: predict?.metrics?.rife?.psnr?.toFixed(2) ?? "—", accent: "var(--good)" },
                  { label: "RIFE MSE", value: predict?.metrics?.rife?.mse?.toFixed(6) ?? "—", accent: "var(--cool)" },
                  { label: "RIFE FSIM", value: predict?.metrics?.rife?.fsim?.toFixed(4) ?? "—", accent: "var(--cool)" },
                  {
                    label: "Δ SSIM vs linear",
                    value: predict?.metrics
                      ? `+${(predict.metrics.rife.ssim - predict.metrics.linear.ssim).toFixed(4)}`
                      : "—",
                    accent: "var(--good)",
                  },
                  { label: "Temporal step", value: triplet?.meta ? `${triplet.meta.cadence_to} min` : gapInfo ? `${gapInfo.actual_step_min.toFixed(0)} min` : "—", accent: "var(--warm)" },
                  { label: "AMV vectors", value: flow?.arrow_stats?.n_shown?.toString() ?? "—", accent: "var(--cool)" },
                  { label: "Batch triplets", value: batch?.summary?.n?.toString() ?? "—", accent: "var(--muted)" },
                ].map((k) => (
                  <div key={k.label} className="panel p-4">
                    <div className="panel-header">{k.label}</div>
                    <div className="text-2xl font-semibold mt-2 font-[family-name:var(--font-mono)]" style={{ color: k.accent }}>
                      {k.value}
                    </div>
                  </div>
                ))}
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                <div className="panel p-6 space-y-3">
                  <span className="panel-header">Dataset & run configuration</span>
                  <dl className="grid grid-cols-[120px_1fr] gap-y-2 text-xs">
                    <dt className="text-[var(--muted)]">Satellite</dt>
                    <dd className="font-[family-name:var(--font-mono)]">{sourceLabel}</dd>
                    <dt className="text-[var(--muted)]">UTC date</dt>
                    <dd className="font-[family-name:var(--font-mono)]">{sourceMode === "catalog" ? day : sourceMode === "upload" ? "User upload" : "Local folder"}</dd>
                    <dt className="text-[var(--muted)]">Gap step</dt>
                    <dd className="font-[family-name:var(--font-mono)]">{gapMin} min · {gapInfo?.n_triplets ?? "—"} triplets</dd>
                    <dt className="text-[var(--muted)]">Image size</dt>
                    <dd className="font-[family-name:var(--font-mono)]">{size} px · RIFE scale {scale}</dd>
                    <dt className="text-[var(--muted)]">Checkpoint</dt>
                    <dd className="font-[family-name:var(--font-mono)]">
                      {formatCheckpointLabel(predict?.checkpoint ?? health?.checkpoint_path)}
                    </dd>
                    <dt className="text-[var(--muted)]">Device</dt>
                    <dd className="font-[family-name:var(--font-mono)]">{health?.device?.toUpperCase() ?? "—"}</dd>
                  </dl>
                </div>
                <div className="panel p-6 space-y-3">
                  <span className="panel-header">Temporal resolution enhancement</span>
                  {triplet?.meta ? (
                    <>
                      <p className="text-xs text-[var(--muted)] leading-relaxed">
                        Input scans at ~{triplet.meta.cadence_from} min cadence; model predicts midpoint frame
                        to achieve ~{triplet.meta.cadence_to} min effective temporal resolution without
                        additional satellite resources.
                      </p>
                      <dl className="grid grid-cols-[100px_1fr] gap-y-2 text-xs font-[family-name:var(--font-mono)]">
                        <dt className="text-[var(--muted)]">t0</dt>
                        <dd>{fmtTime(triplet.meta.t0)}</dd>
                        <dt className="text-[var(--warm)]">t1 pred</dt>
                        <dd className="text-[var(--warm)]">{fmtTime(triplet.meta.t1_pred)}</dd>
                        {showGroundTruth && triplet.meta.t1_gt && (
                          <>
                            <dt className="text-[var(--muted)]">t1 GT</dt>
                            <dd>{fmtTime(triplet.meta.t1_gt)}</dd>
                          </>
                        )}
                        <dt className="text-[var(--muted)]">t2</dt>
                        <dd>{fmtTime(triplet.meta.t2)}</dd>
                      </dl>
                    </>
                  ) : (
                    <p className="text-xs text-[var(--muted)]">Load a triplet to see cadence timeline and predicted midpoint timestamp.</p>
                  )}
                </div>
              </div>

              {predict && (
                <div className="panel p-6">
                  <span className="panel-header">Frame interpolation metrics · RIFE vs linear baseline</span>
                  {predict.metrics ? (
                  <div className="mt-4 overflow-x-auto">
                    <table className="w-full text-xs font-[family-name:var(--font-mono)]">
                      <thead>
                        <tr className="text-[var(--muted)] text-left border-b border-[var(--border)]">
                          <th className="py-2 pr-6">Metric</th>
                          <th className="py-2 pr-6">RIFE (predicted t1)</th>
                          <th className="py-2 pr-6">Linear baseline</th>
                          <th className="py-2">Δ (RIFE − linear)</th>
                        </tr>
                      </thead>
                      <tbody>
                        {(["ssim", "psnr", "mse", "fsim"] as const).map((key) => {
                          const r = predict.metrics!.rife[key];
                          const l = predict.metrics!.linear[key];
                          const delta = r - l;
                          const fmt = key === "psnr" ? (v: number) => v.toFixed(2) : (v: number) => v.toFixed(6);
                          const deltaFmt = key === "psnr" ? delta.toFixed(2) : delta.toFixed(6);
                          const better = key === "mse" ? delta < 0 : delta > 0;
                          return (
                            <tr key={key} className="border-b border-[var(--border)]/50">
                              <td className="py-2 pr-6 uppercase">{key}</td>
                              <td className="py-2 pr-6 text-[var(--warm)]">{fmt(r)}{key === "psnr" ? " dB" : ""}</td>
                              <td className="py-2 pr-6">{fmt(l)}{key === "psnr" ? " dB" : ""}</td>
                              <td className={`py-2 ${better ? "text-[var(--good)]" : "text-[var(--bad)]"}`}>
                                {delta > 0 && key !== "mse" ? "+" : ""}{deltaFmt}
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                  ) : (
                    <p className="text-xs text-[var(--muted)] mt-3">
                      Predicted midpoint frame is shown below. Upload an optional ground-truth t1 to compute SSIM, PSNR, MSE, and FSIM.
                    </p>
                  )}
                  {ablationData.length > 0 && (
                    <p className="text-[10px] text-[var(--muted)] mt-3">
                      Method ablation (optical-flow variants): see Motion / AMV tab · best RIFE SSIM{" "}
                      {Math.max(...ablationData.map((r) => r.ssim)).toFixed(4)}
                    </p>
                  )}
                </div>
              )}

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                <div className="panel p-6">
                  <span className="panel-header">Optical flow / AMV summary</span>
                  {flow ? (
                    <dl className="mt-4 grid grid-cols-[140px_1fr] gap-y-2 text-xs">
                      <dt className="text-[var(--muted)]">Vectors shown</dt>
                      <dd className="font-[family-name:var(--font-mono)]">{flow.arrow_stats.n_shown} / {flow.arrow_stats.n_grid} grid cells</dd>
                      <dt className="text-[var(--muted)]">Mean AMV speed</dt>
                      <dd className="font-[family-name:var(--font-mono)]">{flow.speed_stats.mean_ms?.toFixed(2) ?? "—"} m/s</dd>
                      <dt className="text-[var(--muted)]">P99 / max speed</dt>
                      <dd className="font-[family-name:var(--font-mono)]">
                        {flow.speed_stats.p99_ms?.toFixed(2) ?? "—"} / {flow.speed_stats.max_ms?.toFixed(2) ?? "—"} m/s
                      </dd>
                      <dt className="text-[var(--muted)]">Projection</dt>
                      <dd className="text-[10px]">{flow.geo_note}</dd>
                    </dl>
                  ) : (
                    <p className="text-xs text-[var(--muted)] mt-3">Run Predict t1 to estimate bidirectional optical flow and AMV overlays.</p>
                  )}
                </div>
                <div className="panel p-6">
                  <span className="panel-header">Batch validation summary</span>
                  {batch ? (
                    <dl className="mt-4 grid grid-cols-[160px_1fr] gap-y-2 text-xs font-[family-name:var(--font-mono)]">
                      <dt className="text-[var(--muted)]">Triplets evaluated</dt>
                      <dd>{batch.summary.n}</dd>
                      <dt className="text-[var(--muted)]">Mean RIFE SSIM</dt>
                      <dd className="text-[var(--warm)]">{batch.summary.rife_ssim_mean.toFixed(6)}</dd>
                      <dt className="text-[var(--muted)]">Mean linear SSIM</dt>
                      <dd>{batch.summary.linear_ssim_mean.toFixed(6)}</dd>
                      <dt className="text-[var(--muted)]">Mean Δ SSIM</dt>
                      <dd className="text-[var(--good)]">+{batch.summary.delta_ssim.toFixed(6)}</dd>
                      <dt className="text-[var(--muted)]">Mean RIFE FSIM</dt>
                      <dd>{batch.summary.rife_fsim_mean.toFixed(6)}</dd>
                    </dl>
                  ) : (
                    <p className="text-xs text-[var(--muted)] mt-3">
                      Run batch validation on the Batch Validate tab to aggregate metrics across multiple gap-aligned triplets.
                    </p>
                  )}
                </div>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                <div className="panel p-6">
                  <span className="panel-header">INSAT-3DS / 3DR deployment (Step 4)</span>
                  <p className="text-xs text-[var(--muted)] mt-3 leading-relaxed">
                    {isInsatSource
                      ? predict
                        ? "Model applied on INSAT TIR1 channel data. Intermediate frames synthesized at enhanced temporal resolution (target 15 min cadence). Export .nc frames and timelapse GIFs from Motion tab."
                        : "INSAT-3DS/3DR TIR1 selected. Load triplet and run Predict t1 to generate 15 min intermediate frames."
                      : "Steps 1–3 use GOES-19 or Himawari high-cadence data for training/validation. Switch satellite source to INSAT-3DS TIR1 in the sidebar to deploy the fine-tuned RIFE model on INSAT imagery."}
                  </p>
                  <div className="mt-3">
                    <StatusPill ok={isInsatSource && !!predict} label={isInsatSource ? (predict ? "INSAT active" : "INSAT ready") : "GOES validation mode"} />
                  </div>
                </div>
                <div className="panel p-6">
                  <span className="panel-header">Visualization deliverables</span>
                  <ul className="mt-3 space-y-2 text-xs text-[var(--muted)] list-disc list-inside">
                    {showGroundTruth && (
                      <li>Side-by-side predicted vs ground-truth t1 comparison (below)</li>
                    )}
                    <li>
                      {showGroundTruth
                        ? "Ground-truth and predicted timelapse GIFs with AMV arrows (Motion tab)"
                        : "Predicted timelapse GIF with AMV arrows (Motion tab)"}
                    </li>
                    <li>Per-frame .nc export for t0, t1_gt, t1_pred, t2 (Motion tab downloads)</li>
                    <li>PDF evaluation report with metrics, ablation, and frame panels</li>
                  </ul>
                  {flow?.gifs?.predicted && (
                    <p className="text-[10px] font-[family-name:var(--font-mono)] text-[var(--good)] mt-3">
                      Timelapse GIFs ready · open Motion / AMV tab
                    </p>
                  )}
                </div>
              </div>

              <ScanTimeline meta={triplet?.meta ?? null} />
              <CompareSlider
                predictedB64={predict?.frames?.predicted ?? null}
                groundTruthB64={groundTruthB64}
                metrics={predict?.metrics?.rife ?? null}
              />
            </>
          )}

          {tab === "interpolation" && (
            <>
              {scans.length > 0 && (
                <div className="panel p-4 overflow-x-auto">
                  <div className="panel-header mb-3">Scan catalog ({scanTotal} total, showing {scans.length})</div>
                  <table className="w-full text-xs font-[family-name:var(--font-mono)]">
                    <thead>
                      <tr className="text-[var(--muted)] text-left border-b border-[var(--border)]">
                        <th className="py-2 pr-4">File</th>
                        <th className="py-2">UTC</th>
                      </tr>
                    </thead>
                    <tbody>
                      {scans.slice(0, 15).map((s) => (
                        <tr key={s.uri} className="border-b border-[var(--border)]/50">
                          <td className="py-1.5 pr-4 truncate max-w-xs">{s.file}</td>
                          <td className="py-1.5 text-[var(--muted)]">{s.scan_utc}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}

              {triplet && (
                <div className={`grid gap-4 ${showGroundTruth ? "grid-cols-3" : "grid-cols-2"}`}>
                  {(["t0", ...(showGroundTruth ? (["t1_gt"] as const) : []), "t2"] as const).map((k) => {
                    const b64 = triplet.frames[k];
                    if (!b64) return null;
                    return (
                    <div key={k} className="panel p-3">
                      <div className="panel-header mb-2">
                        {k === "t0" ? "Input t0" : k === "t1_gt" ? "Ground truth t1" : "Input t2"}
                      </div>
                      <img src={imgSrc(b64)} alt={k} className="rounded-lg w-full aspect-square object-contain bg-black" />
                      <p className="text-[9px] font-[family-name:var(--font-mono)] text-[var(--muted)] mt-2 truncate">
                        {triplet.names[k === "t1_gt" ? "t1" : k]}
                      </p>
                    </div>
                    );
                  })}
                </div>
              )}

              {triplet?.timing && triplet.timing.length > 0 && (
                <div className="panel p-4">
                  <div className="panel-header mb-3">Timing</div>
                  <table className="w-full text-xs">
                    <tbody>
                      {triplet.timing.map((r) => (
                        <tr key={r.frame} className="border-b border-[var(--border)]/50">
                          <td className="py-1.5 text-[var(--muted)]">{r.frame}</td>
                          <td className="py-1.5 font-[family-name:var(--font-mono)] text-right">{r.time_utc}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}

              <CompareSlider
                predictedB64={predict?.frames?.predicted ?? null}
                groundTruthB64={groundTruthB64}
                metrics={predict?.metrics?.rife ?? null}
              />

              {predict && (
                <div className="panel p-6">
                  <div className="panel-header mb-4">Method comparison (current triplet)</div>
                  {predict.metrics && (
                  <div className="grid grid-cols-2 gap-6">
                    <MetricsGrid metrics={predict.metrics.rife} prefix="RIFE" />
                    <MetricsGrid metrics={predict.metrics.linear} prefix="Linear" />
                  </div>
                  )}
                  <div className="grid grid-cols-2 gap-4 mt-6">
                    <div>
                      <p className="text-xs text-[var(--muted)] mb-2">Linear baseline</p>
                      <img src={imgSrc(predict.frames.linear)} alt="Linear" className="rounded-lg w-full aspect-video object-contain bg-black" />
                    </div>
                    <div>
                      <p className="text-xs text-[var(--muted)] mb-2">RIFE predicted</p>
                      <img src={imgSrc(predict.frames.predicted)} alt="RIFE" className="rounded-lg w-full aspect-video object-contain bg-black" />
                    </div>
                  </div>
                </div>
              )}
            </>
          )}

          {tab === "batch" && (
            <>
              <div className="panel p-6 flex flex-wrap items-end gap-4">
                <div>
                  <label className="panel-header block mb-2">Max triplets</label>
                  <input
                    type="number"
                    min={1}
                    max={100}
                    value={maxBatch}
                    onChange={(e) => setMaxBatch(Number(e.target.value))}
                    className="rounded-lg bg-[var(--bg)] border border-[var(--border)] px-3 py-2 text-sm font-[family-name:var(--font-mono)] w-24"
                  />
                </div>
                <button
                  onClick={runBatch}
                  disabled={!!loading}
                  className="rounded-lg bg-[var(--warm)] text-[#1a0f08] px-6 py-2 text-sm font-semibold hover:opacity-90 disabled:opacity-50"
                >
                  Run batch validation
                </button>
              </div>

              {batch && (
                <>
                  <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                    {[
                      { l: "Mean RIFE SSIM", v: batch.summary.rife_ssim_mean.toFixed(4) },
                      { l: "Mean Linear SSIM", v: batch.summary.linear_ssim_mean.toFixed(4) },
                      { l: "Δ SSIM", v: `+${batch.summary.delta_ssim.toFixed(4)}` },
                      { l: "Mean RIFE FSIM", v: batch.summary.rife_fsim_mean.toFixed(4) },
                      { l: "Triplets", v: String(batch.summary.n) },
                    ].map((m) => (
                      <div key={m.l} className="panel p-4">
                        <div className="panel-header">{m.l}</div>
                        <div className="text-xl font-[family-name:var(--font-mono)] mt-2 text-[var(--warm)]">{m.v}</div>
                      </div>
                    ))}
                  </div>

                  <div className="panel p-6">
                    <span className="panel-header">SSIM by triplet index</span>
                    <div className="h-72 mt-4">
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={batchRows}>
                          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" vertical={false} />
                          <XAxis dataKey="i" tick={{ fill: "#7C8698", fontSize: 11 }} />
                          <YAxis domain={[0.7, 1]} tick={{ fill: "#7C8698", fontSize: 11 }} />
                          <Tooltip contentStyle={ChartTooltipStyle()} />
                          <Legend wrapperStyle={{ fontSize: 11 }} />
                          <Line type="monotone" dataKey="rife_ssim" name="RIFE" stroke="#FF7A45" strokeWidth={2} dot={false} />
                          <Line type="monotone" dataKey="linear_ssim" name="Linear" stroke="#7C8698" strokeWidth={2} strokeDasharray="4 4" dot={false} />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                </>
              )}
            </>
          )}

          {tab === "motion" && (
            <>
              {!flow && !loading && (
                <div className="panel p-6 text-sm text-[var(--muted)]">
                  Click <strong className="text-[var(--warm)]">Predict t1</strong> in the sidebar —
                  motion vectors and timelapse GIFs are generated automatically right after the
                  midpoint frame is synthesized.
                </div>
              )}

              {flow && (
                <p className="text-xs text-[var(--muted)] font-[family-name:var(--font-mono)] px-1">
                  {flow.geo_note}
                </p>
              )}

              {flow && (
                <>
                  <MotionOverlayFrames
                    flow={flow}
                    downloadReq={predictReq}
                    onNcDownload={setNcDownload}
                    uploadId={uploadId}
                    ncEnabled={sourceMode !== "upload" || uploadHasNc}
                    showGroundTruth={showGroundTruth}
                  />
                  <MotionTimelapses flow={flow} showGroundTruth={showGroundTruth} />
                </>
              )}

              {ablationData.length > 0 && (
                <div className="panel p-6">
                  <span className="panel-header">Ablation · SSIM / FSIM by method</span>
                  <div className="h-64 mt-4">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={ablationData} barGap={6}>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" vertical={false} />
                        <XAxis dataKey="method" tick={{ fill: "#7C8698", fontSize: 10 }} />
                        <YAxis domain={[0.7, 1]} tick={{ fill: "#7C8698", fontSize: 11 }} />
                        <Tooltip contentStyle={ChartTooltipStyle()} />
                        <Legend wrapperStyle={{ fontSize: 11 }} />
                        <Bar dataKey="ssim" name="SSIM" fill="#FF7A45" radius={[4, 4, 0, 0]} />
                        <Bar dataKey="fsim" name="FSIM" fill="#29D3C6" radius={[4, 4, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              )}

              {flow && (
                <div className="panel p-6">
                  <span className="panel-header mb-4 block">Method comparison (no arrows)</span>
                  <div className={`grid gap-4 ${showGroundTruth ? "grid-cols-2 lg:grid-cols-4" : "grid-cols-2 lg:grid-cols-3"}`}>
                    {[
                      { label: "Farneback", key: "farneback" as const, file: "farneback_t1.png" },
                      { label: "RIFE flow-only", key: "flow_only" as const, file: "rife_flow_only_t1.png" },
                      { label: "RIFE full", key: "full" as const, file: "rife_full_t1.png" },
                      ...(showGroundTruth
                        ? [{ label: "Ground truth", key: "ground_truth" as const, file: "ground_truth_t1.png" }]
                        : []),
                    ]
                      .map(({ label, key, file }) => {
                        const b64 = frameB64(flow.frames[key]);
                        if (!b64) return null;
                        return (
                      <div key={key} className="space-y-2">
                        <img
                          src={imgSrc(b64)}
                          alt={label}
                          className="rounded-lg w-full aspect-square object-contain bg-black"
                        />
                        <div className="flex items-center justify-between">
                          <span className="text-[10px] text-[var(--muted)]">{label}</span>
                          <DownloadBtn label="PNG" b64={b64} filename={file} />
                        </div>
                      </div>
                        );
                      })}
                  </div>
                </div>
              )}
            </>
          )}

          {tab === "training" && training && (
            <div className="space-y-6">
              <div className="panel p-6">
                <span className="panel-header">Checkpoint status</span>
                <div className="mt-4 space-y-2 text-sm">
                  <p className="font-[family-name:var(--font-mono)] text-xs text-[var(--muted)]">
                    {formatCheckpointLabel(training.checkpoint_path)}
                  </p>
                  <div className="flex gap-2">
                    <StatusPill ok={training.checkpoint_exists} label="Fine-tuned" />
                    <StatusPill ok={training.stage1_backup_exists} label="Stage 1 backup" />
                  </div>
                  {training.training_mode && (
                    <p className="text-[var(--muted)] text-xs mt-2">
                      Mode: {training.training_mode} · Stage {training.stage}
                    </p>
                  )}
                </div>
              </div>

              {Object.keys(training.full_eval).length > 0 && (
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  {[
                    { l: "RIFE SSIM", k: "rife_ssim" },
                    { l: "Δ SSIM", k: "delta_ssim" },
                    { l: "Photo err", k: "photo_err" },
                    { l: "Flow smooth", k: "flow_smooth" },
                  ].map(({ l, k }) => (
                    <div key={k} className="panel p-4">
                      <div className="panel-header">{l}</div>
                      <div className="text-xl font-[family-name:var(--font-mono)] mt-2">
                        {training.full_eval[k] != null
                          ? typeof training.full_eval[k] === "number"
                            ? training.full_eval[k].toFixed(4)
                            : String(training.full_eval[k])
                          : "—"}
                      </div>
                    </div>
                  ))}
                </div>
              )}

              <div className="panel p-6 text-sm text-[var(--muted)]">
                <p>
                  Stage 1 (statistical) and Stage 2 (PINN physics) fine-tuning run via the Streamlit tab or CLI scripts{" "}
                  <code className="text-[var(--text)]">finetune_goes.py</code> /{" "}
                  <code className="text-[var(--text)]">finetune_physics.py</code>.
                </p>
                <p className="mt-2">
                  Runtime inference uses the dashboard API — start with{" "}
                  <code className="text-[var(--text)]">uvicorn api_server:app --port 8000</code> alongside{" "}
                  <code className="text-[var(--text)]">npm run dev</code>.
                </p>
              </div>
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
