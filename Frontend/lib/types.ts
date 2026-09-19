export interface HealthStatus {
  status: string;
  rife_ready: boolean;
  checkpoint_exists: boolean;
  checkpoint_path: string;
  device: string;
  cuda_available: boolean;
  aws_mode: boolean;
  practical_rife: string;
}

export interface SatelliteSource {
  id: string;
  label: string;
  default_cadence_min: number;
  fmt: string;
  bucket: string;
}

export interface LocalFolder {
  path: string;
  label: string;
  n_scans: number;
}

export interface ScanRow {
  file: string;
  scan_utc: string;
  uri: string;
}

export interface GapInfo {
  cadence_min: number;
  skip: number;
  actual_step_min: number;
  total_span_min: number;
  n_triplets: number;
  min_files_needed: number;
  n_triplets_listed?: number;
}

export interface TimingRow {
  frame: string;
  time_utc: string;
}

export interface TripletMeta {
  t0: string | null;
  t1_pred: string | null;
  t1_gt: string | null;
  t2: string | null;
  cadence_from: number;
  cadence_to: number;
  satellite: string;
}

export interface Metrics {
  mse: number;
  psnr: number;
  ssim: number;
  fsim: number;
}

export interface TripletLoadResult {
  triplet_index: number;
  n_triplets: number;
  indices: { i0: number; i1: number; i2: number };
  names: { t0: string; t1: string; t2: string };
  gap_info: GapInfo;
  timing: TimingRow[];
  meta: TripletMeta;
  frames: { t0: string; t1_gt: string; t2: string };
  has_ground_truth?: boolean;
}

export interface PredictResult {
  checkpoint: string;
  names: { t0: string; t1: string; t2: string };
  metrics: { rife: Metrics; linear: Metrics } | null;
  has_ground_truth?: boolean;
  upload_id?: string;
  frames: {
    predicted: string;
    linear: string;
    ground_truth: string;
    t0: string;
    t2: string;
  };
}

export interface UploadPredictOptions {
  t0: File;
  t2: File;
  t1?: File | null;
  size: number;
  scale: number;
  gap_min: number;
}

export interface BatchRow {
  i: number;
  file_i0: number;
  file_i1: number;
  file_i2: number;
  gap_t0_t1_min: number | null;
  rife_ssim: number;
  rife_fsim: number;
  rife_psnr: number;
  linear_ssim: number;
  linear_fsim: number;
}

export interface BatchSummary {
  rife_ssim_mean: number;
  linear_ssim_mean: number;
  delta_ssim: number;
  rife_fsim_mean: number;
  n: number;
}

export interface BatchResult {
  rows: BatchRow[];
  summary: BatchSummary;
}

export interface FlowArrow {
  x: number;
  y: number;
  u: number;
  v: number;
}

export interface FlowResult {
  checkpoint: string;
  geo_note: string;
  ablation_rows: Record<string, unknown>[];
  ablation_chart: { method: string; ssim: number; fsim: number }[];
  speed_stats: { mean_ms?: number; p99_ms?: number; max_ms?: number };
  arrows: FlowArrow[];
  n_arrows: number;
  arrow_stats: { n_shown: number; n_grid: number };
  flow_rgb: string;
  overlay_frames: {
    t0?: string;
    t1_gt?: string;
    t2?: string;
    t1_pred?: string;
  };
  gifs: {
    predicted?: string;
    ground_truth?: string;
  };
  frames: {
    linear: string;
    farneback: string;
    flow_only: string;
    full: string;
    ground_truth: string;
    background: string;
  };
}

export interface TrainingStatus {
  checkpoint_exists: boolean;
  checkpoint_path: string;
  stage1_backup_exists: boolean;
  training_mode?: string;
  stage?: string | number;
  full_eval: Record<string, number>;
}

export interface TripletRequest {
  source_id?: string | null;
  day?: string | null;
  hour?: number | null;
  gap_min: number;
  triplet_index: number;
  size: number;
  local_folder?: string | null;
}

export interface PredictRequest extends TripletRequest {
  scale: number;
}

export type FrameDownloadKey = "t0" | "t1_gt" | "t2" | "t1_pred";

export interface FrameDownloadResult {
  filename: string;
  mime: string;
  data_b64: string;
  cached?: boolean;
}

export interface ReportScorecardRow {
  area: string;
  criterion: string;
  status: string;
  note: string;
}

export interface ReportPayload {
  config: Record<string, unknown>;
  triplet?: {
    meta?: TripletMeta;
    timing?: TimingRow[];
    names?: { t0: string; t1: string; t2: string };
  } | null;
  predict?: {
    metrics: { rife: Metrics; linear: Metrics };
    checkpoint: string;
    names: { t0: string; t1: string; t2: string };
  } | null;
  flow?: {
    arrow_stats: FlowResult["arrow_stats"];
    speed_stats: FlowResult["speed_stats"];
    ablation_chart: FlowResult["ablation_chart"];
    geo_note: string;
    n_arrows?: number;
  } | null;
  batch?: BatchResult | null;
  health?: HealthStatus | null;
  training?: TrainingStatus | null;
  scorecard?: ReportScorecardRow[];
  images?: Record<string, string>;
}

export interface ReportDownloadResult {
  filename: string;
  mime: string;
  data_b64: string;
}
