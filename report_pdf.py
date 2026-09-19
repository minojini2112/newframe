"""Generate ISRO PS12 evaluation PDF reports from dashboard payload."""

from __future__ import annotations

import base64
import io
from datetime import datetime
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle
from PIL import Image

# US Letter — fixed size for every page (no bbox_inches="tight")
PAGE_W, PAGE_H = 8.5, 11.0

COLORS = {
    "header": "#1a2332",
    "accent": "#d4845c",
    "text": "#1c2430",
    "muted": "#5c6675",
    "line": "#dde2e8",
    "row_alt": "#f5f7fa",
    "good": "#2d8a5e",
}


def _checkpoint_label(path: str | None) -> str:
    if not path:
        return "—"
    normalized = path.replace("\\", "/")
    marker = "checkpoints/"
    idx = normalized.find(marker)
    if idx >= 0:
        return normalized[idx + len(marker) :]
    parts = [p for p in normalized.split("/") if p]
    if len(parts) >= 2:
        return f"{parts[-2]}/{parts[-1]}"
    return parts[-1] if parts else "Fine-tuned RIFE weights"


def _b64_to_img(b64: str | None) -> np.ndarray | None:
    if not b64:
        return None
    return np.array(Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB"))


def _fmt_ts(iso: str | None) -> str:
    if not iso:
        return "—"
    try:
        return iso.replace("T", " ").replace("Z", " UTC")[:22]
    except Exception:
        return str(iso)


class _ReportBuilder:
    """Consistent letter-size pages with shared header/footer styling."""

    def __init__(self) -> None:
        self._page = 0

    def _new_fig(self) -> plt.Figure:
        fig = plt.figure(figsize=(PAGE_W, PAGE_H), facecolor="white")
        self._page += 1
        return fig

    def _draw_header(self, fig: plt.Figure, title: str, subtitle: str = "") -> None:
        ax = fig.add_axes([0, 0.935, 1, 0.065], facecolor=COLORS["header"])
        ax.axis("off")
        ax.add_patch(Rectangle((0, 0), 0.012, 1, transform=ax.transAxes, facecolor=COLORS["accent"], clip_on=False))
        ax.text(0.04, 0.52, title, color="white", fontsize=13, fontweight="bold", va="center", transform=ax.transAxes)
        if subtitle:
            ax.text(0.96, 0.52, subtitle, color="#b8c0cc", fontsize=8, va="center", ha="right", transform=ax.transAxes)

    def _draw_footer(self, fig: plt.Figure, generated: str) -> None:
        fig.text(
            0.5,
            0.028,
            f"FillFrame · ISRO PS12 Frame Interpolation  ·  Generated {generated}  ·  Page {self._page}",
            ha="center",
            va="center",
            fontsize=7,
            color=COLORS["muted"],
        )

    def save(self, pdf: PdfPages, fig: plt.Figure, generated: str) -> None:
        self._draw_footer(fig, generated)
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

    def section_title(self, fig: plt.Figure, y: float, text: str) -> None:
        fig.text(0.075, y, text.upper(), fontsize=9, fontweight="bold", color=COLORS["accent"], ha="left")

    def kv_block(self, fig: plt.Figure, top: float, rows: list[tuple[str, str]], line_h: float = 0.028) -> float:
        """Render key-value rows; return y below last row."""
        y = top
        for i, (key, val) in enumerate(rows):
            if y < 0.12:
                break
            if i % 2 == 0:
                fig.add_artist(
                    Rectangle(
                        (0.075, y - line_h + 0.002),
                        0.85,
                        line_h - 0.004,
                        transform=fig.transFigure,
                        facecolor=COLORS["row_alt"],
                        edgecolor="none",
                        zorder=0,
                    )
                )
            fig.text(0.085, y - line_h * 0.35, key, fontsize=8.5, color=COLORS["muted"], ha="left", va="center", zorder=1)
            fig.text(
                0.38,
                y - line_h * 0.35,
                val,
                fontsize=8.5,
                color=COLORS["text"],
                ha="left",
                va="center",
                family="monospace",
                zorder=1,
            )
            y -= line_h
        return y

    def cover_page(self, pdf: PdfPages, generated: str, lines_sections: list[tuple[str, list[tuple[str, str]]]]) -> None:
        fig = self._new_fig()
        self._draw_header(fig, "Frame Interpolation Report", "ISRO PS12")

        fig.text(0.075, 0.88, "Fill in the Frames Seamlessly", fontsize=20, fontweight="bold", color=COLORS["text"])
        fig.text(
            0.075,
            0.845,
            "AI/ML optical-flow frame interpolation for geostationary satellite thermal infrared imagery",
            fontsize=9,
            color=COLORS["muted"],
        )

        y = 0.80
        for section, rows in lines_sections:
            self.section_title(fig, y, section)
            y -= 0.035
            y = self.kv_block(fig, y, rows)
            y -= 0.02

        fig.text(
            0.075,
            0.11,
            "Objective: estimate motion between consecutive scans, synthesize intermediate frames with RIFE, "
            "and validate against higher-cadence ground truth using SSIM, PSNR, MSE, and FSIM.",
            fontsize=8,
            color=COLORS["muted"],
            wrap=True,
        )
        self.save(pdf, fig, generated)

    def text_page(self, pdf: PdfPages, generated: str, title: str, sections: list[tuple[str, list[str]]]) -> None:
        fig = self._new_fig()
        self._draw_header(fig, title)
        y = 0.88
        for section, lines in sections:
            if y < 0.14:
                break
            self.section_title(fig, y, section)
            y -= 0.04
            for line in lines:
                if y < 0.12:
                    break
                fig.text(0.085, y, line, fontsize=8.5, color=COLORS["text"], ha="left", va="top")
                y -= 0.028
            y -= 0.02
        self.save(pdf, fig, generated)

    def metrics_page(self, pdf: PdfPages, generated: str, predict: dict[str, Any]) -> None:
        metrics = predict.get("metrics", {})
        rife = metrics.get("rife", {})
        linear = metrics.get("linear", {})

        fig = self._new_fig()
        self._draw_header(fig, "Interpolation Metrics", "RIFE vs ground-truth t1")

        ax = fig.add_axes([0.075, 0.42, 0.85, 0.42])
        ax.axis("off")
        ax.set_title("Frame quality vs ground-truth midpoint (t1)", fontsize=10, fontweight="bold", pad=12, color=COLORS["text"])

        rows = [
            ["Metric", "RIFE (predicted)", "Linear baseline", "Δ (RIFE − linear)"],
            ["SSIM", f"{rife.get('ssim', 0):.6f}", f"{linear.get('ssim', 0):.6f}", f"{rife.get('ssim', 0) - linear.get('ssim', 0):+.6f}"],
            ["PSNR (dB)", f"{rife.get('psnr', 0):.2f}", f"{linear.get('psnr', 0):.2f}", f"{rife.get('psnr', 0) - linear.get('psnr', 0):+.2f}"],
            ["MSE", f"{rife.get('mse', 0):.6f}", f"{linear.get('mse', 0):.6f}", f"{rife.get('mse', 0) - linear.get('mse', 0):+.6f}"],
            ["FSIM", f"{rife.get('fsim', 0):.6f}", f"{linear.get('fsim', 0):.6f}", f"{rife.get('fsim', 0) - linear.get('fsim', 0):+.6f}"],
        ]
        table = ax.table(cellText=rows, loc="center", cellLoc="center", edges="horizontal")
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2.0)
        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor(COLORS["line"])
            if row == 0:
                cell.set_facecolor(COLORS["header"])
                cell.set_text_props(color="white", fontweight="bold")
            elif row % 2 == 0:
                cell.set_facecolor(COLORS["row_alt"])
            if col == 3 and row > 0:
                val = rows[row][3]
                if val.startswith("+"):
                    cell.set_text_props(color=COLORS["good"])

        # Summary callout
        delta_ssim = rife.get("ssim", 0) - linear.get("ssim", 0)
        fig.text(
            0.075,
            0.36,
            f"RIFE improves SSIM by {delta_ssim:+.4f} over linear blending on this triplet.",
            fontsize=9,
            color=COLORS["text"],
        )
        self.save(pdf, fig, generated)

    def ablation_page(self, pdf: PdfPages, generated: str, ablation: list[dict[str, Any]]) -> None:
        if not ablation:
            return
        fig = self._new_fig()
        self._draw_header(fig, "Method Ablation", "Optical-flow interpolation variants")

        ax = fig.add_axes([0.075, 0.25, 0.85, 0.58])
        ax.axis("off")
        rows = [["Method", "SSIM", "FSIM"]]
        for row in ablation:
            rows.append([
                row.get("method", "").replace("\n", " "),
                f"{row.get('ssim', 0):.4f}" if row.get("ssim") is not None else "—",
                f"{row.get('fsim', 0):.4f}" if row.get("fsim") is not None else "—",
            ])
        table = ax.table(cellText=rows, loc="center", cellLoc="center", edges="horizontal")
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.7)
        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor(COLORS["line"])
            if row == 0:
                cell.set_facecolor(COLORS["header"])
                cell.set_text_props(color="white", fontweight="bold")
            elif row % 2 == 0:
                cell.set_facecolor(COLORS["row_alt"])

        self.save(pdf, fig, generated)

    def images_page(self, pdf: PdfPages, generated: str, images: dict[str, str], titles: dict[str, str]) -> None:
        keys = [k for k in titles if images.get(k)]
        if not keys:
            return

        # Primary frames (2×2)
        primary = [k for k in ("t0", "ground_truth", "predicted", "t2") if k in keys]
        if primary:
            fig = self._new_fig()
            self._draw_header(fig, "Satellite Frames", "Ground truth vs RIFE predicted")
            n = len(primary)
            cols = 2
            rows_n = (n + 1) // 2
            for i, key in enumerate(primary):
                r, c = divmod(i, cols)
                left = 0.075 + c * 0.44
                bottom = 0.52 - r * 0.38
                ax = fig.add_axes([left, bottom, 0.40, 0.34])
                img = _b64_to_img(images[key])
                if img is not None:
                    ax.imshow(img, aspect="auto")
                ax.set_title(titles[key], fontsize=9, fontweight="bold", color=COLORS["text"], pad=6)
                ax.axis("off")
            self.save(pdf, fig, generated)

        # AMV overlay on separate page if present
        if images.get("overlay_pred"):
            fig = self._new_fig()
            self._draw_header(fig, "Motion Vectors", "Predicted t1 with AMV overlay")
            ax = fig.add_axes([0.1, 0.18, 0.8, 0.68])
            img = _b64_to_img(images["overlay_pred"])
            if img is not None:
                ax.imshow(img, aspect="auto")
            ax.set_title(titles.get("overlay_pred", "AMV overlay"), fontsize=10, fontweight="bold", pad=8)
            ax.axis("off")
            self.save(pdf, fig, generated)

    def batch_page(self, pdf: PdfPages, generated: str, batch: dict[str, Any]) -> None:
        s = batch.get("summary") or {}
        if not s:
            return
        fig = self._new_fig()
        self._draw_header(fig, "Batch Validation", "Aggregated metrics across triplets")
        rows = [
            ("Triplets evaluated", str(s.get("n", "—"))),
            ("Mean RIFE SSIM", f"{s.get('rife_ssim_mean', 0):.6f}"),
            ("Mean linear SSIM", f"{s.get('linear_ssim_mean', 0):.6f}"),
            ("Mean Δ SSIM (RIFE − linear)", f"+{s.get('delta_ssim', 0):.6f}"),
            ("Mean RIFE FSIM", f"{s.get('rife_fsim_mean', 0):.6f}"),
        ]
        self.section_title(fig, 0.82, "Summary statistics")
        self.kv_block(fig, 0.78, rows, line_h=0.055)
        self.save(pdf, fig, generated)


def generate_report_pdf(payload: dict[str, Any]) -> bytes:
    """Build multi-page letter-size PDF for PS12 evaluation deliverable."""
    buf = io.BytesIO()
    generated = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    config = payload.get("config", {})
    triplet = payload.get("triplet") or {}
    predict = payload.get("predict") or {}
    flow = payload.get("flow") or {}
    batch = payload.get("batch") or {}
    health = payload.get("health") or {}
    training = payload.get("training") or {}
    meta = triplet.get("meta") or {}

    rb = _ReportBuilder()

    with PdfPages(buf) as pdf:
        rb.cover_page(
            pdf,
            generated,
            [
                (
                    "Dataset & configuration",
                    [
                        ("Satellite", str(config.get("source_label", config.get("source_id", "—")))),
                        ("UTC date", str(config.get("day", "—"))),
                        ("Gap step", f"{config.get('gap_min', '—')} min"),
                        ("Image size", f"{config.get('size', '—')} px"),
                        ("RIFE scale", str(config.get("scale", "—"))),
                        ("Triplet index", str(config.get("triplet_index", "—"))),
                    ],
                ),
                (
                    "Temporal resolution",
                    [
                        ("Input cadence", f"{meta.get('cadence_from', '—')} min"),
                        ("Target step", f"{meta.get('cadence_to', '—')} min"),
                        ("t0 (input)", _fmt_ts(meta.get("t0"))),
                        ("t1 predicted", _fmt_ts(meta.get("t1_pred"))),
                        ("t1 ground truth", _fmt_ts(meta.get("t1_gt"))),
                        ("t2 (input)", _fmt_ts(meta.get("t2"))),
                    ],
                ),
                (
                    "Model",
                    [
                        ("Checkpoint", _checkpoint_label(predict.get("checkpoint") or health.get("checkpoint_path"))),
                        ("Device", str(health.get("device", "—")).upper()),
                        ("Training mode", str(training.get("training_mode", "—"))),
                    ],
                ),
            ],
        )

        if predict.get("metrics"):
            rb.metrics_page(pdf, generated, predict)

        ablation = flow.get("ablation_chart") or []
        if ablation:
            rb.ablation_page(pdf, generated, ablation)

        images = payload.get("images") or {}
        rb.images_page(
            pdf,
            generated,
            images,
            {
                "t0": "Input t0",
                "ground_truth": "Ground-truth t1",
                "predicted": "RIFE predicted t1",
                "t2": "Input t2",
                "overlay_pred": "Predicted t1 + AMV arrows",
            },
        )

        if flow:
            speed = flow.get("speed_stats") or {}
            flow_rows = [
                ("Vectors shown", str(flow.get("arrow_stats", {}).get("n_shown", "—"))),
                ("Grid cells", str(flow.get("arrow_stats", {}).get("n_grid", "—"))),
                ("Geo projection", str(flow.get("geo_note", "—"))),
            ]
            if speed.get("mean_ms") is not None:
                flow_rows.extend([
                    ("Mean AMV speed", f"{speed['mean_ms']:.2f} m/s"),
                    ("P99 speed", f"{speed.get('p99_ms', 0):.2f} m/s"),
                    ("Max speed", f"{speed.get('max_ms', 0):.2f} m/s"),
                ])
            rb.text_page(
                pdf,
                generated,
                "Motion Vectors & Visualization",
                [
                    ("Optical flow / AMV", [f"{k}: {v}" for k, v in flow_rows]),
                    (
                        "Dashboard deliverables",
                        [
                            "Ground-truth and predicted timelapse GIFs with AMV overlays",
                            "Per-frame .nc export for t0, t1_gt, t1_pred, and t2",
                            "Side-by-side predicted vs ground-truth comparison in the web dashboard",
                        ],
                    ),
                ],
            )

        if batch.get("summary"):
            rb.batch_page(pdf, generated, batch)

    return buf.getvalue()
