# FillFrame

### Physics-Informed Deep Learning for Seamless Temporal Super-Resolution of Geostationary Satellite Imagery

**ISRO Problem Statement 12 — *Fill in the Frames Seamlessly***

> We do not interpolate pixels. We propagate radiance through a flow field constrained by atmospheric physics — so every synthetic frame is a physically admissible state of the cloud field, not a statistical hallucination.

## Problem

Geostationary satellites observe Earth on fixed schedules — **INSAT-3DS/3DR every ~30 minutes**, **GOES-19 and Himawari every ~10 minutes** — leaving blind intervals where fires spread, cyclones intensify, and convective systems reorganize faster than the sensor cadence.

Classical optical-flow interpolation blurs cloud edges, smears divergent convection, and violates thermodynamic consistency.

## Approach

**FillFrame** couples **RIFE** (Real-Time Intermediate Flow Estimation) with a **physics-informed loss stack** (atmospheric continuity, incompressibility, brightness-temperature conservation, vorticity dynamics).

- Train on high-cadence **GOES-19 ABI Channel 13** (~10.3 µm)
- Deploy on **INSAT-3DS TIR1** (~10.8 µm)
- Target effective cadence **30 → 15 → 7.5 minutes** via recursive midpoint synthesis
- Validate with **SSIM, MSE, PSNR, FSIM**

## Satellite targets

| Satellite | TIR band | Native cadence | After FillFrame |
|-----------|----------|----------------|-----------------|
| INSAT-3DS / 3DR | TIR1 (~10.8 µm) | ~30 min | 15 min · 7.5 min |
| GOES-19 ABI | Ch.13 (~10.3 µm) | ~10 min | 5 min |
| Himawari-8/9 AHI | Band 13 (~10.4 µm) | ~10 min | 5 min |
| GK-2A AMI | IR105 (~10.5 µm) | ~10 min | 5 min |

## Multi-satellite sources

Catalog support lives in `s3_catalog.py` and streams radiance **without copying datasets to disk**:

| Source ID | Platform | Access |
|-----------|----------|--------|
| `goes19_c13` | GOES-19 ABI Ch.13 radiance | Public NOAA S3 `noaa-goes19` |
| `goes19_c13_bt` | GOES-19 ABI Ch.13 brightness temp (L2) | Public NOAA S3 |
| `himawari8_b13` | Himawari-8 AHI Band 13 (HSD) | Public NOAA S3 `noaa-himawari8` |
| `himawari9_b13` | Himawari-9 AHI Band 13 (HSD) | Public NOAA S3 `noaa-himawari9` |
| `gk2a_ir105` | GK-2A AMI IR105 | Public NOAA S3 `noaa-gk2a-pds` |
| `insat3ds_tir1` / `insat_mount` | INSAT-3DS/3DR TIR1 | ISRO private bucket or EC2 data mount |

Train / validate on high-cadence GOES or Himawari; deploy the same model on INSAT for the PS12 30→15→7.5 min target.

## Status

Repository scaffold in progress. Core modules (data catalog, physics losses, Streamlit app, API, Next.js dashboard) will land in subsequent commits.
