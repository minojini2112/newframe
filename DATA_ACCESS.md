# Satellite data access (ISRO PS12)

All **public** datasets below work with **no AWS account** from your EC2 instance (or locally). Data is **streamed** into memory — not copied to disk.

---

## 1. GOES-19 ABI Channel 13 (primary training — 10 min)

| | |
|---|---|
| **Bucket** | `noaa-goes19` (us-east-1) |
| **Path** | `ABI-L1b-RadF/{year}/{day-of-year}/{hour}/` |
| **Files** | `*M6C13*.nc` |
| **Variable** | `Rad` |
| **Registry** | https://registry.opendata.aws/noaa-goes/ |

```bash
# Example: list C13 files for 2025 day 171 hour 14 UTC
aws s3 ls --no-sign-request s3://noaa-goes19/ABI-L1b-RadF/2025/171/14/ | findstr C13
```

**L2 brightness temperature (Kelvin):** prefix `ABI-L2-CMIPF/`, variable `CMI` — select **GOES-19 L2 CMIPF** in the app.

---

## 2. Himawari-8 / Himawari-9 Band 13 (secondary — 10 min)

| | Himawari-8 | Himawari-9 |
|---|---|---|
| **Bucket** | `noaa-himawari8` | `noaa-himawari9` |
| **Path** | `AHI-L1b-FLDK/{year}/{month}/{day}/{hhmm}/` | same |
| **Files** | `HS_H08_*_B13_FLDK_*.DAT.bz2` | `HS_H09_*_B13_FLDK_*.DAT.bz2` |
| **Registry** | https://registry.opendata.aws/noaa-himawari/ | same |

```bash
aws s3 ls --no-sign-request s3://noaa-himawari8/AHI-L1b-FLDK/2024/06/30/0200/
aws s3 ls --no-sign-request s3://noaa-himawari9/AHI-L1b-FLDK/2024/06/30/0200/
```

**Note:** Himawari L1b is binary HSD, not NetCDF. The app uses **satpy** to read it:

```bash
pip install satpy
```

Select **Himawari-8** or **Himawari-9** in the mission console satellite dropdown.

---

## 3. GK-2A AMI IR105 (secondary — 10 min)

| | |
|---|---|
| **Bucket** | `noaa-gk2a-pds` |
| **Path** | `AMI/L1B/FD/{yyyymm}/{dd}/{hh}/` |
| **Files** | `gk2a_ami_le1b_ir105_fd020ge_*.nc` |
| **Registry** | https://registry.opendata.aws/noaa-gk2a-pds/ |

```bash
aws s3 ls --no-sign-request s3://noaa-gk2a-pds/AMI/L1B/FD/202406/30/00/ | findstr ir105
```

---

## 4. INSAT-3DS / 3DR TIR1 (deployment target — ~30 min)

**INSAT is not on public AWS.** ISRO provides it on your hackathon EC2 instance.

### Option A — Instance mount (recommended on ISRO AWS)

ISRO typically mounts satellite data on the instance. The app **auto-scans** these paths:

- `/data/insat`
- `/data/INSAT-3DS`
- `/mnt/satellite-data`
- `SATELLITE_DATA_MOUNT` (set this if your path is different)

Files: `3SIMG_*.h5` (INSAT-3DS), `3RIMG_*.h5` (INSAT-3DR)  
HDF5 variable: **`IMG_TIR1`** (thermal infrared ~10.8 µm)

```bash
export SATELLITE_DATA_MOUNT=/your/mount/path
export ISRO_PS12_AWS=1
```

Select **INSAT instance data mount** in the app.

### Option B — ISRO private S3 bucket

If ISRO gave you a bucket name on the same AWS account:

```bash
export ISRO_S3_BUCKET=your-bucket-name
export INSAT_S3_PREFIX=path/to/insat/{year}/{month}/{day}/
# Uses instance IAM role — no access keys in code
```

### Option C — MOSDAC portal (off AWS / registration)

For data outside the hackathon instance:

1. Register at https://mosdac.gov.in
2. Order INSAT-3DS L1B products (`3SIMG_L1B_STD`) or browse https://mosdac.gov.in/internal/catalog-insat3s
3. API docs: https://mosdac.gov.in/user-manual-mosdac-data-download-api

**Workflow:** Train on **GOES-19** (10 min) → validate → apply model to **INSAT** (30 → 15 min interpolation).

---

## App usage

1. Set `ISRO_PS12_AWS=1` on the EC2 instance
2. Run `streamlit run app.py`
3. Choose satellite → **UTC date** → **List scans** → pick triplet gap (10 / 20 / 30 min)
4. No manual download step required for public buckets
