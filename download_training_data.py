"""
Download GOES-19 C13 NetCDF for physics fine-tuning (local cache).

  Calm day  — 2026 DOY 150 (May 30), stable weather, 10 min cadence
  Storm day — 2026 DOY 168 (Arthur), active convection

Usage:
    python download_training_data.py
    python download_training_data.py --calm-only --hours 8 9 10 11 12 13 14 15 16 17 18
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import requests
import s3fs
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent

CALM_YEAR, CALM_DOY = 2026, 150
STORM_YEAR, STORM_DOY = 2026, 168

DEFAULT_CALM_DIR = REPO / "ISRO2026" / "goes19_c13"
DEFAULT_STORM_DIR = REPO / "goes19_c13_arthur"
DEFAULT_CALM_HOURS = list(range(8, 19))
DEFAULT_STORM_HOURS = list(range(12, 23))


def download_file(s3_key: str, local_path: Path) -> None:
    url = f"https://noaa-goes19.s3.amazonaws.com/{s3_key.removeprefix('noaa-goes19/')}"
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        tmp = str(local_path) + ".part"
        with open(tmp, "wb") as f, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=local_path.name[:40],
            leave=False,
        ) as bar:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
        os.replace(tmp, local_path)


def fetch_hours(
    fs: s3fs.S3FileSystem,
    year: int,
    doy: int,
    hours: list[int],
    save_dir: Path,
    max_new: int | None = None,
) -> tuple[int, int]:
    save_dir.mkdir(parents=True, exist_ok=True)
    downloaded = skipped = 0
    for hour in hours:
        prefix = f"noaa-goes19/ABI-L1b-RadF/{year}/{doy:03d}/{hour:02d}/"
        try:
            files = fs.ls(prefix)
        except FileNotFoundError:
            print(f"  hour {hour:02d}: no data")
            continue
        c13 = sorted(f for f in files if "M6C13" in f or "C13" in f)
        print(f"  hour {hour:02d}: {len(c13)} C13 files")
        for key in c13:
            if max_new is not None and downloaded >= max_new:
                return downloaded, skipped
            local = save_dir / os.path.basename(key)
            if local.exists():
                skipped += 1
                continue
            download_file(key, local)
            downloaded += 1
    return downloaded, skipped


def main() -> None:
    p = argparse.ArgumentParser(description="Download GOES C13 for fillframe physics training")
    p.add_argument("--calm-dir", type=Path, default=DEFAULT_CALM_DIR)
    p.add_argument("--storm-dir", type=Path, default=DEFAULT_STORM_DIR)
    p.add_argument("--calm-only", action="store_true")
    p.add_argument("--storm-only", action="store_true")
    p.add_argument("--hours", type=int, nargs="+", default=None, help="Calm-day UTC hours")
    p.add_argument("--storm-hours", type=int, nargs="+", default=DEFAULT_STORM_HOURS)
    p.add_argument("--max-new", type=int, default=None)
    args = p.parse_args()

    fs = s3fs.S3FileSystem(anon=True, client_kwargs={"region_name": "us-east-1"})
    calm_hours = args.hours or DEFAULT_CALM_HOURS

    if not args.storm_only:
        print(f"Calm GOES C13 -> {args.calm_dir} (DOY {CALM_DOY})")
        d, s = fetch_hours(fs, CALM_YEAR, CALM_DOY, calm_hours, args.calm_dir, args.max_new)
        print(f"  calm: new={d}, skipped={s}, total={len(list(args.calm_dir.glob('*.nc')))}")

    if not args.calm_only:
        print(f"Storm GOES C13 -> {args.storm_dir} (DOY {STORM_DOY}, Arthur)")
        d, s = fetch_hours(fs, STORM_YEAR, STORM_DOY, args.storm_hours, args.storm_dir, args.max_new)
        print(f"  storm: new={d}, skipped={s}, total={len(list(args.storm_dir.glob('*.nc')))}")


if __name__ == "__main__":
    main()
