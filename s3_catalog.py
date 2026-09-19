"""
ISRO PS12 — satellite catalog (stream from S3 / instance mount, no local copy).

Public buckets (anonymous, us-east-1):
  GOES-19 C13  → s3://noaa-goes19/ABI-L1b-RadF/{year}/{doy}/{hour}/
  Himawari-8 B13 → s3://noaa-himawari8/AHI-L1b-FLDK/{year}/{mm}/{dd}/{hhmm}/
  Himawari-9 B13 → s3://noaa-himawari9/AHI-L1b-FLDK/{year}/{mm}/{dd}/{hhmm}/
  GK-2A IR105  → s3://noaa-gk2a-pds/AMI/L1B/FD/{yyyymm}/{dd}/{hh}/

INSAT-3DS TIR1 is NOT on public S3 — use ISRO instance mount (auto-scanned) or
set ISRO_S3_BUCKET to the bucket ISRO provides on your hackathon EC2.
MOSDAC portal: https://mosdac.gov.in/insat-3ds
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Literal

import numpy as np

Format = Literal["netcdf", "hdf5", "himawari_hsd", "mount"]

# Common mount roots on ISRO / hackathon EC2 (scanned for INSAT .h5)
DEFAULT_MOUNT_ROOTS = [
    "/data",
    "/data/insat",
    "/data/INSAT-3DS",
    "/data/insat3ds",
    "/mnt/satellite-data",
    "/opt/isro-data",
    "/home/ubuntu/data",
]


@dataclass(frozen=True)
class SatelliteSource:
    id: str
    label: str
    bucket: str
    prefix_template: str
    file_filter: str
    rad_var: str
    anonymous: bool
    default_cadence_min: float
    time_parser: str  # goes | himawari_hsd | gk2a | insat | filename
    fmt: Format = "netcdf"
    help_url: str = ""
    extra_filter: str = ""  # e.g. Himawari segment _S0110.


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def mount_roots() -> list[str]:
    roots: list[str] = []
    extra = _env("SATELLITE_DATA_MOUNT")
    if extra:
        roots.append(extra)
    local_insat = Path(__file__).resolve().parent / "data" / "insat"
    if local_insat.is_dir():
        roots.append(str(local_insat))
    roots.extend(DEFAULT_MOUNT_ROOTS)
    seen: set[str] = set()
    out: list[str] = []
    for r in roots:
        if r and r not in seen and os.path.isdir(r):
            seen.add(r)
            out.append(r)
    return out


def satellite_sources() -> dict[str, SatelliteSource]:
    sources: dict[str, SatelliteSource] = {
        "goes19_c13": SatelliteSource(
            id="goes19_c13",
            label="GOES-19 ABI Ch.13 (10.3 µm) — primary training",
            bucket=_env("GOES_S3_BUCKET", "noaa-goes19"),
            prefix_template=_env("GOES_S3_PREFIX", "ABI-L1b-RadF/{year}/{doy:03d}/"),
            file_filter=_env("GOES_FILE_FILTER", "M6C13"),
            rad_var=_env("GOES_RAD_VAR", "Rad"),
            anonymous=True,
            default_cadence_min=10.0,
            time_parser="goes",
            fmt="netcdf",
            help_url="https://registry.opendata.aws/noaa-goes/",
        ),
        "goes19_c13_bt": SatelliteSource(
            id="goes19_c13_bt",
            label="GOES-19 ABI Ch.13 brightness temp (L2 CMIPF, Kelvin)",
            bucket=_env("GOES_S3_BUCKET", "noaa-goes19"),
            prefix_template=_env("GOES_L2_PREFIX", "ABI-L2-CMIPF/{year}/{doy:03d}/"),
            file_filter="M6C13",
            rad_var="CMI",
            anonymous=True,
            default_cadence_min=10.0,
            time_parser="goes",
            fmt="netcdf",
            help_url="https://registry.opendata.aws/noaa-goes/",
        ),
        "himawari8_b13": SatelliteSource(
            id="himawari8_b13",
            label="Himawari-8 AHI Band 13 (10.4 µm) — archive 2015–2022",
            bucket=_env("HIMAWARI8_S3_BUCKET", "noaa-himawari8"),
            prefix_template="AHI-L1b-FLDK/{year}/{month:02d}/{day:02d}/",
            file_filter="_B13_FLDK_",
            extra_filter="",
            rad_var="B13",
            anonymous=True,
            default_cadence_min=10.0,
            time_parser="himawari_hsd",
            fmt="himawari_hsd",
            help_url="https://registry.opendata.aws/noaa-himawari/",
        ),
        "himawari9_b13": SatelliteSource(
            id="himawari9_b13",
            label="Himawari-9 AHI Band 13 (10.4 µm) — 10 min",
            bucket=_env("HIMAWARI_S3_BUCKET", "noaa-himawari9"),
            prefix_template="AHI-L1b-FLDK/{year}/{month:02d}/{day:02d}/",
            file_filter="_B13_FLDK_",
            extra_filter="",
            rad_var="B13",
            anonymous=True,
            default_cadence_min=10.0,
            time_parser="himawari_hsd",
            fmt="himawari_hsd",
            help_url="https://registry.opendata.aws/noaa-himawari/",
        ),
        "gk2a_ir105": SatelliteSource(
            id="gk2a_ir105",
            label="GK-2A AMI IR105 (10.5 µm) — 10 min",
            bucket="noaa-gk2a-pds",
            prefix_template="AMI/L1B/FD/{year}{month:02d}/{day:02d}/",
            file_filter="ir105",
            rad_var="image_pixel_values",
            anonymous=True,
            default_cadence_min=10.0,
            time_parser="gk2a",
            fmt="netcdf",
            help_url="https://registry.opendata.aws/noaa-gk2a-pds/",
        ),
    }

    # INSAT: ISRO-provided private bucket OR auto-scan mounted data on EC2
    insat_bucket = _env("INSAT_S3_BUCKET") or _env("ISRO_S3_BUCKET")
    if insat_bucket:
        sources["insat3ds_tir1"] = SatelliteSource(
            id="insat3ds_tir1",
            label="INSAT-3DS/3DR TIR1 — ISRO bucket (deployment target)",
            bucket=insat_bucket,
            prefix_template=_env(
                "INSAT_S3_PREFIX",
                "INSAT-3DS/L1B/{year}/{month:02d}/{day:02d}/",
            ),
            file_filter=_env("INSAT_FILE_FILTER", ".h5"),
            rad_var=_env("INSAT_RAD_VAR", "IMG_TIR1"),
            anonymous=_env("INSAT_S3_ANONYMOUS", "0") == "1",
            default_cadence_min=30.0,
            time_parser="insat",
            fmt="hdf5",
            help_url="https://mosdac.gov.in/insat-3ds",
        )

    if mount_roots():
        sources["insat_mount"] = SatelliteSource(
            id="insat_mount",
            label="INSAT-3DS/3DR TIR1 — instance data mount (ISRO EC2)",
            bucket="",
            prefix_template="",
            file_filter=".h5",
            rad_var="IMG_TIR1",
            anonymous=True,
            default_cadence_min=30.0,
            time_parser="insat",
            fmt="mount",
            help_url="https://mosdac.gov.in/insat-3ds",
        )

    return sources


def source_help_text(source_id: str) -> str:
    src = satellite_sources().get(source_id)
    if not src:
        return ""
    lines = {
        "goes19_c13": (
            "**Public NOAA bucket** `noaa-goes19` · prefix `ABI-L1b-RadF/YYYY/DOY/HH/` · "
            "filter `M6C13` · variable `Rad`. No AWS account needed."
        ),
        "goes19_c13_bt": (
            "**L2 brightness temperature** in Kelvin · bucket `noaa-goes19` · "
            "prefix `ABI-L2-CMIPF/YYYY/DOY/HH/` · variable `CMI`."
        ),
        "himawari8_b13": (
            "**Public NOAA bucket** `noaa-himawari8` · path "
            "`AHI-L1b-FLDK/YYYY/MM/DD/HHMM/` · Band 13 HSD (10 `.DAT.bz2` segments per scan). "
            "Archive **2015–2022 only** (use UTC dates in that range). "
            "Requires `satpy` (`pip install satpy`)."
        ),
        "himawari9_b13": (
            "**Public NOAA bucket** `noaa-himawari9` · path "
            "`AHI-L1b-FLDK/YYYY/MM/DD/HHMM/` · Band 13 HSD (10 `.DAT.bz2` segments per scan). "
            "Requires `satpy` (`pip install satpy`). Archive from Jul 2015."
        ),
        "gk2a_ir105": (
            "**Public NOAA bucket** `noaa-gk2a-pds` · path "
            "`AMI/L1B/FD/YYYYMM/DD/HH/` · files `*ir105*.nc`."
        ),
        "insat3ds_tir1": (
            "**ISRO private bucket** on your EC2 — set `ISRO_S3_BUCKET` and `INSAT_S3_PREFIX`. "
            "HDF5 variable `IMG_TIR1` (~30 min cadence). Train on GOES/Himawari first."
        ),
        "insat_mount": (
            "**INSAT HDF5 on the instance** — scans mount paths like `/data/insat`, "
            "`/mnt/satellite-data`. Set `SATELLITE_DATA_MOUNT` if your path differs. "
            "Not on public AWS; register at [MOSDAC](https://mosdac.gov.in) for off-instance access."
        ),
    }
    text = lines.get(source_id, src.label)
    if src.help_url:
        text += f" · [Docs]({src.help_url})"
    return text


def aws_deploy_mode() -> bool:
    if _env("ISRO_PS12_AWS").lower() in ("1", "true", "yes"):
        return True
    if _env("SATELLITE_DATA_MODE").lower() == "s3":
        return True
    return bool(mount_roots() or _env("ISRO_S3_BUCKET"))


def data_mount_root() -> str | None:
    return _env("SATELLITE_DATA_MOUNT") or None


@dataclass(frozen=True)
class NcRef:
    uri: str
    scan_time: datetime | None = None
    source_id: str = ""
    rad_var: str = ""
    segment_uris: tuple[str, ...] = ()

    @property
    def name(self) -> str:
        fname = self.uri.replace("\\", "/").rsplit("/", 1)[-1]
        if self.segment_uris and len(self.segment_uris) > 1:
            m = re.search(r"(HS_H\d{2}_\d{8}_\d{4}_B13_FLDK)", fname)
            if m:
                return f"{m.group(1)}_FULLDISK_{len(self.segment_uris)}seg"
        return fname

    @property
    def is_s3(self) -> bool:
        return self.uri.startswith("s3://")

    def to_dict(self) -> dict:
        d = {
            "uri": self.uri,
            "name": self.name,
            "scan_time": self.scan_time.isoformat() if self.scan_time else None,
            "source_id": self.source_id,
            "rad_var": self.rad_var,
        }
        if self.segment_uris:
            d["segment_uris"] = list(self.segment_uris)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> NcRef:
        st = d.get("scan_time")
        seg = d.get("segment_uris") or ()
        return cls(
            uri=d["uri"],
            scan_time=datetime.fromisoformat(st) if st else None,
            source_id=d.get("source_id", ""),
            rad_var=d.get("rad_var", ""),
            segment_uris=tuple(seg),
        )


def parse_goes_scan_time(filename: str) -> datetime | None:
    m = re.search(r"_s(\d{4})(\d{3})(\d{2})(\d{2})(\d{2})(\d)_", filename)
    if not m:
        return None
    year, doy, hh, mm, ss, ds = (int(x) for x in m.groups())
    base = datetime(year, 1, 1)
    return base + timedelta(
        days=doy - 1, hours=hh, minutes=mm, seconds=ss, milliseconds=ds * 100,
    )


def parse_himawari_hsd_time(name: str, key: str = "") -> datetime | None:
    m = re.search(r"HS_H\d{2}_(\d{4})(\d{2})(\d{2})_(\d{4})_B13", name)
    if m:
        y, mo, d, hhmm = m.groups()
        return datetime(int(y), int(mo), int(d), int(hhmm[:2]), int(hhmm[2:]))
    # path …/2024/06/30/0200/…
    m2 = re.search(r"/(\d{4})/(\d{2})/(\d{2})/(\d{4})/", key.replace("\\", "/"))
    if m2:
        y, mo, d, hhmm = m2.groups()
        return datetime(int(y), int(mo), int(d), int(hhmm[:2]), int(hhmm[2:]))
    return None


def parse_gk2a_time(name: str) -> datetime | None:
    m = re.search(r"_(\d{12})\.nc", name)
    if not m:
        return None
    s = m.group(1)
    return datetime(int(s[:4]), int(s[4:6]), int(s[6:8]), int(s[8:10]), int(s[10:12]))


def parse_insat_time(name: str) -> datetime | None:
    m = re.search(r"(\d{2})([A-Z]{3})(\d{4})_(\d{4})", name.upper())
    if not m:
        return None
    dd, mon, year, hhmm = m.groups()
    months = {
        "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
        "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
    }
    mo = months.get(mon)
    if not mo:
        return None
    return datetime(int(year), mo, int(dd), int(hhmm[:2]), int(hhmm[2:]))


def _parse_scan_time(name: str, parser: str, key: str = "") -> datetime | None:
    if parser == "goes":
        return parse_goes_scan_time(name)
    if parser == "himawari_hsd":
        return parse_himawari_hsd_time(name, key)
    if parser == "gk2a":
        return parse_gk2a_time(name)
    if parser == "insat":
        return parse_insat_time(name)
    m = re.search(r"(\d{4})(\d{2})(\d{2})[_-]?(\d{2})(\d{2})", name)
    if m:
        y, mo, d, h, mi = (int(x) for x in m.groups())
        try:
            return datetime(y, mo, d, h, mi)
        except ValueError:
            pass
    return None


def _format_prefix(template: str, day: date, hour: int | None = None) -> str:
    fmt = dict(
        year=day.year,
        doy=int(day.strftime("%j")),
        month=day.month,
        day=day.day,
        hour=hour if hour is not None else 0,
    )
    prefix = template.format(**fmt)
    return prefix if prefix.endswith("/") else prefix + "/"


def _s3_client(anonymous: bool):
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    if anonymous:
        return boto3.client("s3", config=Config(signature_version=UNSIGNED))
    return boto3.client("s3")


def _mounted_path(bucket: str, key: str) -> str | None:
    for root in mount_roots():
        for candidate in (
            os.path.join(root, bucket, key) if bucket else "",
            os.path.join(root, key),
            os.path.join(root, os.path.basename(key)),
        ):
            if candidate and os.path.isfile(candidate):
                return candidate
    return None


def resolve_read_uri(bucket: str, key: str) -> str:
    local = _mounted_path(bucket, key)
    if local:
        return local
    return f"s3://{bucket}/{key}"


def _s3_anonymous(uri: str, source_id: str = "") -> bool:
    """Public NOAA buckets need unsigned S3; INSAT private bucket uses instance creds."""
    if not uri.startswith("s3://"):
        return True
    src = satellite_sources().get(source_id)
    if src is not None:
        return src.anonymous
    bucket = uri[5:].split("/", 1)[0]
    return bucket.startswith("noaa-")


def _s3_storage_options(uri: str, source_id: str = "") -> dict:
    if not uri.startswith("s3://"):
        return {}
    return {
        "anon": _s3_anonymous(uri, source_id),
        "client_kwargs": {"region_name": "us-east-1"},
    }


def _key_ok(key: str, file_filter: str, extra_filter: str) -> bool:
    low = key.lower()
    if not (
        low.endswith(".nc")
        or low.endswith(".h5")
        or low.endswith(".hdf")
        or low.endswith(".dat.bz2")
    ):
        return False
    if file_filter.startswith(".") and low.endswith(file_filter):
        return True
    if file_filter and file_filter not in key:
        return False
    if extra_filter and extra_filter not in key:
        return False
    return True


def list_s3_keys(
    bucket: str, prefix: str, file_filter: str, anonymous: bool, extra_filter: str = "",
) -> list[str]:
    client = _s3_client(anonymous)
    keys: list[str] = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if _key_ok(key, file_filter, extra_filter):
                keys.append(key)
    return sorted(keys)


def _list_insat_mount(day: date) -> list[NcRef]:
    patterns = ("3SIMG*.h5", "3RIMG*.h5", "3DIMG*.h5", "*TIR1*.h5", "*_L1B_*.h5")
    refs: list[NcRef] = []
    seen: set[str] = set()
    for root in mount_roots():
        base = Path(root)
        for pat in patterns:
            for f in base.rglob(pat):
                if not f.is_file():
                    continue
                p = str(f.resolve())
                if p in seen:
                    continue
                st = parse_insat_time(f.name)
                if st and st.date() != day:
                    continue
                seen.add(p)
                refs.append(
                    NcRef(
                        uri=p,
                        scan_time=st,
                        source_id="insat_mount",
                        rad_var="IMG_TIR1",
                    )
                )
    refs.sort(key=lambda r: (r.scan_time or datetime.min, r.name))
    return refs


def _himawari_scan_group_key(key: str) -> str:
    """Group the 10 HSD segment files that form one full-disk scan."""
    m = re.search(r"/(\d{4})/(\d{2})/(\d{2})/(\d{4})/", key.replace("\\", "/"))
    if m:
        return "/".join(m.groups())
    m2 = re.search(r"HS_H\d{2}_(\d{8}_\d{4})_B13", key)
    return m2.group(1) if m2 else key


def _list_himawari_scan_refs(
    source_id: str,
    bucket: str,
    keys: list[str],
    rad_var: str,
    time_parser: str,
) -> list[NcRef]:
    groups: dict[str, list[str]] = {}
    for key in keys:
        gk = _himawari_scan_group_key(key)
        groups.setdefault(gk, []).append(key)

    refs: list[NcRef] = []
    for gk in sorted(groups):
        seg_keys = sorted(groups[gk])
        first_key = seg_keys[0]
        segment_uris = tuple(resolve_read_uri(bucket, k) for k in seg_keys)
        refs.append(
            NcRef(
                uri=segment_uris[0],
                scan_time=_parse_scan_time(
                    first_key.rsplit("/", 1)[-1], time_parser, first_key,
                ),
                source_id=source_id,
                rad_var=rad_var,
                segment_uris=segment_uris,
            )
        )
    refs.sort(key=lambda r: (r.scan_time or datetime.min, r.name))
    return refs


def _stage_hsd_segments(
    uris: list[str],
    source_id: str = "",
) -> tuple[list[str], object | None]:
    """Download HSD segment URIs to local temp files for satpy."""
    import os
    import tempfile

    local_paths: list[str] = []
    tmp_dir = None
    s3_paths = [u for u in uris if u.startswith("s3://")]
    if s3_paths:
        import fsspec

        opts = _s3_storage_options(s3_paths[0], source_id)
        fs = fsspec.filesystem("s3", **opts)
        tmp_dir = tempfile.TemporaryDirectory(prefix="fillframe_hsd_")
        for uri in uris:
            if uri.startswith("s3://"):
                basename = uri.rsplit("/", 1)[-1]
                local_path = os.path.join(tmp_dir.name, basename)
                with fs.open(uri, "rb") as src, open(local_path, "wb") as dst:
                    dst.write(src.read())
                local_paths.append(local_path)
            else:
                local_paths.append(uri)
    else:
        local_paths = list(uris)
    return local_paths, tmp_dir


def _resolve_himawari_segment_uris(
    uri: str,
    segment_uris: tuple[str, ...] = (),
    source_id: str = "",
) -> list[str]:
    if segment_uris:
        return list(segment_uris)
    if not uri.endswith(".DAT.bz2"):
        return [uri]

    # Legacy single-segment refs: discover all B13 segments in the same HHMM folder.
    if uri.startswith("s3://"):
        bucket, key = uri[5:].split("/", 1)
        folder_prefix = key.rsplit("/", 1)[0] + "/"
        src = satellite_sources().get(source_id)
        anonymous = src.anonymous if src else True
        keys = list_s3_keys(bucket, folder_prefix, "_B13_FLDK_", anonymous, "")
        if keys:
            return [resolve_read_uri(bucket, k) for k in sorted(keys)]
    return [uri]


def list_scans_for_day(source_id: str, day: date, hour_utc: int | None = None) -> list[NcRef]:
    src = satellite_sources()[source_id]

    if src.fmt == "mount":
        return _list_insat_mount(day)

    prefixes: list[str] = []
    if "{hour" in src.prefix_template and hour_utc is not None:
        prefixes = [_format_prefix(src.prefix_template, day, hour_utc)]
    elif src.id in ("himawari8_b13", "himawari9_b13"):
        prefixes = [_format_prefix(src.prefix_template, day)]
    else:
        prefixes = [_format_prefix(src.prefix_template, day)]

    refs: list[NcRef] = []
    seen: set[str] = set()
    all_keys: list[str] = []
    for prefix in prefixes:
        try:
            keys = list_s3_keys(
                src.bucket, prefix, src.file_filter, src.anonymous, src.extra_filter,
            )
        except Exception:
            continue
        for key in keys:
            if key in seen:
                continue
            seen.add(key)
            all_keys.append(key)

    if src.fmt == "himawari_hsd":
        refs = _list_himawari_scan_refs(
            source_id, src.bucket, all_keys, src.rad_var, src.time_parser,
        )
        if hour_utc is not None:
            refs = [r for r in refs if r.scan_time and r.scan_time.hour == hour_utc]
        return refs

    for key in all_keys:
        name = key.rsplit("/", 1)[-1]
        st = _parse_scan_time(name, src.time_parser, key)
        if hour_utc is not None and st and st.hour != hour_utc:
            continue
        refs.append(
            NcRef(
                uri=resolve_read_uri(src.bucket, key),
                scan_time=st,
                source_id=source_id,
                rad_var=src.rad_var,
            )
        )

    refs.sort(key=lambda r: (r.scan_time or datetime.min, r.name))
    return refs


def _load_himawari_hsd(
    uri: str,
    segment_uris: tuple[str, ...] = (),
    source_id: str = "",
) -> np.ndarray:
    try:
        from satpy import Scene
    except ImportError as exc:
        raise ImportError(
            "Himawari HSD needs satpy: pip install satpy"
        ) from exc

    uris = _resolve_himawari_segment_uris(uri, segment_uris, source_id)
    local_paths, tmp_dir = _stage_hsd_segments(uris, source_id)
    try:
        scene = Scene(filenames=local_paths, reader="ahi_hsd")
        scene.load(["B13"])
        data = scene["B13"].values
        if hasattr(data, "mask"):
            data = np.ma.filled(data, np.nan)
        return np.asarray(data, dtype=np.float64)
    finally:
        if tmp_dir is not None:
            tmp_dir.cleanup()


def _himawari_latlon(
    uri: str,
    segment_uris: tuple[str, ...] = (),
    source_id: str = "",
) -> tuple[np.ndarray, np.ndarray]:
    try:
        from satpy import Scene
    except ImportError as exc:
        raise ImportError(
            "Himawari HSD needs satpy: pip install satpy"
        ) from exc

    uris = _resolve_himawari_segment_uris(uri, segment_uris, source_id)
    local_paths, tmp_dir = _stage_hsd_segments(uris, source_id)
    try:
        scene = Scene(filenames=local_paths, reader="ahi_hsd")
        scene.load(["B13"])
        area = scene["B13"].attrs["area"]
        lons, lats = area.get_lonlats()
        return np.asarray(lats, dtype=np.float64), np.asarray(lons, dtype=np.float64)
    finally:
        if tmp_dir is not None:
            tmp_dir.cleanup()


def _load_hdf5(uri: str, rad_var: str, source_id: str = "") -> np.ndarray:
    import h5py

    if uri.startswith("s3://"):
        import fsspec

        opts = _s3_storage_options(uri, source_id)
        fs = fsspec.filesystem("s3", **opts)
        with fs.open(uri, "rb") as f:
            with h5py.File(f, "r") as hf:
                return _read_hdf5_dataset(hf, rad_var)
    with h5py.File(uri, "r") as hf:
        return _read_hdf5_dataset(hf, rad_var)


def _read_hdf5_dataset(hf, rad_var: str) -> np.ndarray:
    import h5py

    for name in (rad_var, "IMG_TIR1", "TIR1", "BrightnessTemperature"):
        if name in hf:
            return np.asarray(hf[name][:], dtype=np.float64)
    # nested groups
    for key in hf.keys():
        if isinstance(hf[key], h5py.Group):
            try:
                return _read_hdf5_dataset(hf[key], rad_var)
            except KeyError:
                continue
    raise KeyError(f"No TIR dataset in HDF5; top keys={list(hf.keys())}")


def _load_netcdf(uri: str, rad_var: str | None, source_id: str = "") -> np.ndarray:
    ds = _open_nc_dataset(uri, source_id)
    try:
        var = rad_var
        if var is None or var not in ds:
            for candidate in (
                "Rad", "CMI", "IMG_TIR1", "TIR1", "tbb",
                "BrightnessTemperature", "image_pixel_values", "radiance",
            ):
                if candidate in ds:
                    var = candidate
                    break
        if var is None or var not in ds:
            raise KeyError(f"No radiance variable in {uri}; vars={list(ds.data_vars)}")
        return np.asarray(ds[var].values, dtype=np.float64)
    finally:
        ds.close()


def load_nc_radiance_uri(
    uri: str,
    rad_var: str | None = None,
    source_id: str = "",
    segment_uris: tuple[str, ...] = (),
) -> np.ndarray:
    src = satellite_sources().get(source_id)
    fmt = src.fmt if src else "netcdf"
    var = rad_var or (src.rad_var if src else None)

    if fmt == "himawari_hsd" or uri.endswith(".DAT.bz2"):
        return _load_himawari_hsd(uri, segment_uris=segment_uris, source_id=source_id)
    if fmt == "hdf5" or uri.endswith(".h5"):
        return _load_hdf5(uri, var or "IMG_TIR1", source_id=source_id)
    return _load_netcdf(uri, var, source_id=source_id)


def load_nc_radiance_ref(ref: NcRef, rad_var: str | None = None) -> np.ndarray:
    return load_nc_radiance_uri(
        ref.uri,
        rad_var=rad_var or ref.rad_var or None,
        source_id=ref.source_id,
        segment_uris=ref.segment_uris,
    )


def _open_nc_dataset(uri: str, source_id: str = ""):
    import xarray as xr

    engine = "h5netcdf" if uri.startswith("s3://") else None
    open_kwargs: dict = {}
    if uri.startswith("s3://"):
        open_kwargs["storage_options"] = _s3_storage_options(uri, source_id)
    return xr.open_dataset(uri, engine=engine, **open_kwargs)


def _goes_latlon_from_ds(ds) -> tuple[np.ndarray, np.ndarray]:
    """Lat/lon (degrees) for GOES ABI L1b/L2 fixed grid."""
    from pyproj import Proj

    if "goes_imager_projection" not in ds:
        raise KeyError("no goes_imager_projection")
    attrs = ds["goes_imager_projection"].attrs
    lon_0 = float(attrs["longitude_of_projection_origin"])
    h = float(attrs["perspective_point_height"])
    a = float(attrs.get("semi_major_axis", 6378137.0))
    b = float(attrs.get("semi_minor_axis", 6356752.31414))
    proj = Proj(
        proj="geos", lon_0=lon_0, h=h, x_0=0, y_0=0, lat_0=0,
        a=a, b=b, sweep="x", units="m",
    )
    x = np.asarray(ds["x"].values, dtype=np.float64)
    y = np.asarray(ds["y"].values, dtype=np.float64)
    if x.ndim == 1 and y.ndim == 1:
        x_m, y_m = np.meshgrid(x * h, y * h)
    else:
        x_m = x * h
        y_m = y * h
    lon, lat = proj(x_m, y_m, inverse=True)
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    bad = ~np.isfinite(lat) | ~np.isfinite(lon) | (np.abs(lat) > 90)
    lat[bad] = np.nan
    lon[bad] = np.nan
    return lat, lon


def load_nc_latlon_uri(
    uri: str,
    source_id: str = "",
    segment_uris: tuple[str, ...] = (),
) -> tuple[np.ndarray, np.ndarray]:
    """Return (lat, lon) in degrees on the native grid."""
    src = satellite_sources().get(source_id)
    if (src and src.fmt == "himawari_hsd") or uri.endswith(".DAT.bz2"):
        return _himawari_latlon(uri, segment_uris=segment_uris, source_id=source_id)

    ds = _open_nc_dataset(uri, source_id)
    try:
        for lat_name, lon_name in (
            ("lat", "lon"), ("latitude", "longitude"), ("Latitude", "Longitude"),
        ):
            if lat_name in ds and lon_name in ds:
                lat = np.asarray(ds[lat_name].values, dtype=np.float64)
                lon = np.asarray(ds[lon_name].values, dtype=np.float64)
                if lat.ndim == 1 and lon.ndim == 1:
                    lon, lat = np.meshgrid(lon, lat)
                return lat, lon
        return _goes_latlon_from_ds(ds)
    finally:
        ds.close()


def load_nc_latlon_ref(ref: NcRef) -> tuple[np.ndarray, np.ndarray]:
    return load_nc_latlon_uri(
        ref.uri,
        source_id=ref.source_id,
        segment_uris=ref.segment_uris,
    )
