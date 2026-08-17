"""Ground-truth normalization for arbitrary AOIs (Phase 6).

Merges per-network PM2.5 observations into one standardized, spatially and
temporally consistent table with schema:

    station_id | source | country | latitude | longitude | timestamp | PM2.5 | units | quality_flag

CPCB observations are preserved unchanged (as the India source) when present.
OpenAQ (global) is optional and disabled by default.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

GROUND_TRUTH_SCHEMA = [
    "station_id", "source", "country", "latitude", "longitude",
    "timestamp", "PM2.5", "units", "quality_flag",
]


def _read_cpcb(config) -> pd.DataFrame:
    ds_cfg = config.get("datasets", {}).get("cpcb", {})
    gt_src = config.get("ground_truth", {}).get("sources", {}).get("cpcb", {})
    input_file = ds_cfg.get("input_file")
    if not input_file or not Path(input_file).exists():
        return pd.DataFrame()

    df = pd.read_csv(input_file)
    ts_col = ds_cfg.get("timestamp_column", "timestamp")
    sid_col = ds_cfg.get("station_id_column", "station_id")
    lat_col = ds_cfg.get("latitude_column", "latitude")
    lon_col = ds_cfg.get("longitude_column", "longitude")
    pm_col = ds_cfg.get("pm25_column", "PM2.5")

    df = df.rename(columns={pm_col: "PM2.5"})
    df["station_id"] = df[sid_col].astype(str)
    df["source"] = "cpcb"
    df["country"] = gt_src.get("country", "India")
    df["units"] = gt_src.get("units", "ug/m3")
    df["quality_flag"] = gt_src.get("quality_flag", "cpcb_reported")
    df["timestamp"] = pd.to_datetime(df[ts_col], errors="coerce")
    df["latitude"] = pd.to_numeric(df[lat_col], errors="coerce")
    df["longitude"] = pd.to_numeric(df[lon_col], errors="coerce")
    df["PM2.5"] = pd.to_numeric(df["PM2.5"], errors="coerce")

    keep = ["station_id", "source", "country", "latitude", "longitude",
            "timestamp", "PM2.5", "units", "quality_flag"]
    return df[keep]


def _read_openaq(config) -> pd.DataFrame:
    gt = config.get("ground_truth", {}).get("sources", {}).get("openaq", {})
    input_file = gt.get("input_file")
    if not input_file or not Path(input_file).exists():
        return pd.DataFrame()

    df = pd.read_csv(input_file)
    rename_map = {
        "location": "station_id",
        "city": "station_id",
        "datetime": "timestamp",
        "value": "PM2.5",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    df["source"] = "openaq"
    df["country"] = gt.get("country")
    df["units"] = gt.get("units", "ug/m3")
    df["quality_flag"] = gt.get("quality_flag", "openaq_reported")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["latitude"] = pd.to_numeric(df.get("latitude"), errors="coerce")
    df["longitude"] = pd.to_numeric(df.get("longitude"), errors="coerce")
    df["PM2.5"] = pd.to_numeric(df["PM2.5"], errors="coerce")

    keep = ["station_id", "source", "country", "latitude", "longitude",
            "timestamp", "PM2.5", "units", "quality_flag"]
    df = df[[c for c in keep if c in df.columns]]
    for col in keep:
        if col not in df.columns:
            df[col] = None
    return df[keep]


def normalize_ground_truth(config, write_path: Optional[str] = None) -> pd.DataFrame:
    """Build the normalized ground-truth table from all enabled sources."""
    gt_cfg = config.get("ground_truth", {})
    frames = []
    for src_id in ("cpcb", "openaq"):
        src_cfg = gt_cfg.get("sources", {}).get(src_id, {})
        if not src_cfg.get("enabled", False):
            continue
        frame = _read_cpcb(config) if src_id == "cpcb" else _read_openaq(config)
        if frame.empty:
            logger.warning("Ground-truth source '%s' produced no rows.", src_id)
            continue
        frames.append(frame)

    if not frames:
        df = pd.DataFrame(columns=GROUND_TRUTH_SCHEMA)
        logger.warning("No enabled ground-truth sources produced rows.")
    else:
        df = pd.concat(frames, ignore_index=True)

    df = df.dropna(subset=["latitude", "longitude", "PM2.5", "timestamp"])
    n_invalid = int(
        ((df["latitude"] < -90.0) | (df["latitude"] > 90.0)
         | (df["longitude"] < -180.0) | (df["longitude"] > 180.0)).sum()
    )
    if n_invalid:
        logger.warning(
            "Dropped %d ground-truth rows with physically invalid coordinates.",
            n_invalid,
        )
        df = df[
            df["latitude"].between(-90.0, 90.0)
            & df["longitude"].between(-180.0, 180.0)
        ]
    df = df.drop_duplicates(subset=["station_id", "source", "timestamp"], keep="first")
    df = df.sort_values(["source", "station_id", "timestamp"]).reset_index(drop=True)

    out_file = write_path or gt_cfg.get("output_file", "data/processed/ground_pm25.parquet")
    if not df.empty:
        out = Path(out_file)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)
        logger.info("Ground truth written to %s (%d rows)", out, len(df))
    else:
        logger.warning("No ground-truth rows to write; %s left untouched.", out_file)

    return df


def ground_truth_summary(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"n_stations": 0, "n_rows": 0, "sources": [], "countries": []}
    return {
        "n_stations": int(df["station_id"].nunique()),
        "n_rows": int(len(df)),
        "sources": sorted(df["source"].unique().tolist()),
        "countries": sorted(df["country"].dropna().unique().tolist()),
        "pm25_min": float(df["PM2.5"].min()),
        "pm25_max": float(df["PM2.5"].max()),
        "pm25_mean": float(df["PM2.5"].mean()),
    }
