"""Readiness gate (Milestone 17).

The readiness gate is fully data-derived and never auto-YES: it measures real
complete-case rows, stations, countries, temporal span, and per-source feature
coverage. With no real data it must say NO / BLOCKED.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from .schema import TARGET_COL

logger = logging.getLogger(__name__)

MIN_COMPLETE_ROWS = 1000
MIN_STATIONS = 20
MIN_COUNTRIES = 5
MIN_TEMPORAL_DAYS = 30


def _source_ready_flag(df: pd.DataFrame, source_features) -> bool:
    """True only when ALL features of a source are present in >= a share of rows."""
    if df.empty:
        return False
    rows_with_all = df[source_features].notna().all(axis=1).sum()
    return bool(rows_with_all >= MIN_COMPLETE_ROWS)


def build_readiness(df: pd.DataFrame, config) -> dict:
    """Compute the data-derived readiness report and write it."""
    n_complete = int(df["complete_case"].sum()) if not df.empty else 0
    n_stations = int(df["station_id"].nunique()) if not df.empty else 0
    n_countries = int(df["country"].dropna().nunique()) if not df.empty else 0

    temporal_days = 0
    date_range = None
    if not df.empty:
        dates = pd.to_datetime(df["date"], errors="coerce")
        if not dates.isna().all():
            temporal_days = int(dates.dt.normalize().nunique())
            date_range = {"min": str(dates.min().date()),
                          "max": str(dates.max().date())}

    source_flags = {
        "pm25": {"present": bool(
            df[TARGET_COL].notna().sum() >= MIN_COMPLETE_ROWS
        ) if not df.empty else False},
        "aod": {"present": _source_ready_flag(df, ["AOD"])},
        "weather": {"present": _source_ready_flag(
            df, ["temperature_c", "relative_humidity_pct", "wind_speed_mps",
                 "wind_direction_deg"])},
        "ndvi": {"present": _source_ready_flag(df, ["NDVI"])},
        "dem": {"present": _source_ready_flag(df, ["elevation_m"])},
        "osm": {"present": _source_ready_flag(df, ["road_density"])},
        "viirs": {"present": _source_ready_flag(df, ["night_lights"])},
    }

    ready = (
        n_complete >= MIN_COMPLETE_ROWS
        and n_stations >= MIN_STATIONS
        and n_countries >= MIN_COUNTRIES
        and temporal_days >= MIN_TEMPORAL_DAYS
        and all(v["present"] for v in source_flags.values())
    )
    reasons = []
    if n_complete < MIN_COMPLETE_ROWS:
        reasons.append(f"complete-case rows {n_complete} < {MIN_COMPLETE_ROWS}")
    if n_stations < MIN_STATIONS:
        reasons.append(f"stations {n_stations} < {MIN_STATIONS}")
    if n_countries < MIN_COUNTRIES:
        reasons.append(f"countries {n_countries} < {MIN_COUNTRIES}")
    if temporal_days < MIN_TEMPORAL_DAYS:
        reasons.append(f"temporal days {temporal_days} < {MIN_TEMPORAL_DAYS}")
    for source, flag in source_flags.items():
        if not flag["present"]:
            reasons.append(f"source '{source}' feature coverage insufficient")

    report = {
        "scope": config.get("global_data", {}).get("scope", "global"),
        "thresholds": {
            "min_complete_rows": MIN_COMPLETE_ROWS,
            "min_stations": MIN_STATIONS,
            "min_countries": MIN_COUNTRIES,
            "min_temporal_days": MIN_TEMPORAL_DAYS,
        },
        "measured": {
            "complete_case_rows": n_complete,
            "stations": n_stations,
            "countries": n_countries,
            "temporal_days": temporal_days,
            "date_range": date_range,
        },
        "sources": source_flags,
        "model_training_ready": "YES" if ready else "NO",
        "reason": None if ready else "; ".join(reasons),
        "rule": "Readiness is fully data-derived and never auto-YES.",
    }

    processed_base = Path(
        config.get("global_data", {}).get("storage", {}).get(
            "processed_base", "data/processed/global")
    )
    reports_dir = processed_base / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / "global_training_readiness.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["path"] = str(path)
    return report
