"""Global station registry and summary (Milestone 16).

Built from the QC'd, unit-normalized observations. The registry maps station
ids to locations and time coverage; the summary aggregates counts per country.
Both are written to ``data/processed/global/stations/`` and are only produced
when real observations were acquired.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

STATION_REGISTRY_SCHEMA = [
    "station_id", "source", "country", "latitude", "longitude",
    "first_timestamp", "last_timestamp", "n_observations", "n_days",
]


def build_station_registry(daily_df: pd.DataFrame) -> pd.DataFrame:
    """One row per station from the daily series (longitude/latitude columns)."""
    if daily_df.empty:
        return pd.DataFrame(columns=STATION_REGISTRY_SCHEMA)

    df = daily_df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["first_timestamp"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df["last_timestamp"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    grouped = df.groupby(["station_id", "source", "country"], as_index=False).agg(
        latitude=("latitude", "first"),
        longitude=("longitude", "first"),
        first_timestamp=("first_timestamp", "min"),
        last_timestamp=("last_timestamp", "max"),
        n_observations=("observation_count", "sum"),
        n_days=("date", "count"),
    )
    grouped = grouped.sort_values("station_id").reset_index(drop=True)
    return grouped[STATION_REGISTRY_SCHEMA]


def build_station_summary(registry: pd.DataFrame) -> pd.DataFrame:
    """Aggregate counts per country (and overall) from the station registry."""
    if registry.empty:
        return pd.DataFrame(columns=["country", "n_stations", "n_observations"])
    summary = registry.groupby("country", as_index=False).agg(
        n_stations=("station_id", "nunique"),
        n_observations=("n_observations", "sum"),
    )
    return summary.sort_values("n_stations", ascending=False).reset_index(drop=True)


def write_station_outputs(registry: pd.DataFrame, daily_df: pd.DataFrame,
                          processed_base: Path) -> dict:
    """Write registry + summary parquet; returns artifact paths."""
    out_dir = Path(processed_base) / "stations"
    out_dir.mkdir(parents=True, exist_ok=True)

    registry_path = out_dir / "global_station_registry.parquet"
    summary_path = out_dir / "global_station_summary.parquet"

    registry.to_parquet(registry_path, index=False)
    summary = build_station_summary(registry)
    summary.to_parquet(summary_path, index=False)

    n_rows = int(len(daily_df)) if not daily_df.empty else 0
    n_stations = int(registry["station_id"].nunique()) if not registry.empty else 0
    countries = (
        sorted(registry["country"].dropna().unique().tolist())
        if not registry.empty else []
    )
    date_range = None
    if not daily_df.empty:
        dates = pd.to_datetime(daily_df["date"])
        date_range = {"min": str(dates.min().date()), "max": str(dates.max().date())}

    return {
        "n_stations": n_stations,
        "n_rows": n_rows,
        "countries": countries,
        "date_range": date_range,
        "registry_path": str(registry_path),
        "summary_path": str(summary_path),
    }
