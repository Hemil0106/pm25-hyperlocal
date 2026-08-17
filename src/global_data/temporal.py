"""Temporal daily normalization (Milestone 16).

Daily aggregation is configurable: ``daily_mean`` produces the mean of valid
observations per (station, day) plus an observation count. Days with fewer than
``min_observations_per_day`` valid measurements are SKIPPED entirely - missing
days are never zero-filled.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def aggregate_daily(df: pd.DataFrame, value_col: str = "pm25_ug_m3",
                    ts_col: str = "timestamp", station_col: str = "station_id",
                    lat_col: str = "latitude", lon_col: str = "longitude",
                    country_col: str = "country", source_col: str = "source",
                    method: str = "daily_mean",
                    min_observations_per_day: int = 1) -> tuple[pd.DataFrame, dict]:
    """Aggregate valid observations to a daily series per station.

    Returns (daily_df, report). A day/station with fewer than
    ``min_observations_per_day`` valid measurements is omitted.
    """
    if df.empty:
        empty = pd.DataFrame(columns=[
            station_col, "date", "pm25_daily", "observation_count",
            lat_col, lon_col, country_col, source_col,
        ])
        return empty, {
            "method": method, "min_observations_per_day": min_observations_per_day,
            "station_days_skipped": 0, "status": "empty",
        }

    out = df.copy()
    out["date"] = pd.to_datetime(out[ts_col]).dt.normalize()

    grouped = out.groupby(
        [station_col, "date"], as_index=False
    ).agg(
        pm25_daily=(value_col, "mean"),
        observation_count=(value_col, "count"),
        latitude=(lat_col, "first"),
        longitude=(lon_col, "first"),
        country=(country_col, "first"),
        source=(source_col, "first"),
    )
    grouped["pm25_daily"] = pd.to_numeric(grouped["pm25_daily"], errors="coerce")

    skipped = grouped["observation_count"] < min_observations_per_day
    n_skipped = int(skipped.sum())
    daily = grouped[~skipped].copy().reset_index(drop=True)
    daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()

    report = {
        "method": method,
        "min_observations_per_day": int(min_observations_per_day),
        "station_days_skipped": n_skipped,
        "station_days_retained": int(len(daily)),
        "stations": int(daily[station_col].nunique()) if not daily.empty else 0,
        "countries": int(daily["country"].nunique()) if not daily.empty else 0,
        "status": "ok" if not daily.empty else "no_sufficient_daily_data",
    }
    return daily, report
