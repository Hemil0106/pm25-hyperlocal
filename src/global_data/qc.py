"""Observation quality control (Milestone 16).

Rules:
  - Physically invalid coordinates / missing PM2.5 / unknown units are REMOVED
    (they are not observations at all).
  - Outliers are FLAGGED, never deleted: an outlier is a valid measurement that
    is unusually high relative to its network's distribution. Flagging keeps the
    data auditable and lets later stages choose whether to include it.
  - Duplicate (station, timestamp) rows keep the first occurrence.

Every run produces a QC report with the exact fields required by the milestone
spec: input_rows, retained_rows, duplicate_rows, invalid_coordinates,
missing_pm25, invalid_units, negative_pm25, outlier_flags, date_range,
station_count, country_count, status.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _mad_outlier_flags(values: pd.Series, threshold_k: float) -> pd.Series:
    """Flag values deviating > k * MAD from the median (nan-safe)."""
    valid = values.dropna()
    if valid.empty:
        return pd.Series(False, index=values.index, dtype=bool)
    median = valid.median()
    deviation = (valid - median).abs()
    mad = (deviation.median() * 1.4826) or 0.0
    if mad == 0.0:
        return pd.Series(False, index=values.index, dtype=bool)
    flags = deviation > (threshold_k * mad)
    return flags.reindex(values.index).fillna(False).astype(bool)


def apply_qc(df: pd.DataFrame, value_col: str = "pm25_ug_m3",
             units_col: str = "units", lat_col: str = "latitude",
             lon_col: str = "longitude", ts_col: str = "timestamp",
             station_col: str = "station_id", country_col: str = "country",
             outlier_method: str = "mad", outlier_threshold_k: float = 5.0,
             max_pm25_ug_m3: float = 1000.0) -> tuple[pd.DataFrame, dict]:
    """Run the QC rules and return (clean_frame, report)."""
    n_input = int(len(df))
    if df.empty:
        return df.copy(), {
            "input_rows": 0, "retained_rows": 0, "duplicate_rows": 0,
            "invalid_coordinates": 0, "missing_pm25": 0, "invalid_units": 0,
            "negative_pm25": 0, "outlier_flags": 0, "date_range": None,
            "station_count": 0, "country_count": 0, "status": "empty",
        }

    out = df.copy()

    # -- duplicates ------------------------------------------------------
    dup_key = [station_col, ts_col]
    n_before = len(out)
    out = out.drop_duplicates(subset=dup_key, keep="first")
    duplicate_rows = n_before - len(out)

    # -- coordinates -----------------------------------------------------
    lat = pd.to_numeric(out[lat_col], errors="coerce")
    lon = pd.to_numeric(out[lon_col], errors="coerce")
    invalid_coords = ~(
        lat.between(-90.0, 90.0) & lon.between(-180.0, 180.0)
    )
    n_invalid_coords = int(invalid_coords.sum())
    out = out[~invalid_coords]

    # -- timestamps ------------------------------------------------------
    ts = pd.to_datetime(out[ts_col], errors="coerce")
    out = out.assign(**{ts_col: ts})
    n_bad_ts = int(out[ts_col].isna().sum())
    out = out[out[ts_col].notna()]

    # -- units -----------------------------------------------------------
    invalid_units = out[units_col].isna() | out[units_col].astype(str).eq("")
    n_invalid_units = int(invalid_units.sum())
    out = out[~invalid_units]

    # -- value validity --------------------------------------------------
    value = pd.to_numeric(out[value_col], errors="coerce")
    out = out.assign(**{value_col: value})
    missing_pm25 = out[value_col].isna()
    n_missing = int(missing_pm25.sum())
    out = out[~missing_pm25]

    negative_pm25 = out[value_col] < 0.0
    n_negative = int(negative_pm25.sum())
    out = out[~negative_pm25]

    above_max = out[value_col] > max_pm25_ug_m3
    n_above_max = int(above_max.sum())
    out = out[~above_max]

    # -- outlier FLAG (not removal) --------------------------------------
    method = str(outlier_method).lower()
    if method == "mad":
        flags = _mad_outlier_flags(out[value_col], float(outlier_threshold_k))
    else:
        flags = pd.Series(False, index=out.index, dtype=bool)
    out = out.assign(outlier_flag=flags)
    n_outliers = int(flags.sum())

    out = out.reset_index(drop=True)

    date_range = None
    if not out.empty:
        dates = out[ts_col].dt.normalize()
        date_range = {
            "min": str(dates.min().date()),
            "max": str(dates.max().date()),
        }

    report = {
        "input_rows": n_input,
        "retained_rows": int(len(out)),
        "duplicate_rows": duplicate_rows,
        "invalid_coordinates": n_invalid_coords,
        "invalid_timestamps": int(n_bad_ts),
        "missing_pm25": n_missing,
        "invalid_units": n_invalid_units,
        "negative_pm25": n_negative,
        "above_max_pm25": n_above_max,
        "outlier_flags": n_outliers,
        "outlier_method": method,
        "outlier_threshold_k": float(outlier_threshold_k),
        "date_range": date_range,
        "station_count": int(out[station_col].nunique()) if not out.empty else 0,
        "country_count": int(out[country_col].nunique()) if not out.empty else 0,
        "status": "ok" if not out.empty else "no_valid_observations",
    }
    return out, report
