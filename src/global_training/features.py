"""Feature engineering (Milestone 17).

Strictly limited and justified:
  - temporal cyclics (month, day_of_year, sin/cos of day-of-year) derived from
    the row's own date only - never from other rows, never from the future
  - meteorological u/v wind components from wind speed/direction when both are
    real values; otherwise NaN with ``*_available == False``

Features are never generated from the PM2.5 target.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import DERIVED_FEATURE_COLS, TEMPORAL_FEATURE_COLS


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add month / day_of_year / sin / cos derived from the row date.

    Rows with an invalid date get NaN for every temporal feature.
    """
    out = df.copy()
    dates = pd.to_datetime(out["date"], errors="coerce")
    month = np.where(dates.isna(), np.nan, dates.dt.month.to_numpy().astype(float))
    doy = np.where(
        dates.isna(),
        np.nan,
        dates.dt.dayofyear.to_numpy().astype(float),
    )
    year_len = 365.25
    sin_doy = np.where(
        pd.isna(doy), np.nan, np.sin(2.0 * np.pi * doy / year_len))
    cos_doy = np.where(
        pd.isna(doy), np.nan, np.cos(2.0 * np.pi * doy / year_len))
    out["month"] = month
    out["day_of_year"] = doy
    out["sin_day_of_year"] = sin_doy
    out["cos_day_of_year"] = cos_doy
    return out


def add_wind_components(df: pd.DataFrame) -> pd.DataFrame:
    """Add u/v wind components from speed + direction (real values only).

    Meteorological convention: u = -V sin(theta), v = -V cos(theta) with theta
    in radians measured clockwise from north. Missing speed/direction -> NaN.
    """
    out = df.copy()
    if "wind_speed_mps" not in out.columns or "wind_direction_deg" not in out.columns:
        out["wind_u_mps"] = np.nan
        out["wind_v_mps"] = np.nan
        return out
    speed = pd.to_numeric(out["wind_speed_mps"], errors="coerce")
    direction = pd.to_numeric(out["wind_direction_deg"], errors="coerce")
    both = speed.notna() & direction.notna()
    theta = np.deg2rad(direction.where(both, np.nan))
    out["wind_u_mps"] = np.where(both, -speed * np.sin(theta), np.nan)
    out["wind_v_mps"] = np.where(both, -speed * np.cos(theta), np.nan)
    return out


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the full, limited feature-engineering set."""
    out = add_temporal_features(df)
    out = add_wind_components(out)
    return out


def availability_flags(df: pd.DataFrame, feature_cols) -> pd.DataFrame:
    """Return per-feature ``<feature>_available`` flags.

    A feature is available when the row value is not NaN.
    """
    flags = pd.DataFrame(index=df.index)
    for col in feature_cols:
        flags[f"{col}_available"] = df[col].notna()
    return flags
