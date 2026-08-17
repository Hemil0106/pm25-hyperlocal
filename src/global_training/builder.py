"""Training-table builder (Milestone 17).

Builds the schema-correct global training table from the real Milestone 16 daily
observations plus real feature artifacts:

  - observations: ``data/processed/global/pm25/global_pm25_daily.parquet``
  - road density:  real OSM tile artifacts (see osm_density.py)
  - gridded features (AOD / weather / NDVI / DEM / night lights): only populated
    from real M16 artifacts; with none acquired they are NaN + available=False

When no observations exist the builder still emits the schema-correct empty
table so downstream reports and the readiness gate run honestly.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.global_data.scope import validate_scope

from .features import add_derived_features
from .osm_density import load_osm_road_segments, road_density_for
from .schema import (
    AVAILABILITY_COLS,
    FEATURE_COLS,
    GRIDDED_FEATURE_COLS,
    TARGET_COL,
    TRAINING_COLUMNS,
)

logger = logging.getLogger(__name__)

EMPTY_REASON_OK = "ok"
EMPTY_REASON_MISSING = "missing"
EMPTY_REASON_EMPTY = "empty_daily"


def empty_training_frame() -> pd.DataFrame:
    """A schema-correct training table with zero rows."""
    return pd.DataFrame(columns=TRAINING_COLUMNS)


def daily_path(config, scope: str) -> Path:
    processed_base = Path(
        config.get("global_data", {}).get("storage", {}).get(
            "processed_base", "data/processed/global")
    )
    return processed_base / "pm25" / "global_pm25_daily.parquet"


def load_daily(config, scope: str) -> tuple[Optional[pd.DataFrame], str]:
    """Load the real M16 daily observations for a scope.

    Returns (None, 'missing') when the acquisition never produced a file,
    (empty_df, 'empty_daily') when the file has no rows, else (df, 'ok').
    """
    validate_scope(scope)
    path = daily_path(config, scope)
    if not path.exists():
        return None, EMPTY_REASON_MISSING
    try:
        df = pd.read_parquet(path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Unable to read daily parquet %s: %s", path, exc)
        return None, "unreadable"
    if df.empty:
        return df, EMPTY_REASON_EMPTY
    return df, EMPTY_REASON_OK


def build_training_table(
    daily: Optional[pd.DataFrame],
    road_segments: dict,
    scope: str,
) -> tuple[pd.DataFrame, dict]:
    """Assemble the training table from real inputs only.

    Returns (table, report). The report documents what was real and what was
    unavailable; nothing is ever fabricated.
    """
    scope = validate_scope(scope)
    if daily is None or daily.empty:
        return empty_training_frame(), {
            "rows": 0,
            "status": "empty",
            "reason": "No real daily observations available for this scope.",
            "complete_case_rows": 0,
            "stations": 0,
            "countries": 0,
            "osm_tile_artifacts": len(road_segments),
        }

    daily = daily.copy()
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce").dt.normalize()
    daily["latitude"] = pd.to_numeric(daily["latitude"], errors="coerce")
    daily["longitude"] = pd.to_numeric(daily["longitude"], errors="coerce")
    daily["pm25_daily"] = pd.to_numeric(daily["pm25_daily"], errors="coerce")

    road = road_density_for(daily, road_segments, scope)
    out = pd.DataFrame(index=daily.index)

    # Identity columns.
    out["station_id"] = daily["station_id"].astype(str)
    out["station_name"] = np.nan  # no display name acquired (OpenAQ id only)
    out["country"] = daily.get("country", pd.Series(np.nan, index=daily.index))
    out["city"] = np.nan  # no reverse-geocoding performed (never fabricated)
    out["latitude"] = daily["latitude"]
    out["longitude"] = daily["longitude"]
    out["date"] = daily["date"]

    # Target: real normalized daily PM2.5 (ug/m3).
    out[TARGET_COL] = daily["pm25_daily"]

    # Gridded features are NaN unless a real artifact was joined in.
    for col in GRIDDED_FEATURE_COLS:
        out[col] = np.nan

    # Real OSM road density per tile (spatial proxy).
    out["road_density"] = road["road_density"].to_numpy()

    # Derived meteorology + temporal features (real wind only, else NaN).
    out = add_derived_features(out)

    # Availability flags per source feature.
    out["wind_u_mps_available"] = out["wind_u_mps"].notna()
    out["wind_v_mps_available"] = out["wind_v_mps"].notna()
    for col in GRIDDED_FEATURE_COLS:
        out[f"{col}_available"] = out[col].notna()

    # NDVI provenance columns (only meaningful when NDVI is real).
    out["NDVI_source_date"] = np.nan
    out["NDVI_offset_days"] = np.nan

    # Complete-case: valid target AND every candidate feature present.
    out["complete_case"] = out[TARGET_COL].notna() & out[FEATURE_COLS].notna().all(axis=1)

    # Number of available source features (identity/target excluded).
    out["n_features_available"] = out[AVAILABILITY_COLS].astype(bool).sum(axis=1)

    out = out[TRAINING_COLUMNS].reset_index(drop=True)

    report = {
        "rows": int(len(out)),
        "status": "ok",
        "complete_case_rows": int(out["complete_case"].sum()),
        "stations": int(out["station_id"].nunique()) if not out.empty else 0,
        "countries": int(out["country"].dropna().nunique()) if not out.empty else 0,
        "osm_tile_artifacts": len(road_segments),
        "note": "Only real acquired data is used; unavailable features are NaN.",
    }
    return out, report


def split_complete_cases(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Complete-case vs incomplete diagnostic frames (both preserved)."""
    if df.empty:
        return empty_training_frame(), empty_training_frame()
    complete = df[df["complete_case"]].copy().reset_index(drop=True)
    incomplete = df[~df["complete_case"]].copy().reset_index(drop=True)
    return complete, incomplete


def write_training_outputs(table, complete, incomplete, config) -> dict:
    """Write the three training-table parquet files; returns artifact paths."""
    processed_base = Path(
        config.get("global_data", {}).get("storage", {}).get(
            "processed_base", "data/processed/global")
    )
    out_dir = processed_base / "training"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "training_dataset": str(out_dir / "global_training_dataset.parquet"),
        "complete": str(out_dir / "global_training_dataset_complete.parquet"),
        "incomplete": str(out_dir / "global_training_dataset_incomplete.parquet"),
    }
    table.to_parquet(paths["training_dataset"], index=False)
    complete.to_parquet(paths["complete"], index=False)
    incomplete.to_parquet(paths["incomplete"], index=False)
    return paths
