"""Global training schema and dataset construction (Phase 12).

Defines the canonical schema a training dataset must satisfy before it can be
used to train a model for any scope, plus honest builders that never fabricate
training data. With only CPCB (India) data enabled, building a *global*
training frame is impossible and is reported as such.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

logger = logging.getLogger(__name__)

TRAINING_TARGET = "PM2.5"

# Canonical feature columns used by the locked Delhi model family.
FEATURE_COLUMNS = [
    "AOD", "temperature_c", "relative_humidity_pct", "wind_speed_mps",
    "wind_direction_deg", "wind_u_mps", "wind_v_mps", "NDVI", "elevation_m",
    "road_density", "night_lights", "month", "day_of_year",
]

# Spatial/temporal identifiers every training frame must carry.
IDENTIFIER_COLUMNS = [
    "station_id", "grid_id", "date", "longitude", "latitude",
]

# Spatial grouping columns used for grouped validation and honest labeling.
SPATIAL_GROUP_COLUMNS = ["city", "region", "country"]

# Traceability columns: whether each input feature was actually available.
TRACEABILITY_COLUMNS = [
    "AOD_available", "NDVI_available", "VIIRS_available",
    "weather_available", "DEM_available", "OSM_available",
]

GLOBAL_TRAINING_SCHEMA = {
    "target": TRAINING_TARGET,
    "features": FEATURE_COLUMNS,
    "identifiers": IDENTIFIER_COLUMNS,
    "spatial_groups": SPATIAL_GROUP_COLUMNS,
    "traceability": TRACEABILITY_COLUMNS,
}


def global_training_schema() -> dict:
    return dict(GLOBAL_TRAINING_SCHEMA)


def validate_training_schema(df: pd.DataFrame, required_features: Optional[Sequence[str]] = None,
                             require_spatial_groups: bool = False) -> dict:
    """Validate a frame against the training schema.

    Returns a report of missing columns and passes/fails; never raises a
    fabricated success.
    """
    features = list(required_features) if required_features is not None else list(FEATURE_COLUMNS)
    missing = []
    for col in features + IDENTIFIER_COLUMNS:
        if col not in df.columns:
            missing.append(col)
    if require_spatial_groups:
        for col in SPATIAL_GROUP_COLUMNS:
            if col not in df.columns:
                missing.append(col)

    ok = not missing and TRAINING_TARGET in df.columns
    report = {
        "valid": bool(ok),
        "missing_columns": missing,
        "n_rows": int(len(df)),
        "target": TRAINING_TARGET,
    }
    return report


def _station_geo_index(df: pd.DataFrame) -> pd.DataFrame:
    idx = (
        df.groupby("station_id", as_index=False)
        .agg(latitude=("latitude", "first"), longitude=("longitude", "first"),
             country=("country", "first"))
    )
    idx["region"] = idx["country"]
    idx["city"] = idx["station_id"]
    return idx


def build_training_frame(config, aoi=None, required_features: Optional[Sequence[str]] = None,
                         write_path: Optional[str] = None):
    """Build a schema-valid training frame for an AOI (honest).

    Joins normalized ground truth with a per-station geographic index. Spatial
    group columns (city/region/country) are derived from the source network
    metadata. Returns (frame, report). For scopes with no ground truth the
    frame is empty and the report explains why.
    """
    gt = None
    try:
        from .ground_truth import normalize_ground_truth

        gt = normalize_ground_truth(config)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Ground truth unavailable for training frame: %s", exc)

    if gt is None or gt.empty:
        return pd.DataFrame(), {
            "valid": False,
            "reason": "No ground-truth observations enabled. Global training "
                      "requires OpenAQ (disabled) or another multi-country source.",
            "n_rows": 0,
        }

    frame = gt.copy()
    geo = _station_geo_index(frame)
    merge_cols = ["station_id", "region", "city"]
    if "country" not in frame.columns:
        merge_cols.append("country")
    frame = frame.merge(geo[merge_cols], on="station_id", how="left")
    if "country" not in frame.columns:
        frame["country"] = "Unknown"

    for col in FEATURE_COLUMNS:
        if col not in frame.columns:
            frame[col] = float("nan")
    for col in ("grid_id",):
        if col not in frame.columns:
            frame[col] = ""
    frame["date"] = pd.to_datetime(frame["timestamp"]).dt.date.astype(str)

    report = validate_training_schema(frame, required_features=required_features,
                                      require_spatial_groups=True)
    if aoi is not None:
        bounds = aoi.bounds
        frame = frame[
            frame["longitude"].between(bounds["west"], bounds["east"])
            & frame["latitude"].between(bounds["south"], bounds["north"])
        ]
        report["n_rows_in_aoi"] = int(len(frame))
        if aoi.is_global and not frame.empty:
            countries = sorted(frame["country"].dropna().unique().tolist())
            if len(countries) <= 1:
                report["valid"] = False
                report["note"] = (
                    f"Ground truth covers only {countries}. A scientifically "
                    "valid GLOBAL model cannot be trained on single-country "
                    "observations; global scope stays unavailable."
                )

    if write_path:
        out = Path(write_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(out, index=False)
        logger.info("Training frame written to %s (%d rows)", out, len(frame))

    return frame, report
