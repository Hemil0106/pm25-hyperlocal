"""Uncertainty quantification framework (Stage 6).

Provides honest, statistically defensible uncertainty estimation for PM2.5
predictions. When data or models are insufficient, the framework reports
DEFERRED with a clear explanation — never fabricating confidence intervals.

Uncertainty states:
  - DEFERRED  : No real data or model; uncertainty cannot be estimated
  - PARTIAL   : Some data exists but insufficient for full estimation
  - ESTIMATED : Cross-validation residuals available; error scale computed
  - VALIDATED : Held-out spatial/temporal validation confirms error scale

Uncertainty sources (when data allows):
  1. Cross-validation residual spread (aleatoric + epistemic)
  2. Spatial interpolation distance (kriging or IDW)
  3. Temporal variability (day-to-day PM2.5 variance)
  4. Feature coverage gaps (missing covariates per cell)
  5. AOD cloud-masking gaps (fraction of cells without AOD)
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

UNCERTAINTY_STATES = ("DEFERRED", "PARTIAL", "ESTIMATED", "VALIDATED")


def _cv_residual_uncertainty(
    df: pd.DataFrame,
    target_col: str = "PM2.5",
    prediction_col: Optional[str] = None,
) -> dict:
    """Compute uncertainty from cross-validation residuals.

    If a prediction column exists, residuals are computed directly.
    Otherwise, returns DEFERRED with explanation.
    """
    if df.empty or target_col not in df.columns:
        return {
            "status": "DEFERRED",
            "rmse_ug_m3": None,
            "mae_ug_m3": None,
            "iqr_ug_m3": None,
            "n_residuals": 0,
            "reason": "no_target_data",
        }

    if prediction_col and prediction_col in df.columns:
        residuals = (
            df[target_col].dropna() - df[prediction_col].dropna()
        ).dropna()
        if len(residuals) < 10:
            return {
                "status": "DEFERRED",
                "rmse_ug_m3": None,
                "mae_ug_m3": None,
                "iqr_ug_m3": None,
                "n_residuals": int(len(residuals)),
                "reason": f"too_few_residuals ({len(residuals)} < 10)",
            }
        rmse = float(np.sqrt((residuals ** 2).mean()))
        mae = float(residuals.abs().mean())
        iqr = float(residuals.quantile(0.75) - residuals.quantile(0.25))
        return {
            "status": "ESTIMATED",
            "rmse_ug_m3": round(rmse, 2),
            "mae_ug_m3": round(mae, 2),
            "iqr_ug_m3": round(iqr, 2),
            "n_residuals": int(len(residuals)),
            "reason": None,
        }

    return {
        "status": "DEFERRED",
        "rmse_ug_m3": None,
        "mae_ug_m3": None,
        "iqr_ug_m3": None,
        "n_residuals": 0,
        "reason": "no_prediction_column_available",
    }


def _feature_coverage_uncertainty(df: pd.DataFrame) -> dict:
    """Uncertainty contribution from missing features per cell.

    Cells with more missing features have higher spatial uncertainty.
    """
    if df.empty:
        return {
            "status": "DEFERRED",
            "mean_missing_features_pct": None,
            "cells_with_gaps": 0,
            "total_cells": 0,
        }

    feature_cols = [
        "AOD", "temperature_c", "relative_humidity_pct",
        "wind_speed_mps", "wind_direction_deg", "NDVI",
        "elevation_m", "road_density", "night_lights",
    ]
    present_cols = [c for c in feature_cols if c in df.columns]
    if not present_cols:
        return {
            "status": "DEFERRED",
            "mean_missing_features_pct": None,
            "cells_with_gaps": 0,
            "total_cells": len(df),
        }

    missing_per_row = df[present_cols].isna().sum(axis=1)
    total_features = len(present_cols)
    missing_pct = (missing_per_row / total_features * 100).mean()
    cells_with_gaps = int((missing_per_row > 0).sum())

    return {
        "status": "ESTIMATED",
        "mean_missing_features_pct": round(float(missing_pct), 1),
        "cells_with_gaps": cells_with_gaps,
        "total_cells": len(df),
        "features_checked": present_cols,
    }


def _spatial_coverage_uncertainty(df: pd.DataFrame) -> dict:
    """Uncertainty from spatial coverage: how far is each prediction from data.

    Reports the median and max distance to nearest station for the grid.
    """
    if df.empty or "latitude" not in df.columns or "longitude" not in df.columns:
        return {
            "status": "DEFERRED",
            "median_distance_to_station_km": None,
            "max_distance_to_station_km": None,
        }

    station_cols = [c for c in ["station_lat", "station_lon"] if c in df.columns]
    if not station_cols or len(station_cols) < 2:
        return {
            "status": "DEFERRED",
            "median_distance_to_station_km": None,
            "max_distance_to_station_km": None,
            "reason": "no_station_location_columns",
        }

    try:
        from math import radians, cos, sin, asin, sqrt

        def haversine(lat1, lon1, lat2, lon2):
            lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
            return 2 * 6371 * asin(sqrt(a))

        distances = []
        for _, row in df.iterrows():
            lat, lon = row["latitude"], row["longitude"]
            if pd.isna(lat) or pd.isna(lon):
                continue
            min_dist = float("inf")
            for _, srow in df[df["station_lat"].notna()].iterrows():
                d = haversine(lat, lon, srow["station_lat"], srow["station_lon"])
                min_dist = min(min_dist, d)
            if min_dist < float("inf"):
                distances.append(min_dist)

        if not distances:
            return {
                "status": "DEFERRED",
                "median_distance_to_station_km": None,
                "max_distance_to_station_km": None,
                "reason": "no_valid_distances",
            }

        return {
            "status": "ESTIMATED",
            "median_distance_to_station_km": round(float(np.median(distances)), 1),
            "max_distance_to_station_km": round(float(max(distances)), 1),
            "n_cells": len(distances),
        }
    except Exception as exc:
        logger.debug("Spatial uncertainty computation failed: %s", exc)
        return {
            "status": "DEFERRED",
            "reason": f"computation_failed: {exc}",
        }


def build_uncertainty_report(
    df: pd.DataFrame,
    config: dict,
    prediction_col: Optional[str] = None,
    write_path: Optional[Path] = None,
) -> dict:
    """Build the comprehensive uncertainty report.

    Args:
        df: Training table with target and optional prediction columns.
        config: Project config dict.
        prediction_col: Column name with model predictions (if available).
        write_path: Optional path to write the JSON report.

    Returns:
        Uncertainty report with status, sources, and overall assessment.
    """
    # Compute per-source uncertainty
    cv_uncertainty = _cv_residual_uncertainty(df, "PM2.5", prediction_col)
    feature_uncertainty = _feature_coverage_uncertainty(df)
    spatial_uncertainty = _spatial_coverage_uncertainty(df)

    # Determine overall uncertainty state
    statuses = [
        cv_uncertainty["status"],
        feature_uncertainty["status"],
        spatial_uncertainty["status"],
    ]
    if all(s == "DEFERRED" for s in statuses):
        overall_status = "DEFERRED"
    elif all(s in ("ESTIMATED", "VALIDATED") for s in statuses):
        overall_status = "ESTIMATED"
    else:
        overall_status = "PARTIAL"

    # Compute overall uncertainty score (0-100, higher = more uncertain)
    uncertainty_score = 0.0
    n_components = 0
    if cv_uncertainty["status"] == "ESTIMATED" and cv_uncertainty.get("rmse_ug_m3"):
        # Normalize RMSE to 0-100 scale (RMSE of 100 ug/m3 = score 100)
        rmse_score = min(cv_uncertainty["rmse_ug_m3"] / 100.0, 1.0) * 100
        uncertainty_score += rmse_score
        n_components += 1
    if feature_uncertainty["status"] == "ESTIMATED":
        missing_pct = feature_uncertainty.get("mean_missing_features_pct", 0)
        uncertainty_score += missing_pct
        n_components += 1
    if spatial_uncertainty["status"] == "ESTIMATED":
        max_dist = spatial_uncertainty.get("max_distance_to_station_km", 0)
        dist_score = min(max_dist / 50.0, 1.0) * 100
        uncertainty_score += dist_score
        n_components += 1

    if n_components > 0:
        uncertainty_score = round(uncertainty_score / n_components, 1)
    else:
        uncertainty_score = None

    report = {
        "report_version": 1,
        "status": overall_status,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "uncertainty_score": uncertainty_score,
        "sources": {
            "cross_validation_residuals": cv_uncertainty,
            "feature_coverage_gaps": feature_uncertainty,
            "spatial_coverage": spatial_uncertainty,
        },
        "methodology": (
            "Uncertainty is estimated from three orthogonal sources: "
            "(1) cross-validation residual spread, "
            "(2) missing-feature gaps per cell, and "
            "(3) spatial distance to nearest station. "
            "When insufficient data exists, uncertainty is DEFERRED with a "
            "clear explanation — never fabricated."
        ),
        "data_requirements": (
            [] if overall_status == "ESTIMATED" else [
                "real PM2.5 observations (OpenAQ credentials)",
                "model predictions on held-out data",
                "spatial station registry with coordinates",
            ]
        ),
        "rule": (
            "Uncertainty is DEFERRED when data is insufficient. "
            "An RMSE would not constitute a confidence percentage. "
            "The uncertainty score is an error scale, NOT a confidence level."
        ),
    }

    if write_path:
        out = Path(write_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logger.info("Uncertainty report written to %s", out)

    return report


def uncertainty_status_for_api(report: dict) -> dict:
    """Format the uncertainty report for the API response."""
    return {
        "status": report.get("status", "DEFERRED"),
        "method": report.get("methodology", "none"),
        "reason": (
            report.get("sources", {})
            .get("cross_validation_residuals", {})
            .get("reason")
        ),
        "data_requirements": report.get("data_requirements", []),
        "future_method": report.get("methodology", ""),
        "uncertainty_score": report.get("uncertainty_score"),
        "sources": report.get("sources", {}),
    }
