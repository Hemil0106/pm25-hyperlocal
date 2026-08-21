"""Enhanced readiness gate (Stage 3).

The readiness gate is fully data-derived and never auto-YES. It measures:
  - Real complete-case rows, stations, countries, temporal span
  - Per-source feature coverage with configurable thresholds
  - Spatial coverage (geographic spread)
  - Temporal coverage (daily completeness)
  - Data quality flags (outlier ratio, missing ratio)
  - Data freshness (how recent the data is)
  - Confidence levels from the availability registry

All thresholds are configurable via config.yaml under
  global_training.readiness. When no config is provided, sensible defaults
  are used. With no real data, it MUST say NO / BLOCKED.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

import pandas as pd

from .schema import TARGET_COL

logger = logging.getLogger(__name__)

# Default thresholds (overridable via config.yaml).
DEFAULT_THRESHOLDS = {
    "min_complete_rows": 1000,
    "min_stations": 20,
    "min_countries": 5,
    "min_temporal_days": 30,
    "min_unique_latitudes": 3,
    "min_unique_longitudes": 3,
    "min_daily_completeness_pct": 50.0,
    "max_outlier_ratio": 0.10,
    "max_missing_target_ratio": 0.05,
    "min_feature_sources_ready": 4,
    "freshness_max_age_days": 365,
}

# Required features per source (minimum coverage threshold).
SOURCE_FEATURE_REQUIREMENTS = {
    "pm25": {
        "features": [TARGET_COL],
        "min_coverage_pct": 80.0,
    },
    "aod": {
        "features": ["AOD"],
        "min_coverage_pct": 40.0,
    },
    "weather": {
        "features": ["temperature_c", "relative_humidity_pct",
                     "wind_speed_mps", "wind_direction_deg"],
        "min_coverage_pct": 40.0,
    },
    "ndvi": {
        "features": ["NDVI"],
        "min_coverage_pct": 30.0,
    },
    "dem": {
        "features": ["elevation_m"],
        "min_coverage_pct": 50.0,
    },
    "osm": {
        "features": ["road_density"],
        "min_coverage_pct": 50.0,
    },
    "viirs": {
        "features": ["night_lights"],
        "min_coverage_pct": 30.0,
    },
}


def _load_thresholds(config: dict) -> dict:
    """Load readiness thresholds from config, falling back to defaults."""
    defaults = dict(DEFAULT_THRESHOLDS)
    cfg_thresholds = (
        config.get("global_training", {})
        .get("readiness", {})
        .get("thresholds", {})
    )
    if isinstance(cfg_thresholds, dict):
        defaults.update(cfg_thresholds)
    return defaults


def _spatial_coverage(df: pd.DataFrame, thresholds: dict) -> dict:
    """Check geographic spread of the dataset."""
    if df.empty:
        return {"pass": True, "unique_latitudes": 0, "unique_longitudes": 0,
                "reason": None}
    if "latitude" not in df.columns or "longitude" not in df.columns:
        return {"pass": True, "unique_latitudes": 0, "unique_longitudes": 0,
                "reason": None}
    lat_nunique = int(df["latitude"].nunique())
    lon_nunique = int(df["longitude"].nunique())
    min_lats = int(thresholds.get("min_unique_latitudes", 3))
    min_lons = int(thresholds.get("min_unique_longitudes", 3))
    passed = lat_nunique >= min_lats and lon_nunique >= min_lons
    reason = None if passed else (
        f"latitudes {lat_nunique} < {min_lats}" if lat_nunique < min_lats
        else f"longitudes {lon_nunique} < {min_lons}"
    )
    return {
        "pass": passed,
        "unique_latitudes": lat_nunique,
        "unique_longitudes": lon_nunique,
        "reason": reason,
    }


def _temporal_coverage(df: pd.DataFrame, thresholds: dict) -> dict:
    """Check daily completeness of the dataset."""
    if df.empty or "date" not in df.columns:
        return {"pass": True, "daily_completeness_pct": 0.0,
                "unique_dates": 0, "reason": None}
    dates = pd.to_datetime(df["date"], errors="coerce")
    valid_dates = dates.dropna()
    if valid_dates.empty:
        return {"pass": False, "daily_completeness_pct": 0.0,
                "unique_dates": 0, "reason": "all_dates_invalid"}
    unique_dates = int(valid_dates.dt.normalize().nunique())
    date_min = valid_dates.min()
    date_max = valid_dates.max()
    span_days = max(1, int((date_max - date_min).total_seconds() / 86400) + 1)
    completeness_pct = (unique_dates / span_days) * 100.0
    min_pct = float(thresholds.get("min_daily_completeness_pct", 50.0))
    passed = completeness_pct >= min_pct
    return {
        "pass": passed,
        "daily_completeness_pct": round(completeness_pct, 1),
        "unique_dates": unique_dates,
        "span_days": span_days,
        "date_range": {"min": str(date_min.date()), "max": str(date_max.date())},
        "reason": None if passed else f"completeness {completeness_pct:.1f}% < {min_pct}%",
    }


def _quality_flags(df: pd.DataFrame, thresholds: dict) -> dict:
    """Check data quality: outlier ratio, missing target ratio."""
    if df.empty:
        return {"pass": True, "outlier_ratio": 0.0, "missing_target_ratio": 0.0,
                "reason": None}
    if TARGET_COL not in df.columns:
        return {"pass": True, "outlier_ratio": 0.0, "missing_target_ratio": 0.0,
                "reason": None}
    total = len(df)
    missing_target = int(df[TARGET_COL].isna().sum()) if TARGET_COL in df.columns else total
    missing_ratio = missing_target / max(total, 1)
    max_missing = float(thresholds.get("max_missing_target_ratio", 0.05))
    missing_pass = missing_ratio <= max_missing

    outlier_ratio = 0.0
    outlier_pass = True
    if TARGET_COL in df.columns and df[TARGET_COL].notna().any():
        values = df[TARGET_COL].dropna()
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        if iqr > 0:
            lower = q1 - 3.0 * iqr
            upper = q3 + 3.0 * iqr
            n_outliers = int(((values < lower) | (values > upper)).sum())
            outlier_ratio = n_outliers / len(values)
            max_outlier = float(thresholds.get("max_outlier_ratio", 0.10))
            outlier_pass = outlier_ratio <= max_outlier

    passed = missing_pass and outlier_pass
    reason = None
    if not missing_pass:
        reason = f"missing_target_ratio {missing_ratio:.2%} > {max_missing:.2%}"
    elif not outlier_pass:
        reason = f"outlier_ratio {outlier_ratio:.2%} > {thresholds.get('max_outlier_ratio', 0.10):.2%}"

    return {
        "pass": passed,
        "outlier_ratio": round(outlier_ratio, 4),
        "missing_target_ratio": round(missing_ratio, 4),
        "reason": reason,
    }


def _source_coverage(df: pd.DataFrame, thresholds: dict) -> dict:
    """Check per-source feature coverage with configurable thresholds."""
    if df.empty:
        return {src: {"pass": False, "coverage_pct": 0.0, "reason": "empty_dataframe"}
                for src in SOURCE_FEATURE_REQUIREMENTS}

    total = len(df)
    results = {}
    for source_id, req in SOURCE_FEATURE_REQUIREMENTS.items():
        features = req["features"]
        min_pct = float(req.get("min_coverage_pct", 40.0))
        present_features = [f for f in features if f in df.columns]
        if not present_features:
            results[source_id] = {
                "pass": False, "coverage_pct": 0.0,
                "reason": f"features {features} not in dataframe",
                "features_checked": features,
                "features_present": [],
            }
            continue
        rows_with_all = df[present_features].notna().all(axis=1).sum()
        coverage_pct = (rows_with_all / total) * 100.0
        passed = coverage_pct >= min_pct
        results[source_id] = {
            "pass": passed,
            "coverage_pct": round(coverage_pct, 1),
            "min_required_pct": min_pct,
            "features_checked": features,
            "features_present": present_features,
            "rows_with_all_features": int(rows_with_all),
            "reason": None if passed else (
                f"coverage {coverage_pct:.1f}% < {min_pct}% for {source_id}"
            ),
        }
    return results


def build_readiness(df: pd.DataFrame, config: dict,
                    availability_registry: Optional[dict] = None) -> dict:
    """Compute the enhanced data-derived readiness report.

    Args:
        df: Training table with complete-case column and all features.
        config: Project config dict.
        availability_registry: Optional production-grade availability registry
            (from Stage 1). Used for freshness and confidence checks.

    Returns:
        Comprehensive readiness report with per-check pass/fail.
    """
    thresholds = _load_thresholds(config)
    scope = config.get("global_data", {}).get("scope", "global")

    # --- Core metrics ---
    n_complete = int(df["complete_case"].sum()) if not df.empty and "complete_case" in df.columns else 0
    n_stations = int(df["station_id"].nunique()) if not df.empty else 0
    n_countries = int(df["country"].dropna().nunique()) if not df.empty else 0

    temporal_days = 0
    date_range = None
    if not df.empty and "date" in df.columns:
        dates = pd.to_datetime(df["date"], errors="coerce")
        valid_dates = dates.dropna()
        if not valid_dates.empty:
            temporal_days = int(valid_dates.dt.normalize().nunique())
            date_range = {"min": str(valid_dates.min().date()),
                          "max": str(valid_dates.max().date())}

    # --- Per-check results ---
    min_rows = int(thresholds["min_complete_rows"])
    min_stations = int(thresholds["min_stations"])
    min_countries = int(thresholds["min_countries"])
    min_days = int(thresholds["min_temporal_days"])

    checks = {
        "complete_case_rows": {
            "pass": n_complete >= min_rows,
            "measured": n_complete,
            "threshold": min_rows,
            "reason": None if n_complete >= min_rows else f"{n_complete} < {min_rows}",
        },
        "stations": {
            "pass": n_stations >= min_stations,
            "measured": n_stations,
            "threshold": min_stations,
            "reason": None if n_stations >= min_stations else f"{n_stations} < {min_stations}",
        },
        "countries": {
            "pass": n_countries >= min_countries,
            "measured": n_countries,
            "threshold": min_countries,
            "reason": None if n_countries >= min_countries else f"{n_countries} < {min_countries}",
        },
        "temporal_days": {
            "pass": temporal_days >= min_days,
            "measured": temporal_days,
            "threshold": min_days,
            "date_range": date_range,
            "reason": None if temporal_days >= min_days else f"{temporal_days} < {min_days}",
        },
        "spatial_coverage": _spatial_coverage(df, thresholds),
        "temporal_coverage": _temporal_coverage(df, thresholds),
        "quality_flags": _quality_flags(df, thresholds),
        "source_coverage": _source_coverage(df, thresholds),
    }

    # --- Feature sources readiness ---
    # ALL sources must pass for model training readiness.
    sources_ready = sum(
        1 for src, res in checks["source_coverage"].items()
        if res.get("pass", False)
    )
    total_sources = len(checks["source_coverage"])
    failed_sources = [
        src for src, res in checks["source_coverage"].items()
        if not res.get("pass", False)
    ]
    checks["feature_sources_ready"] = {
        "pass": sources_ready == total_sources,
        "measured": sources_ready,
        "threshold": total_sources,
        "reason": None if sources_ready == total_sources else (
            f"insufficient coverage for: {', '.join(failed_sources)}"
        ),
    }

    # --- Freshness check (from availability registry) ---
    if availability_registry is not None:
        max_age_days = int(thresholds.get("freshness_max_age_days", 365))
        pm25_entry = availability_registry.get("sources", {}).get("pm25", {})
        freshness = pm25_entry.get("freshness", {})
        age_s = freshness.get("age_s")
        age_days = (age_s / 86400) if age_s is not None else None
        fresh = freshness.get("fresh", False)
        checks["data_freshness"] = {
            "pass": fresh or age_days is None,
            "age_days": round(age_days, 1) if age_days is not None else None,
            "max_age_days": max_age_days,
            "reason": None if fresh or age_days is None else (
                f"pm25 age {age_days:.0f}d exceeds {max_age_days}d"
            ),
        }
    else:
        checks["data_freshness"] = {
            "pass": True,
            "reason": "availability_registry_not_provided_assumed_fresh",
        }

    # --- Overall verdict ---
    all_pass = all(
        c.get("pass", False) for c in checks.values()
        if isinstance(c, dict) and "pass" in c
    )
    reasons = []
    for check_name, check in checks.items():
        if isinstance(check, dict) and not check.get("pass", True):
            reasons.append(check.get("reason", f"{check_name} failed"))

    report = {
        "report_version": 2,
        "scope": scope,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "thresholds": thresholds,
        "measured": {
            "complete_case_rows": n_complete,
            "stations": n_stations,
            "countries": n_countries,
            "temporal_days": temporal_days,
            "date_range": date_range,
            "total_rows": len(df) if not df.empty else 0,
        },
        "checks": checks,
        "model_training_ready": "YES" if all_pass else "NO",
        "readiness_score": round(
            sum(1 for c in checks.values()
                if isinstance(c, dict) and c.get("pass", False))
            / max(len(checks), 1) * 100, 1
        ),
        "reason": None if all_pass else "; ".join(reasons),
        "rule": (
            "Readiness is fully data-derived and never auto-YES. "
            "All checks must pass for model training to proceed."
        ),
    }

    # --- Write report to disk ---
    processed_base = Path(
        config.get("global_data", {}).get("storage", {}).get(
            "processed_base", "data/processed/global")
    )
    reports_dir = processed_base / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / "global_training_readiness.json"

    class _SafeEncoder(json.JSONEncoder):
        def default(self, o):
            import numpy as _np
            if isinstance(o, (_np.bool_,)):
                return bool(o)
            if isinstance(o, (_np.integer,)):
                return int(o)
            if isinstance(o, (_np.floating,)):
                return float(o)
            return super().default(o)

    path.write_text(json.dumps(report, indent=2, cls=_SafeEncoder), encoding="utf-8")
    report["path"] = str(path)

    logger.info(
        "Readiness gate: %s (score=%.1f%%, reason=%s)",
        report["model_training_ready"],
        report["readiness_score"],
        report["reason"] or "all checks pass",
    )
    return report
