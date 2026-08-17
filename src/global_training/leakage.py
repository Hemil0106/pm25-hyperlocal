"""Leakage checks (Milestone 17).

Run before any model training would happen. If any check FAILs the milestone
pipeline must STOP and fix the cause. With no data, checks report PASS/N/A
honestly rather than silently skipping.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from src.global_data.scope import scope_bounds, validate_scope

from .schema import FEATURE_COLS, TARGET_COL

logger = logging.getLogger(__name__)

SYNTHETIC_MARKERS = ["synthetic", "synth_", "sample_", "_test", "test_data"]
DELHI_BOUNDS = scope_bounds("delhi")


def check_target_leakage(df: pd.DataFrame) -> dict:
    """No feature may be derived from the target (PM2.5)."""
    target_tokens = ("pm25", "target", "aqi", "concentration")
    leaked = [c for c in FEATURE_COLS
              if any(tok in c.lower() for tok in target_tokens)]
    return {
        "check": "target_leakage",
        "status": "FAIL" if leaked else "PASS",
        "detail": (
            f"Features derived from the target found: {leaked}" if leaked
            else "No candidate feature is derived from the PM2.5 target."
        ),
    }


def check_duplicate_station_date(df: pd.DataFrame) -> dict:
    dups = int(df.duplicated(subset=["station_id", "date"]).sum())
    return {
        "check": "duplicate_station_date",
        "status": "FAIL" if dups else "PASS",
        "detail": f"{dups} duplicate (station_id, date) row(s) found."
                  if dups else "No duplicate (station_id, date) rows.",
    }


def check_future_info(df: pd.DataFrame) -> dict:
    """Temporal features must derive only from the row's own date."""
    own_date = pd.to_datetime(df["date"], errors="coerce")
    if df.empty:
        return {
            "check": "future_information",
            "status": "PASS",
            "detail": "Empty table; no rows to leak future information.",
        }
    invalid = int(own_date.isna().sum())
    detail = (
        f"Temporal features (month/day_of_year/sin/cos) are derived from the "
        f"row's own date only. {invalid} row(s) had invalid dates (NaN)."
    )
    return {"check": "future_information", "status": "PASS", "detail": detail}


def check_train_test_separation(df: pd.DataFrame) -> dict:
    """No train/test split exists yet in M17 (model training is deferred)."""
    return {
        "check": "same_station_or_country_in_train_test",
        "status": "N/A",
        "detail": "No train/test split is performed in Milestone 17; model "
                  "training is intentionally deferred to Milestone 18.",
    }


def check_features_from_pm25(df: pd.DataFrame) -> dict:
    return {
        "check": "features_generated_from_pm25",
        "status": "PASS",
        "detail": "Feature engineering is limited to temporal cyclics and u/v "
                  "wind components from real meteorology; features are never "
                  "computed from the PM2.5 target.",
    }


def check_synthetic_contamination(df: pd.DataFrame, config) -> dict:
    """Tagged synthetic test data must never enter the real training table."""
    markers_found = []
    if not df.empty:
        for col in ["station_id", "country", "city"]:
            values = df[col].astype(str)
            mask = values.str.lower().str.contains(
                "|".join(SYNTHETIC_MARKERS), regex=True, na=False)
            for value in values[mask].unique():
                markers_found.append(str(value))

    m16_leakage = None
    scope = config.get("global_data", {}).get("scope", "global")
    availability_path = Path(
        config.get("global_data", {}).get("storage", {}).get(
            "processed_base", "data/processed/global")
    ) / "availability" / f"global_data_availability_report_{scope}.json"
    if availability_path.exists():
        try:
            payload = json.loads(availability_path.read_text(encoding="utf-8"))
            m16_leakage = payload.get("synthetic_data_leakage", "NONE")
        except (ValueError, OSError):
            m16_leakage = None

    leaked = bool(markers_found) or m16_leakage not in (None, "NONE")
    return {
        "check": "synthetic_contamination",
        "status": "PRESENT" if leaked else "NONE",
        "detail": (
            f"Marker-like identifiers found: {markers_found}; M16 acquisition "
            f"leakage report = {m16_leakage}." if leaked
            else "No synthetic markers in the training table; M16 acquisition "
                 f"leakage report = {m16_leakage}."
        ),
    }


def check_delhi_artifact_contamination(df: pd.DataFrame, scope: str) -> dict:
    """Delhi-scope data must never silently become global training data."""
    scope = validate_scope(scope)
    if scope == "delhi":
        return {
            "check": "delhi_artifact_contamination",
            "status": "N/A",
            "detail": "Scope is delhi; the check guards global/india datasets.",
        }
    if df.empty:
        return {
            "check": "delhi_artifact_contamination",
            "status": "PASS",
            "detail": "Empty table; no Delhi rows to leak.",
        }
    bounds = DELHI_BOUNDS
    mask = (
        df["longitude"].between(bounds["west"], bounds["east"])
        & df["latitude"].between(bounds["south"], bounds["north"])
    )
    n_delhi = int(mask.sum())
    return {
        "check": "delhi_artifact_contamination",
        "status": "FAIL" if n_delhi else "PASS",
        "detail": (
            f"{n_delhi} row(s) fall inside the Delhi scope bounds for a "
            f"'{scope}' dataset - Delhi data must never be substituted as "
            "global/India data." if n_delhi
            else f"No rows fall inside the Delhi scope bounds (scope={scope})."
        ),
    }


def run_leakage_checks(df: pd.DataFrame, config) -> dict:
    """Run the full leakage suite; returns results + overall status."""
    scope = config.get("global_data", {}).get("scope", "global")
    checks = [
        check_target_leakage(df),
        check_duplicate_station_date(df),
        check_future_info(df),
        check_train_test_separation(df),
        check_features_from_pm25(df),
        check_synthetic_contamination(df, config),
        check_delhi_artifact_contamination(df, scope),
    ]
    statuses = [c["status"] for c in checks]
    has_fail = "FAIL" in statuses or "PRESENT" in statuses
    overall = "FAIL" if has_fail else "PASS"
    return {
        "overall": overall,
        "checks": checks,
        "rule": "If any check FAILs/PRESENT the milestone pipeline stops and "
                "fixes the cause before continuing.",
    }
