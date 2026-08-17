"""Target variable report (Milestone 17).

Statistical summary of the real PM2.5 target plus distributions by country,
station, and date. No fabricated thresholds are introduced - the completeness
flag only reflects how many real complete-case rows exist.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from .schema import TARGET_COL

logger = logging.getLogger(__name__)

MIN_COMPLETE_ROWS_FOR_MODELING = 20


def target_report(df: pd.DataFrame, config) -> dict:
    """Build the honest target report."""
    report = {
        "target": TARGET_COL,
        "rows": int(len(df)),
        "n_target_valid": int(df[TARGET_COL].notna().sum()) if not df.empty else 0,
        "n_complete_case_rows": int(df["complete_case"].sum()) if not df.empty else 0,
        "complete_case_ratio": (
            round(float(df["complete_case"].mean()), 4) if not df.empty else 0.0
        ),
    }

    valid = df[df[TARGET_COL].notna()][TARGET_COL].astype(float) if not df.empty else pd.Series(dtype=float)
    if valid.empty:
        report["distribution"] = None
        report["completeness_flag"] = "insufficient"
        report["reason"] = "No valid PM2.5 target values; no thresholds invented."
        report["min_complete_rows_for_modeling"] = MIN_COMPLETE_ROWS_FOR_MODELING
        _write_target_report(report, config)
        return report

    report["distribution"] = {
        "count": int(valid.count()),
        "mean": round(float(valid.mean()), 4),
        "median": round(float(valid.median()), 4),
        "min": round(float(valid.min()), 4),
        "max": round(float(valid.max()), 4),
        "std": round(float(valid.std()), 4),
        "units": "ug/m3",
    }

    by_country = (
        df[df[TARGET_COL].notna()].groupby(
            df["country"].fillna("(unknown)"))[TARGET_COL]
        .agg(count="count", mean="mean").round(4)
        .to_dict(orient="index")
        if not df.empty else {}
    )
    report["by_country"] = by_country
    report["by_station"] = (
        df[df[TARGET_COL].notna()].groupby("station_id")[TARGET_COL]
        .agg(count="count", mean="mean").round(4)
        .to_dict(orient="index")
        if not df.empty else {}
    )
    dates = pd.to_datetime(df["date"], errors="coerce")
    report["date_range"] = {
        "min": str(dates.min().date()) if not dates.isna().all() else None,
        "max": str(dates.max().date()) if not dates.isna().all() else None,
        "n_days": int(dates.dt.normalize().nunique()) if not dates.isna().all() else 0,
    }

    sufficient = report["n_complete_case_rows"] >= MIN_COMPLETE_ROWS_FOR_MODELING
    report["completeness_flag"] = "sufficient" if sufficient else "insufficient"
    report["min_complete_rows_for_modeling"] = MIN_COMPLETE_ROWS_FOR_MODELING
    report["reason"] = (
        "At least the configured minimum complete-case rows exist."
        if sufficient else
        "Too few complete-case rows to claim a trainable global target."
    )
    _write_target_report(report, config)
    return report


def _write_target_report(report: dict, config) -> None:
    processed_base = Path(
        config.get("global_data", {}).get("storage", {}).get(
            "processed_base", "data/processed/global")
    )
    reports_dir = processed_base / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / "global_target_report.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["path"] = str(path)
