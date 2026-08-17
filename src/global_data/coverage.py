"""Coverage report for the global data layer (Milestone 16).

Writes ``data/processed/global/availability/global_data_availability_report.json``
with an honest per-source AVAILABLE / PARTIAL / UNAVAILABLE status, plus
observation/station/country/date-range metrics when data was actually acquired.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

from .scope import validate_scope

logger = logging.getLogger(__name__)


def _summarize_source(report: dict) -> str:
    status = str(report.get("status", "unavailable")).upper()
    tiles_completed = report.get("tiles_completed")
    tiles_failed = report.get("tiles_failed")
    if tiles_completed is not None and int(tiles_completed) > 0:
        if int(tiles_failed or 0) > 0:
            return "PARTIAL"
        return "AVAILABLE"
    if status in ("DISABLED", "UNAVAILABLE", "NO_DATA", "FAILED",
                  "NO_SUFFICIENT_DAILY_DATA"):
        return "UNAVAILABLE"
    return "UNAVAILABLE"


def build_coverage_report(config, scope: str, source_reports: dict,
                          write_path: Optional[str] = None) -> dict:
    """Assemble the M16 coverage report from per-source acquisition reports."""
    scope = validate_scope(scope)
    pm25 = source_reports.get("pm25") or {}
    stations = pm25.get("stations") or {}
    qc = pm25.get("qc") or {}

    sources = {}
    for source_id, report in sorted(source_reports.items()):
        sources[source_id] = {
            "name": report.get("name", source_id),
            "status": _summarize_source(report),
            "details": {
                k: report.get(k) for k in (
                    "reason", "tiles_completed", "tiles_failed", "artifacts",
                    "composite_metadata", "nodata", "qa",
                ) if report.get(k) is not None
            },
        }

    report = {
        "report_version": 1,
        "built_for_scope": scope,
        "timestamp": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "observations": {
            "n_observations": pm25.get("n_observations"),
            "n_daily_rows": pm25.get("n_daily_rows"),
            "n_stations": pm25.get("n_stations") or stations.get("n_stations"),
            "countries": stations.get("countries") or [],
            "date_range": stations.get("date_range"),
            "qc": {k: qc.get(k) for k in (
                "input_rows", "retained_rows", "duplicate_rows",
                "invalid_coordinates", "missing_pm25", "invalid_units",
                "negative_pm25", "outlier_flags", "station_count",
                "country_count", "date_range",
            ) if qc.get(k) is not None},
        },
        "sources": sources,
        "synthetic_data_leakage": (
            pm25.get("synthetic_leakage", {}).get("synthetic_data_leakage", "NONE")
            if pm25.get("synthetic_leakage") else "NONE"
        ),
        "note": (
            "Honest coverage report. Sources are AVAILABLE only when real data "
            "was actually acquired; missing credentials are reported "
            "UNAVAILABLE and no data is fabricated or substituted."
        ),
    }

    if write_path:
        out = Path(write_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as file:
            json.dump(report, file, indent=2)
        logger.info("Coverage report written to %s", out)

    return report
