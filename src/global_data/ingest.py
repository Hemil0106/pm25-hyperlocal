"""Global data acquisition orchestrator (Milestone 16).

Runs the whole M16 pipeline for a scope:

  ensure dirs -> source registry -> PM2.5 acquisition (if available) ->
  satellite adapters (tile-based, graceful) -> coverage report -> manifest.

No ML training, no predictions. Every source reports AVAILABLE / PARTIAL /
UNAVAILABLE honestly; no data is fabricated and no source is silently
substituted for another.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from .coverage import build_coverage_report
from .manifest import build_manifest
from .scope import validate_scope
from .sources import build_data_source_registry

logger = logging.getLogger(__name__)


def ensure_global_dirs(config) -> dict:
    """Create the M16 storage layout under data/raw/global and data/processed/global."""
    storage = config.get("global_data", {}).get("storage", {})
    raw_base = Path(storage.get("raw_base", "data/raw/global"))
    processed_base = Path(storage.get("processed_base", "data/processed/global"))

    raw_subdirs = ["pm25", "aod", "weather", "ndvi", "dem", "osm", "viirs"]
    processed_subdirs = ["stations", "pm25", "availability", "qc"]

    created = []
    for name in raw_subdirs:
        path = raw_base / name
        path.mkdir(parents=True, exist_ok=True)
        created.append(str(path))
    for name in processed_subdirs:
        path = processed_base / name
        path.mkdir(parents=True, exist_ok=True)
        created.append(str(path))

    logger.info("Global data directories ensured under %s / %s", raw_base, processed_base)
    return {"raw_base": str(raw_base), "processed_base": str(processed_base),
            "created": created}


def _adapter_acquisition(config, scope: str, date: str) -> dict:
    """Attempt every satellite adapter; each returns a status dict."""
    from .aod_global import acquire as acquire_aod
    from .dem_global import acquire as acquire_dem
    from .ndvi_global import acquire as acquire_ndvi
    from .osm_global import acquire as acquire_osm
    from .viirs_global import acquire as acquire_viirs
    from .weather_global import acquire as acquire_weather

    adapters = {
        "aod": (acquire_aod, config),
        "weather": (acquire_weather, config),
        "ndvi": (acquire_ndvi, config),
        "dem": (acquire_dem, config),
        "osm": (acquire_osm, config),
        "viirs": (acquire_viirs, config),
    }
    reports = {}
    for source_id, (acquire_fn, cfg) in adapters.items():
        try:
            reports[source_id] = acquire_fn(cfg, scope=scope, date=date)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Adapter %s raised during acquisition: %s", source_id, exc)
            reports[source_id] = {
                "source": source_id, "scope": scope, "status": "failed",
                "reason": f"Adapter error: {exc}", "tiles_completed": 0,
                "tiles_failed": 0, "artifacts": [],
            }
    return reports


def run_global_data_pipeline(config, scope: str = "global",
                             start_date: Optional[str] = None,
                             end_date: Optional[str] = None,
                             write: bool = True) -> dict:
    """Execute the M16 global data acquisition for a scope.

    Returns a summary dict; never raises on source unavailability.
    """
    scope = validate_scope(scope)
    logger.info("=" * 70)
    logger.info("GLOBAL DATA ACQUISITION (Milestone 16) - scope=%s", scope)
    logger.info("=" * 70)

    storage = ensure_global_dirs(config)
    processed_base = Path(storage["processed_base"])
    availability_dir = processed_base / "availability"
    availability_dir.mkdir(parents=True, exist_ok=True)

    # Fresh failure log per run: the manifest documents THIS acquisition run.
    failed_downloads_path = availability_dir / f"global_failed_downloads_{scope}.json"
    failed_downloads_path.write_text(json.dumps({"failures": []}),
                                     encoding="utf-8")

    source_registry = build_data_source_registry(config, scope=scope)

    # -- PM2.5 (only when creds + network allow) -------------------------
    from .pm25_global import acquire_pm25

    try:
        pm25_report = acquire_pm25(
            config, scope=scope, start_date=start_date, end_date=end_date,
            write=write, failed_downloads_path=failed_downloads_path,
        )
    except Exception as exc:
        logger.error("PM2.5 acquisition crashed: %s", exc)
        pm25_report = {
            "source": "pm25", "scope": scope, "status": "failed",
            "reason": str(exc), "observations_written": False,
        }

    # -- Satellite adapters (tile-based, graceful) -----------------------
    target_date = (start_date or config.get("time", {}).get("start_date", "2025-01-01"))[:10]
    satellite_reports = _adapter_acquisition(config, scope, target_date)

    source_reports = {"pm25": pm25_report, **satellite_reports}

    # -- Coverage report + manifest (filenames are scope-scoped so a report
    #    is never silently served for a different scope) --------------------
    coverage_path = availability_dir / f"global_data_availability_report_{scope}.json"
    coverage = build_coverage_report(config, scope, source_reports,
                                     write_path=str(coverage_path) if write else None)

    manifest_path = processed_base.parent / f"global_data_manifest_{scope}.json"
    manifest = build_manifest(config, scope, source_reports, coverage,
                              write_path=str(manifest_path) if write else None)

    summary = {
        "stage": "GLOBAL_DATA_ACQUISITION",
        "scope": scope,
        "source_registry": source_registry,
        "pm25": pm25_report,
        "satellites": satellite_reports,
        "coverage_report": coverage,
        "manifest_path": str(manifest_path) if write else None,
        "failed_downloads_path": str(failed_downloads_path),
        "ml_not_implemented": True,
        "prediction_not_implemented": True,
    }

    logger.info("=" * 70)
    logger.info("GLOBAL DATA ACQUISITION SUMMARY (scope=%s)", scope)
    for source_id, report in sorted(source_reports.items()):
        logger.info("  %-8s status=%s", source_id, report.get("status"))
    logger.info("=" * 70)
    return summary
