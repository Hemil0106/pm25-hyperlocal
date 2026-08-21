"""Global data acquisition API endpoints.

Serves data acquisition metadata: source registry, production-grade
availability, coverage reports, and overall status. These endpoints only READ
already-generated artifacts or compute availability from config + environment;
they never trigger downloads and never fabricate data.

Status codes: AVAILABLE | PARTIAL | UNAVAILABLE | FAILED | STALE
"""

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

from src.config import load_config
from src.global_data.availability import build_availability_registry
from src.global_data.coverage import build_coverage_report
from src.global_data.scope import SUPPORTED_SCOPES, validate_scope
from src.global_data.sources import build_data_source_registry

logger = logging.getLogger("pm25-mapping-global-data")

router = APIRouter(prefix="/data", tags=["global-data"])

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _config():
    return load_config()


def _availability_path(scope: str) -> Path:
    scope = validate_scope(scope)
    base = Path(_config().get("global_data", {}).get("storage", {}).get(
        "processed_base", "data/processed/global"))
    return PROJECT_ROOT / base / "availability" / f"global_data_availability_report_{scope}.json"


@router.get("/sources")
def get_data_sources(scope: str = Query("global", description="global | india | delhi")):
    """Legacy M16 data-source registry with credential/availability status."""
    try:
        validate_scope(scope)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))  # noqa: B904
    return build_data_source_registry(_config(), scope=scope)


@router.get("/availability")
def get_data_availability(scope: str = Query("global", description="global | india | delhi")):
    """Production-grade availability registry with status, freshness, confidence.

    Returns the full availability registry including:
      - Canonical status codes: AVAILABLE | PARTIAL | UNAVAILABLE | FAILED | STALE
      - Per-source staleness tracking (age_s, fresh flag, window_s)
      - Confidence flags (HIGH | MEDIUM | LOW | NONE)
      - Artifact checksums
      - Readiness summary
    """
    try:
        scope = validate_scope(scope)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))  # noqa: B904
    return build_availability_registry(
        _config(),
        scope=scope,
        processed_base=PROJECT_ROOT / _config().get("global_data", {}).get(
            "storage", {}).get("processed_base", "data/processed/global"),
    )


@router.get("/coverage")
def get_data_coverage(scope: str = Query("global", description="global | india | delhi")):
    """Coverage report for the requested scope (read from the last run)."""
    try:
        scope = validate_scope(scope)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))  # noqa: B904
    path = _availability_path(scope)
    if not path.exists():
        return {
            "built_for_scope": scope,
            "status": "not_run",
            "note": "No acquisition has run for this scope. "
                    "Run `python run.py --global-data-only` first. "
                    "Sources are never assumed available.",
        }
    import json

    with open(path, "r", encoding="utf-8") as file:
        report = json.load(file)

    if report.get("built_for_scope") != scope:
        return {
            "built_for_scope": scope,
            "status": "not_run",
            "note": f"Latest report on disk is for scope "
                    f"'{report.get('built_for_scope')}', not '{scope}'. "
                    "Re-run the acquisition for this scope before reading it.",
        }
    return report


@router.get("/status")
def get_data_status(scope: str = Query("global", description="global | india | delhi")):
    """Compact overall acquisition status for the requested scope."""
    try:
        scope = validate_scope(scope)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))  # noqa: B904

    registry = build_availability_registry(
        _config(),
        scope=scope,
        processed_base=PROJECT_ROOT / _config().get("global_data", {}).get(
            "storage", {}).get("processed_base", "data/processed/global"),
    )

    sources = registry.get("sources", {})
    available = [sid for sid, s in sources.items() if s.get("status") == "AVAILABLE"]
    stale = [sid for sid, s in sources.items() if s.get("status") == "STALE"]
    unavailable = [
        sid for sid, s in sources.items()
        if s.get("status") in ("UNAVAILABLE", "FAILED")
    ]

    return {
        "scope": scope,
        "overall": registry.get("overall_status", "UNAVAILABLE"),
        "available_sources": available,
        "stale_sources": stale,
        "unavailable_sources": unavailable,
        "readiness_summary": registry.get("readiness_summary"),
        "ml_not_implemented": True,
        "prediction_not_implemented": True,
    }
