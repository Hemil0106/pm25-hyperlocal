"""Global data acquisition API endpoints (Milestone 16).

Serves the M16 acquisition metadata: source registry, coverage report,
availability and overall status. These endpoints only READ already-generated
M16 artifacts (data/processed/global/) or compute availability from config +
environment; they never trigger downloads and never fabricate data.

Kept separate from the /global/* endpoints (Milestone 15), which describe the
model/scope side; /data/* describes the data-acquisition side.
"""

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

from src.config import load_config
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
    """M16 data-source registry with honest credential/availability status."""
    try:
        validate_scope(scope)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))  # noqa: B904
    return build_data_source_registry(_config(), scope=scope)


@router.get("/availability")
def get_data_availability(scope: str = Query("global", description="global | india | delhi")):
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
            "note": "No Milestone 16 acquisition has run for this scope. "
                    "Run `python run.py --global-data-only` (or --india-data-only / "
                    "--delhi-data-only) first. Sources are never assumed available.",
        }
    import json

    with open(path, "r", encoding="utf-8") as file:
        report = json.load(file)

    # Scope isolation: never serve a report built for another scope.
    if report.get("built_for_scope") != scope:
        return {
            "built_for_scope": scope,
            "status": "not_run",
            "note": f"Latest report on disk is for scope "
                    f"'{report.get('built_for_scope')}', not '{scope}'. "
                    "Re-run the acquisition for this scope before reading it.",
        }
    return report


@router.get("/coverage")
def get_data_coverage(scope: str = Query("global", description="global | india | delhi")):
    """Alias of /data/availability (explicit coverage semantics)."""
    return get_data_availability(scope)


@router.get("/status")
def get_data_status(scope: str = Query("global", description="global | india | delhi")):
    """Compact overall M16 acquisition status for the requested scope."""
    try:
        scope = validate_scope(scope)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))  # noqa: B904

    report = None
    path = _availability_path(scope)
    if path.exists():
        import json

        with open(path, "r", encoding="utf-8") as file:
            candidate = json.load(file)
        # Scope isolation: only a report built for this scope is authoritative.
        if candidate.get("built_for_scope") == scope:
            report = candidate

    if report is None:
        return {
            "scope": scope,
            "overall": "not_run",
            "available_sources": [],
            "unavailable_sources": [],
            "ml_not_implemented": True,
            "prediction_not_implemented": True,
        }

    sources = report.get("sources", {})
    available = [sid for sid, spec in sources.items() if spec.get("status") == "AVAILABLE"]
    unavailable = [sid for sid, spec in sources.items() if spec.get("status") != "AVAILABLE"]
    any_available = bool(available)
    return {
        "scope": scope,
        "overall": "available" if any_available else "unavailable",
        "available_sources": available,
        "unavailable_sources": unavailable,
        "observations": report.get("observations"),
        "synthetic_data_leakage": report.get("synthetic_data_leakage", "NONE"),
        "ml_not_implemented": True,
        "prediction_not_implemented": True,
    }
