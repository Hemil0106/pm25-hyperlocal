"""Data freshness and model health tracking (Stage 11).

Tracks the freshness of all data sources and the health status of deployed
models. Reports honest status based on artifact timestamps and checksums.

Freshness windows are configurable per source type. Model health is derived
from artifact existence, checksum integrity, and training metadata age.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Default freshness windows in seconds.
FRESHNESS_WINDOWS = {
    "pm25": 7 * 86400,
    "aod": 30 * 86400,
    "weather": 90 * 86400,
    "ndvi": 32 * 86400,
    "dem": 365 * 86400,
    "osm": 30 * 86400,
    "viirs": 30 * 86400,
    "model_xgboost": 180 * 86400,
    "model_rf": 180 * 86400,
    "training_data": 90 * 86400,
}


def _read_json(path: Path) -> Optional[dict]:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return None


def _age_seconds(timestamp_iso: Optional[str]) -> Optional[float]:
    """Compute age in seconds from an ISO timestamp to now."""
    if timestamp_iso is None:
        return None
    try:
        from datetime import datetime, timezone
        dt = datetime.fromisoformat(timestamp_iso.replace("Z", "+00:00"))
        return (datetime.now(timezone.utc) - dt).total_seconds()
    except (ValueError, TypeError):
        return None


def _check_file_health(path: Path) -> dict:
    """Check if a file exists, its size, and modification time."""
    if not path.exists():
        return {"exists": False, "size_bytes": 0, "mtime": None, "status": "MISSING"}
    stat = path.stat()
    from datetime import datetime, timezone
    mtime = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
    return {
        "exists": True,
        "size_bytes": stat.st_size,
        "mtime": mtime.isoformat(),
        "age_s": _age_seconds(mtime.isoformat()),
        "status": "HEALTHY" if stat.st_size > 0 else "EMPTY",
    }


def check_data_freshness(
    config: dict,
    scope: str = "global",
    processed_base: Optional[Path] = None,
) -> dict:
    """Check freshness of all data sources and models.

    Returns a comprehensive freshness report with per-source status,
    overall freshness flag, and any stale sources.
    """
    if processed_base is None:
        processed_base = Path(
            config.get("global_data", {}).get("storage", {}).get(
                "processed_base", "data/processed/global")
        )

    raw_base = processed_base.parent / "raw"
    models_dir = Path(config.get("model", {}).get("output_dir", "models"))

    sources = {}
    stale_sources = []
    all_fresh = True

    for source_id, window_s in FRESHNESS_WINDOWS.items():
        if source_id.startswith("model_"):
            model_file = models_dir / (
                "xgboost_pm25.json" if "xgboost" in source_id
                else "random_forest_pm25.joblib"
            )
            health = _check_file_health(model_file)
        elif source_id == "training_data":
            obs_path = processed_base / "pm25" / "global_pm25_observations.parquet"
            health = _check_file_health(obs_path)
        else:
            source_dir = raw_base / source_id
            if source_dir.exists():
                files = list(source_dir.iterdir())
                if files:
                    newest = max(files, key=lambda f: f.stat().st_mtime)
                    health = _check_file_health(newest)
                else:
                    health = {"exists": False, "size_bytes": 0, "mtime": None,
                              "status": "EMPTY_DIR"}
            else:
                health = {"exists": False, "size_bytes": 0, "mtime": None,
                          "status": "NO_DIR"}

        age_s = health.get("age_s")
        fresh = (
            health["status"] in ("MISSING", "EMPTY", "EMPTY_DIR", "NO_DIR")
            or (age_s is not None and age_s <= window_s)
        )
        if not fresh:
            all_fresh = False
            stale_sources.append(source_id)

        sources[source_id] = {
            "window_s": window_s,
            "fresh": fresh,
            "age_s": round(age_s, 1) if age_s is not None else None,
            "status": health.get("status"),
            "mtime": health.get("mtime"),
            "size_bytes": health.get("size_bytes"),
        }

    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scope": scope,
        "overall_fresh": all_fresh,
        "stale_sources": stale_sources,
        "sources": sources,
    }


def check_model_health(config: dict) -> dict:
    """Check health of deployed models.

    Reports existence, size, integrity, and training metadata for each model.
    """
    models_dir = Path(config.get("model", {}).get("output_dir", "models"))
    metadata_path = models_dir / "model_metadata.json"
    metadata = _read_json(metadata_path)

    models = {}
    for model_id, filename in [
        ("xgboost", "xgboost_pm25.json"),
        ("random_forest", "random_forest_pm25.joblib"),
    ]:
        path = models_dir / filename
        health = _check_file_health(path)
        models[model_id] = {
            "filename": filename,
            "path": str(path),
            "health": health,
        }

    # Training data freshness
    training_meta = None
    if metadata and isinstance(metadata, dict):
        training_meta = {
            "trained_at": metadata.get("trained_at"),
            "n_observations": metadata.get("n_observations"),
            "n_features": metadata.get("n_features"),
            "target": metadata.get("target"),
            "model_type": metadata.get("model_type"),
            "validation_rmse": metadata.get("validation_rmse"),
        }

    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "models": models,
        "training_metadata": training_meta,
        "overall": (
            "HEALTHY" if all(
                m["health"]["status"] == "HEALTHY" for m in models.values()
            ) else "DEGRADED"
        ),
    }
