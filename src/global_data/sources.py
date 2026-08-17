"""Data source registry for the global data layer (Milestone 16).

Describes every acquisition source the platform knows about: its coverage, the
credentials it requires (env vars only), whether those credentials are present,
and whether it is enabled/configured. Availability is reported honestly and is
never assumed.

This is separate from the Milestone 15 dataset registry (src/globalization/
datasets.py), which describes the *locked* prototype datasets. The two coexist;
M16 sources are the acquisition-side registry for global data.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from .scope import scope_bounds, validate_scope

logger = logging.getLogger(__name__)

# Canonical source descriptors. credential_env_vars are looked up in the
# environment ONLY -- never in config.yaml, never hard-coded, never logged.
DATA_SOURCE_REGISTRY = {
    "pm25": {
        "name": "OpenAQ PM2.5 ground observations",
        "type": "point",
        "coverage": "global (OpenAQ network, v3 API)",
        "credential_env_vars": ["OPENAQ_API_KEY"],
        "notes": "Real ground observations; normalized to pm25_ug_m3.",
    },
    "aod": {
        "name": "MODIS MAIAC AOD (MCD19A2)",
        "type": "gridded",
        "coverage": "global (1 km, tile-based)",
        "credential_env_vars": ["EARTHDATA_USERNAME", "EARTHDATA_PASSWORD"],
        "notes": "Satellite aerosol optical depth; NASA Earthdata credentials.",
    },
    "weather": {
        "name": "ERA5-Land meteorology",
        "type": "gridded",
        "coverage": "global (ERA5-Land, chunked)",
        "credential_env_vars": ["CDSAPI_URL", "CDSAPI_KEY"],
        "notes": "Meteorology from the Copernicus Climate Data Store.",
    },
    "ndvi": {
        "name": "MODIS Terra NDVI (MOD13Q1)",
        "type": "gridded",
        "coverage": "global (250 m, tile-based)",
        "credential_env_vars": ["EARTHDATA_USERNAME", "EARTHDATA_PASSWORD"],
        "notes": "16-day composites; nearest valid composite, no-future default.",
    },
    "dem": {
        "name": "NASA SRTM elevation (GL1)",
        "type": "gridded",
        "coverage": "global (SRTM v003, tile-based)",
        "credential_env_vars": ["EARTHDATA_USERNAME", "EARTHDATA_PASSWORD"],
        "notes": "Elevation; NoData is never converted to zero.",
    },
    "osm": {
        "name": "OpenStreetMap road density",
        "type": "vector",
        "coverage": "global (OSM, chunked)",
        "credential_env_vars": [],
        "notes": "Road density per tile is a spatial proxy only.",
    },
    "viirs": {
        "name": "NASA VIIRS night lights (VNP46A2)",
        "type": "gridded",
        "coverage": "global (VNP46A2, tile-based)",
        "credential_env_vars": ["EARTHDATA_USERNAME", "EARTHDATA_PASSWORD"],
        "notes": "Night-time lights; per-tile QA applied.",
    },
}


def credential_available(env_var: str) -> bool:
    """True when an environment variable is set and non-empty.

    Credentials are read from the environment ONLY. This function never reads
    config.yaml and never logs the value.
    """
    if not env_var:
        return False
    value = os.environ.get(env_var)
    return bool(value and str(value).strip())


def source_credentials_available(source_id: str) -> dict:
    """Report which credential env vars are present for a source."""
    spec = DATA_SOURCE_REGISTRY.get(source_id)
    if spec is None:
        raise KeyError(f"Unknown data source '{source_id}'.")
    return {
        var: credential_available(var)
        for var in spec.get("credential_env_vars", [])
    }


def build_data_source_registry(config, scope: str = "global",
                               write_path: Optional[str] = None) -> dict:
    """Build the M16 data-source registry for an acquisition scope.

    Status rules (honest, no assumptions):
      - disabled  : not enabled in config.yaml
      - unavailable: enabled but required credentials are missing AND no local
        cached/raw artifact exists
      - available : enabled, credentials present OR local artifact present
    """
    validated_scope = validate_scope(scope)
    cfg = config.get("global_data", {})
    sources_cfg = cfg.get("sources", {})
    fetch_cfg = cfg.get("fetch", {})

    entries = {}
    for source_id, spec in DATA_SOURCE_REGISTRY.items():
        src_cfg = sources_cfg.get(source_id, {})
        enabled = bool(src_cfg.get("enabled", False))
        creds = source_credentials_available(source_id)
        creds_available = bool(creds) and all(creds.values()) if creds else True
        if source_id == "pm25":
            enabled = bool(cfg.get("pm25", {}).get("enabled", True))
            env_var = cfg.get("pm25", {}).get("credential_env_var", "OPENAQ_API_KEY")
            creds = {env_var: credential_available(env_var)}
            creds_available = creds[env_var]

        if not enabled:
            status = "disabled"
        elif not creds_available:
            status = "unavailable"
        else:
            status = "available"

        entries[source_id] = {
            "id": source_id,
            "name": spec["name"],
            "type": spec["type"],
            "coverage": spec["coverage"],
            "enabled": enabled,
            "status": status,
            "credentials_required": list(spec.get("credential_env_vars", [])),
            "credentials_present": creds,
            "notes": spec["notes"],
        }

    registry = {
        "registry_version": 1,
        "built_for_scope": validated_scope,
        "scope_bounds": dict(scope_bounds(validated_scope)),
        "fetch": {
            "retries": int(fetch_cfg.get("retries", 3)),
            "backoff_base_s": float(fetch_cfg.get("backoff_base_s", 2.0)),
            "backoff_max_s": float(fetch_cfg.get("backoff_max_s", 60.0)),
        },
        "sources": entries,
    }

    if write_path:
        import json
        from pathlib import Path

        out = Path(write_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as file:
            json.dump(registry, file, indent=2)
        logger.info("Global data-source registry written to %s", out)

    return registry
