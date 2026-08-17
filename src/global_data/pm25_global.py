"""Global PM2.5 ground-observation adapter: OpenAQ v3 (Milestone 16).

Acquires real PM2.5 observations from the OpenAQ v3 API. v2 is retired (HTTP
410), so v3 is the only real path. v3 requires an API key which MUST come from
the environment (``OPENAQ_API_KEY``). When the key is absent the source is
reported UNAVAILABLE - never fabricated and never substituted with Delhi data.

The pipeline:
  fetch raw rows (paginated, retried) -> normalize to observation schema ->
  unit-normalize to pm25_ug_m3 -> QC -> daily aggregation -> station outputs.
"""

from __future__ import annotations

import logging
import os
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from .cache import DownloadError, fetch_with_retry, record_failed_download
from .integrity import sha256_file, synthetic_leakage_report
from .qc import apply_qc
from .scope import scope_bounds, validate_scope
from .sources import credential_available
from .stations import write_station_outputs
from .temporal import aggregate_daily
from .units import normalize_pm25_units

logger = logging.getLogger(__name__)

OBSERVATION_SCHEMA = [
    "station_id", "source", "country", "latitude", "longitude",
    "timestamp", "PM2.5", "units", "quality_flag",
]


class CredentialsUnavailable(RuntimeError):
    """Raised when a required credential env var is missing."""


def _config_of(config) -> dict:
    return config.get("global_data", {}).get("pm25", {})


def check_availability(config) -> dict:
    """Honest availability check for the OpenAQ PM2.5 source."""
    cfg = _config_of(config)
    enabled = bool(cfg.get("enabled", True))
    env_var = cfg.get("credential_env_var", "OPENAQ_API_KEY")
    present = credential_available(env_var)
    if not enabled:
        status = "disabled"
    elif not present:
        status = "unavailable"
    else:
        status = "available"
    return {
        "source": "pm25",
        "name": "OpenAQ PM2.5 ground observations",
        "api_version": cfg.get("api_version", "v3"),
        "enabled": enabled,
        "credential_env_var": env_var,
        "credential_present": present,
        "status": status,
        "reason": None if status == "available" else (
            "OpenAQ v3 requires an API key; set OPENAQ_API_KEY (environment only)."
            if status == "unavailable" else "OpenAQ PM2.5 source is disabled."
        ),
    }


def _normalize_row(row: dict) -> Optional[dict]:
    """Map one OpenAQ v3 measurement result to the observation schema."""
    try:
        value = row.get("value")
        if value is None:
            return None
        coords = row.get("coordinates") or {}
        latitude = coords.get("latitude")
        longitude = coords.get("longitude")
        if latitude is None or longitude is None:
            return None
        station = row.get("location") or {}
        station_id = station.get("id") if isinstance(station, dict) else None
        if station_id is None:
            station_id = row.get("location_id")
        if station_id is None:
            return None
        return {
            "station_id": str(station_id),
            "source": "openaq",
            "country": row.get("country") or station.get("country") or "",
            "latitude": float(latitude),
            "longitude": float(longitude),
            "timestamp": row.get("datetime") or row.get("period", {}).get("datetimeFrom"),
            "PM2.5": float(value),
            "units": row.get("unit") or "ug/m3",
            "quality_flag": "openaq_reported",
        }
    except (TypeError, ValueError):
        return None


def _fetch_page(base_url: str, api_key: str, params: dict, attempt: int):
    """Fetch a single OpenAQ v3 measurements page (raises on HTTP error)."""
    import requests

    url = f"{base_url}/measurements"
    headers = {
        "X-API-Key": api_key,
        "User-Agent": "pm25-hyperlocal-m16/0.1",
        "Accept": "application/json",
    }
    timeout = 30.0
    response = requests.get(url, params=params, headers=headers, timeout=timeout)
    if response.status_code == 401:
        raise CredentialsUnavailable("OpenAQ API key rejected (401 Unauthorized).")
    response.raise_for_status()
    return response.json()


def fetch_openaq_pm25(config, start_date: str, end_date: str,
                      bounds: Optional[dict] = None,
                      max_pages: int = 50) -> pd.DataFrame:
    """Download normalized OpenAQ PM2.5 observations in [start, end].

    Pagination follows the v3 ``links.next`` cursor. Every page is retried with
    backoff; exhausted downloads raise DownloadError (logged as a failure).
    """
    cfg = _config_of(config)
    env_var = cfg.get("credential_env_var", "OPENAQ_API_KEY")
    api_key = os.environ.get(env_var)
    if not api_key:
        raise CredentialsUnavailable(
            f"OpenAQ API key missing (env '{env_var}'). "
            "Global PM2.5 acquisition is UNAVAILABLE - no data fabricated."
        )
    base_url = cfg.get("base_url", "https://api.openaq.org/v3")
    fetch_cfg = config.get("global_data", {}).get("fetch", {})

    params = {
        "parameter": "pm25",
        "datetime_from": f"{start_date}T00:00:00Z",
        "datetime_to": f"{end_date}T23:59:59Z",
        "limit": 1000,
        "offset": 0,
    }
    if bounds is not None:
        center_lat = (bounds["south"] + bounds["north"]) / 2.0
        center_lon = (bounds["west"] + bounds["east"]) / 2.0
        radius = 1_500_000  # km-aware upper bound; bbox filtering below is exact.
        params.update({"coordinates": f"{center_lon},{center_lat}", "radius": radius})

    rows: list[dict] = []
    seen_pages = 0
    while seen_pages < max_pages:
        page = fetch_with_retry(
            lambda attempt, p=params: _fetch_page(base_url, api_key, p, attempt),
            attempts=int(fetch_cfg.get("retries", 3)),
            backoff_base_s=float(fetch_cfg.get("backoff_base_s", 2.0)),
            backoff_max_s=float(fetch_cfg.get("backoff_max_s", 60.0)),
        )
        for item in page.get("results", []):
            row = _normalize_row(item)
            if row is not None:
                rows.append(row)
        seen_pages += 1

        links = page.get("links") or {}
        next_link = links.get("next")
        if not next_link:
            break
        next_url = next_link.get("url") if isinstance(next_link, dict) else str(next_link)
        if not next_url or "offset=" not in next_url:
            # Fallback: bump offset when the cursor is opaque.
            params["offset"] = params.get("offset", 0) + params.get("limit", 1000)
        else:
            from urllib.parse import parse_qs, urlparse

            query = parse_qs(urlparse(next_url).query)
            params["offset"] = int(query.get("offset", [0])[0])

    frame = pd.DataFrame(rows, columns=[c for c in OBSERVATION_SCHEMA if True])
    for col in OBSERVATION_SCHEMA:
        if col not in frame.columns:
            frame[col] = None
    frame = frame[OBSERVATION_SCHEMA]

    # Exact spatial filter (API radius is an approximation).
    if bounds is not None and not frame.empty:
        mask = (
            frame["longitude"].between(bounds["west"], bounds["east"])
            & frame["latitude"].between(bounds["south"], bounds["north"])
        )
        frame = frame[mask].copy()

    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame["PM2.5"] = pd.to_numeric(frame["PM2.5"], errors="coerce")
    frame["latitude"] = pd.to_numeric(frame["latitude"], errors="coerce")
    frame["longitude"] = pd.to_numeric(frame["longitude"], errors="coerce")
    return frame.reset_index(drop=True)


def acquire_pm25(config, scope: str = "global", start_date: Optional[str] = None,
                 end_date: Optional[str] = None, write: bool = True,
                 failed_downloads_path: Optional[Path] = None) -> dict:
    """Full PM2.5 acquisition pipeline for a scope.

    Returns an honest status dict. Outputs under data/processed/global/ are only
    written when real observations were acquired; otherwise the report explains
    UNAVAILABLE.
    """
    scope = validate_scope(scope)
    availability = check_availability(config)
    report = {
        "source": "pm25",
        "scope": scope,
        "status": availability["status"],
        "reason": availability["reason"],
        "observations_written": False,
        "qc": None,
        "units": None,
        "daily": None,
        "stations": None,
        "synthetic_leakage": None,
    }

    if availability["status"] != "available":
        logger.warning("PM2.5 global acquisition: %s", availability["reason"])
        return report

    bounds = scope_bounds(scope)
    time_cfg = config.get("global_data", {}).get("temporal", {})
    default_start = start_date or config.get("time", {}).get("start_date", "2025-01-01")
    default_end = end_date or config.get("time", {}).get("end_date", default_start)

    try:
        raw = fetch_openaq_pm25(config, default_start, default_end, bounds=bounds)
    except CredentialsUnavailable as exc:
        report["status"] = "unavailable"
        report["reason"] = str(exc)
        return report
    except DownloadError as exc:
        if failed_downloads_path is not None:
            record_failed_download(failed_downloads_path, "pm25", "openaq",
                                   default_start, "n/a", {"error": str(exc)})
        report["status"] = "failed"
        report["reason"] = f"Download failed after retries: {exc}"
        return report

    if raw.empty:
        report["status"] = "no_data"
        report["reason"] = "OpenAQ returned no PM2.5 observations in the requested window."
        return report

    # Unit normalization -> QC -> daily -> stations.
    normalized, units_report = normalize_pm25_units(raw)
    clean, qc_report = apply_qc(
        normalized,
        outlier_method=str(config.get("global_data", {}).get("qc", {}).get("outlier_method", "mad")),
        outlier_threshold_k=float(config.get("global_data", {}).get("qc", {}).get("outlier_threshold_k", 5.0)),
        max_pm25_ug_m3=float(config.get("global_data", {}).get("qc", {}).get("max_pm25_ug_m3", 1000.0)),
    )
    min_obs = int(config.get("global_data", {}).get("temporal", {}).get("min_observations_per_day", 1))
    daily, daily_report = aggregate_daily(clean, min_observations_per_day=min_obs)

    leakage = synthetic_leakage_report(raw, "global_pm25_observations")
    report["synthetic_leakage"] = leakage
    report["qc"] = qc_report
    report["units"] = units_report
    report["daily"] = daily_report

    if daily.empty:
        report["status"] = "no_sufficient_daily_data"
        report["reason"] = daily_report["status"]
        return report

    registry = None
    if write:
        from .ingest import ensure_global_dirs

        ensure_global_dirs(config)
        processed_base = Path(
            config.get("global_data", {}).get("storage", {}).get("processed_base", "data/processed/global")
        )
        pm_dir = processed_base / "pm25"
        pm_dir.mkdir(parents=True, exist_ok=True)
        obs_path = pm_dir / "global_pm25_observations.parquet"
        daily_path = pm_dir / "global_pm25_daily.parquet"
        clean.to_parquet(obs_path, index=False)
        daily.to_parquet(daily_path, index=False)

        from .stations import build_station_registry

        registry = build_station_registry(daily)
        station_artifacts = write_station_outputs(registry, daily, processed_base)
        report["stations"] = station_artifacts
        report["artifact_checksums"] = {
            "global_pm25_observations.parquet": sha256_file(obs_path),
            "global_pm25_daily.parquet": sha256_file(daily_path),
            "global_station_registry.parquet": sha256_file(
                Path(station_artifacts["registry_path"])
            ),
        }
        report["observations_written"] = True
    else:
        from .stations import build_station_registry

        registry = build_station_registry(daily)

    report["status"] = "available"
    report["n_observations"] = int(len(clean)) if not clean.empty else 0
    report["n_daily_rows"] = int(len(daily))
    report["n_stations"] = int(registry["station_id"].nunique()) if registry is not None and not registry.empty else 0
    return report
