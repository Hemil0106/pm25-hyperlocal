"""Global PM2.5 ground-observation adapter: OpenAQ v3 (Milestone 16).

Acquires real PM2.5 observations from the OpenAQ v3 API. v2 is retired (HTTP
410), so v3 is the only real path. v3 requires an API key which MUST come from
the environment (``OPENAQ_API_KEY``). When the key is absent the source is
reported UNAVAILABLE - never fabricated and never substituted with Delhi data.

v3 API workflow:
  1. GET /v3/locations?parameter=pm25 — find all PM2.5 stations (paginate)
  2. Filter by bbox client-side
  3. GET /v3/sensors/{id}/days — get daily measurements per sensor
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


import time as _time


def _v3_get(base_url: str, api_key: str, endpoint: str, params: dict,
            max_retries: int = 5) -> dict:
    """Make a single OpenAQ v3 GET request with 429 rate-limit and connection retry."""
    import requests

    url = f"{base_url}/{endpoint.lstrip('/')}"
    headers = {
        "X-API-Key": api_key,
        "User-Agent": "pm25-hyperlocal-m16/0.1",
        "Accept": "application/json",
    }
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=30)
        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.ChunkedEncodingError) as exc:
            wait = min(30 * (attempt + 1), 120)
            logger.info("OpenAQ network error (attempt %d/%d), waiting %ds: %s",
                        attempt + 1, max_retries, wait, exc)
            _time.sleep(wait)
            continue
        if response.status_code == 401:
            raise CredentialsUnavailable("OpenAQ API key rejected (401 Unauthorized).")
        if response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", 10))
            wait = max(retry_after, 5 * (attempt + 1))
            logger.info("OpenAQ rate limit hit (429), waiting %ds...", wait)
            _time.sleep(wait)
            continue
        response.raise_for_status()
        return response.json()
    raise RuntimeError(f"OpenAQ: failed after {max_retries} retries on {endpoint}")


def _find_pm25_locations(base_url: str, api_key: str,
                         bounds: Optional[dict] = None) -> list[dict]:
    """Find all PM2.5 monitoring locations within bounds.

    Uses /v3/locations?parameter=pm25 and paginates, filtering by bbox.
    For each location, picks the NEWEST PM2.5 sensor (highest datetimeLast).
    Returns list of dicts with keys:
    location_id, name, latitude, longitude, country, sensor_id.
    """
    if bounds is None:
        return []

    stations = []
    seen_loc_ids = set()
    page = 1
    max_pages = 200
    while page <= max_pages:
        try:
            data = _v3_get(base_url, api_key, "locations", {
                "parameter": "pm25",
                "limit": 100,
                "page": page,
            })
        except Exception as exc:
            logger.warning("OpenAQ: page %d failed after retries: %s; returning %d stations so far",
                           page, exc, len(stations))
            break
        results = data.get("results", [])
        if not results:
            break

        for loc in results:
            coords = loc.get("coordinates") or {}
            lat = coords.get("latitude")
            lon = coords.get("longitude")
            if lat is None or lon is None:
                continue
            if not (bounds["west"] <= lon <= bounds["east"]
                    and bounds["south"] <= lat <= bounds["north"]):
                continue

            loc_id = loc.get("id")
            if loc_id in seen_loc_ids:
                continue
            seen_loc_ids.add(loc_id)

            # Pick the NEWEST PM2.5 sensor (by datetimeLast)
            pm25_sensors = []
            for sensor in loc.get("sensors", []):
                param = sensor.get("parameter", {})
                if param.get("name") == "pm25":
                    pm25_sensors.append(sensor)
            if not pm25_sensors:
                continue

            best_sensor = max(
                pm25_sensors,
                key=lambda s: (s.get("datetimeLast") or {}).get("utc", ""),
            )

            country_info = loc.get("country") or {}
            stations.append({
                "location_id": loc_id,
                "name": loc.get("name", ""),
                "latitude": float(lat),
                "longitude": float(lon),
                "country": country_info.get("code", "")
                           if isinstance(country_info, dict)
                           else str(country_info),
                "sensor_id": best_sensor["id"],
            })

        if page % 20 == 0:
            logger.info("OpenAQ: scanned %d pages, found %d PM2.5 stations in bounds",
                        page, len(stations))
        meta = data.get("meta", {})
        total_found = meta.get("found", 0)
        if isinstance(total_found, str):
            try:
                total_found = int(total_found)
            except (ValueError, TypeError):
                total_found = page * 100 + 1
        if page * 100 >= total_found:
            break
        page += 1
        _time.sleep(1.1)

    logger.info("OpenAQ: found %d PM2.5 stations in bounds", len(stations))
    return stations


def _fetch_sensor_days(base_url: str, api_key: str, sensor_id: int,
                       date_from: str, date_to: str) -> list[dict]:
    """Fetch daily measurements for a specific sensor.

    Uses /v3/sensors/{id}/days endpoint. Rate-limited to respect 429s.
    """
    all_results = []
    page = 1
    while page <= 50:
        data = _v3_get(base_url, api_key, f"sensors/{sensor_id}/days", {
            "date_from": date_from,
            "date_to": date_to,
            "limit": 100,
            "page": page,
        })
        results = data.get("results", [])
        if not results:
            break
        all_results.extend(results)
        meta = data.get("meta", {})
        total = meta.get("found", 0)
        if isinstance(total, str):
            total = len(all_results) + 1
        if page * 100 >= total:
            break
        page += 1
        _time.sleep(0.5)
    return all_results


def _normalize_sensor_day(day_row: dict, station: dict) -> Optional[dict]:
    """Map one OpenAQ v3 /sensors/{id}/days result to the observation schema."""
    try:
        value = day_row.get("value")
        if value is None or value < 0:
            return None
        period = day_row.get("period", {})
        datetime_obj = period.get("datetimeFrom") or period.get("datetimeTo")
        if datetime_obj is None:
            return None
        if isinstance(datetime_obj, dict):
            datetime_obj = datetime_obj.get("utc", "")

        return {
            "station_id": str(station["location_id"]),
            "source": "openaq",
            "country": station.get("country", ""),
            "latitude": station["latitude"],
            "longitude": station["longitude"],
            "timestamp": datetime_obj,
            "PM2.5": float(value),
            "units": day_row.get("unit") or "ug/m3",
            "quality_flag": "openaq_reported",
        }
    except (TypeError, ValueError):
        return None


def fetch_openaq_pm25(config, start_date: str, end_date: str,
                      bounds: Optional[dict] = None) -> pd.DataFrame:
    """Download normalized OpenAQ PM2.5 observations in [start, end].

    v3 workflow: find locations → filter by bounds → fetch days per sensor.
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

    stations = _find_pm25_locations(base_url, api_key, bounds=bounds)
    if not stations:
        return pd.DataFrame(columns=OBSERVATION_SCHEMA)

    logger.info("OpenAQ: fetching daily data for %d stations", len(stations))
    rows: list[dict] = []
    for i, station in enumerate(stations):
        try:
            days = _fetch_sensor_days(
                base_url, api_key, station["sensor_id"],
                start_date, end_date,
            )
            for day_row in days:
                row = _normalize_sensor_day(day_row, station)
                if row is not None:
                    rows.append(row)
        except Exception as exc:
            logger.warning("OpenAQ sensor %d (%s) fetch failed: %s",
                           station["sensor_id"], station["name"], exc)

        if (i + 1) % 10 == 0:
            logger.info("OpenAQ: processed %d/%d stations, %d rows so far",
                        i + 1, len(stations), len(rows))
        _time.sleep(1.1)

    frame = pd.DataFrame(rows)
    for col in OBSERVATION_SCHEMA:
        if col not in frame.columns:
            frame[col] = None
    frame = frame[OBSERVATION_SCHEMA]

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
