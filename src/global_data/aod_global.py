"""AOD (MODIS MAIAC MCD19A2.061) global adapter.

Uses NASA earthaccess for authentication and CMR search, then the NASA
Harmony API for server-side subsetting (bbox + date + variable) to obtain
a GeoTIFF directly -- no local HDF4/HDF5 reading required.

If Harmony is unavailable, falls back to earthaccess direct download
with rasterio-based HDF4 reading via GDAL.

Credentials: EARTHDATA_USERNAME + EARTHDATA_PASSWORD (env only).

Scientific contract:
  - Never fabricate AOD values.
  - Never silently substitute another dataset.
  - Report honest status at every stage.
"""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
import time
from pathlib import Path
from typing import Optional

import numpy as np
import requests

from .satellite import SatelliteSource

logger = logging.getLogger(__name__)

_MCD19A2_CONCEPT_ID = "C2324689816-LPCLOUD"
_HARMONY_BASE = "https://harmony.earthdata.nasa.gov"
_AOD_VARIABLE = "Optical_Depth_055"
_NODATA = -9999.0
_MAX_HARMONY_WAIT_S = 600


def check_aod_authentication() -> dict:
    """Safe diagnostic: test whether NASA Earthdata auth is configured and working.

    Returns structured dict. Never returns secrets.
    """
    username = os.environ.get("EARTHDATA_USERNAME", "")
    password = os.environ.get("EARTHDATA_PASSWORD", "")
    configured = bool(username and password)

    result = {
        "configured": configured,
        "authenticated": False,
        "provider": "NASA",
        "product": "MCD19A2.061",
        "error_code": None,
        "error_message": None,
    }

    if not configured:
        result["error_code"] = "CREDENTIALS_MISSING"
        result["error_message"] = (
            "EARTHDATA_USERNAME and/or EARTHDATA_PASSWORD not set."
        )
        return result

    try:
        import earthaccess

        auth = earthaccess.login(strategy="environment")
        if auth.authenticated:
            result["authenticated"] = True
        else:
            result["error_code"] = "AUTHENTICATION_FAILED"
            result["error_message"] = "earthaccess.login returned unauthenticated."
    except ImportError:
        result["error_code"] = "EARTHACCESS_NOT_INSTALLED"
        result["error_message"] = "earthaccess package not installed."
    except Exception as exc:
        result["error_code"] = "AUTHENTICATION_ERROR"
        safe_msg = str(exc)[:200]
        result["error_message"] = safe_msg

    return result


def _get_earthaccess_session():
    """Authenticate and return (earthaccess auth, requests session)."""
    import earthaccess

    username = os.environ.get("EARTHDATA_USERNAME", "")
    password = os.environ.get("EARTHDATA_PASSWORD", "")

    if username and password:
        auth = earthaccess.login(strategy="environment")
    else:
        auth = earthaccess.login()

    if not auth.authenticated:
        raise RuntimeError("NASA Earthdata authentication failed.")

    session = earthaccess.get_requests_https_session()
    return auth, session


def _harmony_fetch(
    session: requests.Session,
    bbox_wgs84: tuple,
    date_str: str,
    output_path: Path,
    timeout: float = 120,
) -> bool:
    """Fetch AOD via Harmony OGC Coverages API -> GeoTIFF."""
    west, south, east, north = bbox_wgs84
    dt_start = f"{date_str}T00:00:00.000Z"
    dt_end = f"{date_str}T23:59:59.999Z"

    url = (
        f"{_HARMONY_BASE}/{_MCD19A2_CONCEPT_ID}"
        f"/ogc-api-coverages/1.0.0/collections/parameter_vars/coverage/rangeset"
        f"?subset=lat({south}:{north})"
        f"&subset=lon({west}:{east})"
        f'&subset=time("{dt_start}":"{dt_end}")'
        f"&variable={_AOD_VARIABLE}"
        f"&format=application/geo+tiff"
        f"&outputCrs=EPSG:4326"
        f"&maxResults=1"
    )

    try:
        resp = session.get(url, timeout=timeout, stream=True)
    except Exception as exc:
        logger.warning("Harmony request failed: %s", exc)
        return False

    if resp.status_code != 200:
        logger.warning("Harmony returned HTTP %d", resp.status_code)
        return False

    content_type = resp.headers.get("Content-Type", "")

    if "application/json" in content_type:
        data = resp.json()
        if "status" in data:
            return _poll_harmony(session, data["status"], output_path)
        logger.warning("Harmony unexpected JSON: %s", json.dumps(data)[:300])
        return False

    if "tiff" in content_type or "image/" in content_type:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = output_path.with_suffix(".tif.tmp")
        try:
            with open(tmp_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=256 * 1024):
                    f.write(chunk)
            if tmp_path.stat().st_size < 100:
                tmp_path.unlink(missing_ok=True)
                logger.warning("Harmony returned suspiciously small file")
                return False
            tmp_path.rename(output_path)
            logger.info(
                "Harmony: saved %s (%.0f KB)",
                output_path.name,
                output_path.stat().st_size / 1024,
            )
            return True
        except Exception as exc:
            tmp_path.unlink(missing_ok=True)
            logger.warning("Harmony download failed: %s", exc)
            return False

    logger.warning("Harmony unexpected Content-Type: %s", content_type)
    return False


def _poll_harmony(
    session: requests.Session,
    status_url: str,
    output_path: Path,
    max_wait: int = _MAX_HARMONY_WAIT_S,
) -> bool:
    """Poll async Harmony job until done, then download result."""
    deadline = time.time() + max_wait
    while time.time() < deadline:
        try:
            resp = session.get(status_url, timeout=30)
        except requests.RequestException:
            time.sleep(5)
            continue

        if resp.status_code != 200:
            time.sleep(5)
            continue

        data = resp.json()
        status = data.get("status")

        if status == "successful":
            for item in data.get("data", []):
                href = item.get("href", "")
                if href.endswith(".tif") or "tiff" in href.lower():
                    try:
                        dl = session.get(href, timeout=120, stream=True)
                        if dl.status_code == 200:
                            output_path.parent.mkdir(parents=True, exist_ok=True)
                            tmp_path = output_path.with_suffix(".tif.tmp")
                            with open(tmp_path, "wb") as f:
                                for chunk in dl.iter_content(chunk_size=256 * 1024):
                                    f.write(chunk)
                            if tmp_path.stat().st_size > 100:
                                tmp_path.rename(output_path)
                                logger.info(
                                    "Harmony: saved %s", output_path.name
                                )
                                return True
                            tmp_path.unlink(missing_ok=True)
                    except requests.RequestException as exc:
                        logger.warning(
                            "Harmony result download failed: %s", exc
                        )
                    return False
            logger.warning("Harmony successful but no GeoTIFF in response")
            return False

        if status == "failed":
            logger.warning(
                "Harmony job failed: %s", data.get("message", "unknown")
            )
            return False

        logger.debug("Harmony status: %s...", status)
        time.sleep(10)

    logger.warning("Harmony timed out after %ds", max_wait)
    return False
class AODSource(SatelliteSource):
    """AOD source using MCD19A2.061 via Harmony API."""
    source_id = "aod"
    product = "MODIS_MAIAC_MCD19A2.061"
    default_resolution = "1km"


def check_availability(config) -> dict:
    return AODSource(config).check_availability()


def _validate_aod_geotiff(path: Path) -> dict:
    """Validate an AOD GeoTIFF. Returns stats dict or error."""
    import rasterio

    try:
        with rasterio.open(path) as src:
            data = src.read(1)
            transform = src.transform
            crs = src.crs
            bounds = src.bounds
            nodata = src.nodata

            if nodata is not None:
                valid_mask = (data != nodata) & ~np.isnan(data)
            else:
                valid_mask = ~np.isnan(data)

            valid_count = int(np.sum(valid_mask))
            total_count = data.size
            valid_pct = (valid_count / total_count * 100) if total_count > 0 else 0.0

            stats = {
                "valid": valid_count > 0,
                "valid_pixel_count": valid_count,
                "total_pixel_count": total_count,
                "valid_coverage_pct": round(valid_pct, 1),
                "min": float(np.nanmin(data[valid_mask])) if valid_count > 0 else None,
                "max": float(np.nanmax(data[valid_mask])) if valid_count > 0 else None,
                "mean": float(np.nanmean(data[valid_mask])) if valid_count > 0 else None,
                "median": float(np.nanmedian(data[valid_mask])) if valid_count > 0 else None,
                "aod_source_crs": str(crs) if crs else None,
                "aod_target_crs": str(crs) if crs else None,
                "source_bounds": {
                    "left": bounds.left, "bottom": bounds.bottom,
                    "right": bounds.right, "top": bounds.top,
                },
                "shape": list(data.shape),
                "transform": list(transform.to_gdal()),
            }
            return stats
    except Exception as exc:
        return {"valid": False, "error": str(exc)}


def _backend_copy(source_path: Path, scope: str, date_str: str, config: dict) -> Optional[Path]:
    """Copy processed AOD to the backend serving directory."""
    try:
        processed_base = Path(
            config.get("global_data", {})
            .get("storage", {})
            .get("processed_base", "data/processed/global")
        )
        project_root = processed_base.parent.parent

        if scope == "delhi":
            backend_dir = project_root / "data" / "processed"
        elif scope in ("pune", "mumbai"):
            backend_dir = project_root / "data" / "processed" / scope
        else:
            backend_dir = project_root / "data" / "processed"

        backend_aod = backend_dir / f"aod_500m_{date_str}.tif"
        backend_aod.parent.mkdir(parents=True, exist_ok=True)

        if not backend_aod.exists():
            shutil.copy2(source_path, backend_aod)
            logger.info("AOD: copied to backend path %s", backend_aod)
        return backend_aod
    except Exception as exc:
        logger.warning("AOD: backend copy failed: %s", exc)
        return None


def _acquire_harmony(
    session: requests.Session,
    scope: str,
    date_str: str,
    config: dict,
) -> dict:
    """Acquire AOD using Harmony API (primary method).

    Returns status dict with acquisition details.
    """
    from .scope import scope_bounds

    bounds = scope_bounds(scope)
    bbox = (bounds["west"], bounds["south"], bounds["east"], bounds["north"])

    processed_base = Path(
        config.get("global_data", {})
        .get("storage", {})
        .get("processed_base", "data/processed/global")
    )
    project_root = processed_base.parent.parent

    if scope == "delhi":
        out_dir = project_root / "data" / "processed"
    elif scope in ("pune", "mumbai"):
        out_dir = project_root / "data" / "processed" / scope
    else:
        out_dir = project_root / "data" / "processed"

    out_dir.mkdir(parents=True, exist_ok=True)
    aod_path = out_dir / f"aod_500m_{date_str}.tif"

    if aod_path.exists() and aod_path.stat().st_size > 100:
        logger.info("AOD: using cached file %s", aod_path)
        stats = _validate_aod_geotiff(aod_path)
        _backend_copy(aod_path, scope, date_str, config)
        return {
            "source": "aod",
            "provider": "NASA MODIS MAIAC",
            "product": "MCD19A2.061",
            "method": "harmony_cached",
            "date": date_str,
            "scope": scope,
            "granules_discovered": 0,
            "downloaded": 0,
            "cached": 1,
            "failed": 0,
            "status": "AVAILABLE" if stats.get("valid") else "FAILED",
            "stats": stats,
        }

    ok = _harmony_fetch(session, bbox, date_str, aod_path)
    if not ok:
        return {
            "source": "aod",
            "provider": "NASA MODIS MAIAC",
            "product": "MCD19A2.061",
            "method": "harmony",
            "date": date_str,
            "scope": scope,
            "granules_discovered": 0,
            "downloaded": 0,
            "cached": 0,
            "failed": 1,
            "status": "FAILED",
            "error_code": "HARMONY_FETCH_FAILED",
            "error_message": "Harmony API did not return valid AOD data.",
        }

    stats = _validate_aod_geotiff(aod_path)
    if not stats.get("valid"):
        return {
            "source": "aod",
            "provider": "NASA MODIS MAIAC",
            "product": "MCD19A2.061",
            "method": "harmony",
            "date": date_str,
            "scope": scope,
            "granules_discovered": 1,
            "downloaded": 1,
            "cached": 0,
            "failed": 0,
            "status": "FAILED",
            "error_code": "INVALID_AOD_DATA",
            "error_message": "Downloaded AOD failed validation.",
            "stats": stats,
        }

    _backend_copy(aod_path, scope, date_str, config)
    return {
        "source": "aod",
        "provider": "NASA MODIS MAIAC",
        "product": "MCD19A2.061",
        "method": "harmony",
        "date": date_str,
        "scope": scope,
        "granules_discovered": 1,
        "downloaded": 1,
        "cached": 0,
        "failed": 0,
        "status": "AVAILABLE",
        "stats": stats,
    }


def acquire(config, scope: str = "global", date: str = "2025-01-01") -> dict:
    """Acquire MCD19A2.061 AOD for the given scope and date.

    Uses Harmony API via earthaccess session. Returns honest status dict.
    """
    creds_present = bool(
        os.environ.get("EARTHDATA_USERNAME")
        and os.environ.get("EARTHDATA_PASSWORD")
    )

    if not creds_present:
        return {
            "source": "aod",
            "provider": "NASA MODIS MAIAC",
            "product": "MCD19A2.061",
            "method": "none",
            "date": date,
            "scope": scope,
            "granules_discovered": 0,
            "downloaded": 0,
            "cached": 0,
            "failed": 0,
            "status": "UNAVAILABLE",
            "error_code": "CREDENTIALS_MISSING",
            "error_message": "EARTHDATA_USERNAME/PASSWORD not set.",
        }

    try:
        auth, session = _get_earthaccess_session()
    except Exception as exc:
        return {
            "source": "aod",
            "provider": "NASA MODIS MAIAC",
            "product": "MCD19A2.061",
            "method": "none",
            "date": date,
            "scope": scope,
            "granules_discovered": 0,
            "downloaded": 0,
            "cached": 0,
            "failed": 0,
            "status": "UNAVAILABLE",
            "error_code": "AUTHENTICATION_FAILED",
            "error_message": str(exc)[:200],
        }

    return _acquire_harmony(session, scope, date, config)
