"""ERA5-Land weather global adapter (Milestone 16).

Chunked acquisition of ERA5-Land meteorology from the Copernicus Climate Data
Store. Requires CDS API credentials (environment only, ``CDSAPI_URL`` +
``CDSAPI_KEY``). Missing credentials -> graceful UNAVAILABLE.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .satellite import SatelliteSource

logger = logging.getLogger(__name__)


class WeatherSource(SatelliteSource):
    source_id = "weather"
    product = "ERA5-Land"
    default_resolution = "0.1deg"


def check_availability(config) -> dict:
    return WeatherSource(config).check_availability()


def _cds_request_payload(tile: dict, date: str, variables) -> dict:
    """Build the ERA5-Land CDS API request body for a chunk (tile)."""
    return {
        "product_type": "reanalysis",
        "variable": variables,
        "year": date[:4],
        "month": date[5:7],
        "day": date[8:10],
        "time": [f"{h:02d}:00" for h in range(0, 24, 6)],
        "area": [tile["north"], tile["west"], tile["south"], tile["east"]],
        "data_format": "netcdf",
    }


def acquire(config, scope: str = "global", date: str = "2025-01-01") -> dict:
    source = WeatherSource(config)

    def on_tile(tile: object, _date: str) -> Path:
        tile_bbox = tile.bbox
        tile_id = tile.tile_id
        raw_base = Path(config.get("global_data", {}).get("storage", {}).get(
            "raw_base", "data/raw/global"))
        out_dir = raw_base / "weather" / scope / str(tile_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        dest = out_dir / f"era5_land_{_date}.nc"
        if dest.exists():
            return dest

        import json
        import os

        import requests

        variables = config.get("datasets", {}).get("weather", {}).get(
            "variables",
            ["2m_temperature", "2m_relative_humidity", "10m_u_component_of_wind",
             "10m_v_component_of_wind"],
        )
        url = os.environ.get("CDSAPI_URL", "https://cds.climate.copernicus.eu/api")
        key = os.environ.get("CDSAPI_KEY", "")
        payload = _cds_request_payload(tile_bbox, _date, variables)

        # CDS API v2: submit a retrieval job, then poll for the result.
        auth = requests.auth.HTTPBasicAuth(key.split(":")[0], key.split(":")[1] if ":" in key else key)
        response = requests.post(f"{url.rstrip('/')}/retrieve", json=payload, auth=auth, timeout=60)
        response.raise_for_status()
        job = response.json()
        job_url = f"{url.rstrip('/')}/tasks/{job['request_id']}"

        import time

        for _ in range(120):
            state = requests.get(job_url, auth=auth, timeout=60).json()
            if state.get("state") == "completed":
                break
            time.sleep(5)
        else:
            raise TimeoutError(f"CDS job {job['request_id']} did not complete.")

        download_url = state.get("links", [{}])[0].get("href")
        if not download_url:
            raise RuntimeError("CDS job completed without a download link.")
        source._download(download_url, dest, tile, _date)
        return dest

    return source.attempt_acquire(scope, date, on_tile)
