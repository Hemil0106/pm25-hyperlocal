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
        "variable": variables,
        "year": date[:4],
        "month": date[5:7],
        "day": date[8:10],
        "time": [f"{h:02d}:00" for h in range(0, 24, 6)],
        "area": [tile["north"], tile["west"], tile["south"], tile["east"]],
        "format": "netcdf",
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

        import os

        variables = config.get("datasets", {}).get("weather", {}).get(
            "variables",
            ["2m_temperature", "2m_relative_humidity", "10m_u_component_of_wind",
             "10m_v_component_of_wind"],
        )
        url = os.environ.get("CDSAPI_URL", "https://cds.climate.copernicus.eu/api")
        key = os.environ.get("CDSAPI_KEY", "")

        import cdsapi

        import time as _time

        client = cdsapi.Client(url=url, key=key)
        payload = _cds_request_payload(tile_bbox, _date, variables)
        logger.info("Weather: CDS request for tile %s payload=%s", tile_id, payload)

        for attempt in range(3):
            try:
                client.retrieve("reanalysis-era5-land", payload, str(dest))
                return dest
            except Exception as exc:
                if attempt < 2:
                    wait = 30 * (attempt + 1)
                    logger.warning("Weather CDS attempt %d failed: %s; retry in %ds",
                                   attempt + 1, exc, wait)
                    _time.sleep(wait)
                else:
                    raise

    return source.attempt_acquire(scope, date, on_tile)
