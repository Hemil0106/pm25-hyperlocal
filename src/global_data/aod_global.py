"""AOD (MODIS MAIAC MCD19A2) global adapter (Milestone 16).

Tile-based acquisition of aerosol optical depth. Requires NASA Earthdata
credentials (environment only). Missing credentials -> graceful UNAVAILABLE.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .satellite import SatelliteSource, SourceUnavailable

logger = logging.getLogger(__name__)


class AODSource(SatelliteSource):
    source_id = "aod"
    product = "MODIS_MAIAC_MCD19A2"
    default_resolution = "1km"


def check_availability(config) -> dict:
    return AODSource(config).check_availability()


def _granule_url_for_tile(tile_bbox: dict) -> str:
    """LAADS DAAC file search URL for a tile (needs Earthdata bearer token)."""
    center_lat = (tile_bbox["south"] + tile_bbox["north"]) / 2.0
    center_lon = (tile_bbox["west"] + tile_bbox["east"]) / 2.0
    # MCD19A2 granules are 1200km MODIS tiles; the search API locates the
    # granule containing the tile center.
    return (
        "https://ladsweb.modaps.eosdis.nasa.gov/api/v2/content/details/"
        "MCD19A2.061/2025-01-01"
        f"?center={center_lat},{center_lon}"
    )


def acquire(config, scope: str = "global", date: str = "2025-01-01") -> dict:
    source = AODSource(config)

    def on_tile(tile: object, _date: str) -> Path:
        tile_id = tile.tile_id
        raw_base = Path(config.get("global_data", {}).get("storage", {}).get(
            "raw_base", "data/raw/global"))
        out_dir = raw_base / "aod" / scope / str(tile_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        dest = out_dir / "MCD19A2.tif"
        if dest.exists():
            return dest
        # Real download path (exercised when Earthdata credentials are set).
        source._download(_granule_url_for_tile(tile.bbox), dest, tile, _date)
        return dest

    return source.attempt_acquire(scope, date, on_tile)
