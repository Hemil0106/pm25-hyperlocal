"""AOD (MODIS MAIAC MCD19A2) global adapter (Milestone 16).

Tile-based acquisition of aerosol optical depth. Requires NASA Earthdata
credentials (environment only). Missing credentials -> graceful UNAVAILABLE.
Uses NASA CMR for granule search + Earthdata bearer token for download.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .satellite import SatelliteSource, SourceUnavailable

logger = logging.getLogger(__name__)

_MCD19A2_CONCEPT_ID = "C2324689816-LPCLOUD"


class AODSource(SatelliteSource):
    source_id = "aod"
    product = "MODIS_MAIAC_MCD19A2"
    default_resolution = "1km"


def check_availability(config) -> dict:
    return AODSource(config).check_availability()


def acquire(config, scope: str = "global", date: str = "2025-01-01") -> dict:
    source = AODSource(config)

    def on_tile(tile: object, _date: str) -> Path:
        tile_id = tile.tile_id
        tile_bbox = tile.bbox
        raw_base = Path(config.get("global_data", {}).get("storage", {}).get(
            "raw_base", "data/raw/global"))
        out_dir = raw_base / "aod" / scope / str(tile_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        dest = out_dir / "MCD19A2.hdf"
        if dest.exists():
            return dest

        from .nasa_auth import (cmr_search_granules, find_download_url,
                                 download_with_earthdata)

        granules = cmr_search_granules(
            _MCD19A2_CONCEPT_ID,
            bbox=tile_bbox,
            temporal_start=_date,
            temporal_end=_date,
        )
        url = find_download_url(granules)
        if not url:
            raise RuntimeError(
                f"No MCD19A2 granule found for tile {tile_id} on {_date}")

        download_with_earthdata(url, str(dest))
        return dest

    return source.attempt_acquire(scope, date, on_tile)
