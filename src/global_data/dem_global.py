"""DEM (NASA SRTM GL1) global adapter (Milestone 16).

Tile-based elevation acquisition. SRTM NoData values are preserved -- elevation
is never zero-filled. Requires NASA Earthdata credentials (env only).
"""

from __future__ import annotations

import logging
from pathlib import Path

from .satellite import SatelliteSource

logger = logging.getLogger(__name__)

SRTM_NODATA = -32768


def check_availability(config) -> dict:
    return DEMSource(config).check_availability()


def srtm_tile_name(lon_center: float, lat_center: float) -> str:
    """SRTM tile naming (N/E 1-degree tiles), e.g. N28E077."""
    lat_suffix = "S" if lat_center < 0 else "N"
    lon_suffix = "W" if lon_center < 0 else "E"
    lat = int(abs(lat_center) // 1)
    lon = int(abs(lon_center) // 1)
    return f"{lat_suffix}{lat:02d}{lon_suffix}{lon:03d}"


class DEMSource(SatelliteSource):
    source_id = "dem"
    product = "NASA_SRTM_GL1_v003"
    default_resolution = "30m"


def acquire(config, scope: str = "global", date: str = "2025-01-01") -> dict:
    source = DEMSource(config)

    def on_tile(tile: object, _date: str) -> Path:
        tile_bbox = tile.bbox
        tile_id = tile.tile_id
        center_lat = (tile_bbox["south"] + tile_bbox["north"]) / 2.0
        center_lon = (tile_bbox["west"] + tile_bbox["east"]) / 2.0
        name = srtm_tile_name(center_lon, center_lat)
        raw_base = Path(config.get("global_data", {}).get("storage", {}).get(
            "raw_base", "data/raw/global"))
        out_dir = raw_base / "dem" / scope
        out_dir.mkdir(parents=True, exist_ok=True)
        dest = out_dir / f"{name}.hgt"
        if dest.exists():
            return dest
        # NASA LP DAAC Cloud (SRTMGL1.003) download via CMR search.
        from .nasa_auth import (cmr_search_granules, find_download_url,
                                 download_with_earthdata, earthdata_auth_headers)

        _SRTM_CID = "C2763266360-LPCLOUD"
        granules = cmr_search_granules(
            _SRTM_CID,
            bbox=tile_bbox,
            page_size=5,
        )
        url = find_download_url(granules)
        if not url:
            raise RuntimeError(
                f"No SRTM granule found for tile {tile_id}")
        download_with_earthdata(url, str(dest),
                                extra_headers=earthdata_auth_headers())
        return dest

    report = source.attempt_acquire(scope, date, on_tile)
    report["nodata"] = SRTM_NODATA
    report["nodata_note"] = "NoData is preserved; elevation is never zero-filled."
    return report
