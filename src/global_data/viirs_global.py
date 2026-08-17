"""VIIRS night-time lights global adapter (Milestone 16).

Tile-based acquisition of VNP46A2 with per-tile quality assurance. Missing
Earthdata credentials -> graceful UNAVAILABLE.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .satellite import SatelliteSource

logger = logging.getLogger(__name__)

VIIRS_KEEP_QUALITY = [0]  # Mandatory_Quality_Flag == 0 (good quality)


def check_availability(config) -> dict:
    return VIIRSSource(config).check_availability()


def qa_pass_mask(quality_values, keep_quality) -> list[bool]:
    """Per-pixel QA mask: keep only pixels whose flag is in keep_quality."""
    return [int(v) in set(keep_quality) for v in quality_values]


class VIIRSSource(SatelliteSource):
    source_id = "viirs"
    product = "NASA_VIIRS_VNP46A2_v2.0"
    default_resolution = "daily"


def acquire(config, scope: str = "global", date: str = "2025-01-01") -> dict:
    source = VIIRSSource(config)

    def on_tile(tile: object, _date: str) -> Path:
        tile_id = tile.tile_id
        tile_bbox = tile.bbox
        raw_base = Path(config.get("global_data", {}).get("storage", {}).get(
            "raw_base", "data/raw/global"))
        out_dir = raw_base / "viirs" / scope / str(tile_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        dest = out_dir / f"VNP46A2_{_date}.tif"
        if dest.exists():
            return dest
        center_lat = (tile_bbox["south"] + tile_bbox["north"]) / 2.0
        center_lon = (tile_bbox["west"] + tile_bbox["east"]) / 2.0
        url = (
            "https://ladsweb.modaps.eosdis.nasa.gov/api/v2/content/details/"
            f"VNP46A2.002/{_date}?center={center_lat},{center_lon}"
        )
        source._download(url, dest, tile, _date)
        return dest

    report = source.attempt_acquire(scope, date, on_tile)
    report["qa"] = {
        "flag_band": "Mandatory_Quality_Flag",
        "keep_quality": VIIRS_KEEP_QUALITY,
    }
    return report
