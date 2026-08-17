"""OSM road density global adapter (Milestone 16).

Road density per tile is a spatial proxy only (it is not a direct pollution
measurement). Queries the Overpass API in chunks (tiles) so no global dataset
is held in memory. No credentials required, but failures (network/rate limits)
are recorded gracefully.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .satellite import SatelliteSource

logger = logging.getLogger(__name__)


class OSMSource(SatelliteSource):
    source_id = "osm"
    product = "OpenStreetMap roads"
    default_resolution = "tile_road_density"


def check_availability(config) -> dict:
    result = OSMSource(config).check_availability()
    if result["status"] == "available":
        endpoint = config.get("global_data", {}).get("sources", {}).get(
            "osm", {}).get("endpoint", "https://overpass-api.de/api/interpreter")
        reachable = OSMSource.probe_connectivity(endpoint)
        if not reachable:
            result["status"] = "unavailable"
            result["reason"] = "Overpass endpoint unreachable from this environment."
    return result


def road_density_query(tile_bbox: dict) -> str:
    """Overpass QL: count major road ways intersecting a tile bbox."""
    west, south, east, north = (tile_bbox["west"], tile_bbox["south"],
                                tile_bbox["east"], tile_bbox["north"])
    return f"""
    [out:json][timeout:60];
    (
      way["highway"~"motorway|trunk|primary|secondary|tertiary"]
        ({south},{west},{north},{east});
    );
    out count;
    """


def acquire(config, scope: str = "global", date: str = "2025-01-01") -> dict:
    source = OSMSource(config)

    def on_tile(tile: object, _date: str) -> Path:
        tile_bbox = tile.bbox
        tile_id = tile.tile_id
        raw_base = Path(config.get("global_data", {}).get("storage", {}).get(
            "raw_base", "data/raw/global"))
        out_dir = raw_base / "osm" / scope
        out_dir.mkdir(parents=True, exist_ok=True)
        dest = out_dir / f"roads_{tile_id}.json"
        if dest.exists():
            return dest

        import requests

        endpoint = source.source_cfg.get(
            "endpoint", "https://overpass-api.de/api/interpreter")
        response = requests.post(
            endpoint, data={"data": road_density_query(tile_bbox)},
            headers={"User-Agent": "pm25-hyperlocal-m16/0.1"}, timeout=120,
        )
        response.raise_for_status()
        payload = response.json()
        count = 0
        for element in payload.get("elements", []):
            count += int(element.get("tags", {}).get("count", 0))
        import json as _json

        dest.write_text(_json.dumps({
            "tile_id": tile_id,
            "bbox": tile_bbox,
            "date": _date,
            "road_segments": count,
            "proxy_note": "Road density is a spatial proxy, not a measurement.",
        }), encoding="utf-8")
        return dest

    return source.attempt_acquire(scope, date, on_tile)
