"""OSM road-density tile loader (Milestone 17).

Loads the real Milestone 16 OSM tile artifacts
(``data/raw/global/osm/<scope>/roads_tile_*.json``) into a deterministic
tile-id -> road-segment-count map. Road density is a spatial proxy only.

A station/observation falls in a tile via ``tile_for_point``; when the tile
artifact exists the row gets the real count, otherwise road_density is NaN with
``road_density_available == False``. No values are invented.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from src.geospatial.tiles import tile_for_point
from src.global_data.scope import scope_bounds, validate_scope

logger = logging.getLogger(__name__)


def load_osm_road_segments(config, scope: str) -> dict:
    """Load tile-id -> road_segments for an acquisition scope.

    Returns {} when no real OSM artifacts exist for the scope.
    """
    scope = validate_scope(scope)
    raw_base = Path(
        config.get("global_data", {}).get("storage", {}).get(
            "raw_base", "data/raw/global")
    )
    osm_dir = raw_base / "osm" / scope
    out: dict = {}
    if not osm_dir.is_dir():
        return out
    for path in sorted(osm_dir.glob("roads_tile_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            tile_id = payload.get("tile_id")
            if not tile_id:
                continue
            out[tile_id] = int(payload.get("road_segments", 0))
        except (ValueError, KeyError, TypeError) as exc:  # noqa: BLE001
            logger.warning("Skipping unreadable OSM tile %s: %s", path.name, exc)
    return out


def road_density_for(
    daily_df: pd.DataFrame,
    road_segments: dict,
    scope: str,
    size_deg: float = 10.0,
) -> pd.DataFrame:
    """Return a per-row road_density + road_density_available frame."""
    if daily_df.empty:
        return pd.DataFrame(columns=["road_density", "road_density_available"])

    bounds = scope_bounds(scope)
    values: list = []
    available: list = []
    for lon, lat in zip(daily_df["longitude"], daily_df["latitude"]):
        tile_id = tile_for_point(
            float(lon), float(lat), size_deg=size_deg, bbox=bounds)
        if tile_id in road_segments:
            values.append(road_segments[tile_id])
            available.append(True)
        else:
            values.append(float("nan"))
            available.append(False)
    return pd.DataFrame({
        "road_density": pd.Series(values, dtype="float64"),
        "road_density_available": pd.Series(available, dtype="bool"),
    })


def osm_tiles_present(config, scope: str) -> dict:
    """Honest report of OSM tile artifacts for a scope (used by metadata)."""
    segments = load_osm_road_segments(config, scope)
    return {
        "n_tile_artifacts": len(segments),
        "tile_ids": sorted(segments.keys()),
    }
