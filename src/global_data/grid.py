"""Common grid framework for global data (Milestone 16).

The framework defines a 1000 m coarse grid and a 500 m target grid. For global
scopes the grid is ONLY produced per tile -- a full global 500 m map is NEVER
materialized. The common grid aligns observations and feature rasters so the
temporal join is spatially consistent.

Reuses the locked src/geospatial/grid.generate_grid (no duplicated logic).
"""

from __future__ import annotations

import logging
from typing import Optional

import geopandas as gpd

from ..geospatial.aoi import AOI, resolve_aoi
from ..geospatial.grid import generate_grid
from ..geospatial.tiles import generate_tiles
from .scope import scope_bounds, validate_scope

logger = logging.getLogger(__name__)

DEFAULT_MAX_CELLS = 2_000_000


def _aoi_for_scope(config, scope: str) -> AOI:
    if scope in ("global", "india", "delhi"):
        return resolve_aoi(config, region=scope)
    return resolve_aoi(config)


def grid_for_tile(config, scope: str, resolution_m: int, tile: dict,
                  tile_id: str, max_cells: int = DEFAULT_MAX_CELLS) -> gpd.GeoDataFrame:
    """Per-tile common grid at ``resolution_m`` (EPSG:4326)."""
    aoi = _aoi_for_scope(config, validate_scope(scope))
    return generate_grid(aoi, resolution_m, tile=tile, tile_id=tile_id,
                         max_cells=max_cells)


def common_grid(config, scope: str, resolution_m: int = 1000,
                max_cells: int = DEFAULT_MAX_CELLS) -> tuple[gpd.GeoDataFrame, dict]:
    """Build the common grid for a scope.

    For global/india scopes this iterates the tile grid so no full-resolution
    global frame is allocated at once; for small scopes a single grid is built.
    Returns (concatenated_grid, report).
    """
    scope = validate_scope(scope)
    bounds = scope_bounds(scope)
    tiles = list(generate_tiles(bounds))
    frames = []
    report = {"scope": scope, "resolution_m": int(resolution_m),
              "tiles": len(tiles), "cells": 0, "tile_based": scope == "global"}

    if scope == "global" and tiles:
        max_per_tile = max(1, max_cells // max(1, len(tiles)))
        for tile in tiles:
            try:
                frame = grid_for_tile(config, scope, resolution_m, tile.bbox,
                                      tile.tile_id, max_cells=max_per_tile)
                frames.append(frame)
            except MemoryError:
                logger.warning("Tile %s grid too large; skipped.", tile.tile_id)
    else:
        aoi = _aoi_for_scope(config, scope)
        if scope == "global":
            raise ValueError("Global common grid must be built per tile.")
        frame = generate_grid(aoi, int(resolution_m), max_cells=max_cells)
        frames.append(frame)

    if frames:
        grid = gpd.GeoDataFrame(pd_concat(frames), crs="EPSG:4326")
        report["cells"] = int(len(grid))
    else:
        grid = gpd.GeoDataFrame()
    return grid, report


def pd_concat(frames):
    import pandas as pd

    return pd.concat(frames, ignore_index=True)
