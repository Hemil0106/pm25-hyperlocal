"""AOI-dependent grid generation.

Grids are built from the AOI in the AOI's metric (projected) CRS so that cell
sizes are true distances (1 km / 500 m); centroids are then transformed to
EPSG:4326 for latitude/longitude output.

Every cell carries:
  grid_id        deterministic within a tile (``{tile_id}::{r:05d}_{c:05d}``)
  latitude       centroid latitude (EPSG:4326)
  longitude      centroid longitude (EPSG:4326)
  geometry       cell polygon (EPSG:4326)
  parent_grid_id coarse (1 km) parent cell id for fine (500 m) cells
  tile_id        owning tile ("" for non-tiled AOIs)

Cell-count guard: grids above ``max_cells`` are refused with a clear message
instead of exhausting memory. For global AOIs, and for large AOIs at fine
resolutions (e.g. India at 500 m ~ 50M cells), use tile-based processing.
"""

from __future__ import annotations

import logging
from typing import Optional

import geopandas as gpd
import numpy as np
from shapely.geometry import box

from .aoi import AOI
from .crs import get_metric_crs, transform_coords

logger = logging.getLogger(__name__)

DEFAULT_MAX_CELLS = 2_000_000


def _grid_for_bbox(bbox: dict, resolution_m: int, metric_crs: str, tile_id: str,
                   max_cells: int = DEFAULT_MAX_CELLS):
    west = float(bbox["west"])
    south = float(bbox["south"])
    east = float(bbox["east"])
    north = float(bbox["north"])

    from pyproj import Transformer

    transformer = Transformer.from_crs("EPSG:4326", metric_crs, always_xy=True)
    x0, y0 = transformer.transform(west, south)
    x1, y1 = transformer.transform(east, north)
    x0, x1 = min(x0, x1), max(x0, x1)
    y0, y1 = min(y0, y1), max(y0, y1)

    n_cols = max(1, int(round((x1 - x0) / resolution_m)))
    n_rows = max(1, int(round((y1 - y0) / resolution_m)))
    n_cells = n_rows * n_cols
    if n_cells > max_cells:
        raise MemoryError(
            f"Grid too large: {n_rows:,}x{n_cols:,} = {n_cells:,} cells at "
            f"{resolution_m}m for tile '{tile_id}'. Use tile-based processing "
            f"or a coarser resolution (max_cells={max_cells:,})."
        )

    cell_w = (x1 - x0) / n_cols
    cell_h = (y1 - y0) / n_rows

    rows = np.repeat(np.arange(n_rows), n_cols)
    cols = np.tile(np.arange(n_cols), n_rows)
    xmin = x0 + cols * cell_w
    ymin = y0 + rows * cell_h

    # Cell polygons in the metric CRS
    geoms = [
        box(xmin[i], ymin[i], xmin[i] + cell_w, ymin[i] + cell_h)
        for i in range(n_cells)
    ]
    grid = gpd.GeoDataFrame({"row": rows, "col": cols, "geometry": geoms},
                            crs=metric_crs)

    # Exact centroids computed in the metric CRS, then transformed to 4326.
    cx = xmin + cell_w / 2.0
    cy = ymin + cell_h / 2.0
    back = Transformer.from_crs(metric_crs, "EPSG:4326", always_xy=True)
    lon, lat = back.transform(cx, cy)

    grid["grid_id"] = [
        f"{tile_id}::{int(r):05d}_{int(c):05d}" for r, c in zip(rows, cols)
    ]
    grid["tile_id"] = tile_id
    grid["longitude"] = np.asarray(lon)
    grid["latitude"] = np.asarray(lat)

    return grid.to_crs("EPSG:4326")


def generate_grid(aoi: AOI, resolution_m: int, tile: Optional[dict] = None,
                  tile_id: str = "", max_cells: int = DEFAULT_MAX_CELLS) -> gpd.GeoDataFrame:
    """Generate a resolution_m grid over an AOI (optionally scoped to a tile).

    For global AOIs a tile bbox is required.
    """
    if aoi.is_global and tile is None:
        raise ValueError(
            "Global AOI grids must be generated per tile (tile bbox required). "
            "Never allocate the global grid in memory."
        )

    bbox = tile if tile is not None else aoi.bbox
    metric_crs = get_metric_crs(aoi_bbox=bbox)
    if metric_crs is None:
        raise ValueError(
            f"No single projected CRS for bbox {bbox}; use tile-based processing."
        )
    return _grid_for_bbox(bbox, int(resolution_m), metric_crs, tile_id, max_cells)


def generate_coarse_fine_grids(aoi: AOI, coarse_m: int = 1000, fine_m: int = 500,
                               tile: Optional[dict] = None, tile_id: str = "",
                               max_cells: int = DEFAULT_MAX_CELLS):
    """Generate 1 km (coarse) and 500 m (fine) grids with parent links.

    Each 500 m cell's parent_grid_id is the 1 km cell containing its centroid.
    """
    coarse = generate_grid(aoi, coarse_m, tile=tile, tile_id=tile_id, max_cells=max_cells)
    fine = generate_grid(aoi, fine_m, tile=tile, tile_id=tile_id, max_cells=max_cells)

    if coarse.empty:
        raise ValueError("Coarse grid is empty - check AOI/resolution.")

    if fine.empty:
        raise ValueError("Fine grid is empty - check AOI/resolution.")

    fine["parent_grid_id"] = ""
    if not fine.empty:
        coarse_sindex = coarse.sindex
        coarse_geoms = coarse["geometry"].values
        coarse_ids = coarse["grid_id"].values

        def find_parent(lon_val, lat_val):
            point = box(lon_val - 1e-9, lat_val - 1e-9, lon_val + 1e-9, lat_val + 1e-9)
            hits = list(coarse_sindex.intersection(point.bounds))
            for idx in hits:
                if coarse_geoms[idx].contains(point):
                    return coarse_ids[idx]
            return ""

        fine["parent_grid_id"] = [
            find_parent(lo, la) for lo, la in zip(fine["longitude"], fine["latitude"])
        ]

    return coarse, fine
