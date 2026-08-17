"""Globalized data ingestion for arbitrary AOIs (Phase 8).

Generic, AOI-aware ingestion interfaces:

  - ingest_raster : clip a GeoTIFF/NetCDF to an AOI (or a tile bbox).
  - ingest_tabular: filter a tabular (CSV/parquet) source to an AOI.
  - ingest_vector : clip a vector (GeoJSON/GPKG) to an AOI.

All functions keep results in EPSG:4326 geographic coordinates; projected
(resampled, re-projected) products are handled by the processing stages that
consume them. For global AOIs, callers MUST pass a tile bbox so only that
tile's slice of a scene is materialized.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from ..geospatial.crs import GEOGRAPHIC_CRS

logger = logging.getLogger(__name__)


def _resolve_path(path: Union[str, Path, None]) -> Optional[Path]:
    if path is None:
        return None
    path = Path(path)
    return path if path.is_absolute() else Path.cwd() / path


def _clip_bounds(aoi=None, tile: Optional[dict] = None) -> dict:
    """Effective clip bounds: tile bbox if provided, else AOI bounds."""
    if tile is not None:
        return {
            "west": float(tile["west"]),
            "south": float(tile["south"]),
            "east": float(tile["east"]),
            "north": float(tile["north"]),
        }
    if aoi is None:
        raise ValueError("Either aoi or tile is required.")
    return dict(aoi.bounds)


def ingest_raster(source_path, aoi=None, tile: Optional[dict] = None,
                  band: int = 1, nodata: Optional[float] = None):
    """Clip a single-band raster source to the AOI (or tile) bounds.

    NetCDF sources (regular lat/lon grids) are handled via xarray clipping.
    Returns a dict with windowed array, bounds, transform and crs of the clip.
    """
    import numpy as np
    import rasterio
    from rasterio.windows import from_bounds

    src_path = _resolve_path(source_path)
    if src_path is None or not src_path.exists():
        raise FileNotFoundError(f"Raster source not found: {src_path}")

    if src_path.suffix.lower() in (".nc", ".nc4", ".nc3"):
        return ingest_netcdf(src_path, aoi=aoi, tile=tile, variable=None)

    bounds = _clip_bounds(aoi, tile)
    with rasterio.open(src_path) as src:
        if src.crs is not None and src.crs.to_string() != GEOGRAPHIC_CRS:
            from rasterio.warp import calculate_default_transform, reproject, Resampling

            win = from_bounds(bounds["west"], bounds["south"], bounds["east"], bounds["north"],
                              transform=src.transform)
            windowed = src.read(band, window=win)
            if nodata is None:
                nodata = src.nodata
            dst_crs = src.crs
            transform = src.window_transform(win)
            # Reproject the windowed slice into the source's own CRS so the
            # clip respects the original projection.
            out_transform, width, height = calculate_default_transform(
                GEOGRAPHIC_CRS, dst_crs, width=win.width, height=win.height,
                left=bounds["west"], bottom=bounds["south"],
                right=bounds["east"], top=bounds["north"])
            dst = np.empty((height, width), dtype=windowed.dtype)
            reproject(
                source=windowed, destination=dst,
                src_transform=transform, src_crs=GEOGRAPHIC_CRS,
                src_nodata=nodata,
                dst_transform=out_transform, dst_crs=dst_crs,
                dst_nodata=nodata,
                resampling=Resampling.nearest,
            )
            return {
                "array": dst,
                "bounds": tuple(out_transform * (0, 0, width, height)),
                "transform": out_transform,
                "crs": dst_crs.to_string(),
                "nodata": nodata,
            }

        win = from_bounds(bounds["west"], bounds["south"], bounds["east"], bounds["north"],
                          transform=src.transform)
        array = src.read(band, window=win)
        return {
            "array": array,
            "bounds": bounds,
            "transform": src.window_transform(win),
            "crs": src.crs.to_string() if src.crs else GEOGRAPHIC_CRS,
            "nodata": nodata if nodata is not None else src.nodata,
        }


def ingest_netcdf(source_path, aoi=None, tile: Optional[dict] = None,
                  variable: Optional[str] = None):
    """Clip a NetCDF dataset (regular lat/lon grid) to the AOI (or tile).

    Returns the clipped xarray Dataset (or DataArray for ``variable``).
    """
    import xarray as xr

    src_path = _resolve_path(source_path)
    if src_path is None or not src_path.exists():
        raise FileNotFoundError(f"NetCDF source not found: {src_path}")

    bounds = _clip_bounds(aoi, tile)
    ds = xr.open_dataset(src_path)

    lon_name = "longitude" if "longitude" in ds.coords else "lon"
    lat_name = "latitude" if "latitude" in ds.coords else "lat"
    if lon_name not in ds.coords or lat_name not in ds.coords:
        ds.close()
        raise ValueError(
            f"NetCDF {src_path.name} has no {lon_name}/{lat_name} coordinates."
        )

    clipped = ds.sel(
        {lat_name: slice(bounds["south"], bounds["north"]),
         lon_name: slice(bounds["west"], bounds["east"])},
        drop=True,
    )
    if variable is not None:
        if variable not in clipped.variables:
            ds.close()
            raise ValueError(
                f"Variable '{variable}' not in {src_path.name}; available: "
                f"{sorted(clipped.data_vars)}"
            )
        result = clipped[variable]
        ds.close()
        return result
    return clipped


def ingest_tabular(source_path, aoi=None, tile: Optional[dict] = None,
                   lat_col: str = "latitude", lon_col: str = "longitude"):
    """Filter a tabular source (CSV/parquet) to rows inside the AOI (or tile)."""
    src_path = _resolve_path(source_path)
    if src_path is None or not src_path.exists():
        raise FileNotFoundError(f"Tabular source not found: {src_path}")

    suffix = src_path.suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(src_path)
    else:
        df = pd.read_csv(src_path)

    if lat_col not in df.columns or lon_col not in df.columns:
        raise ValueError(
            f"Source {src_path.name} has no '{lat_col}'/'{lon_col}' columns; "
            f"available: {list(df.columns)[:12]}"
        )

    bounds = _clip_bounds(aoi, tile)
    mask = (
        df[lon_col].between(bounds["west"], bounds["east"])
        & df[lat_col].between(bounds["south"], bounds["north"])
    )
    return df[mask].copy()


def ingest_vector(source_path, aoi=None, tile: Optional[dict] = None):
    """Clip a vector source (GeoJSON/GPKG/...) to the AOI (or tile)."""
    src_path = _resolve_path(source_path)
    if src_path is None or not src_path.exists():
        raise FileNotFoundError(f"Vector source not found: {src_path}")

    gdf = gpd.read_file(src_path)
    if gdf.crs is not None and gdf.crs.to_string() != GEOGRAPHIC_CRS:
        gdf = gdf.to_crs(GEOGRAPHIC_CRS)

    bounds = _clip_bounds(aoi, tile)
    clip_box = box(bounds["west"], bounds["south"], bounds["east"], bounds["north"])
    return gdf[gdf.geometry.intersects(clip_box)].copy()
