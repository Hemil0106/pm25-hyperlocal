"""Spatial normalization of feature rasters to the common grid (Milestone 16).

- Continuous features (elevation, AOD, NDVI, temperature, night lights) are
  sampled with bilinear interpolation.
- Categorical/integer features (land class, QA flags) are sampled with nearest
  neighbor.

Sampling is performed per grid centroid so results are deterministic and
aligned with the observation-feature temporal join.
"""

from __future__ import annotations

import logging
from typing import Optional

import geopandas as gpd
import numpy as np

logger = logging.getLogger(__name__)

CONTINUOUS_FEATURES = ("elevation_m", "aod", "ndvi", "temperature_c",
                       "relative_humidity", "wind_speed", "night_lights",
                       "road_density")
CATEGORICAL_FEATURES = ("land_class", "qa_flag")


def sample_raster_at_points(raster_path, grid: gpd.GeoDataFrame, band: int = 1,
                            method: str = "bilinear",
                            nodata: Optional[float] = None) -> np.ndarray:
    """Sample a single-band raster at grid centroid points.

    Returns a float array aligned with ``grid`` row order; missing/NoData
    samples are NaN.
    """
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import transform as warp_transform

    grid = grid.to_crs("EPSG:4326")
    xs = grid["longitude"].astype(float).values
    ys = grid["latitude"].astype(float).values

    resampling = Resampling.nearest if method == "nearest" else Resampling.bilinear

    with rasterio.open(raster_path) as src:
        if src.count < band:
            raise ValueError(f"Raster has {src.count} bands; band {band} requested.")
        if nodata is None:
            nodata = src.nodata
        if src.crs is not None and src.crs.to_string() != "EPSG:4326":
            sx, sy = warp_transform("EPSG:4326", src.crs, xs.tolist(), ys.tolist())
        else:
            sx, sy = xs.tolist(), ys.tolist()

        values = list(
            src.sample(zip(sx, sy), indexes=band, resampling=resampling)
        )

    out = np.asarray([float(v[0]) for v in values], dtype=float)
    if nodata is not None:
        out[out == float(nodata)] = np.nan
    return out


def normalize_feature_to_grid(raster_path, grid: gpd.GeoDataFrame,
                              feature_name: str, band: int = 1,
                              method: Optional[str] = None,
                              nodata: Optional[float] = None) -> np.ndarray:
    """Pick the sampling method by feature semantics, then sample."""
    if method is None:
        method = (
            "nearest" if feature_name in CATEGORICAL_FEATURES else "bilinear"
        )
    if method not in ("bilinear", "nearest"):
        raise ValueError(f"Unsupported sampling method '{method}'.")
    return sample_raster_at_points(raster_path, grid, band=band, method=method,
                                   nodata=nodata)
