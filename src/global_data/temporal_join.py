"""Temporal join: observations x common grid -> observation-feature table (M16).

Each daily observation row is joined to the common-grid cell containing its
location, then augmented with feature values (spatial features like elevation,
and features sampled for the observation's date). Every feature column carries
an explicit ``<feature>_available`` flag so callers know exactly which features
were actually present - missing features are never silently filled.
"""

from __future__ import annotations

import logging
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

logger = logging.getLogger(__name__)


def nearest_grid_cell_id(grid: gpd.GeoDataFrame, longitude, latitude) -> str:
    """grid_id of the cell containing a point ("" when outside the grid)."""
    point = Point(longitude, latitude)
    candidates = list(grid.sindex.intersection((longitude, latitude, longitude, latitude)))
    for idx in candidates:
        geom = grid.iloc[idx]["geometry"]
        if geom is not None and geom.contains(point):
            return str(grid.iloc[idx]["grid_id"])
    return ""


def spatial_join_to_grid(daily_df: pd.DataFrame, grid: gpd.GeoDataFrame) -> pd.DataFrame:
    """Attach grid_id to each daily observation via spatial containment."""
    if daily_df.empty or grid is None or grid.empty:
        out = daily_df.copy()
        if not out.empty:
            out["grid_id"] = ""
        else:
            out = out.assign(grid_id=pd.Series(dtype=str))
        return out

    grid = grid.copy()
    if grid.crs is not None and grid.crs.to_string() != "EPSG:4326":
        grid = grid.to_crs("EPSG:4326")
    grid = grid.set_index(pd.RangeIndex(len(grid)))

    out = daily_df.copy()
    grid_ids = []
    for _, row in out.iterrows():
        grid_ids.append(nearest_grid_cell_id(grid, row["longitude"], row["latitude"]))
    out["grid_id"] = grid_ids
    return out


def add_feature_to_observations(obs: pd.DataFrame, feature_name: str,
                                values: np.ndarray, grid: gpd.GeoDataFrame,
                                source_status: str = "available") -> pd.DataFrame:
    """Add a feature column + ``<feature>_available`` flag to the obs table.

    Feature values are assumed to be aligned with ``grid`` rows; each
    observation takes the value of the feature at its grid cell. When the
    source is unavailable, the column is NaN and the flag is False.
    """
    out = obs.copy()
    flag_col = f"{feature_name}_available"
    if source_status != "available" or values is None:
        out[feature_name] = np.nan
        out[flag_col] = False
        return out

    lookup = {}
    if len(values) == len(grid):
        for idx, grid_id in enumerate(grid["grid_id"]):
            lookup[str(grid_id)] = float(values[idx])
    out[feature_name] = out["grid_id"].map(lookup).astype(float)
    out[flag_col] = out[feature_name].notna()
    return out


def temporal_join(daily_df: pd.DataFrame, grid: gpd.GeoDataFrame,
                  feature_layers: Optional[dict] = None) -> tuple[pd.DataFrame, dict]:
    """Build the observation-feature table.

    ``feature_layers``: {feature_name: {"values": np.ndarray, "grid": gpd.GeoDataFrame,
    "source_status": str}}. Returns (table, report) with per-feature availability.
    """
    feature_layers = feature_layers or {}
    table = spatial_join_to_grid(daily_df, grid)
    report = {
        "n_observations": int(len(table)),
        "n_grid_cells": int(len(grid)) if grid is not None else 0,
        "features": {},
    }
    for feature_name, layer in feature_layers.items():
        layer_grid = layer.get("grid")
        table = add_feature_to_observations(
            table, feature_name,
            layer.get("values"),
            layer_grid if layer_grid is not None else grid,
            source_status=layer.get("source_status", "available"),
        )
        report["features"][feature_name] = {
            "available": int(table[f"{feature_name}_available"].sum()),
            "source_status": layer.get("source_status", "available"),
        }
    return table, report
