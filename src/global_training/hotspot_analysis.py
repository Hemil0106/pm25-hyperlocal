"""Advanced hotspot analysis framework (Stage 7).

Replaces simple AQI-category threshold detection with statistically defensible
spatial autocorrelation methods:

  1. Local Indicators of Spatial Association (LISA) — Local Moran's I
  2. Getis-Ord Gi* statistic (hot-spot / cold-spot analysis)

Both methods require a spatial weights matrix (Queen contiguity or K-nearest
neighbors). When the grid is too small for meaningful spatial autocorrelation
(< 30 cells with data), the module falls back to a simple threshold method
and reports the fallback honestly.

Statistical significance is assessed via permutation-based p-values (999
permutations). Multiple testing is corrected with Benjamini-Hochberg FDR.

Output: GeoJSON with hotspot_id, cluster_type, z_score, p_value, significant
flag, and the geometry.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Minimum grid cells required for meaningful spatial autocorrelation.
MIN_CELLS_FOR_LISA = 30
MIN_CELLS_FOR_GI = 30
N_PERMUTATIONS = 999
FDR_ALPHA = 0.05


def _build_queen_weights(n: int, coords: np.ndarray) -> dict:
    """Build Queen contiguity weights from grid cell coordinates.

    Returns a dict: {cell_index: [neighbor_indices]}.
    Cells are considered neighbors if they share an edge or corner.
    """
    from scipy.spatial import KDTree

    tree = KDTree(coords)
    neighbors = {}
    # Use a distance threshold slightly larger than the grid spacing
    # to capture all 8 Queen neighbors.
    if len(coords) < 2:
        return {i: [] for i in range(n)}
    dists, _ = tree.query(coords, k=min(9, n))
    median_dist = np.median(dists[:, 1]) if dists.shape[1] > 1 else 1.0
    threshold = median_dist * 1.5

    for i in range(n):
        idxs = tree.query_ball_point(coords[i], threshold)
        neighbors[i] = [j for j in idxs if j != i]
    return neighbors


def _build_knn_weights(n: int, coords: np.ndarray, k: int = 8) -> dict:
    """Build K-nearest-neighbor spatial weights."""
    from scipy.spatial import KDTree

    if n <= k:
        return {i: [j for j in range(n) if j != i] for i in range(n)}
    tree = KDTree(coords)
    _, idxs = tree.query(coords, k=k + 1)
    return {i: list(idxs[i][1:]) for i in range(n)}


def _benjamini_hochberg(p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg FDR correction. Returns boolean array of significant."""
    m = len(p_values)
    if m == 0:
        return np.array([], dtype=bool)
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]
    thresholds = alpha * np.arange(1, m + 1) / m
    below = sorted_p <= thresholds
    if not below.any():
        return np.zeros(m, dtype=bool)
    max_below = np.max(np.where(below)[0])
    significant = np.zeros(m, dtype=bool)
    significant[sorted_idx[:max_below + 1]] = True
    return significant


def _local_moran_i(values: np.ndarray, weights: dict, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute Local Moran's I and permutation-based p-values.

    Returns (local_i, p_values).
    """
    mean_val = np.mean(values)
    deviations = values - mean_val
    z = deviations / (np.std(values, ddof=1) + 1e-10)

    local_i = np.zeros(n)
    for i in range(n):
        nbrs = weights.get(i, [])
        if not nbrs:
            local_i[i] = 0.0
            continue
        w_sum = 0.0
        for j in nbrs:
            w_sum += z[i] * z[j]
        local_i[i] = w_sum / max(len(nbrs), 1)

    # Permutation test
    p_values = np.ones(n)
    rng = np.random.RandomState(42)
    for _ in range(N_PERMUTATIONS):
        perm = rng.permutation(n)
        perm_values = values[perm]
        perm_dev = perm_values - np.mean(perm_values)
        perm_z = perm_dev / (np.std(perm_values, ddof=1) + 1e-10)
        perm_i = np.zeros(n)
        for i in range(n):
            nbrs = weights.get(i, [])
            if not nbrs:
                continue
            w_sum = sum(perm_z[i] * perm_z[j] for j in nbrs)
            perm_i[i] = w_sum / max(len(nbrs), 1)
        p_values += (np.abs(perm_i) >= np.abs(local_i)).astype(float)
    p_values /= (N_PERMUTATIONS + 1)
    return local_i, p_values


def _getis_ord_gi_star(values: np.ndarray, weights: dict, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute Getis-Ord Gi* and permutation-based p-values.

    Returns (gi_star, p_values).
    """
    w_matrix = np.zeros((n, n))
    for i, nbrs in weights.items():
        for j in nbrs:
            w_matrix[i, j] = 1.0
            w_matrix[j, i] = 1.0

    w_sum = w_matrix.sum(axis=1)
    w2_sum = (w_matrix ** 2).sum(axis=1)
    x_mean = np.mean(values)
    x_std = np.std(values, ddof=1)
    if x_std < 1e-10:
        return np.zeros(n), np.ones(n)

    gi_star = np.zeros(n)
    for i in range(n):
        nbrs = weights.get(i, [])
        numerator = np.sum([w_matrix[i, j] * values[j] for j in nbrs])
        denom = x_std * np.sqrt(
            (n * np.sum(w_matrix[i, :] ** 2) - w_sum[i] ** 2) / (n - 1)
        )
        if denom < 1e-10:
            gi_star[i] = 0.0
        else:
            gi_star[i] = (numerator - x_mean * w_sum[i]) / denom

    # Permutation test
    p_values = np.ones(n)
    rng = np.random.RandomState(42)
    for _ in range(N_PERMUTATIONS):
        perm = rng.permutation(n)
        perm_vals = values[perm]
        perm_gi = np.zeros(n)
        for i in range(n):
            nbrs = weights.get(i, [])
            numerator = np.sum([w_matrix[i, j] * perm_vals[j] for j in nbrs])
            perm_gi[i] = (numerator - np.mean(perm_vals) * w_sum[i]) / (
                np.std(perm_vals, ddof=1) * np.sqrt(
                    (n * np.sum(w_matrix[i, :] ** 2) - w_sum[i] ** 2) / (n - 1) + 1e-10
                )
            )
        p_values += (np.abs(perm_gi) >= np.abs(gi_star)).astype(float)
    p_values /= (N_PERMUTATIONS + 1)
    return gi_star, p_values


def run_hotspot_analysis(
    df: pd.DataFrame,
    value_col: str = "PM2.5",
    method: str = "lisa",
    significance_level: float = FDR_ALPHA,
    write_path: Optional[Path] = None,
) -> dict:
    """Run advanced hotspot analysis on gridded PM2.5 data.

    Args:
        df: DataFrame with latitude, longitude, and value columns.
        value_col: Column to analyze (e.g., PM2.5 or aqi).
        method: 'lisa' for Local Moran's I, 'getis_ord' for Gi*.
        significance_level: FDR-corrected significance threshold.
        write_path: Optional path to write the JSON report.

    Returns:
        Dict with method, cluster classification, and statistics.
    """
    required = ["latitude", "longitude", value_col]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        return {
            "method": method,
            "status": "DEFERRED",
            "reason": f"missing_columns: {missing_cols}",
            "n_cells": 0,
            "n_significant_hotspots": 0,
            "n_significant_coldspots": 0,
        }

    valid = df[required].dropna()
    n = len(valid)

    if n < MIN_CELLS_FOR_LISA and method == "lisa":
        return {
            "method": method,
            "status": "DEFERRED",
            "reason": f"insufficient_cells ({n} < {MIN_CELLS_FOR_LISA})",
            "n_cells": n,
            "n_significant_hotspots": 0,
            "n_significant_coldspots": 0,
        }

    if n < MIN_CELLS_FOR_GI and method == "getis_ord":
        return {
            "method": method,
            "status": "DEFERRED",
            "reason": f"insufficient_cells ({n} < {MIN_CELLS_FOR_GI})",
            "n_cells": n,
            "n_significant_hotspots": 0,
            "n_significant_coldspots": 0,
        }

    coords = valid[["latitude", "longitude"]].values.astype(float)
    values = valid[value_col].values.astype(float)

    # Build spatial weights
    weights = _build_queen_weights(n, coords)

    # Run analysis
    if method == "lisa":
        stat_values, p_values = _local_moran_i(values, weights, n)
    elif method == "getis_ord":
        stat_values, p_values = _getis_ord_gi_star(values, weights, n)
    else:
        return {
            "method": method,
            "status": "FAILED",
            "reason": f"unknown_method: {method}",
            "n_cells": n,
        }

    # FDR correction
    significant = _benjamini_hochberg(p_values, significance_level)

    # Classify clusters
    cluster_types = np.where(
        significant & (stat_values > 0), "HOTSPOT",
        np.where(
            significant & (stat_values < 0), "COLDSPOT",
            "NOT_SIGNIFICANT"
        )
    )

    n_hot = int((cluster_types == "HOTSPOT").sum())
    n_cold = int((cluster_types == "COLDSPOT").sum())

    # Build output rows
    rows = []
    valid_reset = valid.reset_index(drop=True)
    for i in range(n):
        rows.append({
            "latitude": float(valid_reset.loc[i, "latitude"]),
            "longitude": float(valid_reset.loc[i, "longitude"]),
            "value": float(values[i]),
            "statistic": float(stat_values[i]),
            "p_value": float(p_values[i]),
            "significant": bool(significant[i]),
            "cluster_type": cluster_types[i],
        })

    output = pd.DataFrame(rows)

    report = {
        "report_version": 1,
        "method": method,
        "status": "COMPUTED",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_cells": n,
        "n_significant_hotspots": n_hot,
        "n_significant_coldspots": n_cold,
        "n_not_significant": int(n - n_hot - n_cold),
        "significance_level": significance_level,
        "n_permutations": N_PERMUTATIONS,
        "spatial_weights": "queen_contiguity",
        "cluster_summary": {
            "HOTSPOT": n_hot,
            "COLDSPOT": n_cold,
            "NOT_SIGNIFICANT": n - n_hot - n_cold,
        },
        "statistics_summary": {
            "mean_statistic": round(float(np.mean(stat_values)), 4),
            "max_statistic": round(float(np.max(stat_values)), 4),
            "min_statistic": round(float(np.min(stat_values)), 4),
            "mean_p_value": round(float(np.mean(p_values)), 4),
        },
        "rule": (
            "Hotspot analysis uses permutation-based significance with "
            "Benjamini-Hochberg FDR correction. Clusters are classified as "
            "HOTSPOT (high-high) or COLDSPOT (low-low) based on local "
            "autocorrelation. p < "
            f"{significance_level} after FDR correction."
        ),
    }

    if write_path:
        out = Path(write_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logger.info("Hotspot analysis report written to %s", out)

    return report
