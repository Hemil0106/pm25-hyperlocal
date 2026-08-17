"""Data availability manifest for the globalized platform (Phase 7).

Answers, honestly: for a given AOI + date, which datasets are actually
available (locally present, enabled, and covering the AOI)? Ground-truth
observations inside the AOI are counted. Model scope availability is also
reported so callers never present "available" when only local data exists.
"""

from __future__ import annotations

import json
import logging
from datetime import date as _date
from pathlib import Path
from typing import Optional

import geopandas as gpd

from .datasets import build_dataset_registry
from .ground_truth import normalize_ground_truth, ground_truth_summary

logger = logging.getLogger(__name__)


def _gt_rows_in_aoi(df, aoi) -> int:
    if df.empty:
        return 0
    bounds = aoi.bounds
    mask = (
        df["longitude"].between(bounds["west"], bounds["east"])
        & df["latitude"].between(bounds["south"], bounds["north"])
    )
    return int(mask.sum())


def build_data_availability(config, aoi, date: Optional[str] = None,
                            write_path: Optional[str] = None) -> dict:
    date = date or config.get("time", {}).get("start_date", "2025-01-01")
    registry = build_dataset_registry(config, aoi)

    gt_df = None
    gt_stats = {"n_rows_in_aoi": 0, "n_stations_in_aoi": 0}
    try:
        gt_df = normalize_ground_truth(config)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Ground-truth normalization failed during availability check: %s", exc)
    if gt_df is not None:
        in_aoi = _gt_rows_in_aoi(gt_df, aoi)
        stats = ground_truth_summary(gt_df)
        gt_stats = {
            "n_rows_total": stats["n_rows"],
            "n_stations_total": stats["n_stations"],
            "n_rows_in_aoi": in_aoi,
            "n_stations_in_aoi": int(gt_df[gt_df["longitude"].between(
                aoi.bounds["west"], aoi.bounds["east"])
                & gt_df["latitude"].between(
                aoi.bounds["south"], aoi.bounds["north"])]["station_id"].nunique())
            if in_aoi else 0,
            "sources": stats["sources"],
            "countries": stats["countries"],
        }

    model_scopes = config.get("model_scopes", {})
    scopes = model_scopes.get("scopes", {})
    scope_status = {}
    for scope_id, scope in scopes.items():
        status = scope.get("status")
        reason = scope.get("reason", "")
        if status is None:
            model_file = scope.get("model_file")
            if model_file and Path(model_file).exists():
                status = "available"
            else:
                status = "unavailable"
                reason = reason or f"No trained model for scope '{scope_id}'."
        scope_status[scope_id] = {
            "label": scope.get("label", scope_id),
            "status": status,
            "reason": reason,
        }

    manifest = {
        "manifest_version": 1,
        "date": date,
        "aoi": {"name": aoi.name, "mode": aoi.mode, "bounds": aoi.bounds},
        "ground_truth": gt_stats,
        "model_scopes": scope_status,
        "datasets": registry["datasets"],
        "overall": "partial",
        "notes": [],
    }

    if aoi.mode == "global":
        manifest["notes"].append(
            "Global ground-truth PM2.5 is not configured (OpenAQ disabled). "
            "No scientifically valid global model exists; global scope is unavailable."
        )
        manifest["overall"] = "data_limited"
    elif aoi.name in ("Delhi", "India"):
        if gt_stats["n_rows_in_aoi"] > 0:
            manifest["overall"] = "available"
        else:
            manifest["overall"] = "no_observations"

    if write_path:
        out = Path(write_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as file:
            json.dump(manifest, file, indent=2)
        logger.info("Data availability manifest written to %s", out)

    return manifest
