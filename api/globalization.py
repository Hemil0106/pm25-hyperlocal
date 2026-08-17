"""Globalization API endpoints (Phase 18-19).

AOI-aware endpoints: /regions, /countries, /aoi, /datasets, /data-availability,
/model-scopes, /output-metadata and a generalized /pm25/bbox query.

All parameters are validated; region names are resolved only against the
configured catalog (no filesystem injection). PM2.5 bbox queries are gated by
model scope - a bbox outside Delhi returns an honest "unavailable" response,
never fabricated predictions.
"""

import logging
from datetime import date
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

from src.config import load_config
from src.geospatial.aoi import resolve_aoi
from src.globalization.availability import build_data_availability
from src.globalization.datasets import build_dataset_registry
from src.globalization.ground_truth import normalize_ground_truth
from src.globalization.model_scopes import scopes_summary, resolve_model_scope
from src.globalization.inference import inference_plan
from src.globalization.downscaling import downscaling_plan
from src.globalization.hotspots import hotspot_plan

logger = logging.getLogger("pm25-mapping-globalization")

router = APIRouter(prefix="/global", tags=["globalization"])

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _config():
    return load_config()


@router.get("/regions")
def get_regions():
    """Catalog of named regions (AOI selector options)."""
    config = _config()
    aoi_cfg = config.get("aoi", {})
    regions = aoi_cfg.get("regions", {})
    out = {}
    for region_id, region_cfg in regions.items():
        name = region_cfg.get("name", region_id)
        try:
            aoi = resolve_aoi(config, region=region_id)
            bounds = aoi.bounds
            area_km2 = round(aoi.area_km2(), 2)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Region %s unavailable: %s", region_id, exc)
            bounds = None
            area_km2 = None
        out[region_id] = {
            "name": name,
            "mode": region_cfg.get("mode"),
            "aqi_scheme": region_cfg.get("aqi_scheme"),
            "bounds": bounds,
            "area_km2": area_km2,
        }
    return {"regions": out, "default": aoi_cfg.get("named_region", {}).get("name", "Delhi")}


@router.get("/countries")
def get_countries():
    """Countries present in the normalized ground-truth table."""
    config = _config()
    gt = normalize_ground_truth(config)
    if gt.empty:
        return {"countries": [], "n_rows": 0, "note": "No ground truth enabled."}
    countries = sorted(gt["country"].dropna().unique().tolist())
    return {"countries": countries, "n_rows": int(len(gt))}


@router.get("/aoi")
def get_aoi_info(region: str = Query(None, description="Named region id"),
                 bbox: str = Query(None, description="min_lon,min_lat,max_lon,max_lat")):
    """Resolve and describe an AOI by region id or explicit bbox."""
    config = _config()
    bbox_dict = None
    if bbox:
        parts = bbox.split(",")
        if len(parts) != 4:
            raise HTTPException(status_code=400, detail="bbox must be min_lon,min_lat,max_lon,max_lat")
        try:
            min_lon, min_lat, max_lon, max_lat = (float(p) for p in parts)
        except ValueError:
            raise HTTPException(status_code=400, detail="bbox values must be numeric")  # noqa: B904
        if not (-180 <= min_lon < max_lon <= 180) or not (-90 <= min_lat < max_lat <= 90):
            raise HTTPException(status_code=400, detail="bbox out of range or inverted")
        bbox_dict = {"min_lon": min_lon, "min_lat": min_lat, "max_lon": max_lon, "max_lat": max_lat}

    if region and bbox_dict:
        raise HTTPException(status_code=400, detail="Provide either region or bbox, not both")
    if not region and not bbox_dict:
        region = None  # default AOI

    try:
        aoi = resolve_aoi(config, region=region, bbox=bbox_dict)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))  # noqa: B904

    scope = resolve_model_scope(config, aoi)
    return {
        "name": aoi.name,
        "mode": aoi.mode,
        "bounds": aoi.bounds,
        "area_km2": round(aoi.area_km2(), 2),
        "is_global": aoi.is_global,
        "metric_crs": None if aoi.is_global else aoi.centroid_crs(),
        "model_scope": {"id": scope.get("scope_id"), "status": scope.get("status"),
                        "can_predict": scope.get("can_predict", False),
                        "reason": scope.get("reason")},
    }


@router.get("/datasets")
def get_datasets():
    """Dataset registry with local availability."""
    config = _config()
    aoi = resolve_aoi(config)
    reg = build_dataset_registry(config, aoi, write_path=None)
    reg.pop("built_for_aoi", None)
    return reg


@router.get("/data-availability")
def get_data_availability(date: date = Query(None),
                          region: str = Query(None, description="Named region id"),
                          bbox: str = Query(None)):
    """Honest data availability manifest for an AOI + date."""
    config = _config()
    bbox_dict = None
    if bbox:
        parts = bbox.split(",")
        if len(parts) != 4:
            raise HTTPException(status_code=400, detail="bbox must be min_lon,min_lat,max_lon,max_lat")
        try:
            min_lon, min_lat, max_lon, max_lat = (float(p) for p in parts)
        except ValueError:
            raise HTTPException(status_code=400, detail="bbox values must be numeric")  # noqa: B904
        bbox_dict = {"min_lon": min_lon, "min_lat": min_lat, "max_lon": max_lon, "max_lat": max_lat}
    try:
        aoi = resolve_aoi(config, region=region, bbox=bbox_dict)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))  # noqa: B904

    date_str = str(date) if date else None
    manifest = build_data_availability(config, aoi, date=date_str, write_path=None)
    return manifest


@router.get("/model-scopes")
def get_model_scopes():
    """Model scope catalog with honest availability."""
    config = _config()
    return {
        "default_scope": config.get("model_scopes", {}).get("default_scope"),
        "scopes": scopes_summary(config),
    }


@router.get("/output-metadata")
def get_output_metadata(date: date = Query(None),
                        region: str = Query(None)):
    """Scope-tagged output metadata: what exists, for which scope (Phase 19)."""
    config = _config()
    try:
        aoi = resolve_aoi(config, region=region)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))  # noqa: B904

    inference = inference_plan(config, aoi, date=str(date) if date else None)
    downscaling = downscaling_plan(config, aoi, date=str(date) if date else None)
    hotspots = hotspot_plan(config, aoi, date=str(date) if date else None)

    return {
        "aoi": {"name": aoi.name, "mode": aoi.mode},
        "date": str(date) if date else None,
        "inference": {
            "can_predict": inference["can_predict"],
            "scope_status": inference["scope_status"],
            "reason": inference["reason"],
            "resolution_500m_output_exists": inference["resolution"]["500m"] == "True",
        },
        "downscaling": {"available": downscaling["can_downscale"],
                        "reason": downscaling["reason"],
                        "caveats": downscaling["caveats"]},
        "hotspots": {"available": hotspots["available"],
                     "reason": hotspots["reason"],
                     "definition": hotspots["definition"]},
        "note": "Global/regional scopes are reported unavailable until a "
                "validated model trained on that scope's observations exists.",
    }


@router.get("/pm25/bbox")
def get_pm25_bbox(date: date,
                  west: float = Query(..., ge=-180, lt=180),
                  south: float = Query(..., ge=-90, lt=90),
                  east: float = Query(..., ge=-180, le=180),
                  north: float = Query(..., ge=-90, le=90)):
    """Predictions inside a bounding box, gated by model scope."""
    config = _config()
    if west >= east or south >= north:
        raise HTTPException(status_code=400, detail="west<east and south<north required")
    if east - west > 10 or north - south > 10:
        raise HTTPException(status_code=400, detail="bbox too large; max 10 degrees per side")

    aoi = resolve_aoi(config, bbox={
        "min_lon": west, "min_lat": south, "max_lon": east, "max_lat": north})
    scope = resolve_model_scope(config, aoi, date=str(date))
    if not scope.get("can_predict", False):
        return {
            "status": "unavailable",
            "reason": scope.get("reason"),
            "date": str(date),
            "bbox": {"west": west, "south": south, "east": east, "north": north},
            "n_cells": 0,
            "cells": [],
        }

    predictions_file = PROJECT_ROOT / "data" / "processed" / "pm25_500m_predictions.parquet"
    if not predictions_file.exists():
        raise HTTPException(status_code=404, detail="Delhi predictions not found; run python run.py.")
    import pandas as pd
    import geopandas as gpd

    frame = gpd.read_parquet(predictions_file)
    frame = frame[frame["date"].astype(str) == str(date)].copy()
    if frame.empty:
        return {"status": "NoData", "date": str(date), "n_cells": 0, "cells": []}

    # Locked predictions geometry is in EPSG:32643 (metric); bbox is 4326.
    # Compute cell centroids in the metric CRS (exact) and transform to 4326.
    if "latitude" not in frame.columns and frame.crs is not None:
        from pyproj import Transformer

        metric_crs = frame.crs.to_string()
        transformer = Transformer.from_crs(metric_crs, "EPSG:4326", always_xy=True)
        cx = frame.geometry.centroid.x
        cy = frame.geometry.centroid.y
        frame["longitude"] = transformer.transform(cx, cy)[0]
        frame["latitude"] = transformer.transform(cx, cy)[1]
    if frame.crs is not None and frame.crs.to_string() != "EPSG:4326":
        frame = frame.to_crs("EPSG:4326")
    if "pm25" not in frame.columns and "pm25_500m_final" in frame.columns:
        frame["pm25"] = pd.to_numeric(frame["pm25_500m_final"], errors="coerce")
    mask = (
        frame["longitude"].between(west, east)
        & frame["latitude"].between(south, north)
    )
    cells = frame.loc[mask]
    cells = cells[~cells["pm25"].isna()] if "pm25" in cells.columns else cells
    records = [
        {
            "latitude": round(float(row["latitude"]), 6),
            "longitude": round(float(row["longitude"]), 6),
            "pm25": round(float(row["pm25"]), 2),
        }
        for _, row in cells.iterrows()
    ]
    return {
        "status": "available",
        "date": str(date),
        "bbox": {"west": west, "south": south, "east": east, "north": north},
        "n_cells": len(records),
        "cells": records,
        "scope": {"id": scope.get("scope_id"), "status": scope.get("status")},
    }
