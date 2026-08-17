"""Dataset registry for the globalized platform (Phase 5).

The registry describes every dataset the platform knows about, its coverage
scope, whether it is enabled, and whether its raw inputs are present locally
(download.enabled=false means availability reflects *local* data only, which
is reported honestly rather than assumed).
"""

from __future__ import annotations

import json
import logging
from datetime import date as _date
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _path_exists(config, path_key: str) -> bool:
    value = config.get(path_key)
    if not value:
        return False
    p = Path(value)
    if not p.is_absolute():
        p = Path.cwd() / p
    return p.exists()


def build_dataset_registry(config, aoi, write_path: Optional[str] = None) -> dict:
    datasets_cfg = config.get("datasets", {})
    ground_truth_cfg = config.get("ground_truth", {})

    def file_check(ds_cfg: dict) -> dict:
        input_file = ds_cfg.get("input_file")
        raw_dir = ds_cfg.get("raw_dir")
        present = False
        present_path = None
        if input_file:
            present = _path_exists(ds_cfg, "input_file")
            present_path = input_file if present else None
        elif raw_dir:
            present = _path_exists(ds_cfg, "raw_dir")
            present_path = raw_dir if present else None
        return {"local_data_present": present, "local_path": present_path}

    registry = {
        "registry_version": 1,
        "built_for_aoi": {
            "name": aoi.name,
            "mode": aoi.mode,
            "bounds": aoi.bounds,
            "area_km2": round(aoi.area_km2(), 2),
        },
        "download_enabled": bool(config.get("download", {}).get("enabled", False)),
        "datasets": {},
    }

    specs = {
        "aod": {
            "name": "MODIS MAIAC AOD (MCD19A2)",
            "type": "gridded",
            "coverage": "global (1 km)",
            "config": datasets_cfg.get("aod", {}),
        },
        "cpcb": {
            "name": "CPCB PM2.5 ground observations",
            "type": "point",
            "coverage": "India (CPCB network)",
            "config": datasets_cfg.get("cpcb", {}),
        },
        "weather": {
            "name": "ERA5-Land meteorology",
            "type": "gridded",
            "coverage": "global (ERA5-Land)",
            "config": datasets_cfg.get("weather", {}),
        },
        "ndvi": {
            "name": "MODIS Terra NDVI (MOD13Q1)",
            "type": "gridded",
            "coverage": "global (250 m)",
            "config": datasets_cfg.get("ndvi", {}),
        },
        "osm": {
            "name": "OpenStreetMap roads",
            "type": "vector",
            "coverage": "global (OpenStreetMap)",
            "config": datasets_cfg.get("osm", {}),
        },
        "dem": {
            "name": "NASA SRTM elevation (GL1)",
            "type": "gridded",
            "coverage": "global (SRTM v003)",
            "config": datasets_cfg.get("dem", {}),
        },
        "viirs": {
            "name": "NASA VIIRS night lights (VNP46A2)",
            "type": "gridded",
            "coverage": "global (VNP46A2)",
            "config": datasets_cfg.get("viirs", {}),
        },
    }

    for ds_id, spec in specs.items():
        ds_cfg = spec["config"]
        entry = {
            "id": ds_id,
            "name": spec["name"],
            "type": spec["type"],
            "coverage": spec["coverage"],
            "enabled": bool(ds_cfg.get("enabled", False)),
        }
        entry.update(file_check(ds_cfg))
        registry["datasets"][ds_id] = entry

    gt_cfg = ground_truth_cfg
    registry["datasets"]["ground_truth"] = {
        "id": "ground_truth",
        "name": "Normalized ground-truth PM2.5",
        "type": "point",
        "coverage": "derived from configured sources",
        "enabled": bool(gt_cfg.get("enabled", False)),
        "local_data_present": _path_exists(gt_cfg, "output_file"),
        "local_path": gt_cfg.get("output_file"),
        "output_file": gt_cfg.get("output_file"),
        "sources": {},
    }
    for src_id, src_cfg in (gt_cfg.get("sources") or {}).items():
        registry["datasets"]["ground_truth"]["sources"][src_id] = {
            "enabled": bool(src_cfg.get("enabled", False)),
            "network": src_cfg.get("network"),
            "country": src_cfg.get("country"),
            "units": src_cfg.get("units"),
            "local_data_present": _path_exists(src_cfg, "input_file") if src_cfg.get("input_file") else False,
        }

    if write_path:
        out = Path(write_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as file:
            json.dump(registry, file, indent=2)
        logger.info("Dataset registry written to %s", out)

    return registry
