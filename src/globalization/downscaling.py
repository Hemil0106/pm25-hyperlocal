"""Generalized, scope-gated downscaling (Phase 15).

Downscaling is AOI-aware and honest:
  - it is only meaningful for scopes with a trained parent model;
  - downscaling != resizing (residual correction against fine-resolution
    predictors, not interpolation);
  - a 500 m product does NOT imply higher accuracy than the 1 km product;
    accuracy is governed by spatial validation of the parent model.
Global downscaling stays unavailable until a validated global parent model
and fine-resolution predictors exist for the target AOI.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from .model_scopes import resolve_model_scope

logger = logging.getLogger(__name__)

# Locked Delhi downscaling outputs reused for the prototype_local scope.
DELHI_DOWNSCALE_FEATURES = "data/processed/downscaling_features_500m.parquet"
DELHI_DOWNSCALE_TRAINING = "data/processed/downscaling_training.parquet"
DELHI_PREDICTIONS_500M = "data/processed/pm25_500m_predictions.parquet"


def downscaling_plan(config, aoi, date: Optional[str] = None) -> dict:
    """Report downscaling availability and honest caveats for an AOI."""
    scope = resolve_model_scope(config, aoi, date=date)
    can_downscale = bool(scope.get("can_predict", False))
    if can_downscale:
        features_ok = Path(DELHI_DOWNSCALE_FEATURES).exists()
        training_ok = Path(DELHI_DOWNSCALE_TRAINING).exists()
        can_downscale = features_ok and training_ok

    return {
        "aoi": {"name": aoi.name, "mode": aoi.mode, "bounds": aoi.bounds},
        "date": date,
        "can_downscale": can_downscale,
        "parent_scope_id": scope.get("scope_id"),
        "parent_scope_status": scope.get("status"),
        "reason": (
            scope.get("reason")
            if not scope.get("can_predict", False)
            else (
                "Delhi prototype parent model available; downscaling reuses "
                "locked residual-correction outputs."
                if can_downscale
                else "Parent model available but downscaling outputs missing; "
                     "run the Delhi pipeline (python run.py)."
            )
        ),
        "caveats": [
            "Downscaling is residual correction against 500 m predictors - "
            "NOT raster resizing.",
            "A 500 m product does NOT imply higher accuracy than 1 km; "
            "accuracy is governed by spatial validation of the parent model.",
        ],
    }


def downscale_for_aoi(config, aoi, date: Optional[str] = None) -> dict:
    """Return 500 m downscaled predictions for an AOI, gated by scope."""
    plan = downscaling_plan(config, aoi, date=date)
    if not plan["can_downscale"]:
        return {
            "status": "unavailable",
            "reason": plan["reason"],
            "aoi": aoi.name,
            "date": date,
            "predictions": None,
            "caveats": plan["caveats"],
        }

    src = Path(DELHI_PREDICTIONS_500M)
    if not src.exists():
        return {"status": "missing_output",
                "reason": f"Missing {src}", "aoi": aoi.name,
                "date": date, "predictions": None, "caveats": plan["caveats"]}

    frame = None
    try:
        import geopandas as gpd

        frame = gpd.read_parquet(src)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Downscaling output read failed: %s", exc)

    if frame is None or frame.empty:
        return {"status": "unavailable",
                "reason": "Downscaled predictions are empty/unreadable.",
                "aoi": aoi.name, "date": date, "predictions": None,
                "caveats": plan["caveats"]}

    return {
        "status": "available",
        "reason": "Delhi prototype downscaled predictions (locked output).",
        "aoi": aoi.name,
        "date": date,
        "resolution": "500m",
        "n_cells": int(len(frame)),
        "predictions": frame,
        "caveats": plan["caveats"],
    }
