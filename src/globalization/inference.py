"""Generalized, scope-gated inference (Phase 14).

predict_for_aoi resolves the AOI's model scope. Only scopes with a trained,
validated model can produce predictions. For the locked Delhi prototype the
existing processed 500 m predictions are served (tagged with scope metadata);
for every other AOI a structured, honest "unavailable" result is returned -
never fabricated predictions.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from .model_scopes import resolve_model_scope, prediction_scope_metadata

logger = logging.getLogger(__name__)

# Existing locked Delhi outputs reused for the prototype_local scope.
DELHI_PREDICTIONS_500M = "data/processed/pm25_500m_predictions.parquet"
DELHI_PREDICTIONS_1KM = "data/processed/pm25_1km_predictions.parquet"


def inference_plan(config, aoi, date: Optional[str] = None) -> dict:
    """Report whether inference is possible for the AOI/date, and why."""
    scope = resolve_model_scope(config, aoi, date=date)
    plan = {
        "aoi": {"name": aoi.name, "mode": aoi.mode, "bounds": aoi.bounds},
        "date": date,
        "scope_id": scope.get("scope_id"),
        "scope_status": scope.get("status"),
        "can_predict": bool(scope.get("can_predict", False)),
        "reason": scope.get("reason"),
        "resolution": {"500m": str(Path(DELHI_PREDICTIONS_500M).exists()),
                       "1km": str(Path(DELHI_PREDICTIONS_1KM).exists())},
    }
    return plan


def predict_for_aoi(config, aoi, date: Optional[str] = None,
                    resolution: str = "500m") -> dict:
    """Return predictions for an AOI, gated by model scope.

    Returns an "unavailable" dict for scopes without a validated model.
    """
    scope = resolve_model_scope(config, aoi, date=date)
    if not scope.get("can_predict", False):
        return {
            "status": "unavailable",
            "reason": scope.get("reason"),
            "aoi": aoi.name,
            "date": date,
            "predictions": None,
            "scope_metadata": prediction_scope_metadata(config, aoi, date=date),
        }

    if resolution not in ("500m", "1km"):
        raise ValueError("resolution must be '500m' or '1km'.")
    path = DELHI_PREDICTIONS_500M if resolution == "500m" else DELHI_PREDICTIONS_1KM
    src = Path(path)
    if not src.exists():
        return {
            "status": "missing_output",
            "reason": f"Prediction output not found: {src}. Run the Delhi "
                      "pipeline (python run.py) first.",
            "aoi": aoi.name,
            "date": date,
            "predictions": None,
            "scope_metadata": prediction_scope_metadata(config, aoi, date=date),
        }

    frame = pd.read_parquet(src)
    if "date" in frame.columns:
        frame = frame[frame["date"].astype(str) == str(date)].reset_index(drop=True)

    return {
        "status": "available",
        "reason": "Delhi prototype predictions (locked pipeline output).",
        "aoi": aoi.name,
        "date": date,
        "resolution": resolution,
        "n_cells": int(len(frame)),
        "predictions": frame,
        "scope_metadata": prediction_scope_metadata(config, aoi, date=date),
    }
