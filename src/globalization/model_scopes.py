"""Model scope resolution and enforcement (Phase 10, 13).

A model scope is the explicit geographic/training claim attached to a model.
The locked Delhi model has scope ``prototype_local`` and MUST NEVER be
presented as global/regional. Global/regional scopes are ``unavailable``
until a model is actually trained and spatially validated on observations
from that scope - this module is what makes that claim structurally
impossible to violate.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from .ground_truth import normalize_ground_truth

logger = logging.getLogger(__name__)


def get_model_scopes(config) -> dict:
    return config.get("model_scopes", {}).get("scopes", {})


def resolve_model_scope(config, aoi, date: Optional[str] = None,
                        dataset_mode: Optional[str] = None) -> dict:
    """Determine the applicable model scope for an AOI.

    Returns a dict describing the scope (available or unavailable) with an
    honest reason. Never fabricates availability for untrained scopes.
    """
    scopes = get_model_scopes(config)
    default_scope = config.get("model_scopes", {}).get("default_scope", "prototype_local")
    scope_cfg = scopes.get(default_scope, {})

    # prototype_local is the Delhi model: available only when its model files
    # actually exist and the AOI is the Delhi region.
    model_file = scope_cfg.get("model_file")
    rf_model_file = scope_cfg.get("rf_model_file")
    trained = bool(model_file and Path(model_file).exists())

    is_delhi = False
    if aoi.name == "Delhi":
        is_delhi = True
    elif not aoi.is_global:
        b = aoi.bounds
        is_delhi = (
            b["west"] >= 77.0 - 1e-6 and b["east"] <= 77.4 + 1e-6
            and b["south"] >= 28.4 - 1e-6 and b["north"] <= 28.8 + 1e-6
        )

    if default_scope == "prototype_local":
        if not trained:
            return {
                "scope_id": "prototype_local",
                "label": scope_cfg.get("label", "Delhi Prototype"),
                "status": "unavailable",
                "reason": "Delhi prototype model files not found.",
                "can_predict": False,
                "model_file": model_file,
                "rf_model_file": rf_model_file,
            }
        if not is_delhi:
            return {
                "scope_id": "prototype_local",
                "label": scope_cfg.get("label", "Delhi Prototype"),
                "status": "unavailable_for_aoi",
                "reason": (
                    f"Model scope 'prototype_local' is trained on Delhi only; "
                    f"it cannot predict for AOI '{aoi.name}'. "
                    "Global/regional PM2.5 prediction unavailable - no "
                    "validated model trained on observations in this scope."
                ),
                "can_predict": False,
                "model_file": model_file,
                "rf_model_file": rf_model_file,
            }
        return {
            "scope_id": "prototype_local",
            "label": scope_cfg.get("label", "Delhi Prototype"),
            "status": "available",
            "reason": "Delhi prototype model trained and validated (leave-one-group-out CV).",
            "can_predict": True,
            "model_file": model_file,
            "rf_model_file": rf_model_file,
        }

    # Explicit other scopes
    status = scope_cfg.get("status", "unavailable")
    return {
        "scope_id": default_scope,
        "label": scope_cfg.get("label", default_scope),
        "status": status,
        "reason": scope_cfg.get("reason", "No validated model for this scope."),
        "can_predict": False,
        "model_file": scope_cfg.get("model_file"),
        "rf_model_file": scope_cfg.get("rf_model_file"),
    }


def assert_scope_allows_inference(config, aoi, date=None, dataset_mode=None):
    """Raise if the AOI's scope cannot produce predictions."""
    scope = resolve_model_scope(config, aoi, date=date, dataset_mode=dataset_mode)
    if not scope.get("can_predict", False):
        raise ValueError(scope.get("reason", "Model scope unavailable."))
    return scope


def prediction_scope_metadata(config, aoi, date=None) -> dict:
    """Scope metadata attached to every prediction output."""
    scope = resolve_model_scope(config, aoi, date=date)
    return {
        "scope_id": scope.get("scope_id"),
        "scope_label": scope.get("label"),
        "scope_status": scope.get("status"),
        "reason": scope.get("reason"),
        "model_file": scope.get("model_file"),
        "date": date,
        "trained_on": "Delhi CPCB observations (prototype_local)" if scope.get("scope_id") == "prototype_local" else None,
    }


def scopes_summary(config) -> dict:
    """Expose the full scope catalog with availability for API/dashboard."""
    scopes = get_model_scopes(config)
    summary = {}
    for scope_id, scope in scopes.items():
        if scope.get("status"):
            status = scope.get("status")
            reason = scope.get("reason", "")
        else:
            model_file = scope.get("model_file")
            status = "available" if model_file and Path(model_file).exists() else "unavailable"
            reason = "" if status == "available" else f"No trained model for scope '{scope_id}'."
        summary[scope_id] = {
            "label": scope.get("label", scope_id),
            "status": status,
            "reason": reason,
            "training_region": scope.get("training_region"),
        }
    return summary


def attach_scope_tag(prediction_records, scope: dict) -> list:
    """Annotate prediction records with the producing scope (never fabricate)."""
    tag = {
        "scope_id": scope.get("scope_id"),
        "scope_status": scope.get("status"),
    }
    tagged = []
    for record in prediction_records:
        record = dict(record)
        record["model_scope"] = tag
        tagged.append(record)
    return tagged
