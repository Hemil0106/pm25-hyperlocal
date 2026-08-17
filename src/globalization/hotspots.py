"""Generalized, scope-gated hotspot detection (Phase 17).

Hotspots are AQI-category threshold zones ("predicted high-pollution zones"),
NOT statistical hotspot analysis and NOT emission sources. Detection reuses
the locked Delhi approach (VERY_POOR+ category mask, connected-component
polygonization) and is gated by model scope: no validated global model, no
global hotspots.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from .model_scopes import resolve_model_scope
from .aqi import CATEGORY_ORDER

logger = logging.getLogger(__name__)

DELHI_HOTSPOTS = "data/processed/hotspots_500m.geojson"


def hotspot_plan(config, aoi, date: Optional[str] = None) -> dict:
    scope = resolve_model_scope(config, aoi, date=date)
    available = bool(scope.get("can_predict", False)) and Path(DELHI_HOTSPOTS).exists()
    return {
        "aoi": {"name": aoi.name, "mode": aoi.mode, "bounds": aoi.bounds},
        "date": date,
        "available": available,
        "scope_id": scope.get("scope_id"),
        "scope_status": scope.get("status"),
        "reason": (
            "Delhi prototype hotspots available (AQI-threshold high-pollution "
            "zones)."
            if available
            else (scope.get("reason")
                  if not scope.get("can_predict", False)
                  else f"Hotspot output missing: {DELHI_HOTSPOTS}")
        ),
        "definition": "Predicted high-pollution zone: AQI category >= "
                      "VERY_POOR on predicted PM2.5. Not a statistical "
                      "hotspot analysis and not an emission source.",
        "minimum_category": "VERY_POOR",
        "categories": CATEGORY_ORDER,
    }


def hotspots_for_aoi(config, aoi, date: Optional[str] = None) -> dict:
    plan = hotspot_plan(config, aoi, date=date)
    if not plan["available"]:
        return {
            "status": "unavailable",
            "reason": plan["reason"],
            "aoi": aoi.name,
            "date": date,
            "hotspots": None,
            "definition": plan["definition"],
        }

    try:
        import geopandas as gpd

        hs = gpd.read_file(DELHI_HOTSPOTS)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Hotspot read failed: %s", exc)
        return {"status": "unavailable", "reason": f"Hotspot read failed: {exc}",
                "aoi": aoi.name, "date": date, "hotspots": None,
                "definition": plan["definition"]}

    return {
        "status": "available",
        "reason": "Delhi prototype hotspots (locked output).",
        "aoi": aoi.name,
        "date": date,
        "n_hotspots": int(len(hs)) if hs is not None else 0,
        "hotspots": hs if (hs is not None and not hs.empty) else None,
        "definition": plan["definition"],
    }
