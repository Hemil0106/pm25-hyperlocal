"""Generalized AQI scheme handling (Phase 16).

The AQI standard is explicit and never silently mixed. Schemes:
  - india_cpcb : reuses the locked ``config.aqi.breakpoints`` (source of truth)
  - us_epa     : inline US EPA 24h breakpoints
  - none       : concentration only (no AQI)

The linear-interpolation formula matches the locked Delhi implementation.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

CATEGORY_ORDER = [
    "GOOD", "SATISFACTORY", "MODERATELY_POLLUTED",
    "POOR", "VERY_POOR", "SEVERE",
]
US_EPA_CATEGORY_ORDER = [
    "GOOD", "MODERATE", "UNHEALTHY_SENSITIVE_GROUPS", "UNHEALTHY",
    "VERY_UNHEALTHY", "HAZARDOUS",
]

US_EPA_CATEGORIES = {
    "GOOD": (0, 50),
    "MODERATE": (51, 100),
    "UNHEALTHY_SENSITIVE_GROUPS": (101, 150),
    "UNHEALTHY": (151, 200),
    "VERY_UNHEALTHY": (201, 300),
    "HAZARDOUS": (301, 500),
}


def get_breakpoints(config, scheme_name: str) -> list:
    """Resolve the breakpoints table for a scheme (India CPCB / US EPA / none)."""
    schemes = config.get("aqi_scheme", {}).get("options", {})
    scheme = schemes.get(scheme_name, {})
    method = scheme.get("method", "")

    if scheme_name == "india_cpcb" or method == "CPCB_NATIONAL_AQI":
        return _breakpoints_from_config_aqi(config)

    if scheme_name == "us_epa" or method == "US_EPA_PM25_AQI":
        return [
            {
                "concentration_low": float(b["concentration_low"]),
                "concentration_high": float(b["concentration_high"]),
                "aqi_low": float(b["aqi_low"]),
                "aqi_high": float(b["aqi_high"]),
            }
            for b in scheme.get("breakpoints", [])
        ]

    raise ValueError(
        f"AQI scheme '{scheme_name}' has no breakpoints (use 'none' for "
        "concentration-only output)."
    )


def _breakpoints_from_config_aqi(config) -> list:
    rows = []
    for b in config.get("aqi", {}).get("breakpoints", []):
        rows.append({
            "concentration_low": float(b["concentration_low"]),
            "concentration_high": float(b["concentration_high"]),
            "aqi_low": float(b["aqi_low"]),
            "aqi_high": float(b["aqi_high"]),
        })
    rows.sort(key=lambda r: r["concentration_low"])
    return rows


def get_categories(config, scheme_name: str) -> dict:
    schemes = config.get("aqi_scheme", {}).get("options", {})
    scheme = schemes.get(scheme_name, {})
    method = scheme.get("method", "")
    if scheme_name == "india_cpcb" or method == "CPCB_NATIONAL_AQI":
        return {
            c["name"]: (float(c["aqi_min"]), float(c["aqi_max"]))
            for c in config.get("aqi", {}).get("categories", [])
        }
    return dict(US_EPA_CATEGORIES)


def pm25_to_aqi(value, breakpoints) -> Optional[float]:
    """Locked-compatible PM2.5 -> AQI sub-index (linear, clamped to [0,500])."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return None
    c = float(value)
    if c < 0:
        return None
    selected = None
    for row in breakpoints:
        if c <= row["concentration_high"]:
            selected = row
            break
    if selected is None:
        selected = breakpoints[-1]
    if selected["concentration_high"] <= selected["concentration_low"]:
        aqi = selected["aqi_high"]
    else:
        aqi = (
            (selected["aqi_high"] - selected["aqi_low"])
            / (selected["concentration_high"] - selected["concentration_low"])
            * (c - selected["concentration_low"])
            + selected["aqi_low"]
        )
    return float(np.clip(aqi, 0.0, 500.0))


def aqi_category_name(aqi, categories, scheme_name: str = "india_cpcb") -> Optional[str]:
    if aqi is None or not np.isfinite(aqi):
        return None
    order = US_EPA_CATEGORY_ORDER if scheme_name == "us_epa" else CATEGORY_ORDER
    for name in order:
        lo, hi = categories[name]
        if aqi >= lo:
            if name == order[-1]:
                return name
            nxt = categories[order[order.index(name) + 1]]
            if aqi < nxt[0]:
                return name
    return order[-1]


def aqi_scheme_for(config, aoi) -> dict:
    """Resolve the AQI scheme configured for an AOI's region."""
    aoi_cfg = config.get("aoi", {})
    regions = aoi_cfg.get("regions", {})
    default = config.get("aqi_scheme", {}).get("default", "india_cpcb")

    scheme_name = default
    for region_id, region_cfg in regions.items():
        if region_cfg.get("name") == aoi.name:
            scheme_name = region_cfg.get("aqi_scheme", default)
            break

    return {
        "scheme": scheme_name,
        "label": config.get("aqi_scheme", {}).get("options", {}).get(
            scheme_name, {}).get("label", scheme_name),
    }


def compute_aqi_frame(frame, config, scheme_name: str = "india_cpcb",
                      pm25_col: str = "pm25") -> dict:
    """Compute AQI + category columns on a PM2.5 frame for a scheme.

    Returns a dict {frame, scheme, method, rounding}. Raises for 'none'.
    """
    rounding = config.get("aqi", {}).get("rounding", "nearest_integer")
    if scheme_name == "none":
        raise ValueError("AQI computation is disabled for scheme 'none' "
                         "(concentration-only output).")

    breakpoints = get_breakpoints(config, scheme_name)
    categories = get_categories(config, scheme_name)

    aqi_vals = []
    cat_vals = []
    for v in frame[pm25_col]:
        aqi = pm25_to_aqi(v, breakpoints)
        if aqi is not None and rounding == "nearest_integer":
            aqi = float(round(aqi))
        elif aqi is not None and rounding == "floor":
            aqi = float(np.floor(aqi))
        aqi_vals.append(aqi)
        cat_vals.append(aqi_category_name(aqi, categories, scheme_name))

    out = frame.copy()
    out["pm25_aqi"] = aqi_vals
    out["aqi_category"] = cat_vals
    out["aqi_scheme"] = scheme_name
    if rounding in ("nearest_integer", "floor"):
        out["pm25_aqi"] = out["pm25_aqi"].astype("Int64")
    return {"frame": out, "scheme": scheme_name,
            "method": config.get("aqi_scheme", {}).get("options", {}).get(
                scheme_name, {}).get("method", ""), "rounding": rounding}
