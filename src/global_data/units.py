"""PM2.5 unit normalization (Milestone 16).

All observations are normalized to ``pm25_ug_m3``. Known units are converted
with documented factors. Unknown units are NOT guessed: they are marked invalid
and excluded by QC (with a report line), never silently treated as ug/m3.
"""

from __future__ import annotations

import pandas as pd

TARGET_UNIT = "pm25_ug_m3"

# Canonical unit -> conversion factor to ug/m3. Keys are normalized (lowercase,
# whitespace trimmed, "^3" collapsed to "3", "µg" -> "ug").
UNIT_CONVERSION_FACTORS = {
    "ug/m3": 1.0,
    "mg/m3": 1000.0,
    "g/m3": 1_000_000.0,
    "ng/m3": 0.001,
    "ppm": None,  # PM2.5 is a mass concentration; ppm is not convertible.
}


def normalize_unit_string(raw) -> str:
    """Normalize a unit string for lookup.

    Only typographic variants are collapsed (superscripts, Greek/Unicode mu,
    caret exponents) - never unit semantics. Unknown unit strings remain
    unchanged and are rejected (never guessed).
    """
    if raw is None:
        return ""
    text = str(raw).strip().lower()
    text = text.replace(" ", "")
    text = text.replace("^", "")
    text = text.replace("³", "3")
    text = text.replace("²", "2")
    text = text.replace("μ", "µ")
    text = text.replace("µg", "ug")
    text = text.replace("microgramsper", "ug")
    return text


def conversion_factor(unit) -> float | None:
    """Conversion factor to ug/m3; None for unknown units (never guessed)."""
    if unit is None:
        return None
    normalized = normalize_unit_string(unit)
    return UNIT_CONVERSION_FACTORS.get(normalized)


def is_known_unit(unit) -> bool:
    return conversion_factor(unit) is not None


def normalize_pm25_units(df: pd.DataFrame, value_col: str = "PM2.5",
                         unit_col: str = "units") -> tuple[pd.DataFrame, dict]:
    """Normalize a PM2.5 value column to ug/m3 in-place copy.

    Returns (normalized_df, report). Rows with unknown units get a NaN value and
    are counted in ``unknown_units``; they are excluded later by QC. The
    original value is preserved in ``pm25_raw`` and the factor in
    ``unit_factor`` so the transformation is traceable.
    """
    if df.empty:
        return df.copy(), {"rows": 0, "unknown_units": 0, "target_unit": TARGET_UNIT}

    out = df.copy()
    out["pm25_raw"] = pd.to_numeric(out.get(value_col), errors="coerce")
    out["unit_factor"] = out[unit_col].map(conversion_factor)
    out["pm25_ug_m3"] = out["pm25_raw"] * out["unit_factor"]

    unknown_mask = out["unit_factor"].isna()
    out.loc[unknown_mask, "pm25_ug_m3"] = None
    out[TARGET_UNIT] = out["pm25_ug_m3"]

    report = {
        "rows": int(len(out)),
        "unknown_units": int(unknown_mask.sum()),
        "unknown_unit_values": sorted(
            out.loc[unknown_mask, unit_col].dropna().astype(str).unique().tolist()
        ),
        "target_unit": TARGET_UNIT,
    }
    return out, report
