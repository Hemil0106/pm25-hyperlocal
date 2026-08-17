"""Training-table QC (Milestone 17).

Validates the assembled table: schema presence, dtypes, duplicate
station-date keys, and honest missingness counts. Nothing is silently dropped;
the diagnostic incomplete file preserves every non-complete row.
"""

from __future__ import annotations

import pandas as pd

from .schema import FEATURE_COLS, TARGET_COL, TRAINING_COLUMNS

MAX_PM25_UG_M3 = 1000.0


def check_schema(df: pd.DataFrame) -> dict:
    """Verify the required column set is present with expected dtypes."""
    missing = [c for c in TRAINING_COLUMNS if c not in df.columns]
    if missing:
        return {"ok": False, "reason": f"Missing columns: {missing}"}
    if df.empty:
        return {"ok": True, "reason": None}

    dtype_errors = []
    for col in ["latitude", "longitude", TARGET_COL] + FEATURE_COLS:
        if not pd.api.types.is_numeric_dtype(df[col]):
            dtype_errors.append(col)
    if df["date"].isna().all() and not df.empty:
        dtype_errors.append("date (all invalid)")
    if dtype_errors:
        return {"ok": False, "reason": f"Non-numeric columns: {dtype_errors}"}
    return {"ok": True, "reason": None}


def check_duplicates(df: pd.DataFrame) -> dict:
    """(station_id, date) must be unique per row (one daily value per station)."""
    if df.empty:
        return {"duplicate_station_dates": 0, "ok": True}
    dups = int(df.duplicated(subset=["station_id", "date"]).sum())
    return {"duplicate_station_dates": dups, "ok": dups == 0}


def missingness(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column missing counts/percent for the target and features."""
    if df.empty:
        rows = []
        for col in [TARGET_COL] + FEATURE_COLS:
            rows.append({"column": col, "n_non_null": 0, "n_null": 0, "pct_missing": 1.0})
        return pd.DataFrame(rows)
    cols = [c for c in ([TARGET_COL] + FEATURE_COLS) if c in df.columns]
    n = len(df)
    return pd.DataFrame({
        "column": cols,
        "n_non_null": [int(df[c].notna().sum()) for c in cols],
        "n_null": [int(df[c].isna().sum()) for c in cols],
        "pct_missing": [float(round(df[c].isna().mean(), 4)) for c in cols],
    })


def run_qc(df: pd.DataFrame) -> dict:
    """Run the QC suite; returns an honest report (never auto-deletes rows)."""
    schema = check_schema(df)
    dups = check_duplicates(df)

    if not df.empty:
        target = pd.to_numeric(df[TARGET_COL], errors="coerce")
        negative = int((target < 0).sum())
        above_max = int((target > MAX_PM25_UG_M3).sum())
    else:
        negative = 0
        above_max = 0

    missing = missingness(df)
    report = {
        "schema": schema,
        "duplicates": dups,
        "target_flags": {
            "negative_values": negative,
            "above_max_ug_m3": above_max,
            "max_pm25_ug_m3": MAX_PM25_UG_M3,
            "rule": "Values are flagged, never silently removed.",
        },
        "missingness": missing.to_dict(orient="records"),
        "note": "QC flags conditions honestly; non-complete rows are preserved "
                "in the incomplete diagnostic file.",
    }
    report["ok"] = bool(schema["ok"] and dups["ok"])
    return report
