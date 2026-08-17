"""Feature coverage reports (Milestone 17).

Coverage is broken down by country, month, and station, with per-feature
missingness. Missingness may be geographically biased (e.g., a feature only
acquired for some countries); the report surfaces that explicitly instead of
hiding it.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from .qc import missingness
from .schema import FEATURE_COLS, TARGET_COL

logger = logging.getLogger(__name__)


def coverage_by_country(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["country", "n_rows", "n_stations",
                                     "n_complete_rows", "pct_complete"])
    grouped = (
        df.assign(_country=df["country"].fillna("(unknown)"))
        .groupby("_country", as_index=False)
        .agg(
            n_rows=("station_id", "count"),
            n_stations=("station_id", "nunique"),
            n_complete_rows=("complete_case", "sum"),
        )
    )
    grouped = grouped.rename(columns={"_country": "country"})
    grouped["pct_complete"] = (grouped["n_complete_rows"] / grouped["n_rows"]
                               ).round(4)
    return grouped.sort_values("n_rows", ascending=False).reset_index(drop=True)


def coverage_by_month(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["month", "n_rows", "n_stations",
                                     "n_complete_rows"])
    months = pd.to_datetime(df["date"], errors="coerce").dt.to_period("M")
    grouped = df.assign(_month=months.astype(str)).groupby("_month", as_index=False).agg(
        n_rows=("station_id", "count"),
        n_stations=("station_id", "nunique"),
        n_complete_rows=("complete_case", "sum"),
    )
    grouped = grouped.rename(columns={"_month": "month"})
    return grouped.sort_values("month").reset_index(drop=True)


def geographic_bias_flags(df: pd.DataFrame) -> list:
    """Flag features whose missingness varies geographically between countries."""
    if df.empty:
        return []
    flags = []
    by_country = df.assign(_c=df["country"].fillna("(unknown)"))
    present_features = [c for c in FEATURE_COLS if c in df.columns]
    for col in present_features:
        rates = by_country.groupby("_c")[col].apply(
            lambda s: float(s.isna().mean())).sort_values()
        if rates.nunique() > 1:
            flags.append({
                "feature": col,
                "missingness_varies_by_country": True,
                "min_pct_missing": float(rates.min()),
                "max_pct_missing": float(rates.max()),
                "note": "Missingness may be geographically biased.",
            })
    return flags


def build_coverage_report(df: pd.DataFrame, config,
                          write_csv: bool = True) -> dict:
    """Assemble the coverage report + per-country CSV."""
    report = {
        "scope": config.get("global_data", {}).get("scope", "global"),
        "rows": int(len(df)),
        "stations": int(df["station_id"].nunique()) if not df.empty else 0,
        "countries": int(df["country"].dropna().nunique()) if not df.empty else 0,
        "complete_case_rows": int(df["complete_case"].sum()) if not df.empty else 0,
        "missingness_by_feature": missingness(df).to_dict(orient="records"),
        "by_month": coverage_by_month(df).to_dict(orient="records"),
        "geographic_bias": geographic_bias_flags(df),
        "note": "Complete-case rows are preserved alongside the incomplete "
                "diagnostic file; nothing is silently deleted.",
    }
    if not df.empty:
        target = pd.to_numeric(df[TARGET_COL], errors="coerce")
        report["target_non_null"] = int(target.notna().sum())

    processed_base = Path(
        config.get("global_data", {}).get("storage", {}).get(
            "processed_base", "data/processed/global")
    )
    if write_csv:
        reports_dir = processed_base / "reports"
        diagnostics_dir = processed_base / "diagnostics"
        reports_dir.mkdir(parents=True, exist_ok=True)
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        by_country = coverage_by_country(df)
        by_country.to_csv(diagnostics_dir / "global_coverage_by_country.csv",
                          index=False)
        report_path = reports_dir / "global_feature_coverage_report.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        report["paths"] = {
            "feature_coverage_report": str(report_path),
            "coverage_by_country_csv": str(
                diagnostics_dir / "global_coverage_by_country.csv"),
        }
    return report
