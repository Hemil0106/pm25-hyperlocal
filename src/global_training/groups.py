"""Spatial validation groups (Milestone 17).

Builds the hierarchical grouping (station -> country -> region) used for spatial
cross-validation later, plus Leave-One-Station-Out (LOSO) and
Leave-One-Country-Out (LOCO) group definitions.

Rules:
  - groups are built from the real data only
  - LOSO/LOCO report READY only when there is sufficient coverage; otherwise
    they report INSUFFICIENT/BLOCKED with an honest reason
  - folds are never manufactured; no global model is trained in M17
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

MIN_STATIONS_FOR_LOSO = 5
MIN_COUNTRIES_FOR_LOCO = 2

# Minimal country -> region mapping for spatial grouping. Unknown countries are
# placed in "Unclassified"; this is a label only and never alters the data.
REGION_OF_COUNTRY = {
    "India": "South Asia", "Pakistan": "South Asia", "Bangladesh": "South Asia",
    "Nepal": "South Asia", "Sri Lanka": "South Asia", "Afghanistan": "South Asia",
    "China": "East Asia", "Japan": "East Asia", "South Korea": "East Asia",
    "Mongolia": "East Asia", "Taiwan": "East Asia", "Hong Kong": "East Asia",
    "Vietnam": "Southeast Asia", "Thailand": "Southeast Asia",
    "Indonesia": "Southeast Asia", "Malaysia": "Southeast Asia",
    "Philippines": "Southeast Asia", "Singapore": "Southeast Asia",
    "Myanmar": "Southeast Asia", "Cambodia": "Southeast Asia", "Laos": "Southeast Asia",
    "United States": "North America", "United States of America": "North America",
    "Canada": "North America", "Mexico": "North America",
    "Brazil": "South America", "Argentina": "South America",
    "Colombia": "South America", "Chile": "South America", "Peru": "South America",
    "Germany": "Europe", "France": "Europe", "United Kingdom": "Europe",
    "UK": "Europe", "Italy": "Europe", "Spain": "Europe", "Netherlands": "Europe",
    "Poland": "Europe", "Turkey": "Europe", "Russia": "Europe",
    "Egypt": "Middle East & North Africa", "Saudi Arabia": "Middle East & North Africa",
    "United Arab Emirates": "Middle East & North Africa",
    "South Africa": "Sub-Saharan Africa", "Nigeria": "Sub-Saharan Africa",
    "Kenya": "Sub-Saharan Africa", "Ghana": "Sub-Saharan Africa",
    "Australia": "Oceania", "New Zealand": "Oceania",
}


def region_of_country(country: str) -> str:
    if not country or str(country) != str(country):
        return "Unclassified"
    return REGION_OF_COUNTRY.get(str(country), "Unclassified")


def build_assignments(df: pd.DataFrame) -> pd.DataFrame:
    """Per-row split-assignment descriptors (station/country/region/folds)."""
    if df.empty:
        return pd.DataFrame(columns=[
            "station_id", "country", "region", "fold_station", "fold_country",
        ])
    out = pd.DataFrame(index=df.index)
    out["station_id"] = df["station_id"]
    out["country"] = df["country"].fillna("(unknown)")
    out["region"] = df["country"].map(region_of_country)
    out["fold_station"] = "station_" + df["station_id"].astype(str)
    out["fold_country"] = "country_" + out["country"].astype(str)
    return out.reset_index(drop=True)


def group_statistics(df: pd.DataFrame) -> dict:
    """Counts of groups at each hierarchy level."""
    if df.empty:
        return {
            "stations": 0, "countries": 0, "regions": 0,
            "countries_per_region": {}, "stations_per_country": {},
        }
    assignments = build_assignments(df)
    countries = assignments["country"].drop_duplicates().tolist()
    regions = assignments["region"].drop_duplicates().tolist()
    return {
        "stations": int(assignments["station_id"].nunique()),
        "countries": len(countries),
        "regions": len(regions),
        "countries_per_region": {
            str(k): int(v) for k, v in
            assignments.groupby("region")["country"].nunique().to_dict().items()
        },
        "stations_per_country": {
            str(k): int(v) for k, v in
            assignments.groupby("country")["station_id"].nunique().to_dict().items()
        },
    }


def loso_status(df: pd.DataFrame) -> dict:
    n_stations = int(df["station_id"].nunique()) if not df.empty else 0
    n_complete = int(df["complete_case"].sum()) if not df.empty else 0
    ready = n_stations >= MIN_STATIONS_FOR_LOSO and n_complete >= 2 * n_stations
    return {
        "method": "Leave-One-Station-Out",
        "status": "READY" if ready else "INSUFFICIENT_DATA",
        "n_stations": n_stations,
        "min_stations_required": MIN_STATIONS_FOR_LOSO,
        "n_complete_case_rows": n_complete,
        "reason": (
            None if ready else
            f"Only {n_stations} station(s) and {n_complete} complete-case row(s); "
            "a LOSO fold set is not defensible with this data."
        ),
    }


def loco_status(df: pd.DataFrame) -> dict:
    n_countries = int(df["country"].dropna().nunique()) if not df.empty else 0
    n_complete = int(df["complete_case"].sum()) if not df.empty else 0
    ready = n_countries >= MIN_COUNTRIES_FOR_LOCO and n_complete >= 2 * n_countries
    return {
        "method": "Leave-One-Country-Out",
        "status": "READY" if ready else "INSUFFICIENT_DATA",
        "n_countries": n_countries,
        "min_countries_required": MIN_COUNTRIES_FOR_LOCO,
        "n_complete_case_rows": n_complete,
        "reason": (
            None if ready else
            f"Only {n_countries} country/countries and {n_complete} complete-case "
            "row(s); a LOCO fold set is not defensible with this data."
        ),
    }


def write_validation_outputs(df: pd.DataFrame, config) -> dict:
    """Write split assignments parquet + validation groups JSON."""
    processed_base = Path(
        config.get("global_data", {}).get("storage", {}).get(
            "processed_base", "data/processed/global")
    )
    validation_dir = processed_base / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)

    assignments = build_assignments(df)
    assignments_path = validation_dir / "global_spatial_split_assignments.parquet"
    assignments.to_parquet(assignments_path, index=False)

    groups = {
        "scope": config.get("global_data", {}).get("scope", "global"),
        "hierarchy": group_statistics(df),
        "loso": loso_status(df),
        "loco": loco_status(df),
        "note": "Split assignments describe the real data. Model training is "
                "NOT performed in Milestone 17; folds are never manufactured.",
        "path": str(assignments_path),
    }
    groups_path = validation_dir / "global_validation_groups.json"
    groups_path.write_text(json.dumps(groups, indent=2), encoding="utf-8")
    groups["json_path"] = str(groups_path)
    return groups
