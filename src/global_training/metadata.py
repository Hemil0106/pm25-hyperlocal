"""Training-table metadata + source manifest (Milestone 17).

Versioned metadata documenting the exact inputs, schema, feature-engineering
rules, and limitations so any future consumer (Milestone 18+) knows precisely
what the table does and does not contain.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from src.global_data.integrity import sha256_file
from src.global_data.scope import validate_scope

from .schema import FEATURE_SOURCE_MAP, SCHEMA_VERSION, TRAINING_COLUMNS
from .builder import daily_path

logger = logging.getLogger(__name__)

LIMITATIONS = [
    "Unavailable features are NaN with per-feature available flags; nothing is fabricated.",
    "Gridded features (AOD/weather/NDVI/DEM/night lights) depend on real Milestone 16 "
    "acquisition; with no credentials they are unavailable.",
    "road_density is a per-tile OSM road-segment count (a spatial proxy, not a measurement).",
    "station_name/city are NaN because no reverse-geocoding or display-name acquisition "
    "was performed (never fabricated).",
    "Temporal features are derived from the row's own date only (no future information).",
    "No global ML model is trained and no prediction is made in Milestone 17.",
]


def build_metadata(config, scope: str, builder_report: dict,
                   feature_coverage: dict, readiness: dict,
                   osm_report: dict) -> dict:
    """Assemble and write the versioned metadata + source manifest."""
    scope = validate_scope(scope)
    processed_base = Path(
        config.get("global_data", {}).get("storage", {}).get(
            "processed_base", "data/processed/global")
    )
    metadata_dir = processed_base / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    daily = daily_path(config, scope)
    input_manifest = {
        "daily_observations": {
            "path": str(daily),
            "exists": daily.exists(),
            "sha256": sha256_file(daily) if daily.exists() else None,
        },
    }

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "built_for_scope": scope,
        "generated_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input_manifest": input_manifest,
        "osm_tiles": osm_report,
        "columns": TRAINING_COLUMNS,
        "feature_source_map": FEATURE_SOURCE_MAP,
        "feature_engineering_rules": [
            "month/day_of_year/sin/cos derived from the row's own date.",
            "wind_u/v = -speed*sin(dir), -speed*cos(dir) only when both are real.",
            "Features are never generated from the PM2.5 target.",
        ],
        "builder_report": builder_report,
        "feature_coverage_summary": {
            "rows": feature_coverage.get("rows"),
            "stations": feature_coverage.get("stations"),
            "countries": feature_coverage.get("countries"),
            "complete_case_rows": feature_coverage.get("complete_case_rows"),
        },
        "readiness": {
            "model_training_ready": readiness.get("model_training_ready"),
            "reason": readiness.get("reason"),
        },
        "limitations": LIMITATIONS,
        "global_ml_not_implemented": True,
        "global_prediction_not_implemented": True,
    }

    path = metadata_dir / "global_training_metadata.json"
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    metadata["path"] = str(path)
    return metadata
