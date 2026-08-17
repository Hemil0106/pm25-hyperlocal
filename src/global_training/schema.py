"""Training-table schema (Milestone 17).

Defines the single source of truth for the global training dataset column set,
the candidate feature list, and per-feature availability flags.

Rules enforced across the package:
  - the target (PM2.5, ug/m3) is NEVER used as a predictor feature
  - unavailable features are NaN with ``<feature>_available == False``
  - temporal features (month/day_of_year/sin/cos) are derived only from the
    row's own date and are always available when the date is valid
  - ``complete_case`` marks rows with a valid target AND every feature present
"""

from __future__ import annotations

SCHEMA_VERSION = 1

TARGET_COL = "PM2.5"

IDENTITY_COLS = [
    "station_id",
    "station_name",
    "country",
    "city",
    "latitude",
    "longitude",
    "date",
]

# Candidate features. The gridded ones (AOD, weather, NDVI, DEM, night lights)
# come from real Milestone 16 acquisition artifacts and are NaN when the source
# was not acquired. road_density is the real OSM tile road-segment count.
GRIDDED_FEATURE_COLS = [
    "AOD",
    "temperature_c",
    "relative_humidity_pct",
    "wind_speed_mps",
    "wind_direction_deg",
    "NDVI",
    "elevation_m",
    "road_density",
    "night_lights",
]

# Derived meteorology (u/v wind components) computed from wind_speed/direction.
DERIVED_FEATURE_COLS = ["wind_u_mps", "wind_v_mps"]

# Temporal features derived from the row's own date (no future information).
TEMPORAL_FEATURE_COLS = ["month", "day_of_year", "sin_day_of_year", "cos_day_of_year"]

FEATURE_COLS = GRIDDED_FEATURE_COLS + DERIVED_FEATURE_COLS + TEMPORAL_FEATURE_COLS

# Per-feature availability flags (temporal features are always available).
SOURCE_FEATURE_COLS = GRIDDED_FEATURE_COLS + DERIVED_FEATURE_COLS
AVAILABILITY_COLS = [f"{col}_available" for col in SOURCE_FEATURE_COLS]

# Extra metadata columns.
META_COLS = [
    "NDVI_source_date",
    "NDVI_offset_days",
    "complete_case",
    "n_features_available",
]

TRAINING_COLUMNS = (
    IDENTITY_COLS
    + [TARGET_COL]
    + FEATURE_COLS
    + AVAILABILITY_COLS
    + META_COLS
)

# Feature -> originating Milestone 16 source id (for the metadata manifest).
FEATURE_SOURCE_MAP = {
    "AOD": "aod",
    "temperature_c": "weather",
    "relative_humidity_pct": "weather",
    "wind_speed_mps": "weather",
    "wind_direction_deg": "weather",
    "wind_u_mps": "weather",
    "wind_v_mps": "weather",
    "NDVI": "ndvi",
    "elevation_m": "dem",
    "road_density": "osm",
    "night_lights": "viirs",
}
