"""Hyperlocal PM2.5 Mapping API.

FastAPI backend that serves already-generated pipeline outputs through clean
REST endpoints. No model training, no map regeneration, no AQI recalculation
happens inside requests -- this API only reads the canonical Milestone 1-11
products from disk.
"""

import json
import logging
import time
from datetime import date, datetime, timezone
from functools import lru_cache
from pathlib import Path

import pandas as pd
import rasterio
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from src.config import get_project_root, load_config

from api.globalization import router as globalization_router
from api.data import router as global_data_router
from api.schemas import (
    AQIResponse,
    DateResponse,
    FeatureImportanceResponse,
    GridResponse,
    HealthResponse,
    HotspotStatisticsResponse,
    LocationResponse,
    PM25Response,
    StationDetailResponse,
    StationResponse,
    UncertaintyResponse,
)

logger = logging.getLogger("pm25-mapping-api")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

# ---------------------------------------------------------------------------
# Centralized path configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = get_project_root()
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs"
MODEL_DIR = PROJECT_ROOT / "models"

PM25_500M_PATTERN = "pm25_500m_{date}.tif"
PM25_1KM_PATTERN = "pm25_1km_{date}.tif"
AQI_500M_PATTERN = "aqi_500m_{date}.tif"
HOTSPOTS_FILE = PROCESSED_DIR / "hotspots_500m.geojson"
HOTSPOT_STATISTICS_FILE = PROCESSED_DIR / "hotspot_statistics.json"
METADATA_FILE = PROCESSED_DIR / "final_metadata.json"
UNCERTAINTY_FILE = PROCESSED_DIR / "uncertainty_status.json"
STATIONS_SUMMARY_FILE = PROCESSED_DIR / "cpcb_station_summary.parquet"
STATIONS_DAILY_FILE = PROCESSED_DIR / "cpcb_pm25_daily.parquet"
FEATURE_IMPORTANCE_FILE = PROCESSED_DIR / "model_feature_importance.csv"
MODEL_METADATA_FILE = MODEL_DIR / "model_metadata.json"

RESOLUTION_MAP = {"500": 500, "1000": 1000}
DEFAULT_RESOLUTION_M = 500
SERVICE_NAME = "pm25-mapping-api"
API_VERSION = "0.1.0"
PM25_UNITS = "\u00b5g/m\u00b3"
AQI_TYPE = "PM2.5-derived AQI/sub-index"


def _load_api_config():
    import os

    try:
        config = load_config()
        api_cfg = config.get("api", {})
    except Exception:
        api_cfg = {}

    cors_env = os.environ.get("CORS_ORIGINS", "")
    cors_origins = (
        [o.strip() for o in cors_env.split(",") if o.strip()]
        if cors_env
        else api_cfg.get("cors_origins", [])
    )

    return {
        "host": api_cfg.get("host", "127.0.0.1"),
        "port": int(api_cfg.get("port", 8000)),
        "cors_origins": cors_origins,
        "default_resolution": api_cfg.get("default_resolution", "500m"),
    }


API_CONFIG = _load_api_config()

app = FastAPI(
    title="Hyperlocal PM2.5 Mapping API",
    description=(
        "API for serving AI/ML-based satellite and geospatial PM2.5 mapping "
        "outputs. Serves already-generated pipeline products only; it never "
        "retrains models or regenerates maps on request."
    ),
    version=API_VERSION,
)


@app.middleware("http")
async def _request_logger(request, call_next):
    started = time.perf_counter()
    response = await call_next(request)
    duration_ms = round((time.perf_counter() - started) * 1000, 2)
    logger.info(
        "%s %s query=%s status=%s duration_ms=%s",
        request.method,
        request.url.path,
        request.url.query or "-",
        response.status_code,
        duration_ms,
    )
    return response


# ---------------------------------------------------------------------------
# CORS (prototype: local development origins only, configurable)
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=API_CONFIG["cors_origins"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(globalization_router)
app.include_router(global_data_router)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _file_path(pattern: str, target_date: date) -> Path:
    return PROCESSED_DIR / pattern.format(date=str(target_date))


def _resolve_resolution(resolution: str) -> int:
    """Validate/normalize resolution ('500m'/'500'/'1000m'/'1000')."""
    normalized = str(resolution).strip().lower().rstrip("m")
    if normalized not in RESOLUTION_MAP:
        raise HTTPException(
            status_code=400,
            detail="Requested resolution is not currently available.",
        )
    return RESOLUTION_MAP[normalized]


def _check_date_available(target_date: date, pattern: str) -> Path:
    path = _file_path(pattern, target_date)
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                f"Requested date {target_date} output is not available. "
                "Data is not generated dynamically."
            ),
        )
    return path


@lru_cache(maxsize=32)
def _read_raster(path: str):
    """Read a GeoTIFF once and cache (small prototype rasters)."""
    with rasterio.open(path) as src:
        array = src.read(1)
        return (
            array,
            src.transform,
            src.nodata,
            src.crs.to_string(),
            src.bounds,
        )


def _sample_raster(path: Path, lon: float, lat: float):
    array, transform, nodata, crs, bounds = _read_raster(str(path))
    from rasterio.warp import transform as warp_transform

    xs, ys = warp_transform("EPSG:4326", crs, [lon], [lat])
    col, row = rasterio.transform.rowcol(transform, xs[0], ys[0])
    if row < 0 or row >= array.shape[0] or col < 0 or col >= array.shape[1]:
        raise HTTPException(
            status_code=400,
            detail="Point is outside the study area (AOI).",
        )
    value = float(array[row, col])
    if nodata is not None and value == float(nodata):
        return None
    return value


@lru_cache(maxsize=4)
def _aqi_categories():
    """CPCB AQI category table derived from config (name, min, max)."""
    try:
        config = load_config()
        categories = config["aqi"]["categories"]
    except Exception:
        return []
    ordered = []
    for entry in categories:
        ordered.append((str(entry["name"]), int(entry["aqi_min"]), int(entry["aqi_max"])))
    ordered.sort(key=lambda item: item[1])
    return ordered


def _category_for_aqi(aqi_value):
    if aqi_value is None:
        return None
    for name, min_value, max_value in _aqi_categories():
        if min_value <= int(aqi_value) <= max_value:
            return name
    return None


@lru_cache(maxsize=4)
def _read_json(path: str):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


@lru_cache(maxsize=2)
def _stations_summary():
    return pd.read_parquet(STATIONS_SUMMARY_FILE)


@lru_cache(maxsize=2)
def _stations_daily():
    return pd.read_parquet(STATIONS_DAILY_FILE)


@lru_cache(maxsize=2)
def _model_metadata():
    if not MODEL_METADATA_FILE.exists():
        return {}
    return _read_json(str(MODEL_METADATA_FILE))


# ---------------------------------------------------------------------------
# Root / health / dates
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
def root():
    return {
        "service": SERVICE_NAME,
        "version": API_VERSION,
        "documentation": "/docs",
        "health": "/health",
        "available_dates": "/available-dates",
    }


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health():
    """Liveness check for the API service."""
    return HealthResponse(
        status="ok",
        service=SERVICE_NAME,
        version=API_VERSION,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/available-dates", response_model=DateResponse, tags=["system"])
def available_dates():
    """Return dates for which final PM2.5/AQI products actually exist."""
    found = []
    for path in PROCESSED_DIR.glob(AQI_500M_PATTERN.format(date="*")):
        target = path.stem[len("aqi_500m_"):]
        pm25 = _file_path(PM25_500M_PATTERN, date.fromisoformat(target))
        if pm25.exists():
            found.append(target)
    return DateResponse(dates=sorted(found))


@app.get("/metadata", tags=["system"])
def metadata():
    """Return the final pipeline metadata (final_metadata.json)."""
    if not METADATA_FILE.exists():
        raise HTTPException(status_code=404, detail="Metadata not available.")
    return _read_json(str(METADATA_FILE))


# ---------------------------------------------------------------------------
# Point queries
# ---------------------------------------------------------------------------
@app.get("/pm25", response_model=PM25Response, tags=["pm25"])
def get_pm25(
    date: date,
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    resolution: str = "500m",
):
    """Return the predicted PM2.5 raster cell value for a location."""
    res_m = _resolve_resolution(resolution)
    pattern = PM25_500M_PATTERN if res_m == 500 else PM25_1KM_PATTERN
    path = _check_date_available(date, pattern)
    value = _sample_raster(path, lon, lat)
    if value is None:
        return PM25Response(
            date=str(date), latitude=lat, longitude=lon,
            resolution_m=res_m, pm25=None, units=PM25_UNITS, status="NoData",
        )
    return PM25Response(
        date=str(date), latitude=lat, longitude=lon,
        resolution_m=res_m, pm25=round(value, 2), units=PM25_UNITS, status="valid",
    )


@app.get("/pm25/grid", response_model=GridResponse, tags=["pm25"])
def get_pm25_grid(date: date, resolution: str = "500m"):
    """Grid metadata for the requested PM2.5 layer.

    Decision: for this prototype the map layer is served directly as the
    canonical GeoTIFF through /raster/pm25 (compact, no client-side
    polygonization, no huge JSON). This endpoint returns grid metadata so the
    future dashboard can build tile/Leaflet requests.
    """
    res_m = _resolve_resolution(resolution)
    pattern = PM25_500M_PATTERN if res_m == 500 else PM25_1KM_PATTERN
    path = _check_date_available(date, pattern)
    array, _, _, crs, bounds = _read_raster(str(path))
    return GridResponse(
        date=str(date),
        resolution_m=res_m,
        crs=crs,
        n_rows=int(array.shape[0]),
        n_cols=int(array.shape[1]),
        bounds={
            "left": bounds.left, "bottom": bounds.bottom,
            "right": bounds.right, "top": bounds.top,
        },
        raster_url=f"/raster/pm25?date={date}&resolution={res_m}m",
        note=(
            "Grid layer is served as GeoTIFF via /raster/pm25; no "
            "per-cell polygonization is performed."
        ),
    )


@app.get("/aqi", response_model=AQIResponse, tags=["aqi"])
def get_aqi(
    date: date,
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
):
    """Return the PM2.5-derived AQI cell for a location."""
    path = _check_date_available(date, AQI_500M_PATTERN)
    value = _sample_raster(path, lon, lat)
    if value is None:
        return AQIResponse(
            date=str(date), latitude=lat, longitude=lon,
            pm25_aqi=None, category=None, type=AQI_TYPE,
        )
    aqi_int = int(round(value))
    return AQIResponse(
        date=str(date), latitude=lat, longitude=lon,
        pm25_aqi=aqi_int, category=_category_for_aqi(aqi_int), type=AQI_TYPE,
    )


@app.get("/location", response_model=LocationResponse, tags=["location"])
def get_location(
    date: date,
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
):
    """Return PM2.5 and PM2.5-derived AQI for a selected geographic location."""
    pm25_path = _check_date_available(date, PM25_500M_PATTERN)
    aqi_path = _check_date_available(date, AQI_500M_PATTERN)
    pm25 = _sample_raster(pm25_path, lon, lat)
    aqi = _sample_raster(aqi_path, lon, lat)
    aqi_int = None if aqi is None else int(round(aqi))
    uncertainty = _read_json(str(UNCERTAINTY_FILE)) if UNCERTAINTY_FILE.exists() else {}
    model_meta = _model_metadata()
    model_name = model_meta.get("model_type") or model_meta.get("model") or "XGBoost"
    dataset_mode = model_meta.get("dataset_mode", "unknown")
    aod_used = bool(model_meta.get("aod_available", False))
    return LocationResponse(
        date=str(date),
        location={"latitude": lat, "longitude": lon},
        pm25=None if pm25 is None else round(pm25, 2),
        pm25_units=PM25_UNITS,
        pm25_derived_aqi=aqi_int,
        aqi_category=_category_for_aqi(aqi_int),
        aqi_type=AQI_TYPE,
        uncertainty=None,
        uncertainty_status=str(uncertainty.get("status", "DEFERRED")),
        model=model_name,
        dataset_mode=dataset_mode,
        aod_used=aod_used,
    )


# ---------------------------------------------------------------------------
# Hotspots
# ---------------------------------------------------------------------------
@lru_cache(maxsize=2)
def _hotspots_raw():
    with open(HOTSPOTS_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


@app.get("/hotspots", tags=["hotspots"])
def get_hotspots(date: date | None = None):
    """Return predicted high-pollution zones as a GeoJSON FeatureCollection."""
    if not HOTSPOTS_FILE.exists():
        raise HTTPException(status_code=404, detail="Hotspots not available.")
    collection = _hotspots_raw()
    if date is not None:
        day = str(date)
        features = [
            f for f in collection.get("features", [])
            if str(f.get("properties", {}).get("date", "")).startswith(day)
        ]
        collection = {"type": "FeatureCollection", "features": features}
    return Response(
        content=json.dumps(collection),
        media_type="application/geo+json",
    )


@app.get("/hotspots/statistics", response_model=HotspotStatisticsResponse, tags=["hotspots"])
def get_hotspot_statistics(date: date | None = None):
    """Return hotspot statistics from hotspot_statistics.json."""
    if not HOTSPOT_STATISTICS_FILE.exists():
        raise HTTPException(status_code=404, detail="Hotspot statistics not available.")
    stats = _read_json(str(HOTSPOT_STATISTICS_FILE))
    return HotspotStatisticsResponse(**stats)


# ---------------------------------------------------------------------------
# Stations
# ---------------------------------------------------------------------------
@app.get("/stations", response_model=list[StationResponse], tags=["stations"])
def get_stations():
    """Return cleaned CPCB station information (location + basic counts)."""
    if not STATIONS_SUMMARY_FILE.exists():
        raise HTTPException(status_code=404, detail="Station data not available.")
    summary = _stations_summary()
    latest = {}
    if STATIONS_DAILY_FILE.exists():
        daily = _stations_daily().sort_values("date")
        latest = (
            daily.drop_duplicates(subset=["station_id"], keep="last")
            .set_index("station_id")["PM2.5"].to_dict()
        )
    stations = []
    for row in summary.to_dict("records"):
        stations.append(
            StationResponse(
                station_id=str(row["station_id"]),
                latitude=float(row["latitude"]),
                longitude=float(row["longitude"]),
                observation_count=(
                    int(row["observation_count"])
                    if row.get("observation_count") is not None else None
                ),
                first_timestamp=str(row["first_timestamp"]) if row.get("first_timestamp") else None,
                last_timestamp=str(row["last_timestamp"]) if row.get("last_timestamp") else None,
                latest_pm25=(
                    float(latest[str(row["station_id"])])
                    if str(row["station_id"]) in latest else None
                ),
            )
        )
    return stations


@app.get("/stations/{station_id}", response_model=StationDetailResponse, tags=["stations"])
def get_station_detail(station_id: str):
    """Return a single CPCB station with its available dates and PM2.5 series."""
    if not STATIONS_DAILY_FILE.exists():
        raise HTTPException(status_code=404, detail="Station data not available.")
    daily = _stations_daily()
    station = daily[daily["station_id"] == station_id]
    if station.empty:
        raise HTTPException(status_code=404, detail=f"Station {station_id} not found.")
    first = station.iloc[0]
    observations = [
        {
            "date": str(row["date"])[:10],
            "pm25": float(row["PM2.5"]),
            "observation_count": int(row["observation_count"]),
        }
        for row in station.to_dict("records")
    ]
    return StationDetailResponse(
        station_id=station_id,
        latitude=float(first["latitude"]),
        longitude=float(first["longitude"]),
        available_dates=sorted({str(row["date"])[:10] for row in station.to_dict("records")}),
        observations=observations,
    )

# ---------------------------------------------------------------------------
# Model / uncertainty
# ---------------------------------------------------------------------------
@app.get("/feature-importance", response_model=FeatureImportanceResponse, tags=["model"])
def get_feature_importance():
    """Return model feature importance (not causal attribution)."""
    if not FEATURE_IMPORTANCE_FILE.exists():
        raise HTTPException(status_code=404, detail="Feature importance not available.")
    frame = pd.read_csv(FEATURE_IMPORTANCE_FILE)
    features = []
    for row in frame.to_dict("records"):
        features.append({
            "feature": str(row["feature"]),
            "random_forest_importance": (
                float(row["random_forest_importance"])
                if row.get("random_forest_importance") is not None else None
            ),
            "xgboost_gain": (
                float(row["xgboost_gain"]) if row.get("xgboost_gain") is not None else None
            ),
            "rf_relative_share": (
                float(row["rf_relative_share"])
                if row.get("rf_relative_share") is not None else None
            ),
            "xgb_relative_share": (
                float(row["xgb_relative_share"])
                if row.get("xgb_relative_share") is not None else None
            ),
        })
    model_meta = _model_metadata()
    model_name = model_meta.get("model_type") or model_meta.get("model") or "XGBoost"
    dataset_mode = model_meta.get("dataset_mode", "unknown")
    return FeatureImportanceResponse(
        model=model_name,
        dataset_mode=dataset_mode,
        features=features,
        interpretation=(
            "Model feature importance; not causal attribution."
        ),
    )


@app.get("/uncertainty", response_model=UncertaintyResponse, tags=["model"])
def get_uncertainty():
    """Return the current uncertainty status (deferred, no fake confidence)."""
    if not UNCERTAINTY_FILE.exists():
        raise HTTPException(status_code=404, detail="Uncertainty status not available.")
    data = _read_json(str(UNCERTAINTY_FILE))
    return UncertaintyResponse(
        status=str(data.get("status", "DEFERRED")),
        method=data.get("method"),
        reason=data.get("reason"),
        data_requirements=data.get("data_requirements"),
        future_method=data.get("future_method"),
    )


# ---------------------------------------------------------------------------
# Raster serving (controlled, known files only)
# ---------------------------------------------------------------------------
@app.get("/raster/pm25", tags=["raster"])
def serve_pm25_raster(date: date, resolution: str = "500m"):
    """Serve the canonical PM2.5 GeoTIFF for a date (known files only)."""
    res_m = _resolve_resolution(resolution)
    pattern = PM25_500M_PATTERN if res_m == 500 else PM25_1KM_PATTERN
    path = _check_date_available(date, pattern)
    return FileResponse(path, media_type="image/tiff")


@app.get("/raster/aqi", tags=["raster"])
def serve_aqi_raster(date: date):
    """Serve the canonical PM2.5-derived AQI GeoTIFF for a date (known files only)."""
    path = _check_date_available(date, AQI_500M_PATTERN)
    return FileResponse(path, media_type="image/tiff")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host=API_CONFIG["host"],
        port=API_CONFIG["port"],
        reload=True,
    )
