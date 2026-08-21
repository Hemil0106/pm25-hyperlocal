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
AOD_500M_PATTERN = "aod_500m_{date}.tif"

RESOLUTION_MAP = {"500": 500, "1000": 1000}
DEFAULT_RESOLUTION_M = 500
SERVICE_NAME = "pm25-mapping-api"
API_VERSION = "0.1.1"
PM25_UNITS = "\u00b5g/m\u00b3"
AQI_TYPE = "PM2.5-derived AQI/sub-index"

VALID_CITIES = {"delhi", "pune", "mumbai"}


# ---------------------------------------------------------------------------
# City-aware data directory helper
# ---------------------------------------------------------------------------
def _data_dir(city: str | None = None) -> Path:
    """Return the processed data directory for a city.

    city=None or city="delhi" -> the root processed dir (backward compat).
    city="pune" -> data/processed/pune/
    """
    if not city or city == "delhi":
        return PROCESSED_DIR
    return PROCESSED_DIR / city


def _model_metadata_path(city: str | None = None) -> Path:
    if not city or city == "delhi":
        return MODEL_DIR / "model_metadata.json"
    return MODEL_DIR / f"model_metadata_{city}.json"


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
def _file_path(pattern: str, target_date: date, city: str | None = None) -> Path:
    return _data_dir(city) / pattern.format(date=str(target_date))


def _resolve_resolution(resolution: str) -> int:
    normalized = str(resolution).strip().lower().rstrip("m")
    if normalized not in RESOLUTION_MAP:
        raise HTTPException(
            status_code=400,
            detail="Requested resolution is not currently available.",
        )
    return RESOLUTION_MAP[normalized]


def _check_date_available(target_date: date, pattern: str, city: str | None = None) -> Path:
    path = _file_path(pattern, target_date, city)
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                f"Requested date {target_date} output is not available for "
                f"city={city or 'delhi'}. Data is not generated dynamically."
            ),
        )
    return path


@lru_cache(maxsize=64)
def _read_raster(path: str):
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


@lru_cache(maxsize=32)
def _read_json(path: str):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _read_parquet_cached(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def _model_metadata(city: str | None = None):
    p = _model_metadata_path(city)
    if not p.exists():
        return {}
    return _read_json(str(p))


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
    return HealthResponse(
        status="ok",
        service=SERVICE_NAME,
        version=API_VERSION,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/available-dates", response_model=DateResponse, tags=["system"])
def available_dates(city: str | None = Query(None)):
    d = _data_dir(city)
    found = []
    for path in d.glob(AQI_500M_PATTERN.format(date="*")):
        target = path.stem[len("aqi_500m_"):]
        pm25 = _file_path(PM25_500M_PATTERN, date.fromisoformat(target), city)
        if pm25.exists():
            found.append(target)
    return DateResponse(dates=sorted(found))


@app.get("/metadata", tags=["system"])
def metadata(city: str | None = Query(None)):
    d = _data_dir(city)
    meta_file = d / "final_metadata.json"
    if not meta_file.exists():
        raise HTTPException(status_code=404, detail="Metadata not available.")
    return _read_json(str(meta_file))


# ---------------------------------------------------------------------------
# Point queries
# ---------------------------------------------------------------------------
@app.get("/pm25", response_model=PM25Response, tags=["pm25"])
def get_pm25(
    date: date,
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    resolution: str = "500m",
    city: str | None = Query(None),
):
    res_m = _resolve_resolution(resolution)
    pattern = PM25_500M_PATTERN if res_m == 500 else PM25_1KM_PATTERN
    path = _check_date_available(date, pattern, city)
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
def get_pm25_grid(date: date, resolution: str = "500m", city: str | None = Query(None)):
    res_m = _resolve_resolution(resolution)
    pattern = PM25_500M_PATTERN if res_m == 500 else PM25_1KM_PATTERN
    path = _check_date_available(date, pattern, city)
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
        raster_url=f"/raster/pm25?date={date}&resolution={res_m}m&city={city or 'delhi'}",
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
    city: str | None = Query(None),
):
    path = _check_date_available(date, AQI_500M_PATTERN, city)
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
    city: str | None = Query(None),
):
    d = _data_dir(city)
    pm25_path = _check_date_available(date, PM25_500M_PATTERN, city)
    aqi_path = _check_date_available(date, AQI_500M_PATTERN, city)
    pm25 = _sample_raster(pm25_path, lon, lat)
    aqi = _sample_raster(aqi_path, lon, lat)
    aqi_int = None if aqi is None else int(round(aqi))

    uncertainty_file = d / "uncertainty_status.json"
    uncertainty = _read_json(str(uncertainty_file)) if uncertainty_file.exists() else {}

    model_meta = _model_metadata(city)
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
@app.get("/hotspots", tags=["hotspots"])
def get_hotspots(date: date | None = None, city: str | None = Query(None)):
    d = _data_dir(city)
    hotspots_file = d / "hotspots_500m.geojson"
    if not hotspots_file.exists():
        raise HTTPException(status_code=404, detail="Hotspots not available.")
    collection = _read_json(str(hotspots_file))
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
def get_hotspot_statistics(date: date | None = None, city: str | None = Query(None)):
    d = _data_dir(city)
    stats_file = d / "hotspot_statistics.json"
    if not stats_file.exists():
        raise HTTPException(status_code=404, detail="Hotspot statistics not available.")
    stats = _read_json(str(stats_file))
    return HotspotStatisticsResponse(**stats)


# ---------------------------------------------------------------------------
# Stations
# ---------------------------------------------------------------------------
@app.get("/stations", response_model=list[StationResponse], tags=["stations"])
def get_stations(city: str | None = Query(None)):
    d = _data_dir(city)
    summary_file = d / "cpcb_station_summary.parquet"
    daily_file = d / "cpcb_pm25_daily.parquet"
    if not summary_file.exists():
        raise HTTPException(status_code=404, detail="Station data not available.")
    summary = _read_parquet_cached(summary_file)
    latest = {}
    if daily_file.exists():
        daily = _read_parquet_cached(daily_file).sort_values("date")
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
def get_station_detail(station_id: str, city: str | None = Query(None)):
    d = _data_dir(city)
    daily_file = d / "cpcb_pm25_daily.parquet"
    if not daily_file.exists():
        raise HTTPException(status_code=404, detail="Station data not available.")
    daily = _read_parquet_cached(daily_file)
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
def get_feature_importance(city: str | None = Query(None)):
    d = _data_dir(city)
    fi_file = d / "model_feature_importance.csv"
    if not fi_file.exists():
        raise HTTPException(status_code=404, detail="Feature importance not available.")
    frame = pd.read_csv(fi_file)
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
    model_meta = _model_metadata(city)
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
def get_uncertainty(city: str | None = Query(None)):
    d = _data_dir(city)
    unc_file = d / "uncertainty_status.json"
    if not unc_file.exists():
        raise HTTPException(status_code=404, detail="Uncertainty status not available.")
    data = _read_json(str(unc_file))
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
@app.get("/raster/aod", tags=["raster"])
def serve_aod_raster(date: date, city: str | None = Query(None)):
    """Serve MODIS/MAIAC AOD GeoTIFF for the given date and city."""
    path = _check_date_available(date, AOD_500M_PATTERN, city)
    return FileResponse(path, media_type="image/tiff")


@app.get("/raster/pm25", tags=["raster"])
def serve_pm25_raster(date: date, resolution: str = "500m", city: str | None = Query(None)):
    res_m = _resolve_resolution(resolution)
    pattern = PM25_500M_PATTERN if res_m == 500 else PM25_1KM_PATTERN
    path = _check_date_available(date, pattern, city)
    return FileResponse(path, media_type="image/tiff")


@app.get("/raster/aqi", tags=["raster"])
def serve_aqi_raster(date: date, city: str | None = Query(None)):
    path = _check_date_available(date, AQI_500M_PATTERN, city)
    return FileResponse(path, media_type="image/tiff")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host=API_CONFIG["host"],
        port=API_CONFIG["port"],
        reload=True,
    )
