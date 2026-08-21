"""Pydantic response schemas for the PM2.5 Hyperlocal Mapping API.

These models describe the responses served from already-generated pipeline
outputs. Optional fields are used wherever a value may legitimately be
unavailable (e.g. NoData cells, deferred uncertainty).
"""

from typing import Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    build_version: Optional[str] = None
    timestamp: str
    aod: Optional[dict] = None


class DateResponse(BaseModel):
    dates: list[str]


class GridResponse(BaseModel):
    date: str
    resolution_m: int
    crs: str
    n_rows: int
    n_cols: int
    bounds: dict
    raster_url: str
    note: str


class PM25Response(BaseModel):
    date: str
    latitude: float
    longitude: float
    resolution_m: int
    pm25: Optional[float] = Field(default=None, description="Predicted PM2.5 in ug/m3")
    units: str
    status: str


class AQIResponse(BaseModel):
    date: str
    latitude: float
    longitude: float
    pm25_aqi: Optional[int] = Field(default=None, description="PM2.5-derived AQI sub-index value")
    category: Optional[str] = Field(default=None, description="CPCB AQI category")
    type: str = Field(default="PM2.5-derived AQI/sub-index")


class AODInfo(BaseModel):
    aod: Optional[float] = Field(default=None, description="Aerosol Optical Depth at 550nm")
    status: str = Field(
        default="AVAILABLE",
        description="AVAILABLE | NO_VALID_OBSERVATION | DATASET_UNAVAILABLE | API_ERROR",
    )
    source: str = Field(default="MODIS/MAIAC MCD19A2 v061")
    resolution_m: int = Field(default=500, description="Spatial resolution in meters")
    crs: Optional[str] = None
    date: Optional[str] = None
    unit: str = Field(default="unitless", description="AOD is unitless (550nm extinction/vert.column)")
    nodata: bool = Field(default=False, description="True if the exact pixel was nodata")
    lookup: str = Field(default="exact_pixel", description="exact_pixel | nearest_valid_pixel")
    distance_pixels: int = Field(default=0, description="Pixel distance to nearest valid if lookup != exact")


class LocationResponse(BaseModel):
    date: str
    location: dict
    pm25: Optional[float]
    pm25_units: str
    pm25_derived_aqi: Optional[int]
    aqi_category: Optional[str]
    aqi_type: str
    uncertainty: Optional[dict] = None
    uncertainty_status: str
    model: str
    dataset_mode: str
    aod_used: bool
    aod_info: Optional[AODInfo] = None


class HotspotStatisticsResponse(BaseModel):
    method: Optional[str] = None
    minimum_category: Optional[str] = None
    hotspot_cell_count: Optional[int] = None
    hotspot_zone_count: Optional[int] = None
    hotspot_area_km2: Optional[float] = None
    mean_pm25_ug_m3: Optional[float] = None
    max_pm25_ug_m3: Optional[float] = None
    mean_aqi: Optional[float] = None
    max_aqi: Optional[float] = None
    note: Optional[str] = None


class UncertaintyResponse(BaseModel):
    status: str
    method: Optional[str] = None
    reason: Optional[str] = None
    data_requirements: Optional[list[str]] = None
    future_method: Optional[str] = None


class FeatureImportanceResponse(BaseModel):
    model: Optional[str] = None
    dataset_mode: Optional[str] = None
    features: list[dict]
    interpretation: str


class StationResponse(BaseModel):
    station_id: str
    latitude: float
    longitude: float
    observation_count: Optional[int] = None
    first_timestamp: Optional[str] = None
    last_timestamp: Optional[str] = None
    latest_pm25: Optional[float] = None


class StationDetailResponse(BaseModel):
    station_id: str
    latitude: float
    longitude: float
    available_dates: list[str]
    observations: list[dict]
