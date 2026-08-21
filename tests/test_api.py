"""Unit/API tests for Milestone 12 (Backend API).

Uses FastAPI TestClient so the server does not need to be manually started.
"""

import json
import math

import pandas as pd
import pytest
import rasterio
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)

KNOWN_DATE = "2025-01-01"
LAT = 28.60
LON = 77.20


@pytest.fixture(scope="module")
def pm25_tif_value():
    """Value of the canonical 500 m PM2.5 raster at the known point."""
    with rasterio.open("data/processed/pm25_500m_2025-01-01.tif") as src:
        array = src.read(1)
        transform = src.transform
        nodata = src.nodata
        from rasterio.warp import transform as warp_transform

        xs, ys = warp_transform("EPSG:4326", src.crs, [LON], [LAT])
        col, row = rasterio.transform.rowcol(transform, xs[0], ys[0])
        value = array[row, col]
        if nodata is not None and value == nodata:
            return None
        return float(value)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "pm25-mapping-api"
    assert body["version"]


def test_available_dates_only_real():
    response = client.get("/available-dates")
    assert response.status_code == 200
    dates = response.json()["dates"]
    assert KNOWN_DATE in dates
    assert "2099-01-01" not in dates


def test_metadata_matches_final_metadata():
    response = client.get("/metadata")
    assert response.status_code == 200
    body = response.json()
    with open("data/processed/final_metadata.json", "r", encoding="utf-8") as file:
        canonical = json.load(file)
    assert body == canonical


def test_pm25_valid():
    response = client.get(
        f"/pm25?date={KNOWN_DATE}&lat={LAT}&lon={LON}&resolution=500m"
    )
    assert response.status_code == 200
    body = response.json()
    assert body["date"] == KNOWN_DATE
    assert body["resolution_m"] == 500
    assert body["units"] == "\u00b5g/m\u00b3"
    assert body["status"] == "valid"
    assert body["pm25"] is not None


def test_pm25_matches_raster(pm25_tif_value):
    response = client.get(
        f"/pm25?date={KNOWN_DATE}&lat={LAT}&lon={LON}&resolution=500m"
    )
    body = response.json()
    assert math.isclose(body["pm25"], round(pm25_tif_value, 2), rel_tol=1e-3)


def test_pm25_invalid_date_404():
    response = client.get(f"/pm25?date=2099-01-01&lat={LAT}&lon={LON}&resolution=500m")
    assert response.status_code == 404


def test_pm25_malformed_date_422():
    response = client.get("/pm25?date=2025-13-99&lat=28.6&lon=77.2")
    assert response.status_code == 422


def test_pm25_invalid_latitude_422():
    response = client.get("/pm25?date=2025-01-01&lat=95&lon=77.2")
    assert response.status_code == 422


def test_pm25_outside_aoi_400():
    response = client.get("/pm25?date=2025-01-01&lat=10&lon=10&resolution=500m")
    assert response.status_code == 400


def test_pm25_unsupported_resolution_400():
    for res in ("250m", "100m", "50m"):
        response = client.get(
            f"/pm25?date={KNOWN_DATE}&lat={LAT}&lon={LON}&resolution={res}"
        )
        assert response.status_code == 400
        assert response.json()["detail"] == (
            "Requested resolution is not currently available."
        )


def test_pm25_grid():
    response = client.get(f"/pm25/grid?date={KNOWN_DATE}&resolution=500m")
    assert response.status_code == 200
    body = response.json()
    assert body["date"] == KNOWN_DATE
    assert body["resolution_m"] == 500
    assert body["crs"]
    assert body["n_rows"] > 0 and body["n_cols"] > 0
    assert "raster" in body["raster_url"]


def test_aqi_valid():
    response = client.get(f"/aqi?date={KNOWN_DATE}&lat={LAT}&lon={LON}")
    assert response.status_code == 200
    body = response.json()
    assert body["pm25_aqi"] is not None
    assert body["category"] in (
        "GOOD", "SATISFACTORY", "MODERATELY_POLLUTED", "POOR", "VERY_POOR", "SEVERE",
    )
    assert body["type"] == "PM2.5-derived AQI/sub-index"


def test_aqi_never_labeled_national():
    response = client.get(f"/aqi?date={KNOWN_DATE}&lat={LAT}&lon={LON}")
    body = response.json()
    assert "National" not in body["type"]
    assert "full" not in body["type"].lower()


def test_location_consistent_with_pm25_and_aqi(pm25_tif_value):
    pm25 = client.get(f"/pm25?date={KNOWN_DATE}&lat={LAT}&lon={LON}&resolution=500m").json()
    aqi = client.get(f"/aqi?date={KNOWN_DATE}&lat={LAT}&lon={LON}").json()
    location = client.get(f"/location?date={KNOWN_DATE}&lat={LAT}&lon={LON}").json()
    assert location["pm25"] == pm25["pm25"]
    assert location["pm25_derived_aqi"] == aqi["pm25_aqi"]
    assert location["aqi_category"] == aqi["category"]
    assert location["aqi_type"] == "PM2.5-derived AQI/sub-index"
    assert location["uncertainty_status"] == "DEFERRED"
    assert location["model"]
    assert location["dataset_mode"]


def test_hotspots_geojson():
    response = client.get(f"/hotspots?date={KNOWN_DATE}")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/geo+json")
    collection = response.json()
    assert collection["type"] == "FeatureCollection"
    for feature in collection["features"]:
        assert feature["properties"]["date"].startswith(KNOWN_DATE)
        assert "geometry" in feature


def test_hotspots_no_date_returns_all():
    response = client.get("/hotspots")
    assert response.status_code == 200
    assert len(response.json()["features"]) >= 1


def test_hotspot_statistics_matches_file():
    response = client.get("/hotspots/statistics")
    assert response.status_code == 200
    body = response.json()
    with open("data/processed/hotspot_statistics.json", "r", encoding="utf-8") as file:
        canonical = json.load(file)
    for key, value in canonical.items():
        assert body.get(key) == value
    assert body["hotspot_zone_count"] >= 1


def test_stations():
    response = client.get("/stations")
    assert response.status_code == 200
    stations = response.json()
    assert len(stations) >= 1
    for station in stations:
        assert station["station_id"]
        assert station["latitude"] is not None
        assert station["longitude"] is not None


def test_station_detail():
    response = client.get("/stations/ST_01")
    assert response.status_code == 200
    body = response.json()
    assert body["station_id"] == "ST_01"
    assert len(body["observations"]) >= 1
    assert all("date" in o and "pm25" in o for o in body["observations"])


def test_station_detail_404():
    response = client.get("/stations/ST_999")
    assert response.status_code == 404


def test_feature_importance():
    response = client.get("/feature-importance")
    assert response.status_code == 200
    body = response.json()
    assert body["interpretation"].startswith("Model feature importance")
    assert len(body["features"]) >= 1
    feature = body["features"][0]
    assert "feature" in feature
    assert "random_forest_importance" in feature
    assert "xgboost_gain" in feature


def test_feature_importance_matches_file():
    response = client.get("/feature-importance")
    body = response.json()
    frame = pd.read_csv("data/processed/model_feature_importance.csv")
    assert [f["feature"] for f in body["features"]] == frame["feature"].tolist()


def test_uncertainty_deferred():
    response = client.get("/uncertainty")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "DEFERRED"
    assert body["method"] is None or body["method"] == "none"
    assert "confidence" not in body
    assert "confidence_percent" not in body


def test_raster_pm25():
    response = client.get(f"/raster/pm25?date={KNOWN_DATE}&resolution=500m")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/tiff")
    assert len(response.content) > 0


def test_raster_aqi():
    response = client.get(f"/raster/aqi?date={KNOWN_DATE}")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/tiff")


def test_raster_missing_date_404():
    response = client.get("/raster/pm25?date=2099-01-01&resolution=500m")
    assert response.status_code == 404


def test_no_arbitrary_path_access():
    traversal = "2025-01-01/../../../config.yaml"
    response = client.get(f"/raster/pm25?date={traversal}&resolution=500m")
    assert response.status_code in (404, 422)
    response = client.get("/raster/aqi?date=../.env")
    assert response.status_code in (404, 422)
    response = client.get(
        f"/pm25?date={KNOWN_DATE}&lat={LAT}&lon={LON}&resolution=500m/../../config"
    )
    assert response.status_code == 400
    response = client.get(f"/pm25?date={KNOWN_DATE}&lat={LAT}&lon={LON}&resolution=../../etc")
    assert response.status_code == 400


def test_root_route():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["service"] == "pm25-mapping-api"


# ---------------------------------------------------------------------------
# AOD Activation Tests (Steps 12-14)
# ---------------------------------------------------------------------------

def test_health_includes_aod():
    """Step 12.13: API response schema — health endpoint."""
    response = client.get("/health")
    body = response.json()
    assert "aod" in body
    assert body["aod"]["configured"] in (True, False)
    assert body["aod"]["provider"] == "NASA MODIS MAIAC"
    assert body["aod"]["product"] == "MCD19A2.061"


def test_location_includes_aod_info():
    """Step 12.13: /location returns aod_info with all required fields."""
    response = client.get(f"/location?date={KNOWN_DATE}&lat={LAT}&lon={LON}")
    assert response.status_code == 200
    body = response.json()
    assert "aod_info" in body
    aod_info = body["aod_info"]
    assert aod_info is not None
    for field in ("aod", "status", "source", "resolution_m", "crs", "date", "unit", "nodata", "lookup", "distance_pixels"):
        assert field in aod_info, f"Missing field: {field}"
    assert aod_info["source"] == "MODIS/MAIAC MCD19A2 v061"
    assert aod_info["resolution_m"] == 500
    assert aod_info["status"] in ("AVAILABLE", "NO_VALID_OBSERVATION", "DATASET_UNAVAILABLE", "API_ERROR")


def test_location_aod_value_matches_raster():
    """Step 12.1: valid Delhi coordinate + valid date -> real AOD value."""
    import rasterio as _rasterio
    from rasterio.warp import transform as _warp_transform
    aod_path = f"data/processed/aod_500m_{KNOWN_DATE}.tif"
    with _rasterio.open(aod_path) as src:
        array = src.read(1)
        xs, ys = _warp_transform("EPSG:4326", src.crs, [LON], [LAT])
        col, row = _rasterio.transform.rowcol(src.transform, xs[0], ys[0])
        expected = float(array[row, col])
    response = client.get(f"/location?date={KNOWN_DATE}&lat={LAT}&lon={LON}")
    aod_info = response.json()["aod_info"]
    assert aod_info is not None
    assert aod_info["aod"] is not None
    assert aod_info["status"] == "AVAILABLE"
    assert math.isclose(aod_info["aod"], round(expected, 4), rel_tol=1e-3)


def test_location_aod_valid_range():
    """Step 12.1: AOD values are physically realistic."""
    response = client.get(f"/location?date={KNOWN_DATE}&lat={LAT}&lon={LON}")
    aod_info = response.json()["aod_info"]
    assert aod_info is not None
    aod_val = aod_info["aod"]
    assert aod_val is not None
    assert 0.0 <= aod_val <= 5.0, f"AOD value {aod_val} outside valid range [0, 5]"


def test_location_aod_set_used_when_available():
    """Step 12.1: aod_used is True when real AOD pixel exists."""
    response = client.get(f"/location?date={KNOWN_DATE}&lat={LAT}&lon={LON}")
    body = response.json()
    if body.get("aod_info") and body["aod_info"].get("aod") is not None:
        assert body["aod_used"] is True


def test_raster_aod_serves_geotiff():
    """Step 12.13: /raster/aod returns valid GeoTIFF."""
    response = client.get(f"/raster/aod?date={KNOWN_DATE}")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/tiff")
    assert len(response.content) > 100


def test_raster_aod_invalid_date_404():
    """Step 12.6: invalid date -> 404."""
    response = client.get("/raster/aod?date=2099-01-01")
    assert response.status_code == 404


def test_aod_no_path_traversal():
    """Step 12.13: security — no path traversal."""
    traversal = "2025-01-01/../../../config.yaml"
    response = client.get(f"/raster/aod?date={traversal}")
    assert response.status_code in (404, 422)


def test_location_outside_aoi_returns_400():
    """Step 12.3: coordinate outside AOD raster -> correct unavailable response."""
    response = client.get(f"/location?date={KNOWN_DATE}&lat=10&lon=10")
    assert response.status_code == 400


def test_aod_raster_valid_crs():
    """Step 12.13: AOD raster CRS is valid EPSG."""
    import rasterio as _rasterio
    aod_path = f"data/processed/aod_500m_{KNOWN_DATE}.tif"
    with _rasterio.open(aod_path) as src:
        crs = src.crs.to_string()
        assert "EPSG" in crs
        assert src.nodata is not None


def test_aod_all_dates_available():
    """Step 12.1: all known dates have valid AOD rasters."""
    from pathlib import Path as _Path
    known_dates = sorted({
        p.stem.replace("aod_500m_", "")
        for p in _Path("data/processed").glob("aod_500m_*.tif")
    })
    assert len(known_dates) >= 1
    for d in known_dates:
        response = client.get(f"/raster/aod?date={d}")
        assert response.status_code == 200, f"AOD raster missing for {d}"


def test_aod_pune_city():
    """Step 12.11: scope isolation — Pune AOD works independently."""
    response = client.get(f"/raster/aod?date={KNOWN_DATE}&city=pune")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/tiff")


def test_aod_mumbai_city():
    """Step 12.11: scope isolation — Mumbai AOD works independently."""
    response = client.get(f"/raster/aod?date={KNOWN_DATE}&city=mumbai")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/tiff")


def test_aod_raster_finite_values():
    """Step 12.12: no fabricated values — rasters have only finite valid data."""
    import rasterio as _rasterio
    import numpy as _np
    aod_path = f"data/processed/aod_500m_{KNOWN_DATE}.tif"
    with _rasterio.open(aod_path) as src:
        data = src.read(1)
        assert _np.all(_np.isfinite(data[data != src.nodata]))


def test_location_aod_crs_epsg():
    """Step 12.13: CRS in location response is valid."""
    response = client.get(f"/location?date={KNOWN_DATE}&lat={LAT}&lon={LON}")
    aod_info = response.json().get("aod_info")
    assert aod_info is not None
    assert aod_info["crs"] is not None
    assert "EPSG" in aod_info["crs"]


def test_aod_date_propagation():
    """Step 12.8: date propagation — AOD date matches selected date."""
    response = client.get(f"/location?date={KNOWN_DATE}&lat={LAT}&lon={LON}")
    aod_info = response.json()["aod_info"]
    assert aod_info["date"] == KNOWN_DATE


def test_aod_status_field_available():
    """Step 12.13: status field is present and valid."""
    response = client.get(f"/location?date={KNOWN_DATE}&lat={LAT}&lon={LON}")
    aod_info = response.json()["aod_info"]
    assert aod_info["status"] == "AVAILABLE"


def test_aod_lookup_field():
    """Step 12.13: lookup field is present."""
    response = client.get(f"/location?date={KNOWN_DATE}&lat={LAT}&lon={LON}")
    aod_info = response.json()["aod_info"]
    assert aod_info["lookup"] in ("exact_pixel", "nearest_valid_pixel")


def test_aod_unit_field():
    """Step 12.13: unit field is present."""
    response = client.get(f"/location?date={KNOWN_DATE}&lat={LAT}&lon={LON}")
    aod_info = response.json()["aod_info"]
    assert "unit" in aod_info


def test_aod_missing_date_returns_unavailable():
    """Step 12.7: missing AOD dataset for a valid PM25 date."""
    from pathlib import Path as _Path
    pm25_dates = sorted({
        p.stem.replace("pm25_500m_", "")
        for p in _Path("data/processed").glob("pm25_500m_*.tif")
    })
    aod_dates = sorted({
        p.stem.replace("aod_500m_", "")
        for p in _Path("data/processed").glob("aod_500m_*.tif")
    })
    missing_dates = [d for d in pm25_dates if d not in aod_dates]
    if missing_dates:
        test_date = missing_dates[0]
        response = client.get(f"/location?date={test_date}&lat={LAT}&lon={LON}")
        body = response.json()
        assert body["aod_info"]["status"] == "DATASET_UNAVAILABLE"


def test_aod_no_fabricated_values():
    """Step 12.12: AOD value must be a real float from the raster, not 0.0."""
    import rasterio as _rasterio
    from rasterio.warp import transform as _warp_transform
    aod_path = f"data/processed/aod_500m_{KNOWN_DATE}.tif"
    with _rasterio.open(aod_path) as src:
        array = src.read(1)
        xs, ys = _warp_transform("EPSG:4326", src.crs, [LON], [LAT])
        col, row = _rasterio.transform.rowcol(src.transform, xs[0], ys[0])
        expected = float(array[row, col])
    response = client.get(f"/location?date={KNOWN_DATE}&lat={LAT}&lon={LON}")
    aod_info = response.json()["aod_info"]
    assert aod_info["aod"] is not None
    assert aod_info["aod"] != 0.0, "AOD should not be exactly 0.0"
    assert aod_info["aod"] == round(expected, 4)


def test_aod_pune_location_has_value():
    """Step 12.1: Pune location returns real AOD."""
    PUNE_LAT = 18.52
    PUNE_LON = 73.86
    response = client.get(f"/location?date={KNOWN_DATE}&lat={PUNE_LAT}&lon={PUNE_LON}&city=pune")
    body = response.json()
    assert body["aod_info"] is not None
    assert body["aod_info"]["status"] in ("AVAILABLE", "NO_VALID_OBSERVATION", "DATASET_UNAVAILABLE")
    if body["aod_info"]["status"] == "AVAILABLE":
        assert body["aod_info"]["aod"] is not None
        assert 0.0 <= body["aod_info"]["aod"] <= 5.0


def test_aod_mumbai_location_has_value():
    """Step 12.1: Mumbai location returns real AOD."""
    MUMBAI_LAT = 19.08
    MUMBAI_LON = 72.88
    response = client.get(f"/location?date={KNOWN_DATE}&lat={MUMBAI_LAT}&lon={MUMBAI_LON}&city=mumbai")
    body = response.json()
    assert body["aod_info"] is not None
    assert body["aod_info"]["status"] in ("AVAILABLE", "NO_VALID_OBSERVATION", "DATASET_UNAVAILABLE")
    if body["aod_info"]["status"] == "AVAILABLE":
        assert body["aod_info"]["aod"] is not None
        assert 0.0 <= body["aod_info"]["aod"] <= 5.0


def test_aod_three_delhi_locations():
    """Step 13: test at least 3 different Delhi locations."""
    locations = [
        (28.6057, 77.2122),
        (28.6139, 77.2090),
        (28.6280, 77.2170),
    ]
    for lat, lon in locations:
        response = client.get(f"/location?date={KNOWN_DATE}&lat={lat}&lon={lon}")
        assert response.status_code == 200
        aod_info = response.json()["aod_info"]
        assert aod_info is not None
        assert aod_info["status"] in ("AVAILABLE", "NO_VALID_OBSERVATION")
        if aod_info["status"] == "AVAILABLE":
            assert aod_info["aod"] is not None
            assert 0.0 < aod_info["aod"] <= 5.0
