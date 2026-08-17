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
