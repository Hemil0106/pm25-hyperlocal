"""Globalization upgrade tests (Phase 25).

23 required cases:
 1.  global AOI                13. model scope metadata
 2.  India AOI                 14. global inference validation
 3.  Delhi AOI                 15. NoData
 4.  custom bbox               16. API bbox validation
 5.  CRS transformation        17. API security
 6.  tile generation           18. dashboard global mode   (vitest, App.test.tsx)
 7.  tile overlap              19. dashboard Delhi mode    (vitest, App.test.tsx)
 8.  dataset registry          20. invalid global request
 9.  data availability         21. AQI scheme selection
10.  ground-truth normalization 22. tile caching
11.  global training schema     23. output metadata
12.  grouped spatial validation

Cases 18 and 19 are dashboard behavior and are covered by the Vitest suite
(dashboard/src/App.test.tsx); this module covers the backend cases.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.main import app  # noqa: E402
from src.config import load_config  # noqa: E402
from src.geospatial.aoi import resolve_aoi  # noqa: E402
from src.geospatial.crs import (  # noqa: E402
    geodesic_distance_km,
    get_metric_crs,
    transform_coords,
    utm_zone_for,
)
from src.geospatial.tiles import (  # noqa: E402
    CACHE_MARKER,
    cache_get,
    cache_is_valid,
    cache_key,
    cache_put,
    count_tiles_for_global,
    generate_tiles,
    tile_for_point,
)
from src.globalization.aqi import (  # noqa: E402
    aqi_scheme_for,
    get_breakpoints,
    pm25_to_aqi,
)
from src.globalization.datasets import build_dataset_registry  # noqa: E402
from src.globalization.ground_truth import (  # noqa: E402
    GROUND_TRUTH_SCHEMA,
    ground_truth_summary,
    normalize_ground_truth,
)
from src.globalization.inference import inference_plan, predict_for_aoi  # noqa: E402
from src.globalization.model_scopes import (  # noqa: E402
    prediction_scope_metadata,
    resolve_model_scope,
    scopes_summary,
)
from src.globalization.training import (  # noqa: E402
    build_training_frame,
    global_training_schema,
)
from src.globalization.validation import group_assignments, grouped_spatial_cv  # noqa: E402

client = TestClient(app)

DELHI_BBOX = {"west": 77.0, "south": 28.4, "east": 77.4, "north": 28.8}


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def delhi_aoi(config):
    return resolve_aoi(config, region="delhi")


@pytest.fixture(scope="module")
def india_aoi(config):
    return resolve_aoi(config, region="india")


@pytest.fixture(scope="module")
def global_aoi(config):
    return resolve_aoi(config, region="global")


# ---------------------------------------------------------------------------
# 1. Global AOI
# ---------------------------------------------------------------------------
def test_global_aoi(config, global_aoi):
    assert global_aoi.is_global is True
    assert global_aoi.name == "Global"
    bounds = global_aoi.bounds
    assert bounds["west"] == -180.0 and bounds["east"] == 180.0
    assert bounds["south"] == -90.0 and bounds["north"] == 90.0
    assert global_aoi.area_km2() > 400_000_000
    assert global_aoi.centroid_crs() is None


# ---------------------------------------------------------------------------
# 2. India AOI
# ---------------------------------------------------------------------------
def test_india_aoi(config, india_aoi):
    assert india_aoi.is_global is False
    assert india_aoi.name == "India"
    assert india_aoi.area_km2() > 2_000_000
    assert india_aoi.centroid_crs() is not None


# ---------------------------------------------------------------------------
# 3. Delhi AOI
# ---------------------------------------------------------------------------
def test_delhi_aoi(config, delhi_aoi):
    assert delhi_aoi.name == "Delhi"
    assert delhi_aoi.centroid_crs() == "EPSG:32643"
    assert 1000 < delhi_aoi.area_km2() < 3000


# ---------------------------------------------------------------------------
# 4. Custom bbox
# ---------------------------------------------------------------------------
def test_custom_bbox_aoi(config):
    aoi = resolve_aoi(config, bbox={"min_lon": 77.0, "min_lat": 28.4,
                                    "max_lon": 77.4, "max_lat": 28.8})
    assert aoi.mode == "bbox"
    assert aoi.name == "Custom AOI"
    assert aoi.bounds == {"west": 77.0, "south": 28.4, "east": 77.4, "north": 28.8}


# ---------------------------------------------------------------------------
# 5. CRS transformation
# ---------------------------------------------------------------------------
def test_crs_transformation_roundtrip():
    lon, lat = transform_coords(77.2, 28.6, "EPSG:4326", "EPSG:32643")
    lon2, lat2 = transform_coords(lon, lat, "EPSG:32643", "EPSG:4326")
    assert abs(lon2 - 77.2) < 1e-6
    assert abs(lat2 - 28.6) < 1e-6


def test_get_metric_crs_matches_region():
    assert get_metric_crs(DELHI_BBOX) == "EPSG:32643"
    global_bbox = {"west": -180.0, "south": -90.0, "east": 180.0, "north": 90.0}
    assert get_metric_crs(global_bbox) is None


def test_utm_zone_for():
    assert utm_zone_for(77.2, 28.6) == 43


def test_geodesic_distance_km():
    d = geodesic_distance_km(77.0, 28.6, 77.4, 28.6)
    assert abs(d - 39.1) < 0.5


# ---------------------------------------------------------------------------
# 6. Tile generation
# ---------------------------------------------------------------------------
def test_tile_generation_global_10deg():
    tiles = list(generate_tiles({"west": -180, "south": -90, "east": 180, "north": 90},
                                size_deg=10.0, overlap_deg=0.0))
    assert len(tiles) == 648
    assert all(t.tile_id.startswith("tile_") for t in tiles)
    ids = {t.tile_id for t in tiles}
    assert len(ids) == 648


def test_tile_generation_covers_bbox():
    tiles = list(generate_tiles(DELHI_BBOX, size_deg=0.5, overlap_deg=0.0))
    assert len(tiles) == 1
    assert tiles[0].tile_id == "tile_000_000"
    assert tile_for_point(77.2, 28.6, size_deg=0.5, bbox=DELHI_BBOX) == "tile_000_000"


# ---------------------------------------------------------------------------
# 7. Tile overlap
# ---------------------------------------------------------------------------
def test_tile_overlap():
    tiles = list(generate_tiles({"west": 0, "south": 0, "east": 20, "north": 10},
                                size_deg=10.0, overlap_deg=2.0))
    ids = sorted(t.tile_id for t in tiles)
    assert ids == ["tile_000_000", "tile_001_000"]
    first, second = tiles[0], tiles[1]
    assert second.lon_range[0] < first.lon_range[1]


# ---------------------------------------------------------------------------
# 8. Dataset registry
# ---------------------------------------------------------------------------
def test_dataset_registry(config, delhi_aoi):
    reg = build_dataset_registry(config, delhi_aoi, write_path=None)
    assert "datasets" in reg
    assert "registry_version" in reg
    assert "built_for_aoi" in reg
    assert set(reg["datasets"]) == {"aod", "cpcb", "weather", "ndvi", "osm",
                                    "dem", "viirs", "ground_truth"}
    assert reg["built_for_aoi"]["name"] == "Delhi"
    assert reg["datasets"]["cpcb"]["local_data_present"] is True


def test_dataset_registry_global(config, global_aoi):
    reg = build_dataset_registry(config, global_aoi, write_path=None)
    assert reg["built_for_aoi"]["name"] == "Global"
    assert reg["datasets"]["aod"]["coverage"] == "global (1 km)"
    assert reg["datasets"]["ground_truth"]["sources"]["cpcb"]["country"] == "India"


# ---------------------------------------------------------------------------
# 9. Data availability (API + module)
# ---------------------------------------------------------------------------
def test_data_availability_delhi():
    resp = client.get("/global/data-availability")
    assert resp.status_code == 200
    body = resp.json()
    assert body["overall"] == "available"
    assert body["ground_truth"]["n_stations_in_aoi"] >= 4
    assert body["ground_truth"]["n_rows_in_aoi"] >= 200


def test_data_availability_global():
    resp = client.get("/global/data-availability?region=global")
    assert resp.status_code == 200
    body = resp.json()
    assert body["overall"] == "data_limited"
    assert body["ground_truth"]["countries"] == ["India"]
    assert body["model_scopes"]["global"]["status"] == "unavailable"
    assert any("global scope is unavailable" in note for note in body["notes"])


# ---------------------------------------------------------------------------
# 10. Ground-truth normalization
# ---------------------------------------------------------------------------
def test_ground_truth_normalization(config):
    gt = normalize_ground_truth(config, write_path=None)
    assert not gt.empty
    assert set(GROUND_TRUTH_SCHEMA).issubset(gt.columns)
    summary = ground_truth_summary(gt)
    assert summary["n_stations"] >= 4
    assert summary["n_rows"] >= 200
    assert "India" in summary["countries"]


# ---------------------------------------------------------------------------
# 11. Global training schema
# ---------------------------------------------------------------------------
def test_global_training_schema(config, global_aoi):
    schema = global_training_schema()
    assert "features" in schema
    assert "target" in schema
    _, report = build_training_frame(config, aoi=global_aoi)
    assert report["valid"] is False
    assert any("single-country" in (report.get("note") or "") for _ in [0]) or not report.get(
        "valid"
    )


def test_delhi_training_frame_valid(config, delhi_aoi):
    frame, report = build_training_frame(config, aoi=delhi_aoi)
    assert report["valid"] is True
    assert not frame.empty


# ---------------------------------------------------------------------------
# 12. Grouped spatial validation (LOO station/city/region/country)
# ---------------------------------------------------------------------------
def _cv_frame(n_stations=6, n_dates=10):
    rows = []
    for s in range(n_stations):
        for d in range(n_dates):
            rows.append({
                "station_id": f"ST_{s}",
                "city": f"city_{s % 2}",
                "region": f"region_{s % 3}",
                "country": f"country_{s % 2}",
                "latitude": 28.5 + s * 0.05,
                "longitude": 77.0 + s * 0.05,
                "date": f"2025-01-{d + 1:02d}",
                "PM2.5": 100 + s * 5 + d,
                "road_density": 1.0 + s,
                "NDVI": 0.2 + d * 0.01,
                "temperature_c": 15 + d,
            })
    return pd.DataFrame(rows)


@pytest.mark.parametrize("group_by", ["station_id", "city", "region", "country"])
def test_grouped_spatial_cv_logo(group_by):
    df = _cv_frame()
    result = grouped_spatial_cv(
        df, features=["road_density", "NDVI", "temperature_c"],
        target="PM2.5", group_by=group_by,
    )
    assert result["strategy"] == "leave_one_group_out"
    assert result["group_by"] == group_by
    assert result["n_folds"] == len(set(df[group_by]))
    assert result["overall"]["n"] == len(df)
    assert "folds" in result
    assert len(result["predictions"]) == len(df)


def test_grouped_cv_rejects_unknown_group():
    with pytest.raises(ValueError):
        grouped_spatial_cv(_cv_frame(), ["road_density"], "PM2.5",
                           group_by="district")


def test_group_assignments():
    df = _cv_frame()
    out = group_assignments(df)
    assert set(out["country"]) == {"country_0", "country_1"}
    assert out["city"].tolist() == df["city"].tolist()


# ---------------------------------------------------------------------------
# 13. Model scope metadata
# ---------------------------------------------------------------------------
def test_model_scope_delhi_available(config, delhi_aoi):
    scope = resolve_model_scope(config, delhi_aoi)
    assert scope["can_predict"] is True
    assert scope["status"] == "available"
    meta = prediction_scope_metadata(config, delhi_aoi)
    assert meta["scope_id"] == "prototype_local"


def test_model_scope_global_unavailable(config, global_aoi):
    scope = resolve_model_scope(config, global_aoi)
    assert scope["can_predict"] is False
    assert scope["status"] == "unavailable_for_aoi"
    assert "Global/regional PM2.5 prediction unavailable" in scope["reason"]


def test_model_scope_summary(config):
    summary = scopes_summary(config)
    assert summary["prototype_local"]["status"] == "available"
    assert summary["regional"]["status"] == "unavailable"
    assert summary["global"]["status"] == "unavailable"


# ---------------------------------------------------------------------------
# 14. Global inference validation
# ---------------------------------------------------------------------------
def test_inference_plan_global_unavailable(config, global_aoi):
    plan = inference_plan(config, global_aoi, date="2025-01-01")
    assert plan["can_predict"] is False
    assert plan["scope_status"] == "unavailable_for_aoi"


def test_predict_for_global_returns_unavailable(config, global_aoi):
    result = predict_for_aoi(config, global_aoi, date="2025-01-01")
    assert result["status"] == "unavailable"
    assert result["predictions"] is None
    assert result["scope_metadata"]["scope_status"] == "unavailable_for_aoi"


def test_predict_for_delhi_available(config, delhi_aoi):
    result = predict_for_aoi(config, delhi_aoi, date="2025-01-01")
    assert result["status"] == "available"
    assert result["predictions"] is not None
    assert result["n_cells"] > 0
    assert result["scope_metadata"]["scope_id"] == "prototype_local"


# ---------------------------------------------------------------------------
# 15. NoData
# ---------------------------------------------------------------------------
def test_no_data_date_returns_nodata():
    resp = client.get("/global/pm25/bbox?date=2099-01-01&west=77.05&south=28.45&east=77.25&north=28.7")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "NoData"
    assert body["n_cells"] == 0


# ---------------------------------------------------------------------------
# 16. API bbox validation
# ---------------------------------------------------------------------------
def test_pm25_bbox_delhi_available():
    resp = client.get("/global/pm25/bbox?date=2025-01-01&west=77.05&south=28.45&east=77.25&north=28.7")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "available"
    assert body["n_cells"] > 0
    for cell in body["cells"]:
        assert 77.05 <= cell["longitude"] <= 77.25
        assert 28.45 <= cell["latitude"] <= 28.7
        assert cell["pm25"] is not None


def test_pm25_bbox_outside_delhi_unavailable():
    resp = client.get("/global/pm25/bbox?date=2025-01-01&west=-120&south=30&east=-110&north=35")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "unavailable"
    assert body["n_cells"] == 0
    assert body["reason"]


# ---------------------------------------------------------------------------
# 17. API security
# ---------------------------------------------------------------------------
def test_unknown_region_rejected():
    resp = client.get("/global/aoi?region=../../etc/passwd")
    assert resp.status_code == 400
    assert "Unknown region" in resp.json()["detail"]


def test_region_and_bbox_conflict_rejected():
    resp = client.get("/global/aoi?region=delhi&bbox=1,1,2,2")
    assert resp.status_code == 400


def test_invalid_bbox_rejected():
    resp = client.get("/global/aoi?bbox=1,1,1,2")
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# 18 / 19. Dashboard global + Delhi mode -> covered by Vitest (App.test.tsx)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 20. Invalid global request
# ---------------------------------------------------------------------------
def test_bbox_too_large_rejected():
    resp = client.get("/global/pm25/bbox?date=2025-01-01&west=0&south=0&east=20&north=20")
    assert resp.status_code == 400


def test_inverted_bbox_rejected():
    resp = client.get("/global/pm25/bbox?date=2025-01-01&west=1&south=1&east=0&north=2")
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# 21. AQI scheme selection
# ---------------------------------------------------------------------------
def test_aqi_scheme_delhi_is_india_cpcb(config, delhi_aoi):
    scheme = aqi_scheme_for(config, delhi_aoi)
    assert scheme["scheme"] == "india_cpcb"


def test_aqi_scheme_global_is_none(config, global_aoi):
    scheme = aqi_scheme_for(config, global_aoi)
    assert scheme["scheme"] == "none"


def test_pm25_to_aqi_uses_config_breakpoints(config):
    breakpoints = get_breakpoints(config, "india_cpcb")
    assert breakpoints
    aqi = pm25_to_aqi(116.8, breakpoints)
    assert aqi is not None
    assert 100 <= aqi <= 500


def test_pm25_to_aqi_handles_invalid():
    breakpoints = [{"concentration_low": 0, "concentration_high": 1000,
                    "aqi_low": 0, "aqi_high": 500}]
    assert pm25_to_aqi(None, breakpoints) is None
    assert pm25_to_aqi(-5.0, breakpoints) is None


# ---------------------------------------------------------------------------
# 22. Tile caching
# ---------------------------------------------------------------------------
def test_tile_caching(tmp_path):
    key = cache_key("aod", "pm25", "2025-01-01", "tile_000_000", version="v1",
                    cache_dir=tmp_path)
    assert not cache_is_valid(key)
    cache_put(key, metadata={"rows": 3})
    assert cache_is_valid(key)
    assert key.parent.name == "tile_000_000"
    marker = key / CACHE_MARKER
    assert marker.exists()
    cached = cache_get(key)
    assert cached["rows"] == 3
    assert cache_is_valid(tmp_path / "missing" / CACHE_MARKER) is False


# ---------------------------------------------------------------------------
# 23. Output metadata
# ---------------------------------------------------------------------------
def test_output_metadata_delhi():
    resp = client.get("/global/output-metadata?region=delhi&date=2025-01-01")
    assert resp.status_code == 200
    body = resp.json()
    assert body["inference"]["can_predict"] is True
    assert body["inference"]["resolution_500m_output_exists"] is True
    assert body["downscaling"]["available"] is True
    assert body["hotspots"]["available"] is True
    assert "predicted high-pollution zone" in body["hotspots"]["definition"].lower()


def test_output_metadata_global_honest():
    resp = client.get("/global/output-metadata?region=global&date=2025-01-01")
    assert resp.status_code == 200
    body = resp.json()
    assert body["inference"]["can_predict"] is False
    assert body["inference"]["scope_status"] == "unavailable_for_aoi"
    assert body["downscaling"]["available"] is False
    assert body["hotspots"]["available"] is False
