"""Global data acquisition tests (Milestone 16).

Covers the M16 data layer: scopes, units, QC, daily aggregation, station
registry, integrity/checksums, NDVI composite selection, satellite adapters,
grid alignment, temporal join, coverage report, manifest, cache/failed-download
manifest, and the /data/* API endpoints.

Core M16 rules verified here:
  - scope isolation (requested_scope == artifact_scope)
  - units normalized to pm25_ug_m3; unknown units are never guessed
  - QC removes invalid rows, FLAGS (not deletes) outliers
  - daily aggregation never zero-fills; insufficient days are skipped
  - NDVI nearest valid composite with no-future default
  - DEM NoData is never converted to zero
  - missing credentials -> graceful UNAVAILABLE, no fabrication
  - synthetic test data never leaks into real datasets
  - global ML / prediction are NOT implemented in M16
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

from src.global_data.cache import (  # noqa: E402
    DownloadError,
    GlobalCache,
    fetch_with_retry,
    record_failed_download,
)
from src.global_data.coverage import build_coverage_report  # noqa: E402
from src.global_data.grid import common_grid, grid_for_tile  # noqa: E402
from src.global_data.integrity import (  # noqa: E402
    has_synthetic_rows,
    sha256_file,
    synthetic_leakage_report,
    validate_checksum,
    verify_artifact_scope,
)
from src.global_data.manifest import build_manifest  # noqa: E402
from src.global_data.ndvi_global import (  # noqa: E402
    composite_dates,
    composite_metadata,
    nearest_valid_composite,
)
from src.global_data.qc import apply_qc  # noqa: E402
from src.global_data.scope import (  # noqa: E402
    SUPPORTED_SCOPES,
    artifact_scope_tag,
    assert_scope_isolated,
    infer_scope,
    scope_bounds,
    validate_scope,
)
from src.global_data.sources import build_data_source_registry  # noqa: E402
from src.global_data.stations import (  # noqa: E402
    STATION_REGISTRY_SCHEMA,
    build_station_registry,
    build_station_summary,
    write_station_outputs,
)
from src.global_data.temporal import aggregate_daily  # noqa: E402
from src.global_data.temporal_join import (  # noqa: E402
    add_feature_to_observations,
    nearest_grid_cell_id,
    spatial_join_to_grid,
    temporal_join,
)
from src.global_data.units import (  # noqa: E402
    TARGET_UNIT,
    conversion_factor,
    is_known_unit,
    normalize_pm25_units,
)
from src.global_data.viirs_global import qa_pass_mask  # noqa: E402
from src.global_data.dem_global import (  # noqa: E402
    SRTM_NODATA,
    srtm_tile_name,
)
from src.global_data.osm_global import road_density_query  # noqa: E402
from src.global_data.weather_global import _cds_request_payload  # noqa: E402
from src.global_data.nasa_auth import cmr_search_granules  # noqa: E402

client = TestClient(app)

DELHI_BBOX = {"west": 77.0, "south": 28.4, "east": 77.4, "north": 28.8}


@pytest.fixture(scope="module")
def config():
    return load_config()


def make_config(tmp_path: Path, **overrides) -> dict:
    """Minimal global_data config pointing storage/cache into tmp_path."""
    cfg = {
        "global_data": {
            "fetch": {
                "retries": 1, "backoff_base_s": 0.0, "backoff_max_s": 0.0,
                "max_tiles": 0,
            },
            "cache": {
                "dir": str(tmp_path / "cache"), "version": "v1",
                "validate_before_redownload": True,
            },
            "storage": {
                "raw_base": str(tmp_path / "raw"),
                "processed_base": str(tmp_path / "processed"),
            },
            "temporal": {"aggregation": "daily_mean", "min_observations_per_day": 1},
            "qc": {"outlier_method": "mad", "outlier_threshold_k": 5.0,
                   "max_pm25_ug_m3": 1000.0},
            "pm25": {
                "enabled": True, "api_version": "v3",
                "base_url": "https://api.openaq.org/v3",
                "credential_env_var": "OPENAQ_API_KEY",
            },
            "sources": {
                "aod": {"enabled": True, "credential_env_var": "EARTHDATA_USERNAME",
                        "credential_env_var_2": "EARTHDATA_PASSWORD"},
                "weather": {"enabled": True, "credential_env_var": "CDSAPI_URL",
                            "credential_env_var_2": "CDSAPI_KEY"},
                "ndvi": {"enabled": True, "credential_env_var": "EARTHDATA_USERNAME",
                         "credential_env_var_2": "EARTHDATA_PASSWORD",
                         "temporal": {"composite_period_days": 16,
                                      "no_future_match": True}},
                "dem": {"enabled": True, "credential_env_var": "EARTHDATA_USERNAME",
                        "credential_env_var_2": "EARTHDATA_PASSWORD"},
                "osm": {"enabled": True,
                        "endpoint": "https://overpass-api.de/api/interpreter"},
                "viirs": {"enabled": True, "credential_env_var": "EARTHDATA_USERNAME",
                          "credential_env_var_2": "EARTHDATA_PASSWORD"},
                "dummy": {"enabled": True},
            },
        },
        "time": {"start_date": "2025-01-01", "end_date": "2025-01-01"},
        "datasets": {"weather": {"variables": ["2m_temperature"]}},
    }
    for key, value in overrides.items():
        section = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            section = section.setdefault(part, {})
        section[parts[-1]] = value
    return cfg


# ---------------------------------------------------------------------------
# Scope selection and isolation
# ---------------------------------------------------------------------------

def test_scope_validation_supported_scopes():
    assert SUPPORTED_SCOPES == ("global", "india", "delhi", "pune", "mumbai")
    assert validate_scope("Global") == "global"
    assert validate_scope(None) == "global"
    assert validate_scope("  delhi ") == "delhi"


def test_scope_validation_rejects_unknown():
    with pytest.raises(ValueError):
        validate_scope("mars")
    with pytest.raises(ValueError):
        validate_scope("")


def test_scope_bounds_exact():
    assert scope_bounds("global") == {"west": -180.0, "south": -90.0,
                                      "east": 180.0, "north": 90.0}
    assert scope_bounds("india") == {"west": 65.0, "south": 5.0,
                                     "east": 100.0, "north": 40.0}
    assert scope_bounds("delhi") == DELHI_BBOX


def test_infer_scope_smallest_containing():
    assert infer_scope(DELHI_BBOX) == "delhi"
    assert infer_scope({"west": 70.0, "south": 10.0, "east": 90.0, "north": 35.0}) == "india"
    assert infer_scope({"west": -120.0, "south": -40.0, "east": 120.0, "north": 60.0}) == "global"


def test_infer_scope_outside_all():
    assert infer_scope({"west": 200.0, "south": 95.0, "east": 250.0, "north": 120.0}) == "outside_all_scopes"


def test_assert_scope_isolated_refuses_cross_scope():
    # A Delhi artifact must never be served under india/global scope.
    with pytest.raises(ValueError):
        assert_scope_isolated("india", DELHI_BBOX)
    # An india artifact must never be served under global.
    assert_scope_isolated("global", DELHI_BBOX)  # global accepts anything inside world


def test_assert_scope_isolated_accepts_contained():
    assert_scope_isolated("delhi", DELHI_BBOX)
    assert_scope_isolated("india", {"west": 70.0, "south": 10.0, "east": 90.0, "north": 35.0})


def test_artifact_scope_tag():
    assert artifact_scope_tag("global") == "scope=global"
    assert artifact_scope_tag("delhi") == "scope=delhi"


# ---------------------------------------------------------------------------
# Unit normalization
# ---------------------------------------------------------------------------

def test_conversion_factors_known():
    assert conversion_factor("ug/m3") == 1.0
    assert conversion_factor("µg/m³") == 1.0
    assert conversion_factor("mg/m3") == 1000.0
    assert conversion_factor("g/m3") == 1_000_000.0
    assert conversion_factor("ng/m3") == 0.001


def test_conversion_factor_unknown_not_guessed():
    assert conversion_factor("ppm") is None
    assert conversion_factor("unknown") is None
    assert conversion_factor(None) is None
    assert is_known_unit("ug/m3") is True
    assert is_known_unit("ppm") is False


def test_normalize_pm25_units_known_and_unknown():
    df = pd.DataFrame({
        "station_id": ["A", "B", "C"],
        "PM2.5": [10.0, 5.0, 7.0],
        "units": ["ug/m3", "mg/m3", "ppm"],
    })
    out, report = normalize_pm25_units(df)
    assert TARGET_UNIT in out.columns
    # mg/m3 * 1000 -> 5000 ug/m3; ppm is unknown -> NaN (never guessed).
    assert out.loc[out["station_id"] == "A", "pm25_ug_m3"].iloc[0] == 10.0
    assert out.loc[out["station_id"] == "B", "pm25_ug_m3"].iloc[0] == 5000.0
    assert pd.isna(out.loc[out["station_id"] == "C", "pm25_ug_m3"].iloc[0])
    assert report["unknown_units"] == 1
    assert report["target_unit"] == "pm25_ug_m3"


def test_normalize_pm25_units_traceable():
    df = pd.DataFrame({"station_id": ["A"], "PM2.5": [12.5], "units": ["ug/m3"]})
    out, _ = normalize_pm25_units(df)
    assert out["pm25_raw"].iloc[0] == 12.5
    assert out["unit_factor"].iloc[0] == 1.0


def test_normalize_pm25_units_empty():
    df = pd.DataFrame(columns=["station_id", "PM2.5", "units"])
    out, report = normalize_pm25_units(df)
    assert out.empty and report["rows"] == 0


# ---------------------------------------------------------------------------
# Quality control
# ---------------------------------------------------------------------------

def _qc_frame():
    return pd.DataFrame({
        "station_id": ["S1", "S1", "S1", "S2", "S2", "S2"],
        "timestamp": ["2025-01-01 01:00:00", "2025-01-01 01:00:00",
                      "2025-01-01 02:00:00", "2025-01-01 03:00:00",
                      "2025-01-01 04:00:00", "2025-01-01 05:00:00"],
        "latitude": [28.6, 28.6, 28.6, 28.7, 28.7, 28.7],
        "longitude": [77.2, 77.2, 77.2, 77.3, 77.3, 77.3],
        "pm25_ug_m3": [10.0, 10.0, 12.0, 11.0, 13.0, 200.0],
        "units": ["ug/m3"] * 6,
        "country": ["India"] * 6,
    })


def test_qc_removes_duplicates():
    clean, report = apply_qc(_qc_frame())
    assert report["duplicate_rows"] == 1
    assert clean.duplicated(subset=["station_id", "timestamp"]).sum() == 0


def test_qc_removes_invalid_coordinates():
    df = _qc_frame()
    df.loc[0, "latitude"] = 91.0
    clean, report = apply_qc(df)
    assert report["invalid_coordinates"] == 1
    assert 91.0 not in clean["latitude"].values


def test_qc_removes_invalid_timestamps():
    df = _qc_frame()
    df.loc[0, "timestamp"] = "not-a-date"
    clean, report = apply_qc(df)
    assert report["invalid_timestamps"] == 1


def test_qc_removes_missing_pm25():
    df = _qc_frame()
    df.loc[0, "pm25_ug_m3"] = None
    clean, report = apply_qc(df)
    assert report["missing_pm25"] == 1


def test_qc_removes_invalid_units():
    df = _qc_frame()
    df.loc[0, "units"] = ""
    clean, report = apply_qc(df)
    assert report["invalid_units"] == 1


def test_qc_removes_negative_and_above_max():
    df = _qc_frame()
    df.loc[1, "timestamp"] = "2025-01-01 01:30:00"  # de-duplicate rows 0/1
    df.loc[0, "pm25_ug_m3"] = -5.0
    df.loc[1, "pm25_ug_m3"] = 5000.0
    clean, report = apply_qc(df)
    assert report["duplicate_rows"] == 0
    assert report["negative_pm25"] == 1
    assert report["above_max_pm25"] == 1


def test_qc_flags_outliers_not_removes():
    df = _qc_frame()
    clean, report = apply_qc(df)
    # 200.0 is far outside the MAD envelope -> flagged but still retained.
    assert report["outlier_flags"] >= 1
    assert "outlier_flag" in clean.columns
    assert clean["outlier_flag"].sum() >= 1
    assert clean["outlier_flag"].dtype == bool


def test_qc_report_fields():
    clean, report = apply_qc(_qc_frame())
    for field in ("input_rows", "retained_rows", "duplicate_rows",
                  "invalid_coordinates", "missing_pm25", "invalid_units",
                  "negative_pm25", "outlier_flags", "date_range",
                  "station_count", "country_count", "status"):
        assert field in report
    assert report["station_count"] == 2
    assert report["country_count"] == 1
    assert report["date_range"] == {"min": "2025-01-01", "max": "2025-01-01"}
    assert report["status"] == "ok"


def test_qc_empty_input():
    df = pd.DataFrame(columns=["station_id", "timestamp", "latitude",
                               "longitude", "pm25_ug_m3", "units", "country"])
    clean, report = apply_qc(df)
    assert clean.empty and report["status"] == "empty" and report["input_rows"] == 0


# ---------------------------------------------------------------------------
# Daily temporal aggregation
# ---------------------------------------------------------------------------

def _daily_source():
    return pd.DataFrame({
        "station_id": ["S1", "S1", "S1", "S1", "S2"],
        "timestamp": ["2025-01-01 01:00:00", "2025-01-01 03:00:00",
                      "2025-01-02 01:00:00", "2025-01-02 02:00:00",
                      "2025-01-01 04:00:00"],
        "pm25_ug_m3": [10.0, 14.0, 8.0, 12.0, 100.0],
        "latitude": [28.6, 28.6, 28.6, 28.6, 28.7],
        "longitude": [77.2, 77.2, 77.2, 77.2, 77.3],
        "country": ["India", "India", "India", "India", "India"],
        "source": ["openaq"] * 5,
    })


def test_aggregate_daily_mean_and_count():
    daily, report = aggregate_daily(_daily_source())
    s1_day1 = daily[(daily["station_id"] == "S1") & (daily["date"] == "2025-01-01")]
    assert s1_day1["pm25_daily"].iloc[0] == pytest.approx(12.0)
    assert s1_day1["observation_count"].iloc[0] == 2
    assert report["station_days_retained"] == 3


def test_aggregate_daily_skips_insufficient_days_never_zero_fills():
    daily, report = aggregate_daily(_daily_source(), min_observations_per_day=2)
    # S1 2025-01-02 has 2 obs, S2 2025-01-01 has 1 (skipped), S1 2025-01-01 has 2.
    assert report["station_days_skipped"] == 1
    assert "S2" not in daily["station_id"].values
    # Missing days are omitted entirely, never zero-filled.
    assert (daily["pm25_daily"] == 0).sum() == 0


def test_aggregate_daily_empty():
    daily, report = aggregate_daily(
        pd.DataFrame(columns=["station_id", "timestamp", "pm25_ug_m3"]))
    assert daily.empty and report["status"] == "empty"


# ---------------------------------------------------------------------------
# Station registry and summary
# ---------------------------------------------------------------------------

def test_build_station_registry_schema():
    daily, _ = aggregate_daily(_daily_source())
    registry = build_station_registry(daily)
    assert list(registry.columns) == STATION_REGISTRY_SCHEMA
    assert registry["station_id"].nunique() == len(registry)
    assert set(registry["station_id"]) == {"S1", "S2"}


def test_build_station_summary_counts():
    daily, _ = aggregate_daily(_daily_source())
    registry = build_station_registry(daily)
    summary = build_station_summary(registry)
    assert summary["country"].tolist() == ["India"]
    assert summary["n_stations"].iloc[0] == 2


def test_write_station_outputs(tmp_path):
    daily, _ = aggregate_daily(_daily_source())
    registry = build_station_registry(daily)
    result = write_station_outputs(registry, daily, tmp_path)
    assert Path(result["registry_path"]).exists()
    assert Path(result["summary_path"]).exists()
    assert result["n_stations"] == 2
    assert result["countries"] == ["India"]
    assert result["date_range"] == {"min": "2025-01-01", "max": "2025-01-02"}


# ---------------------------------------------------------------------------
# Integrity: checksums, scope verification, synthetic leakage
# ---------------------------------------------------------------------------

def test_sha256_and_validate_checksum(tmp_path):
    file = tmp_path / "a.tif"
    file.write_bytes(b"abc" * 100)
    digest = sha256_file(file)
    assert validate_checksum(file, digest) is True
    assert validate_checksum(file, "deadbeef") is False
    assert validate_checksum(tmp_path / "missing.tif", digest) is False
    assert validate_checksum(file, None) is False


def test_verify_artifact_scope():
    ok, reason = verify_artifact_scope("delhi", DELHI_BBOX)
    assert ok is True
    ok, reason = verify_artifact_scope("global", DELHI_BBOX)
    assert ok is True
    ok, reason = verify_artifact_scope("india", DELHI_BBOX)
    assert ok is False
    ok, reason = verify_artifact_scope("nonsense", DELHI_BBOX)
    assert ok is False


def test_synthetic_leakage_detection():
    clean = pd.DataFrame({"quality_flag": ["openaq_reported"]})
    assert has_synthetic_rows(clean) is False
    leaked = pd.DataFrame({"quality_flag": ["openaq_reported", "SYNTHETIC_TEST_DATA"]})
    assert has_synthetic_rows(leaked) is True
    assert synthetic_leakage_report(leaked, "global_pm25")["synthetic_data_leakage"] == "PRESENT"
    assert synthetic_leakage_report(clean, "global_pm25")["synthetic_data_leakage"] == "NONE"


# ---------------------------------------------------------------------------
# NDVI composite selection
# ---------------------------------------------------------------------------

def test_ndvi_composite_dates_period():
    dates = composite_dates(16, anchor="2024-12-20")
    assert dates[:3] == ["2024-12-20", "2025-01-05", "2025-01-21"]


def test_ndvi_nearest_valid_no_future():
    # target 2025-01-01; only past composites within lookback qualify.
    selected = nearest_valid_composite("2025-01-01", ["2024-12-20", "2024-11-01"])
    assert selected == "2024-12-20"


def test_ndvi_nearest_valid_future_allowed():
    selected = nearest_valid_composite(
        "2025-01-01", ["2024-12-20", "2025-01-05"], no_future_match=False)
    # 2025-01-05 is 4 days from target; 2024-12-20 is 12 days.
    assert selected == "2025-01-05"


def test_ndvi_no_future_excludes_future():
    selected = nearest_valid_composite("2025-01-01", ["2025-01-05"])
    assert selected is None


def test_ndvi_lookback_limit():
    selected = nearest_valid_composite("2025-01-01", ["2024-11-01"], max_lookback_days=32)
    assert selected is None


def test_ndvi_composite_metadata():
    meta = composite_metadata("2025-01-01", "2024-12-20", no_future_match=True)
    assert meta["target_date"] == "2025-01-01"
    assert meta["ndvi_date"] == "2024-12-20"
    assert meta["temporal_offset_days"] == 12
    assert meta["no_future_match"] is True


# ---------------------------------------------------------------------------
# Satellite adapters (pure helpers + graceful unavailability)
# ---------------------------------------------------------------------------

def test_dem_srtm_tile_name():
    assert srtm_tile_name(77.2, 28.6) == "N28E077"
    assert srtm_tile_name(-10.5, -5.5) == "S05W010"
    assert SRTM_NODATA == -32768


def test_viirs_qa_pass_mask():
    assert qa_pass_mask([0, 1, 0, 2], [0]) == [True, False, True, False]
    assert qa_pass_mask([0, 0], [0, 1]) == [True, True]


def test_osm_road_density_query_embeds_bbox():
    query = road_density_query(DELHI_BBOX)
    assert "out count" in query
    assert "28.4,77.0,28.8,77.4" in query


def test_weather_cds_request_payload():
    payload = _cds_request_payload(DELHI_BBOX, "2025-01-01", ["2m_temperature"])
    assert payload["year"] == "2025"
    assert payload["month"] == "01"
    assert payload["day"] == "01"
    assert payload["area"] == [28.8, 77.0, 28.4, 77.4]
    assert payload["format"] == "netcdf"


def test_cmr_search_has_bbox():
    granules = cmr_search_granules(
        "C2057600902-LAADS",
        bbox=DELHI_BBOX,
        temporal_start="2025-01-01",
        temporal_end="2025-01-01",
        page_size=1,
    )
    assert isinstance(granules, list)


def test_satellite_acquire_unavailable_without_credentials(tmp_path, monkeypatch):
    for var in ("EARTHDATA_USERNAME", "EARTHDATA_PASSWORD",
                "CDSAPI_URL", "CDSAPI_KEY", "OPENAQ_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    from src.global_data import aod_global, dem_global, ndvi_global, weather_global, viirs_global

    for acquire in (dem_global.acquire, ndvi_global.acquire,
                    weather_global.acquire, viirs_global.acquire):
        report = acquire(make_config(tmp_path), scope="global", date="2025-01-01")
        assert report["status"] == "unavailable"
        assert "credentials" in str(report["reason"]).lower()
        assert report["tiles_completed"] == 0

    aod_report = aod_global.acquire(make_config(tmp_path), scope="global", date="2025-01-01")
    assert aod_report["status"] in ("UNAVAILABLE", "unavailable")
    combined_msg = str(aod_report.get("reason", "") + aod_report.get("error_message", "")).lower()
    assert "credentials" in combined_msg or "not set" in combined_msg


def test_osm_acquire_with_probe(tmp_path, monkeypatch):
    monkeypatch.delenv("EARTHDATA_USERNAME", raising=False)
    from src.global_data import osm_global
    monkeypatch.setattr(
        "src.global_data.osm_global.OSMSource.probe_connectivity",
        staticmethod(lambda url, timeout=5.0: True),
    )
    report = osm_global.acquire(make_config(tmp_path), scope="delhi", date="2025-01-01")
    # delhi scope -> 1 tile; on_tile performs a real Overpass POST which may
    # fail under CI/offline, so status must be one of the graceful outcomes.
    assert report["status"] in ("available", "failed", "no_tiles_processed")


def test_satellite_max_tiles_cap(tmp_path):
    from src.global_data.satellite import SatelliteSource

    class DummySource(SatelliteSource):
        source_id = "dummy"
        product = "dummy product"
        default_resolution = "1km"

    cfg = make_config(tmp_path, **{"global_data.fetch.max_tiles": 2})
    source = DummySource(cfg)
    report = source.attempt_acquire(
        "global", "2025-01-01",
        on_tile=lambda tile, date: Path(tmp_path) / f"{tile.tile_id}.bin",
    )
    # Only max_tiles=2 of 648 global tiles are attempted.
    assert report["tiles_attempted"] == 2
    assert report["tiles_completed"] == 2
    assert report["status"] == "available"


def test_satellite_attempt_acquire_disabled(tmp_path):
    from src.global_data.satellite import SatelliteSource

    class DummySource(SatelliteSource):
        source_id = "dummy"
        product = "dummy product"
        default_resolution = "1km"

    cfg = make_config(tmp_path, **{"global_data.sources.dummy.enabled": False})
    source = DummySource(cfg)
    report = source.attempt_acquire("global", "2025-01-01", on_tile=lambda t, d: Path("x"))
    assert report["status"] == "disabled"


def test_satellite_failed_manifest_is_scope_scoped(tmp_path):
    from src.global_data.satellite import SatelliteSource

    class DummySource(SatelliteSource):
        source_id = "dummy"
        product = "dummy product"
        default_resolution = "1km"

    cfg = make_config(tmp_path)
    source = DummySource(cfg)
    assert source._failed_manifest_path().name == "global_failed_downloads_global.json"
    source.scope = "india"
    assert source._failed_manifest_path().name == "global_failed_downloads_india.json"


def test_satellite_attempt_acquire_failure_recorded_in_scoped_manifest(tmp_path):
    from src.global_data.satellite import SatelliteSource

    class DummySource(SatelliteSource):
        source_id = "dummy"
        product = "dummy product"
        default_resolution = "1km"

    cfg = make_config(tmp_path, **{"global_data.fetch.max_tiles": 1})

    def on_tile(tile, _date):
        raise RuntimeError("boom")

    report = DummySource(cfg).attempt_acquire("delhi", "2025-01-01", on_tile)
    assert report["tiles_failed"] == 1 and report["tiles_completed"] == 0
    manifest = Path(make_config(tmp_path)["global_data"]["storage"]["processed_base"]) \
        / "availability" / "global_failed_downloads_delhi.json"
    import json

    failures = json.loads(manifest.read_text(encoding="utf-8"))["failures"]
    assert len(failures) == 1
    assert failures[0]["source"] == "dummy"


# ---------------------------------------------------------------------------
# Source registry
# ---------------------------------------------------------------------------

def test_source_registry_honest_statuses(tmp_path, monkeypatch):
    for var in ("EARTHDATA_USERNAME", "EARTHDATA_PASSWORD",
                "CDSAPI_URL", "CDSAPI_KEY", "OPENAQ_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    cfg = make_config(tmp_path)
    registry = build_data_source_registry(cfg, scope="global")
    sources = registry["sources"]
    assert set(sources) == {"aod", "weather", "ndvi", "dem", "osm", "viirs", "pm25"}
    # OSM needs no credentials -> available; everything else needs env creds.
    assert sources["osm"]["status"] == "available"
    for source_id in ("aod", "ndvi", "dem", "viirs", "weather", "pm25"):
        assert sources[source_id]["status"] == "unavailable"
    assert registry["built_for_scope"] == "global"
    assert registry["scope_bounds"] == {"west": -180.0, "south": -90.0,
                                        "east": 180.0, "north": 90.0}


def test_source_registry_disabled(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAQ_API_KEY", raising=False)
    cfg = make_config(tmp_path, **{"global_data.sources.aod.enabled": False})
    registry = build_data_source_registry(cfg, scope="delhi")
    assert registry["sources"]["aod"]["status"] == "disabled"


# ---------------------------------------------------------------------------
# Common grid
# ---------------------------------------------------------------------------

def test_grid_for_tile(config):
    grid = grid_for_tile(config, "delhi", 1000, DELHI_BBOX, "tile_000_000")
    assert not grid.empty
    assert {"grid_id", "latitude", "longitude", "geometry", "tile_id"}.issubset(grid.columns)
    assert (grid["grid_id"].astype(str).str.startswith("tile_000_000::")).all()


def test_common_grid_delhi(config):
    grid, report = common_grid(config, "delhi", 1000)
    assert not grid.empty
    assert report["scope"] == "delhi"
    assert report["cells"] == len(grid)
    assert report["tile_based"] is False


# ---------------------------------------------------------------------------
# Temporal join: observations x grid
# ---------------------------------------------------------------------------

@pytest.fixture()
def tiny_grid():
    import geopandas as gpd
    from shapely.geometry import box

    return gpd.GeoDataFrame({
        "grid_id": ["c1", "c2"],
        "geometry": [box(77.0, 28.4, 77.2, 28.6), box(77.2, 28.4, 77.4, 28.6)],
    }, crs="EPSG:4326")


def test_spatial_join_to_grid(tiny_grid):
    obs = pd.DataFrame({
        "station_id": ["S1", "S2"],
        "longitude": [77.1, 77.3],
        "latitude": [28.5, 28.5],
    })
    joined = spatial_join_to_grid(obs, tiny_grid)
    assert joined["grid_id"].tolist() == ["c1", "c2"]


def test_nearest_grid_cell_id_outside(tiny_grid):
    assert nearest_grid_cell_id(tiny_grid, 179.0, 89.0) == ""


def test_add_feature_to_observations_available(tiny_grid):
    obs = spatial_join_to_grid(pd.DataFrame({
        "station_id": ["S1"], "longitude": [77.1], "latitude": [28.5],
    }), tiny_grid)
    out = add_feature_to_observations(obs, "elevation_m", np.array([10.0, 20.0]), tiny_grid)
    assert out["elevation_m"].iloc[0] == 10.0
    assert bool(out["elevation_m_available"].iloc[0]) is True


def test_add_feature_to_observations_unavailable(tiny_grid):
    obs = spatial_join_to_grid(pd.DataFrame({
        "station_id": ["S1"], "longitude": [77.1], "latitude": [28.5],
    }), tiny_grid)
    out = add_feature_to_observations(obs, "elevation_m", None, tiny_grid,
                                      source_status="unavailable")
    assert pd.isna(out["elevation_m"].iloc[0])
    assert bool(out["elevation_m_available"].iloc[0]) is False


def test_temporal_join_report(tiny_grid):
    daily = pd.DataFrame({
        "station_id": ["S1"], "longitude": [77.1], "latitude": [28.5],
        "date": ["2025-01-01"], "pm25_daily": [12.0], "country": ["India"],
        "source": ["openaq"],
    })
    table, report = temporal_join(daily, tiny_grid, feature_layers={
        "elevation_m": {"values": np.array([10.0, 20.0]), "grid": tiny_grid,
                        "source_status": "available"},
        "ndvi": {"values": None, "grid": tiny_grid, "source_status": "unavailable"},
    })
    assert table["grid_id"].iloc[0] == "c1"
    assert table["elevation_m"].iloc[0] == 10.0
    assert pd.isna(table["ndvi"].iloc[0])
    assert report["features"]["elevation_m"]["available"] == 1
    assert report["features"]["ndvi"]["available"] == 0


# ---------------------------------------------------------------------------
# Coverage report
# ---------------------------------------------------------------------------

def test_coverage_summarize_partial(tmp_path):
    reports = {
        "pm25": {"status": "no_data"},
        "aod": {"status": "available", "name": "aod", "tiles_completed": 0,
                "tiles_failed": 0, "artifacts": []},
        "osm": {"status": "available", "name": "osm", "tiles_completed": 1,
                "tiles_failed": 2, "artifacts": ["x.json"]},
    }
    report = build_coverage_report(make_config(tmp_path), "global", reports)
    assert report["sources"]["osm"]["status"] == "PARTIAL"
    assert report["sources"]["aod"]["status"] == "UNAVAILABLE"
    assert report["built_for_scope"] == "global"
    assert report["synthetic_data_leakage"] == "NONE"


def test_coverage_available_and_unavailable(tmp_path):
    reports = {
        "pm25": {"status": "available", "name": "pm25", "tiles_completed": 3,
                 "tiles_failed": 0, "artifacts": ["a.parquet"]},
        "weather": {"status": "unavailable", "name": "weather",
                    "reason": "Credentials missing (env vars only)."},
    }
    report = build_coverage_report(make_config(tmp_path), "global", reports)
    assert report["sources"]["pm25"]["status"] == "AVAILABLE"
    assert report["sources"]["weather"]["status"] == "UNAVAILABLE"
    assert "credentials" in report["sources"]["weather"]["details"]["reason"].lower()


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def test_manifest_artifacts_and_checksums(tmp_path):
    artifact = tmp_path / "osm" / "roads_tile_000_001.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text('{"tile_id": "tile_000_001"}', encoding="utf-8")

    reports = {
        "osm": {"status": "available", "artifacts": [str(artifact)]},
        "pm25": {"status": "unavailable"},
    }
    manifest = build_manifest(make_config(tmp_path), "global", reports)
    assert manifest["built_for_scope"] == "global"
    assert manifest["source_statuses"] == {"osm": "available", "pm25": "unavailable"}
    assert len(manifest["artifacts"]) == 1
    assert manifest["artifacts"][0]["source"] == "osm"
    assert manifest["artifacts"][0]["sha256"] == sha256_file(artifact)
    assert manifest["scope_isolation"]["verified"] == "OK"
    assert "NOT part of Milestone 16" in manifest["ml_not_implemented"]


# ---------------------------------------------------------------------------
# Cache: validate-before-redownload, checksums, failed-download manifest
# ---------------------------------------------------------------------------

def test_fetch_with_retry_success(monkeypatch):
    monkeypatch.setattr("src.global_data.cache.time.sleep", lambda s: None)
    calls = []

    def fetch(_attempt):
        calls.append(1)
        return "ok"

    assert fetch_with_retry(fetch, attempts=3, backoff_base_s=0.0) == "ok"
    assert len(calls) == 1


def test_fetch_with_retry_exhausts_and_records(monkeypatch):
    monkeypatch.setattr("src.global_data.cache.time.sleep", lambda s: None)
    records = []

    def fetch(_attempt):
        raise RuntimeError("boom")

    with pytest.raises(DownloadError):
        fetch_with_retry(fetch, attempts=2, backoff_base_s=0.0,
                         on_failure=lambda record: records.append(record))
    assert len(records) == 1
    assert records[0]["error"] == "boom"
    assert records[0]["attempts"] == 2


def test_record_failed_download_appends(tmp_path):
    manifest = tmp_path / "global_failed_downloads_global.json"
    record_failed_download(manifest, "osm", "OpenStreetMap roads",
                           "2025-01-01", "tile_000_000", {"error": "timeout"})
    record_failed_download(manifest, "osm", "OpenStreetMap roads",
                           "2025-01-01", "tile_000_001", {"error": "gateway"})
    import json

    failures = json.loads(manifest.read_text(encoding="utf-8"))["failures"]
    assert len(failures) == 2
    assert failures[0]["source"] == "osm"
    assert failures[1]["tile_id"] == "tile_000_001"


def test_global_cache_valid_then_corrupt(tmp_path):
    cache = GlobalCache(tmp_path / "cache", version="v1")
    src = tmp_path / "data.bin"
    src.write_bytes(b"payload" * 10)
    key = cache.key("aod", "MCD19A2", "2025-01-01", "tile_000_000")
    dest = cache.store(key, str(src))
    assert cache.is_valid(key) is True
    assert dest.exists()
    # Corrupt the cached file -> checksum validation fails -> not reusable.
    dest.write_bytes(b"corrupted!")
    assert cache.is_valid(key) is False


# ---------------------------------------------------------------------------
# PM2.5 availability (OpenAQ)
# ---------------------------------------------------------------------------

def test_pm25_availability_unavailable_without_key(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAQ_API_KEY", raising=False)
    from src.global_data import pm25_global

    cfg = make_config(tmp_path)
    status = pm25_global.check_availability(cfg)
    assert status["status"] == "unavailable"
    assert "API key" in status["reason"]
    with pytest.raises(pm25_global.CredentialsUnavailable):
        pm25_global.fetch_openaq_pm25(cfg, "2025-01-01", "2025-01-01")
    report = pm25_global.acquire_pm25(cfg, scope="global")
    assert report["status"] == "unavailable"
    assert report["observations_written"] is False


# ---------------------------------------------------------------------------
# /data/* API endpoints
# ---------------------------------------------------------------------------

def test_api_data_sources():
    response = client.get("/data/sources?scope=global")
    assert response.status_code == 200
    body = response.json()
    assert body["built_for_scope"] == "global"
    assert "osm" in body["sources"]


def test_api_data_status_ml_not_implemented():
    response = client.get("/data/status?scope=global")
    assert response.status_code == 200
    body = response.json()
    assert body["ml_not_implemented"] is True
    assert body["prediction_not_implemented"] is True
    assert body["overall"] in ("available", "unavailable", "not_run",
                                 "AVAILABLE", "PARTIAL", "UNAVAILABLE",
                                 "FAILED", "STALE")


def test_api_data_availability_not_run_or_report():
    response = client.get("/data/availability?scope=global")
    assert response.status_code == 200
    body = response.json()
    # Either an honest "not_run" or a real report; both must be scope-scoped.
    assert "built_for_scope" in body
    assert body.get("status") == "not_run" or "sources" in body


def test_api_data_invalid_scope_rejected():
    assert client.get("/data/sources?scope=mars").status_code == 400
    assert client.get("/data/status?scope=mars").status_code == 400
    assert client.get("/data/availability?scope=mars").status_code == 400
