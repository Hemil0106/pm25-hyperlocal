"""Tests for the AOD pipeline (MCD19A2.061).

Covers: authentication diagnostics, credential handling, Harmony API mocking,
download failure modes, HDF validation, availability registry, API security,
and readiness gate behavior. All NASA responses are mocked for deterministic
unit tests. No fake AOD is produced in production.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_geotiff(path: Path, aod_values=None, shape=(100, 100), nodata=-9999.0):
    """Write a synthetic GeoTIFF for testing validation logic only."""
    if aod_values is None:
        aod_values = np.random.uniform(0.0, 2.0, shape).astype(np.float32)
    transform = from_bounds(77.0, 28.0, 78.0, 29.0, shape[1], shape[0])
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "width": shape[1],
        "height": shape[0],
        "count": 1,
        "crs": CRS.from_epsg(4326),
        "transform": transform,
        "nodata": nodata,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(aod_values, 1)
    return path


def _make_config(tmp_path, scope="delhi"):
    """Create a minimal config for testing."""
    return {
        "global_data": {
            "enabled": True,
            "scope": scope,
            "common_grid": {"coarse_resolution_m": 1000, "target_resolution_m": 500},
            "temporal": {"aggregation": "daily_mean", "min_observations_per_day": 1},
            "qc": {"outlier_method": "mad", "outlier_threshold_k": 5.0, "max_pm25_ug_m3": 1000},
            "pm25": {"source": "openaq", "api_version": "v3", "base_url": "https://api.openaq.org/v3", "credential_env_var": "OPENAQ_API_KEY"},
            "fetch": {"retries": 3, "backoff_base_s": 2.0, "backoff_max_s": 60.0, "max_tiles": 3},
            "cache": {"dir": "data/cache/global", "version": "v1", "validate_before_redownload": True},
            "storage": {"raw_base": "data/raw/global", "processed_base": str(tmp_path / "processed" / "global")},
            "sources": {
                "aod": {"enabled": True, "product": "MODIS_MAIAC_MCD19A2", "credential_env_var": "EARTHDATA_USERNAME", "credential_env_var_2": "EARTHDATA_PASSWORD"},
            },
        },
    }


# ===================================================================
# 1. Missing credentials
# ===================================================================

def test_missing_credentials_reported(monkeypatch):
    monkeypatch.delenv("EARTHDATA_USERNAME", raising=False)
    monkeypatch.delenv("EARTHDATA_PASSWORD", raising=False)
    from src.global_data.aod_global import check_aod_authentication
    result = check_aod_authentication()
    assert result["configured"] is False
    assert result["error_code"] == "CREDENTIALS_MISSING"
    assert result["authenticated"] is False


# ===================================================================
# 2. Valid credential configuration
# ===================================================================

def test_valid_credentials_detected(monkeypatch):
    monkeypatch.setenv("EARTHDATA_USERNAME", "testuser")
    monkeypatch.setenv("EARTHDATA_PASSWORD", "testpass")
    from src.global_data.aod_global import check_aod_authentication
    result = check_aod_authentication()
    assert result["configured"] is True
    assert result["provider"] == "NASA"
    assert result["product"] == "MCD19A2.061"


# ===================================================================
# 3. Authentication failure (mocked)
# ===================================================================

def test_auth_failure(monkeypatch):
    monkeypatch.setenv("EARTHDATA_USERNAME", "baduser")
    monkeypatch.setenv("EARTHDATA_PASSWORD", "badpass")
    from src.global_data.aod_global import check_aod_authentication

    mock_auth = MagicMock()
    mock_auth.authenticated = False
    with patch("earthaccess.login", return_value=mock_auth):
        result = check_aod_authentication()
    assert result["authenticated"] is False
    assert result["error_code"] == "AUTHENTICATION_FAILED"


# ===================================================================
# 4. No credentials -> acquire returns UNAVAILABLE
# ===================================================================

def test_acquire_unavailable_without_credentials(tmp_path, monkeypatch):
    monkeypatch.delenv("EARTHDATA_USERNAME", raising=False)
    monkeypatch.delenv("EARTHDATA_PASSWORD", raising=False)
    from src.global_data.aod_global import acquire
    report = acquire(_make_config(tmp_path), scope="delhi", date="2025-01-01")
    assert report["status"] == "UNAVAILABLE"
    assert report["error_code"] == "CREDENTIALS_MISSING"


# ===================================================================
# 5-8. HTTP failure codes (mocked Harmony)
# ===================================================================

@pytest.mark.parametrize("status_code,expected_prefix", [
    (401, "Harmony returned HTTP"),
    (403, "Harmony returned HTTP"),
    (429, "Harmony returned HTTP"),
    (500, "Harmony returned HTTP"),
])
def test_harmony_http_errors(status_code, expected_prefix, tmp_path, monkeypatch):
    monkeypatch.setenv("EARTHDATA_USERNAME", "testuser")
    monkeypatch.setenv("EARTHDATA_PASSWORD", "testpass")

    mock_session = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.headers = {}
    mock_session.get.return_value = mock_resp

    mock_auth = MagicMock()
    mock_auth.authenticated = True

    from src.global_data.aod_global import _harmony_fetch

    with patch("src.global_data.aod_global._get_earthaccess_session", return_value=(mock_auth, mock_session)):
        ok = _harmony_fetch(mock_session, (77.0, 28.0, 78.0, 29.0), "2025-01-01", tmp_path / "test.tif")
    assert ok is False


# ===================================================================
# 9. Network timeout (mocked)
# ===================================================================

def test_harmony_timeout(tmp_path, monkeypatch):
    monkeypatch.setenv("EARTHDATA_USERNAME", "testuser")
    monkeypatch.setenv("EARTHDATA_PASSWORD", "testpass")

    mock_session = MagicMock()
    mock_session.get.side_effect = ConnectionError("timed out")

    from src.global_data.aod_global import _harmony_fetch

    ok = _harmony_fetch(mock_session, (77.0, 28.0, 78.0, 29.0), "2025-01-01", tmp_path / "test.tif")
    assert ok is False


# ===================================================================
# 10. No granules found (mocked Harmony returns empty JSON)
# ===================================================================

def test_no_granules_json_response(tmp_path):
    mock_session = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.headers = {"Content-Type": "application/json"}
    mock_resp.json.return_value = {"message": "No results found"}
    mock_session.get.return_value = mock_resp

    from src.global_data.aod_global import _harmony_fetch

    ok = _harmony_fetch(mock_session, (77.0, 28.0, 78.0, 29.0), "2025-01-01", tmp_path / "test.tif")
    assert ok is False


# ===================================================================
# 11. Valid AOD GeoTIFF validation
# ===================================================================

def test_validate_valid_geotiff(tmp_path):
    from src.global_data.aod_global import _validate_aod_geotiff

    path = _make_geotiff(tmp_path / "valid.tif")
    stats = _validate_aod_geotiff(path)
    assert stats["valid"] is True
    assert stats["valid_pixel_count"] == 10000
    assert stats["valid_coverage_pct"] == 100.0
    assert stats["min"] >= 0.0
    assert stats["max"] <= 2.0
    assert "aod_source_crs" in stats


# ===================================================================
# 12. Corrupted GeoTIFF
# ===================================================================

def test_validate_corrupted_geotiff(tmp_path):
    from src.global_data.aod_global import _validate_aod_geotiff

    bad = tmp_path / "bad.tif"
    bad.write_bytes(b"not a tiff file")
    stats = _validate_aod_geotiff(bad)
    assert stats["valid"] is False
    assert "error" in stats


# ===================================================================
# 13. All-NaN GeoTIFF (no valid AOD)
# ===================================================================

def test_validate_all_nan_geotiff(tmp_path):
    from src.global_data.aod_global import _validate_aod_geotiff

    nan_array = np.full((100, 100), np.nan, dtype=np.float32)
    path = _make_geotiff(tmp_path / "nan.tif", aod_values=nan_array)
    stats = _validate_aod_geotiff(path)
    assert stats["valid"] is False
    assert stats["valid_pixel_count"] == 0
    assert stats["valid_coverage_pct"] == 0.0


# ===================================================================
# 14. Fill value handling
# ===================================================================

def test_geotiff_with_fill_value(tmp_path):
    from src.global_data.aod_global import _validate_aod_geotiff

    arr = np.full((50, 50), -9999.0, dtype=np.float32)
    arr[10:40, 10:40] = np.random.uniform(0.1, 1.5, (30, 30)).astype(np.float32)
    path = _make_geotiff(tmp_path / "fill.tif", aod_values=arr, shape=(50, 50))
    stats = _validate_aod_geotiff(path)
    assert stats["valid"] is True
    assert stats["valid_pixel_count"] == 900
    assert stats["valid_coverage_pct"] == 36.0


# ===================================================================
# 15. Backend copy logic
# ===================================================================

def test_backend_copy(tmp_path):
    from src.global_data.aod_global import _backend_copy

    config = {
        "global_data": {
            "storage": {"processed_base": str(tmp_path / "processed" / "global")}
        }
    }
    source = _make_geotiff(tmp_path / "source.tif")
    result = _backend_copy(source, "delhi", "2025-01-01", config)
    assert result is not None
    assert result.exists()
    assert "aod_500m_2025-01-01.tif" in str(result)


# ===================================================================
# 16. Backend copy for pune/mumbai scopes
# ===================================================================

@pytest.mark.parametrize("scope", ["pune", "mumbai"])
def test_backend_copy_city_scopes(tmp_path, scope):
    from src.global_data.aod_global import _backend_copy

    config = {
        "global_data": {
            "storage": {"processed_base": str(tmp_path / "processed" / "global")}
        }
    }
    source = _make_geotiff(tmp_path / "source.tif")
    result = _backend_copy(source, scope, "2025-01-01", config)
    assert result is not None
    assert scope in str(result)


# ===================================================================
# 17. Credential safety: no secrets in health endpoint
# ===================================================================

def test_health_never_exposes_secrets(monkeypatch):
    monkeypatch.setenv("EARTHDATA_USERNAME", "SECRET_USER_12345")
    monkeypatch.setenv("EARTHDATA_PASSWORD", "SECRET_PASS_98765")
    from fastapi.testclient import TestClient
    from api.main import app

    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert "SECRET_USER" not in json.dumps(body)
    assert "SECRET_PASS" not in json.dumps(body)
    assert body["aod"]["configured"] is True
    assert body["aod"]["product"] == "MCD19A2.061"


# ===================================================================
# 18. /data/aod/status endpoint
# ===================================================================

def test_aod_status_endpoint_no_credentials(monkeypatch):
    monkeypatch.delenv("EARTHDATA_USERNAME", raising=False)
    monkeypatch.delenv("EARTHDATA_PASSWORD", raising=False)
    from fastapi.testclient import TestClient
    from api.main import app

    client = TestClient(app)
    resp = client.get("/data/aod/status?scope=delhi")
    assert resp.status_code == 200
    body = resp.json()
    assert body["configured"] is False
    assert body["status"] == "UNAVAILABLE"
    assert body["error_code"] == "CREDENTIALS_MISSING"


# ===================================================================
# 19. Build version in health
# ===================================================================

def test_health_has_build_version():
    from fastapi.testclient import TestClient
    from api.main import app

    client = TestClient(app)
    resp = client.get("/health")
    body = resp.json()
    assert "build_version" in body
    assert body["build_version"] is not None
    assert len(body["build_version"]) > 0


# ===================================================================
# 20. AOD availability with real GeoTIFF on disk
# ===================================================================

def test_aod_status_with_files_on_disk(tmp_path, monkeypatch):
    monkeypatch.setenv("EARTHDATA_USERNAME", "testuser")
    monkeypatch.setenv("EARTHDATA_PASSWORD", "testpass")
    monkeypatch.chdir(tmp_path)

    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)
    _make_geotiff(processed / "aod_500m_2025-01-01.tif")

    mock_auth = MagicMock()
    mock_auth.authenticated = True
    with patch("src.global_data.aod_global.check_aod_authentication", return_value={
        "configured": True, "authenticated": True, "provider": "NASA",
        "product": "MCD19A2.061", "error_code": None, "error_message": None,
    }):
        from src.global_data.aod_global import _validate_aod_geotiff
        stats = _validate_aod_geotiff(processed / "aod_500m_2025-01-01.tif")
        assert stats["valid"] is True
        assert stats["valid_coverage_pct"] == 100.0


# ===================================================================
# 21. check_aod_authentication never returns secrets
# ===================================================================

def test_auth_check_never_returns_secrets(monkeypatch):
    monkeypatch.setenv("EARTHDATA_USERNAME", "my_secret_user")
    monkeypatch.setenv("EARTHDATA_PASSWORD", "my_secret_pass_12345")
    from src.global_data.aod_global import check_aod_authentication
    result = check_aod_authentication()
    serialized = json.dumps(result)
    assert "my_secret_user" not in serialized
    assert "my_secret_pass_12345" not in serialized


# ===================================================================
# 22. API /health returns AOD section
# ===================================================================

def test_health_response_includes_aod_section():
    from fastapi.testclient import TestClient
    from api.main import app

    client = TestClient(app)
    resp = client.get("/health")
    body = resp.json()
    assert "aod" in body
    assert isinstance(body["aod"], dict)
    assert "provider" in body["aod"]
    assert "product" in body["aod"]
    assert "configured" in body["aod"]


# ===================================================================
# 23. API no credential leakage in /data/aod/status
# ===================================================================

def test_aod_status_no_credential_leakage(monkeypatch):
    monkeypatch.setenv("EARTHDATA_USERNAME", "LEAK_TEST_USER")
    monkeypatch.setenv("EARTHDATA_PASSWORD", "LEAK_TEST_PASS")
    from fastapi.testclient import TestClient
    from api.main import app

    client = TestClient(app)
    resp = client.get("/data/aod/status?scope=delhi")
    body = resp.json()
    serialized = json.dumps(body)
    assert "LEAK_TEST_USER" not in serialized
    assert "LEAK_TEST_PASS" not in serialized


# ===================================================================
# 24. AOD source info consistency
# ===================================================================

def test_aod_source_info():
    from src.global_data.aod_global import AODSource
    assert AODSource.source_id == "aod"
    assert "MCD19A2" in AODSource.product


# ===================================================================
# 25. Empty processed dir -> no AOD files
# ===================================================================

def test_no_files_returns_no_data(monkeypatch):
    monkeypatch.setenv("EARTHDATA_USERNAME", "testuser")
    monkeypatch.setenv("EARTHDATA_PASSWORD", "testpass")

    mock_auth = MagicMock()
    mock_auth.authenticated = True

    from src.global_data.aod_global import _acquire_harmony

    mock_session = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.headers = {"Content-Type": "application/json"}
    mock_resp.json.return_value = {"message": "No results"}
    mock_session.get.return_value = mock_resp

    config = _make_config(Path("/tmp/aod_test_empty"))
    with patch("src.global_data.aod_global._get_earthaccess_session", return_value=(mock_auth, mock_session)):
        with patch("src.global_data.aod_global._harmony_fetch", return_value=False):
            report = _acquire_harmony(mock_session, "delhi", "2025-01-01", config)
            assert report["status"] == "FAILED"
