"""Tests for Stage 1-11 enhancements.

Covers:
  - Stage 1: Production-grade availability registry (availability.py)
  - Stage 3: Enhanced readiness gate (readiness.py)
  - Stage 6: Uncertainty framework (uncertainty.py)
  - Stage 7: Advanced hotspot analysis (hotspot_analysis.py)
  - Stage 11: Data freshness and model health (freshness.py)
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config  # noqa: E402


# ---------------------------------------------------------------------------
# Stage 1: Availability Registry
# ---------------------------------------------------------------------------

class TestAvailabilityRegistry:
    """Tests for the production-grade data availability registry."""

    def test_import(self):
        from src.global_data.availability import (
            STATUS_AVAILABLE,
            STATUS_FAILED,
            STATUS_PARTIAL,
            STATUS_STALE,
            STATUS_UNAVAILABLE,
            build_availability_registry,
        )
        assert STATUS_AVAILABLE == "AVAILABLE"
        assert STATUS_PARTIAL == "PARTIAL"
        assert STATUS_UNAVAILABLE == "UNAVAILABLE"
        assert STATUS_FAILED == "FAILED"
        assert STATUS_STALE == "STALE"

    def test_registry_structure(self):
        from src.global_data.availability import build_availability_registry
        config = load_config()
        registry = build_availability_registry(config, scope="global")
        assert registry["registry_version"] == 2
        assert "built_for_scope" in registry
        assert "overall_status" in registry
        assert "sources" in registry
        assert "readiness_summary" in registry
        assert registry["overall_status"] in (
            "AVAILABLE", "PARTIAL", "UNAVAILABLE", "FAILED", "STALE"
        )

    def test_registry_sources_have_required_fields(self):
        from src.global_data.availability import build_availability_registry
        config = load_config()
        registry = build_availability_registry(config, scope="global")
        for source_id, entry in registry["sources"].items():
            assert "id" in entry
            assert "status" in entry
            assert "freshness" in entry
            assert "confidence" in entry
            assert entry["status"] in (
                "AVAILABLE", "PARTIAL", "UNAVAILABLE", "FAILED", "STALE"
            )
            assert entry["confidence"]["level"] in ("HIGH", "MEDIUM", "LOW", "NONE")

    def test_readiness_summary_structure(self):
        from src.global_data.availability import build_availability_registry
        config = load_config()
        registry = build_availability_registry(config, scope="global")
        rs = registry["readiness_summary"]
        assert "available_source_count" in rs
        assert "total_source_count" in rs
        assert "pm25_ground_truth_available" in rs
        assert "has_any_real_data" in rs
        assert "all_sources_unavailable" in rs

    def test_osm_no_credentials_needed(self):
        from src.global_data.availability import build_availability_registry
        config = load_config()
        registry = build_availability_registry(config, scope="global")
        osm = registry["sources"].get("osm")
        assert osm is not None
        assert osm["credentials_required"] == []

    def test_registry_json_roundtrip(self, tmp_path):
        from src.global_data.availability import (
            build_availability_registry,
            load_availability_registry,
        )
        config = load_config()
        out_path = tmp_path / "test_registry.json"
        registry = build_availability_registry(
            config, scope="global", write_path=out_path
        )
        loaded = load_availability_registry(out_path)
        assert loaded is not None
        assert loaded["registry_version"] == registry["registry_version"]

    def test_is_source_ready_for_training(self):
        from src.global_data.availability import (
            build_availability_registry,
            is_source_ready_for_training,
        )
        config = load_config()
        registry = build_availability_registry(config, scope="global")
        result = is_source_ready_for_training(registry, "nonexistent")
        assert result is False


# ---------------------------------------------------------------------------
# Stage 3: Enhanced Readiness Gate
# ---------------------------------------------------------------------------

class TestEnhancedReadiness:
    """Tests for the enhanced readiness gate with configurable thresholds."""

    def test_import(self):
        from src.global_training.readiness import (
            DEFAULT_THRESHOLDS,
            SOURCE_FEATURE_REQUIREMENTS,
            build_readiness,
        )
        assert "min_complete_rows" in DEFAULT_THRESHOLDS
        assert "pm25" in SOURCE_FEATURE_REQUIREMENTS

    def test_readiness_no_data(self):
        from src.global_training.readiness import build_readiness
        config = load_config()
        df = pd.DataFrame()
        report = build_readiness(df, config)
        assert report["model_training_ready"] == "NO"
        assert report["readiness_score"] >= 0.0

    def test_readiness_empty_df_has_all_checks(self):
        from src.global_training.readiness import build_readiness
        config = load_config()
        df = pd.DataFrame()
        report = build_readiness(df, config)
        assert "checks" in report
        assert "complete_case_rows" in report["checks"]
        assert "stations" in report["checks"]
        assert "countries" in report["checks"]
        assert "temporal_days" in report["checks"]
        assert "spatial_coverage" in report["checks"]
        assert "temporal_coverage" in report["checks"]
        assert "quality_flags" in report["checks"]
        assert "source_coverage" in report["checks"]
        assert "feature_sources_ready" in report["checks"]

    def test_readiness_with_sufficient_data(self):
        from src.global_training.readiness import build_readiness
        config = load_config()
        np.random.seed(42)
        n = 2000
        df = pd.DataFrame({
            "station_id": [f"station_{i % 30}" for i in range(n)],
            "country": [f"country_{i % 10}" for i in range(n)],
            "latitude": np.random.uniform(28, 30, n),
            "longitude": np.random.uniform(76, 78, n),
            "date": pd.date_range("2025-01-01", periods=35).repeat(n // 35 + 1)[:n],
            "PM2.5": np.random.uniform(10, 200, n),
            "complete_case": [True] * n,
            "AOD": np.random.uniform(0.1, 2.0, n),
            "temperature_c": np.random.uniform(5, 35, n),
            "relative_humidity_pct": np.random.uniform(20, 90, n),
            "wind_speed_mps": np.random.uniform(0, 15, n),
            "wind_direction_deg": np.random.uniform(0, 360, n),
            "NDVI": np.random.uniform(0, 1, n),
            "elevation_m": np.random.uniform(100, 500, n),
            "road_density": np.random.uniform(0, 10, n),
            "night_lights": np.random.uniform(0, 100, n),
        })
        report = build_readiness(df, config)
        assert "readiness_score" in report
        assert report["readiness_score"] >= 0

    def test_readiness_has_thresholds_from_config(self):
        from src.global_training.readiness import build_readiness
        config = load_config()
        df = pd.DataFrame()
        report = build_readiness(df, config)
        assert "thresholds" in report
        assert "min_complete_rows" in report["thresholds"]


# ---------------------------------------------------------------------------
# Stage 6: Uncertainty Framework
# ---------------------------------------------------------------------------

class TestUncertaintyFramework:
    """Tests for the uncertainty quantification framework."""

    def test_import(self):
        from src.global_training.uncertainty import (
            UNCERTAINTY_STATES,
            build_uncertainty_report,
        )
        assert "DEFERRED" in UNCERTAINTY_STATES
        assert "ESTIMATED" in UNCERTAINTY_STATES

    def test_deferred_when_empty(self):
        from src.global_training.uncertainty import build_uncertainty_report
        config = load_config()
        df = pd.DataFrame()
        report = build_uncertainty_report(df, config)
        assert report["status"] == "DEFERRED"
        assert report["uncertainty_score"] is None

    def test_deferred_when_no_predictions(self):
        from src.global_training.uncertainty import build_uncertainty_report
        config = load_config()
        df = pd.DataFrame({
            "PM2.5": np.random.uniform(10, 100, 50),
        })
        report = build_uncertainty_report(df, config)
        assert report["status"] in ("DEFERRED", "PARTIAL")

    def test_report_has_sources(self):
        from src.global_training.uncertainty import build_uncertainty_report
        config = load_config()
        df = pd.DataFrame()
        report = build_uncertainty_report(df, config)
        assert "sources" in report
        assert "cross_validation_residuals" in report["sources"]
        assert "feature_coverage_gaps" in report["sources"]
        assert "spatial_coverage" in report["sources"]

    def test_api_format(self):
        from src.global_training.uncertainty import (
            build_uncertainty_report,
            uncertainty_status_for_api,
        )
        config = load_config()
        df = pd.DataFrame()
        report = build_uncertainty_report(df, config)
        api_response = uncertainty_status_for_api(report)
        assert "status" in api_response
        assert "method" in api_response


# ---------------------------------------------------------------------------
# Stage 7: Advanced Hotspot Analysis
# ---------------------------------------------------------------------------

class TestHotspotAnalysis:
    """Tests for LISA / Getis-Ord hotspot analysis."""

    def test_import(self):
        from src.global_training.hotspot_analysis import (
            MIN_CELLS_FOR_LISA,
            run_hotspot_analysis,
        )
        assert MIN_CELLS_FOR_LISA == 30

    def test_deferred_when_insufficient_cells(self):
        from src.global_training.hotspot_analysis import run_hotspot_analysis
        df = pd.DataFrame({
            "latitude": [28.5, 28.6],
            "longitude": [77.1, 77.2],
            "PM2.5": [100, 150],
        })
        report = run_hotspot_analysis(df, method="lisa")
        assert report["status"] == "DEFERRED"
        assert "insufficient_cells" in report["reason"]

    def test_deferred_when_missing_columns(self):
        from src.global_training.hotspot_analysis import run_hotspot_analysis
        df = pd.DataFrame({"x": [1, 2, 3]})
        report = run_hotspot_analysis(df)
        assert report["status"] == "DEFERRED"
        assert "missing_columns" in report["reason"]

    def test_computed_with_sufficient_data(self):
        from src.global_training.hotspot_analysis import run_hotspot_analysis
        np.random.seed(42)
        n = 50
        lats = np.random.uniform(28.4, 28.8, n)
        lons = np.random.uniform(77.0, 77.4, n)
        values = np.random.uniform(20, 200, n)
        df = pd.DataFrame({
            "latitude": lats,
            "longitude": lons,
            "PM2.5": values,
        })
        report = run_hotspot_analysis(df, method="lisa")
        assert report["status"] == "COMPUTED"
        assert report["n_cells"] == n

    def test_getis_ord_method(self):
        from src.global_training.hotspot_analysis import run_hotspot_analysis
        np.random.seed(42)
        n = 50
        df = pd.DataFrame({
            "latitude": np.random.uniform(28.4, 28.8, n),
            "longitude": np.random.uniform(77.0, 77.4, n),
            "PM2.5": np.random.uniform(20, 200, n),
        })
        report = run_hotspot_analysis(df, method="getis_ord")
        assert report["status"] == "COMPUTED"


# ---------------------------------------------------------------------------
# Stage 11: Data Freshness
# ---------------------------------------------------------------------------

class TestDataFreshness:
    """Tests for data freshness and model health tracking."""

    def test_import(self):
        from src.global_data.freshness import (
            check_data_freshness,
            check_model_health,
        )

    def test_freshness_report_structure(self):
        from src.global_data.freshness import check_data_freshness
        config = load_config()
        report = check_data_freshness(config, scope="global")
        assert "timestamp" in report
        assert "overall_fresh" in report
        assert "stale_sources" in report
        assert "sources" in report

    def test_model_health_report_structure(self):
        from src.global_data.freshness import check_model_health
        config = load_config()
        report = check_model_health(config)
        assert "timestamp" in report
        assert "models" in report
        assert "overall" in report
        assert report["overall"] in ("HEALTHY", "DEGRADED")

    def test_model_health_checks_xgboost(self):
        from src.global_data.freshness import check_model_health
        config = load_config()
        report = check_model_health(config)
        assert "xgboost" in report["models"]
        assert "random_forest" in report["models"]
