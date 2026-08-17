"""Global training dataset tests (Milestone 17).

Covers the M17 data-engineering layer: schema, feature engineering, OSM road
density join, builder, complete-case split, QC, coverage, correlation, target,
representativeness, validation groups (LOSO/LOCO), leakage checks, readiness
gate, and metadata.

Core M17 rules verified here:
  - real data only; unavailable features are NaN with available flags
  - PM2.5 (target) is never a predictor feature
  - complete-case rows are preserved alongside an incomplete diagnostic file
  - LOSO/LOCO report INSUFFICIENT_DATA (never manufactured folds)
  - leakage checks catch target leakage, duplicates, synthetic markers, and
    Delhi->global contamination
  - readiness is data-derived and never auto-YES
  - no global ML training / prediction happens in M17
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config  # noqa: E402

from src.global_training.builder import (  # noqa: E402
    build_training_table,
    empty_training_frame,
    split_complete_cases,
)
from src.global_training.coverage import (  # noqa: E402
    coverage_by_country,
    geographic_bias_flags,
)
from src.global_training.correlation import correlation_frame  # noqa: E402
from src.global_training.features import (  # noqa: E402
    add_derived_features,
    add_temporal_features,
    add_wind_components,
)
from src.global_training.groups import (  # noqa: E402
    loco_status,
    loso_status,
    region_of_country,
)
from src.global_training.leakage import (  # noqa: E402
    check_delhi_artifact_contamination,
    check_duplicate_station_date,
    check_synthetic_contamination,
    check_target_leakage,
    run_leakage_checks,
)
from src.global_training.osm_density import road_density_for  # noqa: E402
from src.global_training.qc import (  # noqa: E402
    check_duplicates,
    check_schema,
    missingness,
)
from src.global_training.readiness import build_readiness  # noqa: E402
from src.global_training.schema import (  # noqa: E402
    AVAILABILITY_COLS,
    FEATURE_COLS,
    TARGET_COL,
    TRAINING_COLUMNS,
)
from src.global_training.target import target_report  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_config(tmp_path: Path) -> dict:
    config = load_config()
    config.setdefault("global_data", {})
    config["global_data"].setdefault("storage", {})
    config["global_data"]["storage"]["processed_base"] = str(tmp_path / "processed")
    config["global_data"]["storage"]["raw_base"] = str(tmp_path / "raw")
    config["global_data"]["scope"] = "global"
    config.setdefault("paths", {})["outputs"] = str(tmp_path / "outputs")
    return config


def synthetic_daily(n_stations: int = 3, start_lat: float = 10.0,
                    start_lon: float = 10.0, n_days: int = 2) -> pd.DataFrame:
    rows = []
    for s in range(n_stations):
        for d in range(n_days):
            rows.append({
                "station_id": f"ST_{s:02d}",
                "date": pd.Timestamp(f"2025-01-{d + 1:02d}"),
                "pm25_daily": 40.0 + s * 10 + d,
                "observation_count": 4,
                "latitude": start_lat + s * 0.5,
                "longitude": start_lon + s * 0.5,
                "country": ["India", "Nepal", "United States"][s],
                "source": "openaq",
            })
    return pd.DataFrame(rows)


def complete_case_frame(n_rows: int = 1000, n_stations: int = 20,
                        n_countries: int = 5, n_days: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    stations = [f"ST_{i:03d}" for i in range(n_stations)]
    countries = [f"Country_{c}" for c in range(n_countries)]
    dates = pd.date_range("2025-01-01", periods=n_days, freq="D")
    rows = []
    for _ in range(n_rows):
        rows.append({
            "station_id": stations[rng.integers(0, n_stations)],
            "country": countries[rng.integers(0, n_countries)],
            "date": dates[rng.integers(0, n_days)],
            TARGET_COL: 50.0 + rng.normal(0, 10),
            **{c: float(rng.normal(0, 1)) for c in FEATURE_COLS},
            "complete_case": True,
        })
    df = pd.DataFrame(rows)
    df["station_id"] = df["station_id"].astype(str)
    return df


def empty_schema_frame() -> pd.DataFrame:
    return empty_training_frame()


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class TestSchema:
    def test_target_is_never_a_feature(self):
        assert TARGET_COL not in FEATURE_COLS
        assert "PM2.5_available" not in AVAILABILITY_COLS

    def test_availability_flags_cover_every_source_feature(self):
        source_feature_cols = [c for c in FEATURE_COLS
                               if c not in ("month", "day_of_year",
                                            "sin_day_of_year", "cos_day_of_year")]
        for col in source_feature_cols:
            assert f"{col}_available" in AVAILABILITY_COLS

    def test_training_columns_are_complete_and_unique(self):
        assert len(TRAINING_COLUMNS) == len(set(TRAINING_COLUMNS))
        assert TARGET_COL in TRAINING_COLUMNS


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

class TestFeatures:
    def test_temporal_features_from_own_date(self):
        df = pd.DataFrame({
            "date": pd.to_datetime(["2025-01-01", "2025-07-15", None]),
        })
        out = add_temporal_features(df)
        assert out.loc[0, "month"] == 1
        assert out.loc[0, "day_of_year"] == 1
        assert np.isclose(out.loc[0, "sin_day_of_year"], np.sin(2 * np.pi / 365.25))
        assert np.isclose(out.loc[0, "cos_day_of_year"], np.cos(2 * np.pi / 365.25))
        assert np.isnan(out.loc[2, "month"])
        assert np.isnan(out.loc[2, "day_of_year"])

    def test_wind_components_require_real_speed_and_direction(self):
        df = pd.DataFrame({
            "wind_speed_mps": [10.0, np.nan, 5.0],
            "wind_direction_deg": [90.0, 180.0, np.nan],
        })
        out = add_wind_components(df)
        # u = -V sin(theta), v = -V cos(theta); theta=90deg => u=-10, v=0
        assert np.isclose(out.loc[0, "wind_u_mps"], -10.0)
        assert np.isclose(out.loc[0, "wind_v_mps"], 0.0, atol=1e-9)
        assert np.isnan(out.loc[1, "wind_u_mps"])
        assert np.isnan(out.loc[2, "wind_u_mps"])

    def test_derived_never_from_target(self):
        df = pd.DataFrame({"date": pd.to_datetime(["2025-01-01"])})
        out = add_derived_features(df)
        assert TARGET_COL not in out.columns


# ---------------------------------------------------------------------------
# OSM road density
# ---------------------------------------------------------------------------

class TestOsmDensity:
    def test_road_density_for_known_tile(self):
        daily = synthetic_daily(n_stations=1, start_lat=10.0, start_lon=10.0, n_days=1)
        segments = {"tile_019_010": 42}
        out = road_density_for(daily, segments, "global")
        assert out["road_density"].iloc[0] == 42
        assert bool(out["road_density_available"].iloc[0]) is True

    def test_road_density_nan_for_missing_tile(self):
        daily = synthetic_daily(n_stations=1, start_lat=10.0, start_lon=10.0, n_days=1)
        out = road_density_for(daily, {}, "global")
        assert np.isnan(out["road_density"].iloc[0])
        assert bool(out["road_density_available"].iloc[0]) is False

    def test_road_density_india_scope_tile(self):
        daily = pd.DataFrame({
            "station_id": ["ST_X"],
            "latitude": [28.6],
            "longitude": [77.2],
        })
        out = road_density_for(daily, {"tile_001_002": 7}, "india")
        assert out["road_density"].iloc[0] == 7

    def test_empty_input(self):
        daily = pd.DataFrame(columns=["station_id", "latitude", "longitude"])
        out = road_density_for(daily, {"tile_019_010": 1}, "global")
        assert out.empty


# ---------------------------------------------------------------------------
# Builder + complete-case split
# ---------------------------------------------------------------------------

class TestBuilder:
    def test_builds_schema_correct_table(self):
        daily = synthetic_daily(n_stations=3, n_days=2)
        segments = {"tile_019_010": 42, "tile_019_011": 8, "tile_019_012": 3}
        table, report = build_training_table(daily, segments, "global")
        assert list(table.columns) == TRAINING_COLUMNS
        assert report["rows"] == 6
        assert report["stations"] == 3
        assert report["countries"] == 3
        assert report["osm_tile_artifacts"] == 3

    def test_gridded_features_are_nan_without_artifacts(self):
        daily = synthetic_daily(n_stations=1, n_days=1)
        table, _ = build_training_table(daily, {}, "global")
        row = table.iloc[0]
        for col in ("AOD", "temperature_c", "relative_humidity_pct",
                    "NDVI", "elevation_m", "night_lights"):
            assert np.isnan(row[col])
            assert bool(row[f"{col}_available"]) is False

    def test_road_density_is_real_when_tile_exists(self):
        daily = synthetic_daily(n_stations=1, n_days=1)
        table, _ = build_training_table(daily, {"tile_019_010": 42}, "global")
        row = table.iloc[0]
        assert row["road_density"] == 42
        assert bool(row["road_density_available"]) is True

    def test_target_is_real_daily_pm25(self):
        daily = synthetic_daily(n_stations=1, n_days=1)
        table, _ = build_training_table(daily, {}, "global")
        assert table.iloc[0][TARGET_COL] == pytest.approx(40.0)

    def test_empty_daily_produces_schema_correct_empty_table(self):
        table, report = build_training_table(None, {}, "global")
        assert table.empty
        assert list(table.columns) == TRAINING_COLUMNS
        assert report["status"] == "empty"

    def test_complete_case_split_preserves_both_files(self):
        daily = synthetic_daily(n_stations=3, n_days=2)
        table, _ = build_training_table(daily, {}, "global")
        complete, incomplete = split_complete_cases(table)
        assert complete.empty  # no gridded features -> no complete cases
        assert len(incomplete) == len(table)
        assert list(complete.columns) == TRAINING_COLUMNS


# ---------------------------------------------------------------------------
# QC
# ---------------------------------------------------------------------------

class TestQc:
    def test_schema_check_on_built_table(self):
        daily = synthetic_daily(n_stations=2, n_days=2)
        table, _ = build_training_table(daily, {}, "global")
        assert check_schema(table)["ok"] is True

    def test_schema_check_on_empty_table(self):
        assert check_schema(empty_schema_frame())["ok"] is True

    def test_duplicate_detection(self):
        df = pd.DataFrame({"station_id": ["A", "A"], "date": ["2025-01-01", "2025-01-01"]})
        assert check_duplicates(df)["duplicate_station_dates"] == 1

    def test_missingness_counts(self):
        df = pd.DataFrame({TARGET_COL: [1.0, np.nan], "AOD": [np.nan, np.nan]})
        missing = missingness(df)
        target = missing[missing["column"] == TARGET_COL].iloc[0]
        assert target["n_null"] == 1


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

class TestCoverage:
    def test_coverage_by_country(self):
        daily = synthetic_daily(n_stations=3, n_days=2)
        table, _ = build_training_table(daily, {}, "global")
        by_country = coverage_by_country(table)
        assert set(by_country["country"]).issuperset({"India", "Nepal", "United States"})
        assert by_country["n_rows"].sum() == 6

    def test_geographic_bias_flags_surface_variation(self):
        df = pd.DataFrame({
            "country": ["A", "A", "B", "B"],
            "road_density": [1.0, 2.0, np.nan, np.nan],
            "complete_case": [True, True, False, False],
            "station_id": ["S1", "S2", "S3", "S4"],
        })
        flags = geographic_bias_flags(df)
        road = [f for f in flags if f["feature"] == "road_density"]
        assert road and road[0]["missingness_varies_by_country"] is True


# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------

class TestCorrelation:
    def test_correlation_on_complete_rows(self):
        df = complete_case_frame(n_rows=50)
        pairs, report = correlation_frame(df)
        assert report["status"] == "ok"
        assert not pairs.empty
        assert set(pairs.columns) == {"feature_a", "feature_b", "pearson",
                                      "spearman", "n"}

    def test_correlation_no_data_is_honest(self):
        pairs, report = correlation_frame(empty_schema_frame())
        assert pairs.empty
        assert report["status"] == "no_data"


# ---------------------------------------------------------------------------
# Target
# ---------------------------------------------------------------------------

class TestTarget:
    def test_target_report_on_empty_is_insufficient_without_thresholds(self, tmp_path):
        config = make_config(tmp_path)
        report = target_report(empty_schema_frame(), config)
        assert report["completeness_flag"] == "insufficient"
        assert report["distribution"] is None
        assert "threshold" not in report

    def test_target_report_distribution(self, tmp_path):
        config = make_config(tmp_path)
        df = complete_case_frame(n_rows=100)
        report = target_report(df, config)
        assert report["distribution"]["count"] == 100
        assert report["distribution"]["mean"] > 0
        assert report["date_range"]["n_days"] > 0


# ---------------------------------------------------------------------------
# Groups (spatial validation)
# ---------------------------------------------------------------------------

class TestGroups:
    def test_region_mapping_and_unclassified(self):
        assert region_of_country("India") == "South Asia"
        assert region_of_country("") == "Unclassified"

    def test_loso_insufficient_when_few_stations(self):
        df = complete_case_frame(n_rows=5, n_stations=2)
        status = loso_status(df)
        assert status["status"] == "INSUFFICIENT_DATA"
        assert "not defensible" in status["reason"]

    def test_loso_ready_with_sufficient_data(self):
        df = complete_case_frame(n_rows=100, n_stations=20)
        assert loso_status(df)["status"] == "READY"

    def test_loco_insufficient_with_single_country(self):
        df = complete_case_frame(n_rows=100, n_stations=10, n_countries=1)
        assert loco_status(df)["status"] == "INSUFFICIENT_DATA"


# ---------------------------------------------------------------------------
# Leakage
# ---------------------------------------------------------------------------

class TestLeakage:
    def test_target_leakage_check_always_passes_for_fixed_feature_set(self):
        df = pd.DataFrame({TARGET_COL: [1.0]})
        result = check_target_leakage(df)
        assert result["status"] == "PASS"
        assert "PM2.5" in result["detail"]

    def test_duplicate_station_date_check(self):
        df = pd.DataFrame({"station_id": ["A", "A"], "date": ["2025-01-01", "2025-01-01"]})
        assert check_duplicate_station_date(df)["status"] == "FAIL"

    def test_synthetic_contamination_detects_marker(self, tmp_path):
        config = make_config(tmp_path)
        df = pd.DataFrame({"station_id": ["SYNTHETIC_TEST_DATA"], "country": ["X"], "city": [None]})
        result = check_synthetic_contamination(df, config)
        assert result["status"] == "PRESENT"

    def test_delhi_contamination_detected_for_global_scope(self):
        df = pd.DataFrame({
            "station_id": ["D1", "G1"],
            "latitude": [28.6, 10.0],
            "longitude": [77.2, 20.0],
        })
        result = check_delhi_artifact_contamination(df, "global")
        assert result["status"] == "FAIL"

    def test_delhi_check_skipped_for_delhi_scope(self):
        df = pd.DataFrame({"station_id": ["D1"], "latitude": [28.6], "longitude": [77.2]})
        assert check_delhi_artifact_contamination(df, "delhi")["status"] == "N/A"

    def test_full_leakage_suite_on_clean_table(self, tmp_path):
        config = make_config(tmp_path)
        daily = synthetic_daily(n_stations=3, n_days=2)
        table, _ = build_training_table(daily, {}, "global")
        result = run_leakage_checks(table, config)
        assert result["overall"] == "PASS"


# ---------------------------------------------------------------------------
# Readiness
# ---------------------------------------------------------------------------

class TestReadiness:
    def test_empty_is_never_ready(self, tmp_path):
        config = make_config(tmp_path)
        report = build_readiness(empty_schema_frame(), config)
        assert report["model_training_ready"] == "NO"
        assert report["reason"] is not None

    def test_readiness_yes_only_with_real_coverage(self, tmp_path):
        config = make_config(tmp_path)
        df = complete_case_frame(n_rows=2000, n_stations=40, n_countries=8, n_days=60)
        report = build_readiness(df, config)
        assert report["model_training_ready"] == "YES"

    def test_partial_coverage_is_not_ready(self, tmp_path):
        config = make_config(tmp_path)
        df = complete_case_frame(n_rows=2000, n_stations=40, n_countries=8, n_days=60)
        df["AOD"] = np.nan  # simulate missing AOD acquisition
        report = build_readiness(df, config)
        assert report["model_training_ready"] == "NO"
        assert any("aod" in reason for reason in [report["reason"]])


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------

class TestPipeline:
    def test_full_pipeline_from_real_daily_input(self, tmp_path):
        from src.global_training import run_global_training_pipeline

        config = make_config(tmp_path)
        daily_dir = Path(config["global_data"]["storage"]["processed_base"]) / "pm25"
        daily_dir.mkdir(parents=True, exist_ok=True)
        synthetic_daily(n_stations=3, n_days=2).to_parquet(
            daily_dir / "global_pm25_daily.parquet", index=False)

        result = run_global_training_pipeline(config, scope="global", write=True)

        assert result["stage"] == "GLOBAL_TRAINING_DATASET"
        assert result["ml_not_implemented"] is True
        assert result["prediction_not_implemented"] is True
        assert result["builder"]["rows"] == 6
        assert result["builder"]["stations"] == 3
        assert result["readiness"]["model_training_ready"] == "NO"
        assert result["leakage"]["overall"] == "PASS"
        assert result["validation"]["loso"]["status"] == "INSUFFICIENT_DATA"

        for key, path in result["report"]["files"]["training_outputs"].items():
            assert Path(path).exists()
        for _, path in result["report"]["files"]["reports"].items():
            assert Path(path).exists()
        assert Path(result["report"]["metadata_path"]).exists()

    def test_pipeline_with_missing_daily_still_completes_honestly(self, tmp_path):
        from src.global_training import run_global_training_pipeline

        config = make_config(tmp_path)
        result = run_global_training_pipeline(config, scope="global", write=True)
        assert result["builder"]["rows"] == 0
        assert result["builder"]["status"] == "empty"
        assert result["readiness"]["model_training_ready"] == "NO"
        assert result["report"]["dataset_summary"]["rows"] == 0
