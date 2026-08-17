"""Milestone 17 orchestrator.

Pipeline: dirs -> daily -> build table -> QC -> feature engineering ->
coverage -> correlation -> target -> representativeness -> validation groups ->
leakage -> readiness -> metadata -> write outputs + visuals -> final report.

Rules honoured here:
  - real data only; unavailable features are NaN with available flags
  - no ML training and no prediction in M17 (never)
  - if a leakage check FAILs the pipeline stops before writing any output
  - with no real data the pipeline still produces honest, schema-correct
    outputs and a NOT READY / BLOCKED result - a valid successful milestone
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from src.global_data.scope import validate_scope

from .builder import (
    build_training_table,
    daily_path,
    load_daily,
    split_complete_cases,
    write_training_outputs,
)
from .coverage import build_coverage_report
from .correlation import write_correlation
from .groups import write_validation_outputs
from .leakage import run_leakage_checks
from .metadata import build_metadata
from .osm_density import load_osm_road_segments, osm_tiles_present
from .qc import run_qc
from .readiness import build_readiness
from .report import build_final_report
from .representativeness import representativeness, write_visualizations
from .target import target_report

logger = logging.getLogger(__name__)


def run_global_training_pipeline(config, scope: str = "global",
                                 write: bool = True) -> dict:
    """Execute the M17 global training-data stage for a scope."""
    scope = validate_scope(scope)
    logger.info("=" * 70)
    logger.info("GLOBAL TRAINING DATASET (Milestone 17) - scope=%s", scope)
    logger.info("=" * 70)

    # -- Inputs -----------------------------------------------------------
    daily, daily_status = load_daily(config, scope)
    road_segments = load_osm_road_segments(config, scope)
    osm_report = osm_tiles_present(config, scope)
    logger.info("Daily observations: status=%s  OSM tile artifacts=%d",
                daily_status, osm_report["n_tile_artifacts"])

    # -- Build + split -----------------------------------------------------
    table, builder_report = build_training_table(daily, road_segments, scope)
    complete, incomplete = split_complete_cases(table)

    # -- QC (schema + duplicates + missingness) -----------------------------
    qc_report = run_qc(table)

    # -- Leakage gate: any FAIL stops the pipeline before writing outputs ----
    leakage = run_leakage_checks(table, config)
    for check in leakage["checks"]:
        logger.info("  leakage %-32s %s", check["check"], check["status"])
    if leakage["overall"] == "FAIL":
        raise RuntimeError(
            "Milestone 17 stopped: leakage check FAILED - fix the cause before "
            "writing training outputs. Details: "
            + str([c for c in leakage["checks"] if c["status"] == "FAIL"])
        )

    # -- Analysis reports ----------------------------------------------------
    feature_coverage = build_coverage_report(table, config, write_csv=write)
    correlation_report = write_correlation(table, config) if write else {}
    target = target_report(table, config)
    representativeness_report = representativeness(table)
    validation_groups = write_validation_outputs(table, config) if write else {}
    readiness = build_readiness(table, config)

    artifacts: dict = {}
    if write:
        artifacts["training_outputs"] = write_training_outputs(
            table, complete, incomplete, config)
        artifacts["reports"] = {
            "feature_coverage": feature_coverage.get("paths", {}).get(
                "feature_coverage_report"),
            "target": target.get("path"),
            "readiness": readiness.get("path"),
            "validation_groups": validation_groups.get("json_path"),
            "correlation_csv": correlation_report.get("path"),
        }
        artifacts["visuals"] = write_visualizations(
            table, config, correlation_report.get("path"))
        logger.info("Visuals written: %s", artifacts["visuals"])

    metadata = build_metadata(config, scope, builder_report,
                              feature_coverage, readiness, osm_report)

    # -- Final M17 report -----------------------------------------------------
    tests_run = {"unit": "tests/test_global_training.py (see milestone report)"}
    regression = {"backend": "pytest -q", "dashboard": "npm test + npm run build"}
    final = build_final_report(
        config, scope, builder_report, feature_coverage, target,
        correlation_report, validation_groups, leakage, readiness,
        metadata, artifacts, tests_run, regression)

    logger.info("=" * 70)
    logger.info("M17 SUMMARY (scope=%s): rows=%d complete=%d stations=%d "
                "countries=%d readiness=%s",
                scope, builder_report["rows"],
                builder_report["complete_case_rows"],
                builder_report["stations"], builder_report["countries"],
                readiness["model_training_ready"])
    logger.info("  LOSO %s | LOCO %s", validation_groups.get("loso", {}).get(
        "status"), validation_groups.get("loco", {}).get("status"))
    logger.info("  Global ML: NOT IMPLEMENTED - intentionally deferred to M18.")
    logger.info("=" * 70)
    return {
        "stage": "GLOBAL_TRAINING_DATASET",
        "scope": scope,
        "builder": builder_report,
        "qc": qc_report,
        "leakage": leakage,
        "coverage": feature_coverage,
        "target": target,
        "validation": validation_groups,
        "readiness": readiness,
        "metadata": metadata,
        "report": final,
        "ml_not_implemented": True,
        "prediction_not_implemented": True,
    }
