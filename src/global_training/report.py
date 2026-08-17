"""Milestone 17 final report writer.

Assembles ``global_training_report.json`` with the honest M17 REPORT fields
(rows, stations, countries, feature coverage, spatial validation, leakage,
readiness, and the explicit "global ML not implemented" statement).
"""

from __future__ import annotations

import json
import time
from pathlib import Path


def build_final_report(config, scope: str, builder_report: dict,
                       feature_coverage: dict, target_report: dict,
                       correlation_report: dict, validation_groups: dict,
                       leakage: dict, readiness: dict,
                       metadata: dict, artifacts: dict,
                       tests_run: dict, regression: dict) -> dict:
    """Compose and write the M17 REPORT JSON."""
    loso = validation_groups.get("loso", {})
    loco = validation_groups.get("loco", {})

    leakage_checks = {c["check"]: c["status"] for c in leakage.get("checks", [])}
    synthetic = next(
        (c["status"] for c in leakage.get("checks", [])
         if c["check"] == "synthetic_contamination"), "NONE")
    delhi_leak = next(
        (c["status"] for c in leakage.get("checks", [])
         if c["check"] == "delhi_artifact_contamination"), "PASS")

    report = {
        "stage": "MILESTONE_17_GLOBAL_TRAINING_DATA",
        "scope": scope,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "files": artifacts,
        "dataset_summary": {
            "rows": builder_report.get("rows", 0),
            "stations": builder_report.get("stations", 0),
            "countries": builder_report.get("countries", 0),
            "date_range": target_report.get("date_range"),
            "complete_case_rows": builder_report.get("complete_case_rows", 0),
        },
        "feature_coverage": {
            "complete_case_rows": feature_coverage.get("complete_case_rows", 0),
            "missingness_by_feature": feature_coverage.get(
                "missingness_by_feature", []),
            "geographic_bias": feature_coverage.get("geographic_bias", []),
        },
        "target": {
            "distribution": target_report.get("distribution"),
            "completeness_flag": target_report.get("completeness_flag"),
        },
        "correlation": {
            "status": correlation_report.get("status"),
            "note": correlation_report.get("note"),
        },
        "spatial_validation": {
            "loso": {
                "status": loso.get("status", "BLOCKED"),
                "reason": loso.get("reason"),
            },
            "loco": {
                "status": loco.get("status", "BLOCKED"),
                "reason": loco.get("reason"),
            },
            "hierarchy": validation_groups.get("hierarchy"),
        },
        "leakage_checks": leakage_checks,
        "leakage_overall": leakage.get("overall"),
        "synthetic_contamination": synthetic,
        "delhi_to_global_leakage": delhi_leak,
        "global_model_readiness": readiness.get("model_training_ready", "NO"),
        "readiness_reason": readiness.get("reason"),
        "global_ml_not_implemented": True,
        "global_ml_statement": "Global ML: NOT IMPLEMENTED - intentionally "
                               "deferred to Milestone 18.",
        "global_prediction_not_implemented": True,
        "tests": tests_run,
        "regression": regression,
        "metadata_path": metadata.get("path"),
        "next_stage": "M18 (do not implement yet)",
    }

    processed_base = Path(
        config.get("global_data", {}).get("storage", {}).get(
            "processed_base", "data/processed/global")
    )
    reports_dir = processed_base / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / "global_training_report.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["path"] = str(path)
    return report
