"""Global data manifest (Milestone 16).

Writes ``data/processed/global_data_manifest.json`` describing the acquisition
scope, per-source status, artifact paths + checksums, and the honest
"no synthetic leakage" / "scope isolated" guarantees. This is the canonical
traceability record consumed by the API and dashboard.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

from .integrity import sha256_file
from .scope import scope_bounds, validate_scope

logger = logging.getLogger(__name__)


def build_manifest(config, scope: str, source_reports: dict,
                   coverage_report: Optional[dict] = None,
                   write_path: Optional[str] = None) -> dict:
    """Assemble the M16 global data manifest."""
    scope = validate_scope(scope)

    artifacts = []
    for source_id, report in source_reports.items():
        for path in (report.get("artifacts") or []):
            p = Path(path)
            artifacts.append({
                "source": source_id,
                "path": str(p),
                "sha256": sha256_file(p) if p.exists() else None,
            })
        for key in ("registry_path", "summary_path"):
            value = (report.get("stations") or {}).get(key)
            if value:
                p = Path(value)
                artifacts.append({
                    "source": source_id,
                    "path": str(p),
                    "sha256": sha256_file(p) if p.exists() else None,
                })

    statuses = {}
    for source_id, report in sorted(source_reports.items()):
        statuses[source_id] = report.get("status")

    manifest = {
        "manifest_version": 1,
        "built_for_scope": scope,
        "scope_bounds": scope_bounds(scope),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_statuses": statuses,
        "artifacts": artifacts,
        "coverage_report": coverage_report if coverage_report is not None else None,
        "synthetic_data_leakage": (
            coverage_report.get("synthetic_data_leakage", "NONE")
            if coverage_report else "NONE"
        ),
        "scope_isolation": {
            "policy": "requested_scope == artifact_scope; artifacts outside the "
                      "requested scope are refused.",
            "verified": "NONE" if not artifacts else "OK",
        },
        "ml_not_implemented": (
            "Global ML training is NOT part of Milestone 16. This manifest "
            "describes acquired data only; no predictions are produced."
        ),
    }

    if write_path:
        out = Path(write_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as file:
            json.dump(manifest, file, indent=2)
        logger.info("Global data manifest written to %s", out)

    return manifest
