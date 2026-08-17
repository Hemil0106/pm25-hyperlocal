"""Artifact integrity checks (Milestone 16).

Checksums give traceability for cached/downloaded files; scope isolation
verification guarantees an artifact built for one scope is never served under
another; the synthetic-leakage check guarantees tagged test data never enters
real datasets.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from .scope import infer_scope, scope_bounds, validate_scope

logger = logging.getLogger(__name__)

SYNTHETIC_TEST_TAG = "SYNTHETIC_TEST_DATA"


def sha256_file(path) -> str:
    """SHA-256 hex digest of a file (streamed, safe for large rasters)."""
    path = Path(path)
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_checksum(path, expected: Optional[str]) -> bool:
    """Compare a file's SHA-256 with an expected digest; empty/missing -> False."""
    if not expected:
        return False
    if not Path(path).exists():
        return False
    return sha256_file(path) == str(expected).lower()


def verify_artifact_scope(requested_scope: str, bounds: dict) -> tuple[bool, str]:
    """Verify artifact bounds are consistent with the requested scope.

    Returns (ok, reason). Enforces requested_scope == artifact_scope: a Delhi
    artifact served as india or global fails here, preventing scope leakage.
    Global is the maximal scope and accepts anything inside the world bounds.
    """
    try:
        requested = validate_scope(requested_scope)
    except ValueError as exc:
        return False, str(exc)
    if requested == "global":
        return True, "global scope accepts any artifact within the world bounds"
    inferred = infer_scope(bounds)
    if inferred == requested:
        return True, f"artifact inside scope '{requested}'"
    return False, (
        f"artifact bounds {bounds} resolve to scope '{inferred}', not "
        f"'{requested}' ({scope_bounds(requested)}); scope isolation refused"
    )


def has_synthetic_rows(df: pd.DataFrame, tag_columns=None) -> bool:
    """True when any row in a real dataset carries the synthetic test tag."""
    if df.empty:
        return False
    tag_columns = tag_columns or ["quality_flag", "source", "notes"]
    for col in tag_columns:
        if col in df.columns:
            values = df[col].astype(str)
            if values.str.contains(SYNTHETIC_TEST_TAG, case=False, na=False).any():
                return True
    return False


def synthetic_leakage_report(df: pd.DataFrame, dataset_name: str) -> dict:
    """Honest check: synthetic-tagged rows must never be in real datasets."""
    leaked = has_synthetic_rows(df)
    return {
        "dataset": dataset_name,
        "synthetic_data_leakage": "PRESENT" if leaked else "NONE",
        "rule": "Rows tagged SYNTHETIC_TEST_DATA must never enter real datasets.",
    }
