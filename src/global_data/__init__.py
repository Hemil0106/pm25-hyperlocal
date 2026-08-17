"""Milestone 16: Global Data Acquisition + Normalized Observation Pipeline.

Data engineering ONLY. No ML training, no global prediction, no AQI/hotspot
rasters. Real sources are acquired with credentials from the environment only;
missing credentials produce graceful UNAVAILABLE status with no fabricated or
substituted data.
"""

from .ingest import ensure_global_dirs, run_global_data_pipeline
from .scope import (
    SCOPE_BOUNDS,
    SUPPORTED_SCOPES,
    assert_scope_isolated,
    infer_scope,
    scope_bounds,
    validate_scope,
)
from .sources import (
    DATA_SOURCE_REGISTRY,
    build_data_source_registry,
    credential_available,
)

__all__ = [
    "SCOPE_BOUNDS",
    "SUPPORTED_SCOPES",
    "DATA_SOURCE_REGISTRY",
    "assert_scope_isolated",
    "build_data_source_registry",
    "credential_available",
    "ensure_global_dirs",
    "infer_scope",
    "run_global_data_pipeline",
    "scope_bounds",
    "validate_scope",
]
