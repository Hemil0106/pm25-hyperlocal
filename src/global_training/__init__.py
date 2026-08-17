"""Milestone 17: Global Training Dataset + Feature Engineering + Spatial
Validation Setup.

Data engineering ONLY. No ML training, no global prediction, no AQI/hotspot
rasters. This stage builds the schema-correct training table from real Milestone
16 outputs; any feature that was not acquired (no credentials / no artifact) is
NaN with a per-feature ``*_available`` flag, and no value is ever fabricated or
substituted.
"""

from .pipeline import run_global_training_pipeline

__all__ = ["run_global_training_pipeline"]
