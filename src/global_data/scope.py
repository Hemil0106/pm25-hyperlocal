"""Acquisition scopes for the global data layer (Milestone 16).

Scopes are explicit: ``global``, ``india``, ``delhi``. Every artifact records
the scope it was built for, and ``assert_scope_isolated`` refuses to serve an
artifact built for one scope under another (this is what prevents Delhi data
from being silently used as global data).
"""

from __future__ import annotations

from typing import Optional

SUPPORTED_SCOPES = ("global", "india", "delhi", "pune", "mumbai")

SCOPE_BOUNDS = {
    "global": {"west": -180.0, "south": -90.0, "east": 180.0, "north": 90.0},
    "india": {"west": 65.0, "south": 5.0, "east": 100.0, "north": 40.0},
    "delhi": {"west": 77.0, "south": 28.4, "east": 77.4, "north": 28.8},
    "pune": {"west": 73.7, "south": 18.4, "east": 74.1, "north": 18.7},
    "mumbai": {"west": 72.7, "south": 18.8, "east": 73.0, "north": 19.3},
}


def validate_scope(scope: Optional[str]) -> str:
    """Normalize and validate an acquisition scope string."""
    if scope is None:
        return "global"
    normalized = str(scope).strip().lower()
    if normalized not in SUPPORTED_SCOPES:
        raise ValueError(
            f"Unknown scope '{scope}'. Supported scopes: {SUPPORTED_SCOPES}"
        )
    return normalized


def scope_bounds(scope: str) -> dict:
    """Bounds (west, south, east, north in EPSG:4326) for a scope."""
    return dict(SCOPE_BOUNDS[validate_scope(scope)])


def is_contained(bounds: dict, container: dict) -> bool:
    """True when ``bounds`` (west/south/east/north) fits inside ``container``."""
    return (
        float(bounds["west"]) >= float(container["west"]) - 1e-9
        and float(bounds["south"]) >= float(container["south"]) - 1e-9
        and float(bounds["east"]) <= float(container["east"]) + 1e-9
        and float(bounds["north"]) <= float(container["north"]) + 1e-9
    )


def infer_scope(bounds: dict) -> str:
    """Smallest configured scope whose bounds contain ``bounds``.

    Used to verify that an acquired artifact really belongs to the requested
    scope (and, crucially, was not silently borrowed from a smaller one).
    Iterated smallest-first so a Delhi artifact is never reported as India or
    Global.
    """
    for scope in ("delhi", "pune", "mumbai", "india", "global"):
        if is_contained(bounds, SCOPE_BOUNDS[scope]):
            return scope
    return "outside_all_scopes"


def assert_scope_isolated(requested_scope: str, artifact_bounds: dict) -> None:
    """Raise when an artifact's scope is not consistent with the requested scope.

    This enforces requested_scope == artifact_scope: a ``delhi`` artifact must
    never be served under ``india`` or ``global``, and an ``india`` artifact
    must never be served under ``global``. ``global`` is the maximal scope and
    accepts anything inside the world bounds.
    """
    requested = validate_scope(requested_scope)
    if requested == "global":
        return  # global is the maximal scope; anything inside is fine.
    if infer_scope(artifact_bounds) != requested:
        raise ValueError(
            f"Scope isolation violated: artifact bounds "
            f"({artifact_bounds}) resolve to scope '{infer_scope(artifact_bounds)}', "
            f"not requested scope '{requested}' "
            f"({SCOPE_BOUNDS[requested]}). Never substitute data from another "
            f"scope as {requested} data."
        )


def artifact_scope_tag(scope: str) -> str:
    """Deterministic scope tag embedded in artifact metadata."""
    return f"scope={validate_scope(scope)}"
