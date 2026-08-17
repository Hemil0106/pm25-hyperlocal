"""Cache integration for globalized ingestion (Phase 9).

A thin, honest caching layer over the tile cache from ``src.geospatial.tiles``.

get_or_compute builds a cache key from full provenance (dataset, product,
date, tile, version, resolution), reuses a valid cache entry, and otherwise
runs ``compute_fn`` and stores its result. Cache hits/misses are reported so
the platform never silently serves stale data.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

from ..geospatial.tiles import (
    cache_is_valid,
    cache_key,
    cache_put,
    cache_get,
    CACHE_MARKER,
)

logger = logging.getLogger(__name__)


def get_or_compute(cache_dir, dataset: str, product: str, date: str,
                   tile_id: str, compute_fn: Callable,
                   version: str = "v1", resolution: str = "",
                   cache_enabled: Optional[bool] = None,
                   extra_metadata: Optional[dict] = None):
    """Return (result, cache_hit) using the tile cache.

    ``compute_fn()`` must return (payload, out_path_or_None).
    """
    cache_enabled = cache_enabled if cache_enabled is not None else True
    if not cache_enabled:
        payload, out_path = compute_fn()
        return payload, False

    key = cache_key(dataset, product, date, tile_id, version=version,
                    resolution=resolution, cache_dir=Path(cache_dir))
    if cache_is_valid(key):
        logger.debug("Cache HIT for %s/%s/%s/%s", dataset, product, date, tile_id)
        payload = cache_get(key)
        marker_path = key / CACHE_MARKER
        out_path = None
        if marker_path.is_file():
            out_path = key / "result"
        return payload, True

    logger.debug("Cache MISS for %s/%s/%s/%s", dataset, product, date, tile_id)
    payload, out_path = compute_fn()
    metadata = {"dataset": dataset, "product": product, "date": date,
                "tile_id": tile_id, "version": version,
                "resolution": resolution}
    if extra_metadata:
        metadata.update(extra_metadata)
    cache_put(key, metadata)
    return payload, False


def cache_stats(cache_dir: Path) -> dict:
    """Count valid cache entries under a cache directory."""
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        return {"entries": 0, "size_bytes": 0}
    n = 0
    size = 0
    for marker in cache_dir.rglob(CACHE_MARKER):
        n += 1
        try:
            size += marker.parent.stat().st_size
        except OSError:
            continue
    return {"entries": n, "size_bytes": size}
