"""Tile-based processing for global/large AOIs.

Global processing must NEVER allocate a single in-memory high-resolution
grid. Instead the AOI is covered by a configurable tile grid; each tile is
processed and validated independently, then merged.

Tile IDs are deterministic: ``tile_{ix:03d}_{iy:03d}`` where ``ix`` indexes
the longitude axis and ``iy`` the latitude axis (starting at the AOI's
southwest corner).

Caching: cache keys are derived from (dataset, product, date, tile, version,
resolution) and stored under ``cache_dir/<dataset>/<product>/<date>/<tile>/
<version>/<resolution>/`` with a ``MARKER`` file. Cached, valid datasets are
reused; otherwise they are (re)downloaded.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

CACHE_MARKER = "COMPLETE.marker"


@dataclass
class Tile:
    tile_id: str
    bbox: dict  # west, south, east, north (EPSG:4326)
    status: str = "pending"
    datasets_available: dict = field(default_factory=dict)
    processing_time_s: Optional[float] = None
    output_path: Optional[Path] = None

    @property
    def lon_range(self):
        return (self.bbox["west"], self.bbox["east"])

    @property
    def lat_range(self):
        return (self.bbox["south"], self.bbox["north"])


def generate_tiles(bbox: dict, size_deg: float = 10.0, overlap_deg: float = 0.0):
    """Generate a deterministic tile grid covering a geographic bbox.

    ``overlap_deg`` extends every tile on all sides (except the outer edge of
    the AOI) so adjacent tiles overlap by ``2 * overlap_deg`` along shared
    edges, enabling consistent seam handling during merges.
    """
    if size_deg <= 0:
        raise ValueError("size_deg must be > 0")
    if overlap_deg < 0:
        raise ValueError("overlap_deg must be >= 0")

    west = float(bbox["west"])
    east = float(bbox["east"])
    south = float(bbox["south"])
    north = float(bbox["north"])

    tiles: list[Tile] = []
    ix = 0
    lon = west
    while lon < east - 1e-9:
        iy = 0
        lat = south
        while lat < north - 1e-9:
            t_west = max(west, lon - overlap_deg)
            t_east = min(east, lon + size_deg + overlap_deg)
            t_south = max(south, lat - overlap_deg)
            t_north = min(north, lat + size_deg + overlap_deg)
            tiles.append(
                Tile(
                    tile_id=f"tile_{ix:03d}_{iy:03d}",
                    bbox={
                        "west": t_west,
                        "south": t_south,
                        "east": t_east,
                        "north": t_north,
                    },
                )
            )
            lat += size_deg
            iy += 1
        lon += size_deg
        ix += 1

    return tiles


def tile_for_point(lon: float, lat: float, size_deg: float, bbox: dict) -> str:
    """Deterministic tile id containing a point (no overlap expansion)."""
    west = float(bbox["west"])
    south = float(bbox["south"])
    ix = int((float(lon) - west) // size_deg)
    iy = int((float(lat) - south) // size_deg)
    return f"tile_{ix:03d}_{iy:03d}"


def count_tiles_for_global(size_deg: float = 10.0) -> int:
    world = {"west": -180.0, "south": -90.0, "east": 180.0, "north": 90.0}
    return len(generate_tiles(world, size_deg=size_deg))


# ---------------------------------------------------------------------------
# Caching (Phase 21)
# ---------------------------------------------------------------------------

def cache_key(dataset: str, product: str, date: str, tile_id: str,
              version: str = "v1", resolution: str = "", cache_dir=None):
    """Build a cache directory path from all provenance components."""
    base = Path(cache_dir) if cache_dir else None
    parts = [dataset, product, date, tile_id, version]
    if resolution:
        parts.append(resolution)
    key = Path(*parts)
    if base is not None:
        return base / key
    return key


def cache_is_valid(key: Path) -> bool:
    return (key / CACHE_MARKER).is_file()


def cache_put(key: Path, metadata: Optional[dict] = None):
    key.mkdir(parents=True, exist_ok=True)
    if metadata is not None:
        with open(key / "provenance.json", "w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2)
    (key / CACHE_MARKER).write_text("complete", encoding="utf-8")


def cache_get(key: Path):
    """Return cached metadata if the cache entry is valid, else None."""
    if not cache_is_valid(key):
        return None
    meta_path = key / "provenance.json"
    if meta_path.is_file():
        with open(meta_path, "r", encoding="utf-8") as file:
            return json.load(file)
    return {}


# ---------------------------------------------------------------------------
# Tile processing helper
# ---------------------------------------------------------------------------

def process_tiles(tiles, processor, cache_dir: Optional[Path] = None):
    """Run a processor on each tile and record status/timing.

    ``processor(tile)`` returns (datasets_available: dict, output_path).
    """
    results = []
    for tile in tiles:
        started = time.time()
        try:
            datasets, output_path = processor(tile)
            tile.datasets_available = datasets
            tile.output_path = Path(output_path) if output_path else None
            tile.status = "ok"
        except Exception as exc:  # noqa: BLE001
            logger.exception("Tile %s failed: %s", tile.tile_id, exc)
            tile.status = f"error: {exc}"
        finally:
            tile.processing_time_s = time.time() - started
        results.append(tile)
    return results


def merge_tile_outputs(tiles, output_dir: Path, pattern: str) -> Path:
    """Merge per-tile GeoTIFFs into a single output raster.

    Each tile file must match ``pattern`` (formatted with tile_id). Tiles are
    mosaicked in order using rasterio.merge. This is a placeholder-friendly,
    honest implementation: if any tile is missing, the merge is skipped.
    """
    import rasterio
    from rasterio.merge import merge

    paths = []
    for tile in tiles:
        candidate = output_dir / pattern.format(tile_id=tile.tile_id)
        if candidate.exists():
            paths.append(candidate)
        else:
            raise FileNotFoundError(
                f"Tile output missing for {tile.tile_id}: {candidate}"
            )

    if not paths:
        raise FileNotFoundError("No tile outputs to merge.")

    merged, out_transform = merge(paths)
    out_meta = rasterio.open(paths[0]).meta.copy()
    out_meta.update(
        {
            "driver": "GTiff",
            "height": merged.shape[1],
            "width": merged.shape[2],
            "transform": out_transform,
        }
    )
    final = output_dir / "merged.tif"
    with rasterio.open(final, "w", **out_meta) as dst:
        dst.write(merged)
    return final
