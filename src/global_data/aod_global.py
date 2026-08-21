"""AOD (MODIS MAIAC MCD19A2) global adapter (Milestone 16).

Tile-based acquisition of aerosol optical depth. Requires NASA Earthdata
credentials (environment only). Missing credentials -> graceful UNAVAILABLE.
Uses NASA CMR for granule search + Earthdata bearer token for download.

After HDF download, converts MCD19A2 HDF to GeoTIFF using pyhdf + numpy.
The HDF subdataset ``Optical_Depth_Land`` (500m MAIAC AOD at 0.47 and 0.55
micron bands, combined) is extracted, tiled to the requested bbox, and saved
as a single-band float32 GeoTIFF.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .satellite import SatelliteSource, SourceUnavailable

logger = logging.getLogger(__name__)

_MCD19A2_CONCEPT_ID = "C2324689816-LPCLOUD"

# MCD19A2 HDF-EOS grid: global sinusoidal tile grid
# Each tile is 2400x2400 pixels at 500m resolution
_GRID_SIZE = 2400
_PIXEL_SIZE_M = 500.0

# MCD19A2 sinusoidal tile for Delhi area is h24v06
# The HDF subdataset name for AOD
_AOD_FIELD = "Optical_Depth_Land"


class AODSource(SatelliteSource):
    source_id = "aod"
    product = "MODIS_MAIAC_MCD19A2"
    default_resolution = "1km"


def check_availability(config) -> dict:
    return AODSource(config).check_availability()


def _sinusoidal_bbox_to_pixel_range(bbox: dict, tile_name: str):
    """Convert a geographic bbox to pixel row/col range within a sinusoidal tile.

    MCD19A2 tiles use the MODIS Sinusoidal projection. For a rough conversion,
    we use the known tile bounds in degrees for each h/v tile index.
    Returns (row_start, row_end, col_start, col_end) in pixel coords.
    """
    import math

    h, v = int(tile_name[1:3]), int(tile_name[4:6])

    # MODIS sinusoidal tile boundaries (approximate in degrees)
    tile_west = -180.0 + h * 10.0
    tile_east = tile_west + 10.0
    tile_north = 90.0 - v * 10.0
    tile_south = tile_north - 10.0

    # Clip bbox to tile bounds
    west = max(bbox["west"], tile_west)
    east = min(bbox["east"], tile_east)
    north = min(bbox["north"], tile_north)
    south = max(bbox["south"], tile_south)

    if west >= east or south >= north:
        return None

    # Convert geographic coords to pixel indices (tile is top-left origin)
    col_start = int((west - tile_west) / (tile_east - tile_west) * _GRID_SIZE)
    col_end = int((east - tile_west) / (tile_east - tile_west) * _GRID_SIZE)
    row_start = int((tile_north - north) / (tile_north - tile_south) * _GRID_SIZE)
    row_end = int((tile_north - south) / (tile_north - tile_south) * _GRID_SIZE)

    col_start = max(0, min(col_start, _GRID_SIZE))
    col_end = max(0, min(col_end, _GRID_SIZE))
    row_start = max(0, min(row_start, _GRID_SIZE))
    row_end = max(0, min(row_end, _GRID_SIZE))

    if col_start >= col_end or row_start >= row_end:
        return None

    return row_start, row_end, col_start, col_end


def _hdf_to_geotiff(hdf_path: Path, bbox: dict, out_path: Path) -> Path:
    """Extract AOD from MCD19A2 HDF and write a GeoTIFF cropped to bbox."""
    import math

    import numpy as np
    import rasterio
    from rasterio.crs import CRS
    from rasterio.transform import from_bounds
    from pyhdf.SD import SD, SDC

    hdf = SD(str(hdf_path), SDC.READ)

    # Get the Optical_Depth_Land field
    try:
        aod_field = hdf.select(_AOD_FIELD)
    except KeyError:
        # Try alternative names
        for alt_name in ["Optical_Depth_Land_And_Water", "Optical_Depth_047", "Optical_Depth_055"]:
            try:
                aod_field = hdf.select(alt_name)
                logger.info("AOD: using subdataset '%s'", alt_name)
                break
            except KeyError:
                continue
        else:
            raise RuntimeError(
                f"MCD19A2 HDF has no AOD subdataset. "
                f"Available fields: {hdf.datasets().keys()}"
            )

    # Read the full AOD array (2400x2400)
    aod_data = np.array(aod_field.get(), dtype=np.float32)
    attrs = aod_field.attributes()
    scale = attrs.get("scale_factor", 0.001)
    fill = attrs.get("_FillValue", -28672)
    aod_valid_range = attrs.get("valid_range", [0, 5000])

    # Read quality field for cloud/shadow mask
    try:
        qa_field = hdf.select("Optical_Depth_QA")
        qa_data = np.array(qa_field.get(), dtype=np.uint8)
    except (KeyError, Exception):
        qa_data = None

    hdf.end()

    # Scale AOD values
    aod_data[aod_data == fill] = np.nan
    aod_data[aod_data < aod_valid_range[0]] = np.nan
    aod_data[aod_data > aod_valid_range[1]] = np.nan
    aod_data = aod_data * scale

    # Apply QA mask if available (bit 0-1: 00 = best quality)
    if qa_data is not None:
        # Bits 0-1: 00=best, 01=good, 10=ok, 11=cloud/shadow
        quality_bits = qa_data & 0x03
        aod_data[quality_bits == 3] = np.nan

    # Extract tile name from filename
    fname = hdf_path.stem
    # MCD19A2.A2025001.h24v06.061.2025003060200
    parts = fname.split(".")
    tile_name = None
    for p in parts:
        if p.startswith("h") and p.startswith("v") is False:
            tile_name = p
            break
    if tile_name is None:
        for p in parts:
            if p[0] == "h" and len(p) > 3 and p[3] == "v":
                tile_name = p
                break

    if tile_name is None:
        raise RuntimeError(f"Cannot parse MODIS tile from filename: {fname}")

    # Get pixel range for bbox
    pixel_range = _sinusoidal_bbox_to_pixel_range(bbox, tile_name)
    if pixel_range is None:
        raise RuntimeError(f"Bbox {bbox} outside tile {tile_name}")

    row_start, row_end, col_start, col_end = pixel_range
    cropped = aod_data[row_start:row_end, col_start:col_end]

    # Calculate geographic bounds for the cropped area
    h, v = int(tile_name[1:3]), int(tile_name[4:6])
    tile_west = -180.0 + h * 10.0
    tile_east = tile_west + 10.0
    tile_north = 90.0 - v * 10.0
    tile_south = tile_north - 10.0

    crop_west = tile_west + (col_start / _GRID_SIZE) * (tile_east - tile_west)
    crop_east = tile_west + (col_end / _GRID_SIZE) * (tile_east - tile_west)
    crop_north = tile_north - (row_start / _GRID_SIZE) * (tile_north - tile_south)
    crop_south = tile_north - (row_end / _GRID_SIZE) * (tile_north - tile_south)

    rows, cols = cropped.shape
    transform = from_bounds(crop_west, crop_south, crop_east, crop_north, cols, rows)
    crs = CRS.from_epsg(4326)

    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "width": cols,
        "height": rows,
        "count": 1,
        "crs": crs,
        "transform": transform,
        "nodata": np.nan,
    }

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(cropped, 1)

    valid_pct = np.count_nonzero(~np.isnan(cropped)) / cropped.size * 100
    logger.info("AOD: wrote GeoTIFF %s (%dx%d, %.1f%% valid, range [%.3f, %.3f])",
                out_path.name, cols, rows, valid_pct,
                np.nanmin(cropped), np.nanmax(cropped))
    return out_path


def acquire(config, scope: str = "global", date: str = "2025-01-01") -> dict:
    source = AODSource(config)

    def on_tile(tile: object, _date: str) -> Path:
        tile_id = tile.tile_id
        tile_bbox = tile.bbox
        raw_base = Path(config.get("global_data", {}).get("storage", {}).get(
            "raw_base", "data/raw/global"))
        out_dir = raw_base / "aod" / scope / str(tile_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        hdf_path = out_dir / "MCD19A2.hdf"
        geotiff_path = out_dir / f"aod_{_date}.tif"

        if not geotiff_path.exists():
            if not hdf_path.exists() or hdf_path.stat().st_size < 1000:
                from .nasa_auth import (cmr_search_granules, find_download_url,
                                         download_with_earthdata)

                granules = cmr_search_granules(
                    _MCD19A2_CONCEPT_ID,
                    bbox=tile_bbox,
                    temporal_start=_date,
                    temporal_end=_date,
                )
                url = find_download_url(granules)
                if not url:
                    raise RuntimeError(
                        f"No MCD19A2 granule found for tile {tile_id} on {_date}")

                download_with_earthdata(url, str(hdf_path))

            _hdf_to_geotiff(hdf_path, tile_bbox, geotiff_path)

        # Also copy to the processed dir so the backend /raster/aod can serve it
        try:
            processed_base = Path(config.get("global_data", {}).get("storage", {}).get(
                "processed_base", "data/processed/global"))
            city = scope if scope != "global" else "delhi"
            backend_dir = Path("data/processed") if scope in ("delhi",) else processed_base.parent
            backend_aod = backend_dir / ("" if scope in ("delhi",) else "global") / f"aod_500m_{_date}.tif"
            if scope == "delhi":
                backend_aod = Path("data/processed") / f"aod_500m_{_date}.tif"
            backend_aod.parent.mkdir(parents=True, exist_ok=True)
            if not backend_aod.exists():
                import shutil
                shutil.copy2(geotiff_path, backend_aod)
                logger.info("AOD: copied to backend path %s", backend_aod)
        except Exception as exc:
            logger.warning("AOD: could not copy to backend path: %s", exc)

        return geotiff_path

    return source.attempt_acquire(scope, date, on_tile)
