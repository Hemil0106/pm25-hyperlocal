"""Centralized CRS handling for AOI-relative geographic processing.

EPSG:32643 (UTM zone 43N) was hard-coded for the Delhi prototype. This module
is the single source of truth for CRS decisions:

  - Geographic CRS: EPSG:4326 (always).
  - Projected (metric) CRS: chosen per-AOI from the AOI centroid using the
    UTM zone rule (EPSG:326xx north / 327xx south). For the Delhi AOI this
    resolves to EPSG:32643, preserving backward compatibility.
  - Equal-area CRS: EPSG:6933 (WGS 84 / NSIDC EASE-Grid 2.0 global) for
    area calculations anywhere on Earth.
  - Geodesic distances/areas: pyproj.Geod (WGS84) for point distances.

Global AOIs have no single projected CRS - callers must use tile-level CRS
(via get_metric_crs on the tile's bbox) or geodesic calculations.
"""

from typing import Optional

GEOGRAPHIC_CRS = "EPSG:4326"
EQUAL_AREA_GLOBAL_CRS = "EPSG:6933"

_GEOD = None


def _geod():
    global _GEOD
    if _GEOD is None:
        from pyproj import Geod

        _GEOD = Geod(ellps="WGS84")
    return _GEOD


def get_geographic_crs() -> str:
    """The project geographic CRS (EPSG:4326)."""
    return GEOGRAPHIC_CRS


def get_equal_area_crs() -> str:
    """Global equal-area CRS for area calculations (EPSG:6933)."""
    return EQUAL_AREA_GLOBAL_CRS


def utm_zone_for(lon: float, lat: float) -> int:
    """UTM zone (1-60) for a point. Handles the -180/180 boundary."""
    lon = float(lon)
    if lon == 180.0:
        lon = 179.9999
    zone = int((lon + 180.0) / 6.0) + 1
    if zone > 60:
        zone = 60
    if zone < 1:
        zone = 1
    return zone


def get_metric_crs(aoi_bbox: Optional[dict] = None,
                   centroid_lon: Optional[float] = None,
                   centroid_lat: Optional[float] = None,
                   lon: Optional[float] = None,
                   lat: Optional[float] = None,
                   aoi=None) -> str:
    """Pick a metric (projected) CRS appropriate for an AOI.

    Accepts either an AOI object, an aoi_bbox dict, an explicit centroid, or
    a lon/lat point. For the Delhi AOI this returns EPSG:32643, matching the
    locked prototype. Returns None for global AOIs (no single projected CRS).
    """
    if aoi is not None:
        if aoi.is_global:
            return None
        west = float(aoi.bbox["west"])
        east = float(aoi.bbox["east"])
        south = float(aoi.bbox["south"])
        north = float(aoi.bbox["north"])
        lon = (west + east) / 2.0
        lat = (south + north) / 2.0
    elif centroid_lon is not None and centroid_lat is not None:
        lon, lat = float(centroid_lon), float(centroid_lat)
    elif aoi_bbox is not None:
        west = float(aoi_bbox["west"])
        east = float(aoi_bbox["east"])
        south = float(aoi_bbox["south"])
        north = float(aoi_bbox["north"])
        if abs(east - west) >= 180.0 or abs(north - south) >= 160.0:
            return None
        lon = (west + east) / 2.0
        lat = (south + north) / 2.0
    elif lon is None or lat is None:
        raise ValueError(
            "get_metric_crs requires an AOI, aoi_bbox, centroid, or lon/lat."
        )

    zone = utm_zone_for(lon, lat)
    if lat >= 0.0:
        epsg = 32600 + zone
    else:
        epsg = 32700 + zone
    return f"EPSG:{epsg}"


def transform_coords(lon: float, lat: float, source_crs: str, target_crs: str):
    """Transform a single (lon, lat) point between CRS strings."""
    from pyproj import Transformer

    transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
    x, y = transformer.transform(float(lon), float(lat))
    return x, y


def transform_geometry(geometry, source_crs: str, target_crs: str):
    """Reproject a shapely geometry between CRS strings."""
    from pyproj import Transformer
    from shapely.ops import transform as shapely_transform

    transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
    return shapely_transform(lambda x, y: transformer.transform(x, y), geometry)


def geodesic_distance_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Great-circle distance between two points in kilometers."""
    return _geod().inv(lon1, lat1, lon2, lat2)[2] / 1000.0


def transform_raster_profile(profile, source_crs: str, target_crs: str, bounds,
                             resolution_m: Optional[float] = None):
    """Reproject raster bounds between CRS strings and return a new profile.

    This is a lightweight helper used by tile/scene preparation. Full raster
    warping is performed by rasterio.warp.reproject; this helper only returns
    the projected bounds and an updated transform.
    """
    from rasterio.warp import transform_bounds

    new_bounds = transform_bounds(source_crs, target_crs, *bounds)
    if resolution_m:
        width = max(1, int(round((new_bounds[2] - new_bounds[0]) / resolution_m)))
        height = max(1, int(round((new_bounds[3] - new_bounds[1]) / resolution_m)))
    else:
        width = profile.get("width")
        height = profile.get("height")
    out_profile = dict(profile)
    out_profile.update(
        {
            "crs": target_crs,
            "width": width,
            "height": height,
            "transform": None,  # recompute after transform_bounds
        }
    )
    return out_profile, new_bounds
