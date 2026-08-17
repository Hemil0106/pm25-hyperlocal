from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import geopandas as gpd
import logging
from shapely.geometry import box
from shapely.ops import transform as shapely_transform


GEOGRAPHIC_CRS = "EPSG:4326"

SUPPORTED_MODES = ("named_region", "bbox", "geojson", "geometry_file", "global")

logger = logging.getLogger(__name__)


@dataclass
class AOI:
    """A configurable area of interest in EPSG:4326.

    Modes:
      - named_region: geometry resolved from a named region entry or the
        configured geometry_file.
      - bbox: rectangle from min/max lon/lat.
      - geojson: loaded from a geometry file.
      - global: full world bounds (-180..180, -90..90). Never used to build a
        single in-memory global raster - processing must be tile-based.
    """

    mode: str
    name: str
    bbox: dict
    geometry: gpd.GeoDataFrame
    crs: str = GEOGRAPHIC_CRS
    config: Optional[dict] = field(default=None)

    def __post_init__(self):
        if self.mode not in SUPPORTED_MODES:
            raise ValueError(
                f"Unsupported AOI mode '{self.mode}'. Supported: {SUPPORTED_MODES}"
            )
        if self.geometry.crs is None:
            self.geometry = self.geometry.set_crs(self.crs)
        if self.geometry.crs.to_string() != self.crs:
            self.geometry = self.geometry.to_crs(self.crs)

    @property
    def is_global(self) -> bool:
        return self.mode == "global"

    @property
    def bounds(self) -> dict:
        west, south, east, north = self.geometry.total_bounds
        return {"west": float(west), "south": float(south), "east": float(east), "north": float(north)}

    def centroid(self):
        if self.is_global:
            return self.geometry.centroid.iloc[0]
        metric_crs = self.centroid_crs()
        if metric_crs:
            return self.geometry.to_crs(metric_crs).centroid.iloc[0]
        return self.geometry.centroid.iloc[0]

    def centroid_crs(self) -> Optional[str]:
        """Metric CRS appropriate for this AOI (None for global)."""
        from .crs import get_metric_crs

        return get_metric_crs(aoi=self)

    def area_km2(self) -> float:
        """Geodesic/equal-area estimate. Uses the WGS 84 / NSIDC EASE-Grid 2.0
        global equal-area projection (EPSG:6933) so the whole-world rectangle
        resolves to Earth's surface area (~510M km2) instead of degree space."""
        try:
            from pyproj import Transformer

            trans = Transformer.from_crs(self.crs, "EPSG:6933", always_xy=True)
            geom = self.geometry.geometry.iloc[0]
            geom_ae = shapely_transform(lambda x, y: trans.transform(x, y), geom)
            return float(geom_ae.area) / 1_000_000.0
        except Exception:
            logger.warning("Area computation failed; returning degree-space area.", exc_info=True)
            return float(self.geometry.area.iloc[0])

    def to_crs(self, target_crs: str) -> gpd.GeoDataFrame:
        return self.geometry.to_crs(target_crs)


def _world_aoi(global_cfg: Optional[dict]) -> gpd.GeoDataFrame:
    cfg = global_cfg or {}
    west = float(cfg.get("min_lon", -180.0))
    south = float(cfg.get("min_lat", -90.0))
    east = float(cfg.get("max_lon", 180.0))
    north = float(cfg.get("max_lat", 90.0))
    geom = box(west, south, east, north)
    return gpd.GeoDataFrame({"name": ["Global"]}, geometry=[geom], crs=GEOGRAPHIC_CRS)


def _bbox_aoi(bbox_cfg: dict, name: str = "Custom AOI") -> gpd.GeoDataFrame:
    geom = box(
        float(bbox_cfg["min_lon"]),
        float(bbox_cfg["min_lat"]),
        float(bbox_cfg["max_lon"]),
        float(bbox_cfg["max_lat"]),
    )
    return gpd.GeoDataFrame({"name": [name]}, geometry=[geom], crs=GEOGRAPHIC_CRS)


def _build_aoi_from_config(cfg: dict) -> AOI:
    mode = str(cfg.get("mode", "named_region"))
    name = str(cfg.get("name", cfg.get("named_region", {}).get("name", "Study area")))
    geometry: gpd.GeoDataFrame

    if mode == "global":
        geometry = _world_aoi(cfg.get("global"))
        west, south, east, north = geometry.total_bounds
        bbox = {"west": float(west), "south": float(south), "east": float(east), "north": float(north)}
        return AOI(mode=mode, name=name, bbox=bbox, geometry=geometry, config=cfg)

    if mode == "bbox":
        geometry = _bbox_aoi(cfg["bbox"], name=name)
    elif mode in ("named_region", "geojson", "geometry_file"):
        boundary_file = cfg.get("geometry_file") or cfg.get("boundary_file")
        if not boundary_file:
            raise ValueError(f"AOI mode '{mode}' requires a geometry_file or boundary_file.")
        geometry = load_aoi(boundary_file, GEOGRAPHIC_CRS)
    else:
        raise ValueError(f"Unsupported AOI mode '{mode}'. Supported: {SUPPORTED_MODES}")

    west, south, east, north = geometry.total_bounds
    bbox = {"west": float(west), "south": float(south), "east": float(east), "north": float(north)}
    return AOI(mode=mode, name=name, bbox=bbox, geometry=geometry, config=cfg)


def resolve_aoi(config, region: Optional[str] = None, bbox: Optional[dict] = None) -> AOI:
    """Resolve an AOI from configuration.

    Priority:
      1. explicit custom `bbox` -> mode 'bbox'
      2. explicit `region` from the region catalog -> that region's config
      3. the top-level `aoi` block (mode named_region/bbox/geojson/global)
    """
    aoi_cfg = config.get("aoi", {})
    if not aoi_cfg:
        raise ValueError("config.yaml is missing the `aoi` block.")

    if bbox:
        required = ("min_lon", "min_lat", "max_lon", "max_lat")
        if not all(key in bbox for key in required):
            raise ValueError(f"bbox must contain {required}.")
        geometry = _bbox_aoi(bbox)
        return AOI(
            mode="bbox",
            name="Custom AOI",
            bbox={k: v for k, v in zip(("west", "south", "east", "north"),
                                       (bbox["min_lon"], bbox["min_lat"], bbox["max_lon"], bbox["max_lat"]))},
            geometry=geometry,
            config={"bbox": bbox},
        )

    if region:
        regions = aoi_cfg.get("regions", {})
        if region not in regions:
            raise KeyError(
                f"Unknown region '{region}'. Available: {sorted(regions)}"
            )
        region_cfg = dict(regions[region])
        region_cfg.setdefault("name", region)
        return _build_aoi_from_config(region_cfg)

    return _build_aoi_from_config(aoi_cfg)


def get_aoi_bounds(config) -> dict:
    """Backward-compatible helper: bounds of the current configured AOI."""
    return resolve_aoi(config).bounds


def load_aoi(boundary_file, target_crs="EPSG:4326"):
    boundary_file = Path(boundary_file)

    if not boundary_file.exists():
        raise FileNotFoundError(
            f"AOI boundary file not found: {boundary_file}"
        )

    logging.info("Loading AOI: %s", boundary_file)

    gdf = gpd.read_file(boundary_file)

    if gdf.empty:
        raise ValueError("AOI file contains no geometry.")

    if gdf.crs is None:
        raise ValueError(
            "AOI has no CRS. Define the CRS before continuing."
        )

    gdf = gdf[gdf.geometry.notnull()].copy()

    if gdf.empty:
        raise ValueError("AOI contains no valid geometries.")

    invalid_count = (~gdf.geometry.is_valid).sum()

    if invalid_count > 0:
        logging.warning(
            "Found %d invalid geometries. Attempting repair.",
            invalid_count
        )

        gdf["geometry"] = gdf.geometry.buffer(0)

    gdf = gdf.to_crs(target_crs)

    gdf = gdf.dissolve()

    gdf = gdf.reset_index(drop=True)

    logging.info(
        "AOI loaded successfully. CRS=%s",
        gdf.crs
    )

    return gdf


def save_aoi(gdf, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    gdf.to_file(output_path, driver="GeoJSON")

    logging.info(
        "AOI saved to: %s",
        output_path
    )


def create_aoi_from_bounds(
    min_lon,
    min_lat,
    max_lon,
    max_lat,
    output_path
):
    from shapely.geometry import box

    geometry = box(
        min_lon,
        min_lat,
        max_lon,
        max_lat
    )

    gdf = gpd.GeoDataFrame(
        {"name": ["study_area"]},
        geometry=[geometry],
        crs="EPSG:4326"
    )

    save_aoi(gdf, output_path)

    return gdf
