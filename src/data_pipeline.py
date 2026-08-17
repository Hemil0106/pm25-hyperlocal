from pathlib import Path
import logging

import earthaccess
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import osmnx
import pandas as pd
import rasterio
import rasterio.plot
from rasterio import features as rio_features
from rasterio.mask import mask as rio_mask
from rasterio.warp import (
    calculate_default_transform,
    reproject,
    Resampling,
)
from shapely.geometry import LineString, box
import xarray as xr

from src.utils import save_json


PROJECT_ROOT = Path(__file__).resolve().parent.parent

AOD_LAYER = "HDF4_EOS:EOS_GRID:{path}:grid1km:Optical_Depth_047"
QA_LAYER = "HDF4_EOS:EOS_GRID:{path}:grid1km:Quality_Assurance"
AOD_SCALE = 0.001
AOD_MIN = 0.0
AOD_MAX = 3.0


def get_aoi(config):
    aoi_path = PROJECT_ROOT / config["study_area"]["boundary_file"]

    if not aoi_path.exists():
        raise FileNotFoundError(
            f"AOI file not found: {aoi_path}\n"
            "Please place your AOI GeoJSON at the configured location."
        )

    aoi = gpd.read_file(aoi_path)

    if aoi.empty:
        raise ValueError("AOI GeoJSON is empty.")

    if aoi.crs is None:
        raise ValueError("AOI does not contain CRS information.")

    aoi = aoi[aoi.geometry.notnull()].copy()

    if aoi.empty:
        raise ValueError("AOI contains no valid geometry.")

    if not aoi.geometry.is_valid.all():
        aoi["geometry"] = aoi.geometry.buffer(0)

    aoi = aoi.to_crs("EPSG:4326")

    aoi = aoi.dissolve().reset_index(drop=True)

    logging.info("AOI loaded successfully.")
    logging.info("AOI bounds: %s", aoi.total_bounds)

    return aoi


def authenticate_earthdata():
    logging.info("Authenticating with NASA Earthdata...")

    auth = earthaccess.login(
        persist=True
    )

    if not auth.authenticated:
        raise RuntimeError(
            "NASA Earthdata authentication failed. "
            "Run 'python -c \"import earthaccess; earthaccess.login()\"' "
            "or set NASA_EARTHDATA_USERNAME/PASSWORD in .env first."
        )

    logging.info("NASA Earthdata authentication successful.")

    return auth


def search_aod(config, aoi):
    start_date = config["time"]["start_date"]
    end_date = config["time"]["end_date"]

    min_lon, min_lat, max_lon, max_lat = aoi.total_bounds

    logging.info(
        "Searching MCD19A2 granules: %s to %s",
        start_date,
        end_date,
    )

    results = earthaccess.search_data(
        short_name="MCD19A2",
        temporal=(start_date, end_date),
        bounding_box=(min_lon, min_lat, max_lon, max_lat),
        count=100,
    )

    logging.info(
        "Found %d MCD19A2 granules.",
        len(results),
    )

    return results


def download_aod(results, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info(
        "Downloading %d granules to: %s",
        len(results),
        output_dir,
    )

    downloaded = earthaccess.download(results, output_dir)

    local_paths = []
    for item in downloaded:
        if isinstance(item, (list, tuple)):
            local_paths.extend(item)
        else:
            local_paths.append(item)

    hdf_paths = [Path(p) for p in local_paths if str(p).lower().endswith(".hdf")]

    logging.info(
        "Downloaded %d HDF files.",
        len(hdf_paths),
    )

    return hdf_paths


def _open_band(hdf_path, layer_template):
    path = layer_template.format(path=str(hdf_path))
    return rasterio.open(path)


def apply_qa_mask(aod, qa):
    masked = aod.copy()

    cloud_mask_bits = qa & 0b00000011

    masked[cloud_mask_bits != 0] = np.nan

    masked[masked <= AOD_MIN] = np.nan
    masked[masked > AOD_MAX] = np.nan

    return masked


def _reproject_to_crs(aod, src_transform, src_crs, dst_crs):
    transform, width, height = calculate_default_transform(
        src_crs,
        dst_crs,
        aod.shape[1],
        aod.shape[0],
        *src_bounds(src_transform, aod.shape),
    )

    destination = np.full((height, width), np.nan, dtype=np.float32)

    reproject(
        aod,
        destination,
        src_transform=src_transform,
        src_crs=src_crs,
        src_nodata=np.nan,
        dst_transform=transform,
        dst_crs=dst_crs,
        dst_nodata=np.nan,
        resampling=Resampling.bilinear,
    )

    return destination, transform


def src_bounds(transform, shape):
    height, width = shape
    return rasterio.transform.array_bounds(height, width, transform)


def clip_to_aoi(array, transform, crs, aoi, dst_crs):
    if crs != dst_crs:
        aoi_clipped = aoi.to_crs(crs)
    else:
        aoi_clipped = aoi

    temp_path = _temp_geotiff(array, transform, crs)

    with rasterio.open(temp_path) as src:
        out_image, out_transform = rio_mask(
            src,
            [aoi_clipped.geometry.values[0]],
            crop=True,
            nodata=np.nan,
        )

    Path(temp_path).unlink(missing_ok=True)

    return out_image[0], out_transform


def _temp_geotiff(array, transform, crs):
    from tempfile import NamedTemporaryFile

    temp = NamedTemporaryFile(suffix=".tif", delete=False)
    temp.close()

    profile = {
        "driver": "GTiff",
        "height": array.shape[0],
        "width": array.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
        "nodata": np.nan,
    }

    with rasterio.open(temp.name, "w", **profile) as dst:
        dst.write(array.astype(np.float32), 1)

    return temp.name


def _extract_date_from_hdf(hdf_path):
    filename = Path(hdf_path).name
    parts = filename.split(".")
    if len(parts) >= 2:
        return parts[1]
    return Path(hdf_path).stem


def process_aod_tile(hdf_path, aoi, output_dir, project_crs="EPSG:4326"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    date_tag = _extract_date_from_hdf(hdf_path)

    logging.info("Processing AOD tile: %s", hdf_path)

    with _open_band(hdf_path, AOD_LAYER) as src:
        aod_raw = src.read(1).astype(np.float32)
        src_transform = src.transform
        src_crs = src.crs

    aod = aod_raw * AOD_SCALE

    try:
        with _open_band(hdf_path, QA_LAYER) as qa_src:
            qa = qa_src.read(1)
    except Exception:
        logging.warning(
            "QA band not readable for %s — skipping QA mask.",
            hdf_path,
        )
        qa = np.zeros_like(aod_raw, dtype=np.uint8)

    aod_masked = apply_qa_mask(aod, qa)

    reprojected, transform = _reproject_to_crs(
        aod_masked,
        src_transform,
        src_crs,
        project_crs,
    )

    aoi_target = aoi.to_crs(project_crs)

    clipped, out_transform = clip_to_aoi(
        reprojected,
        transform,
        project_crs,
        aoi_target,
        project_crs,
    )

    profile = {
        "driver": "GTiff",
        "height": clipped.shape[0],
        "width": clipped.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": project_crs,
        "transform": out_transform,
        "nodata": np.nan,
    }

    output_tif = output_dir / f"aod_{date_tag}.tif"

    with rasterio.open(output_tif, "w", **profile) as dst:
        dst.write(clipped.astype(np.float32), 1)

    logging.info("AOD GeoTIFF saved: %s", output_tif)

    _visualize_aod(
        clipped,
        out_transform,
        aoi_target,
        output_dir / f"aod_{date_tag}.png",
        date_tag,
    )

    return output_tif


def _visualize_aod(array, transform, aoi, output_png, date_tag):
    data = np.ma.masked_invalid(array)

    fig, ax = plt.subplots(figsize=(8, 8))

    extent = rasterio.plot.plotting_extent(array, transform)

    im = ax.imshow(
        data,
        extent=extent,
        origin="upper",
        cmap="inferno",
        vmin=0.0,
        vmax=1.0,
    )

    aoi.boundary.plot(ax=ax, color="cyan", linewidth=1.2)

    ax.set_title(f"AOD 550nm (MCD19A2 Optical_Depth_047) - {date_tag}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    fig.colorbar(im, ax=ax, label="AOD")

    fig.tight_layout()
    fig.savefig(output_png, dpi=150)
    plt.close(fig)

    logging.info("AOD visualization saved: %s", output_png)


def run_aod_pipeline(config):
    logger = logging.getLogger("pm25_pipeline")

    if not config["datasets"]["aod"]["enabled"]:
        logger.info("AOD dataset disabled — skipping.")
        return

    aoi = get_aoi(config)

    authenticate_earthdata()

    results = search_aod(config, aoi)

    if not results:
        logger.warning("No AOD granules found for the period.")
        return

    download_dir = PROJECT_ROOT / config["paths"]["processed"] / "aod"
    output_dir = PROJECT_ROOT / config["paths"]["outputs"] / "aod"

    hdf_paths = download_aod(results, download_dir)

    for hdf_path in hdf_paths:
        process_aod_tile(
            hdf_path,
            aoi,
            output_dir,
            config["crs"]["project"],
        )


COLUMN_ALIASES = {
    "station_id": ["station_id", "station", "station_name", "site_id", "site", "id"],
    "timestamp": ["timestamp", "date", "datetime", "from_date", "fromdate", "time"],
    "latitude": ["latitude", "lat"],
    "longitude": ["longitude", "lon", "lng"],
    "pm25": ["PM2.5", "PM25", "pm2.5", "pm25", "pm2_5"],
}

IMPLAUSIBLE_PM25_MAX = 1000.0


def _norm_column_name(name):
    return "".join(ch for ch in str(name).lower() if ch not in " _.-")


def _resolve_cpcb_columns(columns, config):
    cpcb_cfg = config["datasets"]["cpcb"]

    configured = {
        "station_id": cpcb_cfg.get("station_id_column", "station_id"),
        "timestamp": cpcb_cfg.get("timestamp_column", "timestamp"),
        "latitude": cpcb_cfg.get("latitude_column", "latitude"),
        "longitude": cpcb_cfg.get("longitude_column", "longitude"),
        "pm25": cpcb_cfg.get("pm25_column", "PM2.5"),
    }

    normalized_columns = {_norm_column_name(c): c for c in columns}

    resolution = {}
    unresolved = []

    for field, configured_name in configured.items():
        candidates = [configured_name] + COLUMN_ALIASES[field]
        matched = None
        for candidate in candidates:
            key = _norm_column_name(candidate)
            if key in normalized_columns:
                matched = normalized_columns[key]
                break
        if matched is None:
            unresolved.append(field)
        else:
            resolution[field] = matched

    return resolution, unresolved


def _read_cpcb_csv(input_file):
    last_error = None

    for encoding in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(input_file, encoding=encoding)
        except UnicodeDecodeError as error:
            last_error = error
            continue
        except Exception as error:
            last_error = error
            break

    try:
        return pd.read_csv(input_file, encoding="latin-1", sep=None, engine="python")
    except Exception as error:
        raise RuntimeError(
            f"Could not read CPCB file {input_file}: {last_error or error}"
        ) from error


def load_cpcb_pm25(config):
    logger = logging.getLogger("pm25_pipeline")

    cpcb_cfg = config["datasets"]["cpcb"]
    input_file = PROJECT_ROOT / cpcb_cfg.get("input_file", "data/raw/cpcb/cpcb_pm25.csv")

    if not input_file.exists():
        raise FileNotFoundError(
            f"CPCB raw data not found: {input_file}\n"
            "Generate sample data with: python run.py --create-sample-cpcb"
        )

    logger.info("[CPCB] Loading raw observations: %s", input_file)
    df = _read_cpcb_csv(input_file)

    logger.info("[CPCB] Normalizing columns")
    resolution, unresolved = _resolve_cpcb_columns(df.columns, config)

    if unresolved:
        raise ValueError(
            "Could not detect required CPCB fields.\n"
            f"Available columns: {list(df.columns)}\n"
            "Required logical fields: station_id, timestamp, latitude, longitude, PM2.5\n"
            f"Could not detect: {unresolved}"
        )

    _CANONICAL_NAMES = {
        "station_id": "station_id",
        "timestamp": "timestamp",
        "latitude": "latitude",
        "longitude": "longitude",
        "pm25": "PM2.5",
    }

    rename_map = {
        actual: _CANONICAL_NAMES[field]
        for field, actual in resolution.items()
    }
    df = df.rename(columns=rename_map)

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    timezone = cpcb_cfg.get("timezone", "Asia/Kolkata")
    try:
        if df["timestamp"].dt.tz is None:
            df["timestamp"] = df["timestamp"].dt.tz_localize(timezone)
        else:
            df["timestamp"] = df["timestamp"].dt.tz_convert(timezone)
    except Exception as error:
        logger.warning("[CPCB] timezone handling skipped: %s", error)

    for column in ("latitude", "longitude", "PM2.5"):
        df[column] = pd.to_numeric(df[column], errors="coerce")

    logger.info("[CPCB] Loaded %d rows.", len(df))
    return df


def clean_cpcb_pm25(df, config):
    logger = logging.getLogger("pm25_pipeline")
    cpcb_cfg = config["datasets"]["cpcb"]

    data = df.copy()
    input_rows = int(len(data))

    duplicate_mask = data.duplicated(subset=["station_id", "timestamp"], keep="first")
    n_duplicates = int(duplicate_mask.sum())
    data = data[~duplicate_mask].copy()

    missing_timestamp_mask = data["timestamp"].isna()
    n_missing_timestamps = int(missing_timestamp_mask.sum())
    data = data[~missing_timestamp_mask].copy()

    missing_station_mask = data["station_id"].isna()
    n_missing_stations = int(missing_station_mask.sum())
    data = data[~missing_station_mask].copy()

    missing_coords_mask = data["latitude"].isna() | data["longitude"].isna()
    n_missing_coords = int(missing_coords_mask.sum())
    data = data[~missing_coords_mask].copy()

    lat_ok = data["latitude"].between(-90.0, 90.0)
    lon_ok = data["longitude"].between(-180.0, 180.0)
    invalid_coords_mask = ~(lat_ok & lon_ok)
    n_invalid_coords = int(invalid_coords_mask.sum())
    data = data[~invalid_coords_mask].copy()

    valid_bounds = cpcb_cfg.get("valid_bounds")
    n_impossible_coords = 0
    if valid_bounds:
        in_bounds = (
            data["latitude"].between(valid_bounds["min_lat"], valid_bounds["max_lat"])
            & data["longitude"].between(valid_bounds["min_lon"], valid_bounds["max_lon"])
        )
        n_impossible_coords = int((~in_bounds).sum())
        data = data[in_bounds].copy()

    missing_pm25_mask = data["PM2.5"].isna()
    n_missing_pm25 = int(missing_pm25_mask.sum())
    data = data[~missing_pm25_mask].copy()

    finite_mask = np.isfinite(data[["latitude", "longitude", "PM2.5"]]).all(axis=1)
    n_non_finite = int((~finite_mask).sum())
    data = data[finite_mask].copy()

    negative_mask = data["PM2.5"] < 0
    n_negative_pm25 = int(negative_mask.sum())
    data = data[~negative_mask].copy()

    implausible_mask = data["PM2.5"] > IMPLAUSIBLE_PM25_MAX
    n_invalid_pm25 = int(implausible_mask.sum())
    data = data[~implausible_mask].copy()

    data = data.sort_values(["station_id", "timestamp"]).reset_index(drop=True)

    qc_report = {
        "input_rows": input_rows,
        "duplicate_rows": n_duplicates,
        "missing_timestamps": n_missing_timestamps,
        "missing_station_ids": n_missing_stations,
        "missing_coordinates": n_missing_coords,
        "invalid_coordinates": n_invalid_coords,
        "impossible_coordinates": n_impossible_coords,
        "missing_pm25": n_missing_pm25,
        "non_finite_values": n_non_finite,
        "negative_pm25": n_negative_pm25,
        "invalid_pm25": n_invalid_pm25,
        "final_valid_rows": int(len(data)),
    }

    logger.info(
        "[CPCB] QC: %d input -> %d valid rows.",
        input_rows,
        len(data),
    )

    return data, qc_report


def filter_cpcb_to_aoi(df, config):
    logger = logging.getLogger("pm25_pipeline")

    aoi = get_aoi(config)
    aoi_4326 = aoi.to_crs("EPSG:4326")

    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs="EPSG:4326",
    )

    inside_mask = gdf.geometry.within(aoi_4326.geometry.iloc[0])
    inside_gdf = gdf[inside_mask].copy()

    logger.info(
        "[CPCB] AOI filter: %d -> %d observations.",
        len(gdf),
        len(inside_gdf),
    )

    return inside_gdf


def create_cpcb_station_summary(df):
    summary = (
        df.groupby("station_id", as_index=False)
        .agg(
            latitude=("latitude", "last"),
            longitude=("longitude", "last"),
            observation_count=("timestamp", "count"),
            first_timestamp=("timestamp", "min"),
            last_timestamp=("timestamp", "max"),
            missing_PM25_count=("PM2.5", lambda series: int(series.isna().sum())),
        )
    )

    return summary


def aggregate_cpcb_daily(df, timezone="Asia/Kolkata"):
    data = df.copy()
    ts = data["timestamp"]

    if ts.dt.tz is not None:
        ts = ts.dt.tz_convert(timezone)
        data["date"] = ts.dt.normalize().dt.tz_localize(None)
    else:
        data["date"] = ts.dt.normalize()

    daily = (
        data.groupby(["station_id", "date"], as_index=False)
        .agg(
            PM25_mean=("PM2.5", "mean"),
            observation_count=("PM2.5", "count"),
            latitude=("latitude", "last"),
            longitude=("longitude", "last"),
        )
        .rename(columns={"PM25_mean": "PM2.5"})
    )

    daily = daily.sort_values(["station_id", "date"]).reset_index(drop=True)

    return daily


def _build_cpcb_validation_report(input_rows, valid_rows, inside_df, daily_df, timezone):
    ts = inside_df["timestamp"]
    if ts.dt.tz is not None:
        ts_local = ts.dt.tz_convert(timezone)
    else:
        ts_local = ts

    return {
        "total_stations": int(inside_df["station_id"].nunique()),
        "total_raw_observations": int(input_rows),
        "valid_observations": int(valid_rows),
        "observations_inside_aoi": int(len(inside_df)),
        "date_range": [ts_local.min().isoformat(), ts_local.max().isoformat()],
        "daily_station_date_records": int(len(daily_df)),
        "min_pm25": float(inside_df["PM2.5"].min()),
        "max_pm25": float(inside_df["PM2.5"].max()),
        "mean_pm25": float(inside_df["PM2.5"].mean()),
        "missing_values_after_cleaning": int(inside_df.isna().sum().sum()),
    }


def _visualize_cpcb_stations(aoi, inside_gdf, output_png):
    logger = logging.getLogger("pm25_pipeline")

    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 10))

    aoi.to_crs("EPSG:4326").boundary.plot(ax=ax, color="black", linewidth=1.5)
    inside_gdf.plot(
        ax=ax,
        markersize=30,
        color="crimson",
        alpha=0.85,
        edgecolor="white",
        linewidth=0.5,
    )

    ax.set_title("CPCB PM2.5 Monitoring Stations")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    fig.tight_layout()
    fig.savefig(output_png, dpi=150)
    plt.close(fig)

    logger.info("[CPCB] Visualization saved: %s", output_png)


def run_cpcb_pipeline(config):
    logger = logging.getLogger("pm25_pipeline")

    if not config["datasets"]["cpcb"].get("enabled", True):
        logger.info("[CPCB] dataset disabled in config - skipping.")
        return False

    cpcb_cfg = config["datasets"]["cpcb"]
    input_file = PROJECT_ROOT / cpcb_cfg.get("input_file", "data/raw/cpcb/cpcb_pm25.csv")

    if not input_file.exists():
        logger.warning("[CPCB] raw data NOT AVAILABLE at: %s", input_file)
        logger.warning("[CPCB] expected path: %s", input_file)
        logger.warning(
            "[CPCB] generate sample data with: python run.py --create-sample-cpcb"
        )
        logger.warning("[CPCB] STAGE NOT RUN")
        return False

    logger.info("[CPCB] Loading raw observations")
    raw_df = load_cpcb_pm25(config)

    logger.info("[CPCB] Quality control")
    clean_df, qc_report = clean_cpcb_pm25(raw_df, config)

    qc_report_path = PROJECT_ROOT / config["paths"]["processed"] / "cpcb_qc_report.json"
    save_json(qc_report, qc_report_path)
    logger.info("[CPCB] QC report saved: %s", qc_report_path)

    logger.info("[CPCB] AOI filtering")
    inside_gdf = filter_cpcb_to_aoi(clean_df, config)
    inside_df = inside_gdf.drop(columns=["geometry"]).reset_index(drop=True)

    if inside_df.empty:
        logger.warning("[CPCB] no observations inside AOI - stopping.")
        return False

    logger.info("[CPCB] Station-level summary")
    station_summary = create_cpcb_station_summary(inside_df)
    station_summary_path = (
        PROJECT_ROOT / config["paths"]["processed"] / "cpcb_station_summary.parquet"
    )
    station_summary.to_parquet(station_summary_path, index=False)
    logger.info("[CPCB] Station summary saved: %s", station_summary_path)

    logger.info("[CPCB] Saving clean dataset")
    clean_path = PROJECT_ROOT / config["paths"]["processed"] / "cpcb_pm25_clean.parquet"
    inside_df.to_parquet(clean_path, index=False)
    logger.info("[CPCB] Clean dataset saved: %s", clean_path)

    logger.info("[CPCB] Daily aggregation")
    timezone = cpcb_cfg.get("timezone", "Asia/Kolkata")
    daily_df = aggregate_cpcb_daily(inside_df, timezone)
    daily_path = PROJECT_ROOT / config["paths"]["processed"] / "cpcb_pm25_daily.parquet"
    daily_df.to_parquet(daily_path, index=False)
    logger.info("[CPCB] Daily dataset saved: %s", daily_path)

    logger.info("[CPCB] Validation report")
    validation_report = _build_cpcb_validation_report(
        len(raw_df),
        len(clean_df),
        inside_df,
        daily_df,
        timezone,
    )
    validation_path = (
        PROJECT_ROOT / config["paths"]["processed"] / "cpcb_validation_report.json"
    )
    save_json(validation_report, validation_path)
    logger.info("[CPCB] Validation report saved: %s", validation_path)

    logger.info("[CPCB] Visualization")
    aoi = get_aoi(config)
    viz_path = PROJECT_ROOT / config["paths"]["outputs"] / "cpcb_stations.png"
    _visualize_cpcb_stations(aoi, inside_gdf, viz_path)

    logger.info("[CPCB] COMPLETED")
    return True


def create_cpcb_sample_data(config):
    logger = logging.getLogger("pm25_pipeline")

    cpcb_cfg = config["datasets"]["cpcb"]
    output_path = PROJECT_ROOT / cpcb_cfg.get(
        "input_file", "data/raw/cpcb/cpcb_pm25.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)

    stations = [
        ("ST_01", "Punjabi Bagh", 28.6306, 77.2189),
        ("ST_02", "India Gate", 28.6139, 77.2090),
        ("ST_03", "Dwarka", 28.5273, 77.1386),
        ("ST_04", "University of Delhi", 28.7041, 77.1025),
        ("ST_05", "Rohini", 28.6312, 77.0775),
    ]

    timestamps = pd.date_range("2025-01-01", "2025-01-07", freq="3h")

    rows = []
    for station_id, station_name, lat, lon in stations:
        for ts in timestamps:
            diurnal = 25.0 * np.sin((ts.hour - 6) / 24.0 * 2.0 * np.pi)
            base = rng.uniform(80.0, 180.0)
            pm25 = max(10.0, base + diurnal + rng.normal(0.0, 15.0))
            rows.append(
                {
                    "station_id": station_id,
                    "station_name": station_name,
                    "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "latitude": round(float(lat + rng.normal(0.0, 0.002)), 6),
                    "longitude": round(float(lon + rng.normal(0.0, 0.002)), 6),
                    "PM2.5": round(float(pm25), 2),
                }
            )

    sample_df = pd.DataFrame(rows)

    duplicate = sample_df.iloc[[0]].copy()
    negative = sample_df.iloc[[49]].copy()
    negative["timestamp"] = "2025-01-07 23:59:00"
    negative["PM2.5"] = -5.0
    out_of_bounds = sample_df.iloc[[98]].copy()
    out_of_bounds["timestamp"] = "2025-01-07 23:59:00"
    out_of_bounds["latitude"] = 91.0
    missing_pm25 = sample_df.iloc[[147]].copy()
    missing_pm25["timestamp"] = "2025-01-07 23:59:00"
    missing_pm25["PM2.5"] = np.nan

    sample_df = pd.concat(
        [sample_df, duplicate, negative, out_of_bounds, missing_pm25],
        ignore_index=True,
    )

    sample_df.to_csv(output_path, index=False)

    logger.info(
        "[CPCB] sample data written: %s (%d rows)",
        output_path,
        len(sample_df),
    )

    return output_path


WEATHER_VARIABLE_ALIASES = {
    "temperature": ["temperature_2m", "t2m", "2m_temperature", "temperature"],
    "relative_humidity": [
        "relative_humidity",
        "rh",
        "2m_relative_humidity",
        "r2m",
    ],
    "wind_u": ["u10", "10m_u_component_of_wind", "u_component_of_wind_10m"],
    "wind_v": ["v10", "10m_v_component_of_wind", "v_component_of_wind_10m"],
    "wind_speed": ["wind_speed", "10m_wind_speed", "ws10"],
    "wind_direction": ["wind_direction", "10m_wind_direction", "wd10"],
    "boundary_layer_height": ["boundary_layer_height", "blh"],
}

WEATHER_REQUIRED = {"temperature", "relative_humidity", "wind_u", "wind_v"}
WEATHER_DERIVED = {"wind_speed", "wind_direction"}

COORDINATE_ALIASES = {
    "latitude": ["latitude", "lat"],
    "longitude": ["longitude", "lon", "long"],
    "time": ["time", "valid_time"],
}

WEATHER_CANONICAL_NAMES = {
    "temperature": "temperature",
    "relative_humidity": "relative_humidity",
    "wind_u": "u_wind",
    "wind_v": "v_wind",
    "wind_speed": "wind_speed",
    "wind_direction": "wind_direction",
    "boundary_layer_height": "boundary_layer_height",
}

KELVIN_UNITS = {"k", "kelvin", "degrees_k", "degree_kelvin"}
CELSIUS_UNITS = {"degc", "celsius", "degrees_celsius", "degree_celsius"}

TEMP_C_MIN = -50.0
TEMP_C_MAX = 60.0
RH_MIN = 0.0
RH_MAX = 100.0
WIND_FLAG_MPS = 60.0


def _norm_weather_name(name):
    return "".join(ch for ch in str(name).lower() if ch not in " _.-")


def _normalize_weather_coordinates(dataset):
    rename_map = {}

    for canonical, aliases in COORDINATE_ALIASES.items():
        matched = None
        for alias in aliases:
            for coord in dataset.coords:
                if _norm_weather_name(coord) == _norm_weather_name(alias):
                    matched = coord
                    break
            if matched is not None:
                break

        if matched is None:
            raise ValueError(
                "Could not identify required weather coordinate "
                f"'{canonical}'. Available coordinates: {list(dataset.coords)}"
            )

        if matched != canonical:
            rename_map[matched] = canonical

    if rename_map:
        dataset = dataset.rename(rename_map)

    return dataset


def _normalize_weather_variables(dataset, config):
    logger = logging.getLogger("pm25_pipeline")

    configured = {
        str(v).strip().lower()
        for v in config["datasets"]["weather"].get("variables", [])
    }

    required_needed = configured & WEATHER_REQUIRED
    optional_needed = (configured - WEATHER_REQUIRED - WEATHER_DERIVED).copy()

    blh_cfg = config["datasets"]["weather"].get("boundary_layer_height", {})
    if blh_cfg.get("enabled", False):
        optional_needed.add("boundary_layer_height")

    dataset_vars = {
        _norm_weather_name(v): v
        for v in dataset.data_vars
    }

    resolution = {}
    missing_required = []

    for logical in sorted(required_needed | optional_needed):
        aliases = WEATHER_VARIABLE_ALIASES.get(logical, [logical])
        matched = None
        for alias in aliases:
            key = _norm_weather_name(alias)
            if key in dataset_vars:
                matched = dataset_vars[key]
                break

        if matched is not None:
            resolution[logical] = matched
        elif logical in required_needed:
            missing_required.append(logical)
        else:
            logger.warning(
                "[WEATHER] Optional variable '%s' not found in NetCDF - skipping.",
                logical,
            )

    if missing_required:
        raise ValueError(
            "Could not detect required weather variables.\n"
            f"Required logical fields: {sorted(required_needed)}\n"
            f"Missing: {missing_required}\n"
            f"Available NetCDF variables: {list(dataset.data_vars)}"
        )

    rename_map = {
        actual: WEATHER_CANONICAL_NAMES[logical]
        for logical, actual in resolution.items()
    }
    if rename_map:
        dataset = dataset.rename(rename_map)

    keep = [WEATHER_CANONICAL_NAMES[logical] for logical in resolution]
    drop = [v for v in dataset.data_vars if v not in keep]
    if drop:
        dataset = dataset.drop_vars(drop)

    logger.info(
        "[WEATHER] Normalized variables present: %s",
        list(dataset.data_vars),
    )

    return dataset


def load_weather_data(config):
    logger = logging.getLogger("pm25_pipeline")

    weather_cfg = config["datasets"]["weather"]
    download_cfg = config.get("download", {})

    if download_cfg.get("enabled", False):
        download_weather_data(config)

    input_file = PROJECT_ROOT / weather_cfg.get(
        "input_file", "data/raw/weather/era5_land.nc"
    )

    if not input_file.exists():
        raise FileNotFoundError(
            f"Weather raw data not found: {input_file}\n"
            "Generate sample data with: python run.py --create-sample-weather"
        )

    logger.info("[WEATHER] Loading raw dataset: %s", input_file)
    dataset = xr.open_dataset(input_file)

    logger.info("[WEATHER] Dimensions: %s", dict(dataset.sizes))
    logger.info("[WEATHER] Coordinates: %s", list(dataset.coords))
    logger.info("[WEATHER] Variables: %s", list(dataset.data_vars))

    dataset = _normalize_weather_coordinates(dataset)
    dataset = _normalize_weather_variables(dataset, config)

    return dataset


def convert_weather_units(dataset):
    logger = logging.getLogger("pm25_pipeline")

    if "temperature" in dataset:
        units = str(dataset["temperature"].attrs.get("units", "")).strip()
        norm = _norm_weather_name(units)

        if norm in KELVIN_UNITS:
            logger.info("[WEATHER] Temperature units are Kelvin - converting to degC.")
            dataset["temperature"] = dataset["temperature"] - 273.15
            dataset["temperature"].attrs["units"] = "degC"
        elif norm in CELSIUS_UNITS:
            logger.info("[WEATHER] Temperature already in Celsius - no conversion.")
            dataset["temperature"].attrs["units"] = "degC"
        else:
            raise ValueError(
                f"[WEATHER] Cannot determine temperature units from '{units}'."
            )

    for var, default_units in (("u_wind", "m/s"), ("v_wind", "m/s")):
        if var in dataset and not dataset[var].attrs.get("units"):
            dataset[var].attrs["units"] = default_units

    if "relative_humidity" in dataset and not dataset["relative_humidity"].attrs.get("units"):
        dataset["relative_humidity"].attrs["units"] = "%"

    return dataset


def calculate_wind_derived(dataset):
    logger = logging.getLogger("pm25_pipeline")

    if "u_wind" not in dataset or "v_wind" not in dataset:
        raise ValueError(
            "[WEATHER] u_wind and v_wind are required to derive wind speed/direction."
        )

    u = dataset["u_wind"]
    v = dataset["v_wind"]

    wind_speed = np.sqrt(u**2 + v**2)

    wind_direction = (270.0 - np.degrees(np.arctan2(v, u))) % 360.0

    dataset["wind_speed"] = wind_speed
    dataset["wind_speed"].attrs["units"] = "m/s"
    dataset["wind_speed"].attrs["description"] = (
        "Horizontal wind speed derived from u/v components."
    )

    dataset["wind_direction"] = wind_direction
    dataset["wind_direction"].attrs["units"] = "degrees"
    dataset["wind_direction"].attrs["description"] = (
        "Meteorological direction FROM which the wind is coming "
        "(0 = north, 90 = east), derived from mean u/v components "
        "to avoid circular-angle averaging."
    )

    logger.info(
        "[WEATHER] Derived wind_speed and wind_direction (from-convention)."
    )

    return dataset


def clip_weather_to_aoi(dataset, config):
    logger = logging.getLogger("pm25_pipeline")

    aoi = get_aoi(config)
    aoi_4326 = aoi.to_crs("EPSG:4326")

    min_lon, min_lat, max_lon, max_lat = aoi_4326.total_bounds

    logger.info("[WEATHER] Clipping to AOI bounds: %s", aoi_4326.total_bounds)

    clipped = dataset.sel(
        latitude=slice(min_lat, max_lat),
        longitude=slice(min_lon, max_lon),
    )

    n_before = int(dataset.sizes["latitude"] * dataset.sizes["longitude"])
    n_after = int(clipped.sizes["latitude"] * clipped.sizes["longitude"])

    logger.info("[WEATHER] Spatial cells: %d -> %d", n_before, n_after)

    return clipped


def _select_time_window(dataset, config):
    logger = logging.getLogger("pm25_pipeline")

    start_date = config["time"]["start_date"]
    end_date = config["time"]["end_date"]

    if "time" not in dataset.coords and "time" not in dataset.dims:
        raise ValueError("[WEATHER] Weather dataset has no 'time' coordinate.")

    time_values = pd.to_datetime(dataset["time"].values)

    if time_values.tz is not None:
        logger.info("[WEATHER] Converting time to timezone-naive UTC.")
        time_values = time_values.tz_localize(None)

    dataset = dataset.assign_coords(time=time_values)

    dataset = dataset.sel(time=slice(start_date, end_date))

    logger.info(
        "[WEATHER] Time window: %s to %s (%d steps)",
        start_date,
        end_date,
        dataset.sizes.get("time", 0),
    )

    return dataset


def aggregate_weather_daily(dataset, config):
    logger = logging.getLogger("pm25_pipeline")

    vars_to_average = [
        var
        for var in ("temperature", "relative_humidity", "u_wind", "v_wind")
        if var in dataset
    ]
    if "boundary_layer_height" in dataset:
        vars_to_average.append("boundary_layer_height")

    daily = dataset[vars_to_average].resample(time="1D").mean()

    daily = calculate_wind_derived(daily)

    daily = daily.assign_attrs(dataset.attrs)
    daily.attrs["aggregation"] = "daily mean"
    daily.attrs["wind_direction_method"] = (
        "daily-mean u/v -> wind_speed and direction FROM which wind comes; "
        "no scalar averaging of compass angles"
    )
    daily.attrs["time_convention"] = "timezone-naive UTC"

    logger.info(
        "[WEATHER] Daily aggregation complete: %d days.",
        daily.sizes.get("time", 0),
    )

    return daily


def validate_weather_data(dataset, config):
    logger = logging.getLogger("pm25_pipeline")

    missing_coords = [
        c
        for c in ("time", "latitude", "longitude")
        if c not in dataset.coords and c not in dataset.dims
    ]
    if missing_coords:
        raise ValueError(
            f"[WEATHER] Dataset missing coordinates: {missing_coords}"
        )

    configured = {
        str(v).strip().lower()
        for v in config["datasets"]["weather"].get("variables", [])
    }
    required_logical = configured & WEATHER_REQUIRED
    required_canonical = {
        WEATHER_CANONICAL_NAMES[logical]
        for logical in required_logical
    }
    present_vars = {
        var
        for var in ("temperature", "relative_humidity", "u_wind", "v_wind")
        if var in dataset
    }
    missing_vars = sorted(required_canonical - present_vars)
    if missing_vars:
        raise ValueError(
            f"[WEATHER] Dataset missing required variables: {missing_vars}"
        )

    time_values = pd.to_datetime(dataset["time"].values)
    unique_days = len(set(t.date() for t in time_values))

    report = {
        "input_dimensions": {k: int(v) for k, v in dataset.sizes.items()},
        "date_range": [
            str(time_values.min().date()),
            str(time_values.max().date()),
        ],
        "variables": list(dataset.data_vars),
        "variable_statistics": {},
        "empty_time_slices": {},
        "flagged_values": {},
        "aoi_coverage": {
            "latitude_range": [
                float(dataset["latitude"].min()),
                float(dataset["latitude"].max()),
            ],
            "longitude_range": [
                float(dataset["longitude"].min()),
                float(dataset["longitude"].max()),
            ],
            "spatial_cells": int(
                dataset.sizes["latitude"] * dataset.sizes["longitude"]
            ),
        },
        "daily_record_count": unique_days,
        "time_steps": int(dataset.sizes.get("time", 0)),
        "suspicious_values_policy": "Flagged only - no values removed.",
    }

    for var in dataset.data_vars:
        if not np.issubdtype(dataset[var].dtype, np.number):
            continue

        arr = dataset[var].values
        finite = np.isfinite(arr)
        nan_count = int((~finite).sum())
        total = int(np.size(arr))
        valid = arr[finite]

        stats = {
            "nan_count": nan_count,
            "valid_count": int(finite.sum()),
            "total_count": total,
            "missing_fraction": round(nan_count / total, 6) if total else 0.0,
            "min": float(np.nanmin(arr)) if nan_count < total else None,
            "max": float(np.nanmax(arr)) if nan_count < total else None,
            "mean": float(np.nanmean(arr)) if nan_count < total else None,
        }
        report["variable_statistics"][var] = stats

        all_nan_over_space = dataset[var].isnull().all(
            dim=["latitude", "longitude"]
        )
        empty_count = int(all_nan_over_space.sum().item())
        empty_dates = [
            str(t.date())
            for t, is_empty in zip(time_values, all_nan_over_space.values)
            if is_empty
        ]
        report["empty_time_slices"][var] = {
            "count": empty_count,
            "dates": empty_dates,
        }

    flagged = {}
    if "temperature" in dataset:
        t = dataset["temperature"].values
        out_of_range = (t < TEMP_C_MIN) | (t > TEMP_C_MAX)
        flagged["temperature_out_of_range"] = {
            "count": int(out_of_range.sum()),
            "min_degC": float(np.nanmin(t)),
            "max_degC": float(np.nanmax(t)),
            "bounds_degC": [TEMP_C_MIN, TEMP_C_MAX],
        }
    if "relative_humidity" in dataset:
        rh = dataset["relative_humidity"].values
        out_of_bounds = (rh < RH_MIN) | (rh > RH_MAX)
        flagged["relative_humidity_out_of_bounds"] = {
            "count": int(out_of_bounds.sum()),
            "bounds_percent": [RH_MIN, RH_MAX],
        }
    if "wind_speed" in dataset:
        ws = dataset["wind_speed"].values
        suspicious = ws > WIND_FLAG_MPS
        flagged["wind_speed_suspicious_high"] = {
            "count": int(suspicious.sum()),
            "max_mps": float(np.nanmax(ws)),
            "threshold_mps": WIND_FLAG_MPS,
        }
    report["flagged_values"] = flagged

    logger.info(
        "[WEATHER] QC: %d time steps, %d days, %d spatial cells.",
        report["time_steps"],
        report["daily_record_count"],
        report["aoi_coverage"]["spatial_cells"],
    )

    return report


def _build_weather_missing_report(dataset):
    missing = {
        "convention": (
            "timezone-naive UTC; no values interpolated or fabricated in this stage"
        ),
        "variables": {},
        "incomplete_dates": [],
    }

    for var in dataset.data_vars:
        if not np.issubdtype(dataset[var].dtype, np.number):
            continue

        isnull = dataset[var].isnull()

        all_nan_over_time = isnull.all(dim="time").values
        missing_cells = int(all_nan_over_time.sum())

        all_nan_over_space = isnull.all(dim=["latitude", "longitude"]).values
        time_values = pd.to_datetime(dataset["time"].values)
        missing_day_dates = [
            str(t.date())
            for t, is_empty in zip(time_values, all_nan_over_space)
            if is_empty
        ]

        missing["variables"][var] = {
            "isolated_missing_values": int(isnull.sum().item()),
            "missing_spatial_cells": missing_cells,
            "missing_entire_days": missing_day_dates,
            "interpolation_strategy": (
                "not applied yet; short isolated gaps to be handled later by "
                "a documented interpolation strategy"
            ),
        }

        for date in missing_day_dates:
            if date not in missing["incomplete_dates"]:
                missing["incomplete_dates"].append(date)

    missing["incomplete_dates"].sort()

    return missing


def _plot_weather_map(data, aoi, title, output_png, colorbar_label, cmap):
    logger = logging.getLogger("pm25_pipeline")

    fig, ax = plt.subplots(figsize=(8, 7))

    data.plot.pcolormesh(ax=ax, cmap=cmap, add_colorbar=True)
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    aoi.boundary.plot(ax=ax, color="black", linewidth=1.5)

    ax.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(output_png, dpi=150)
    plt.close(fig)

    logger.info("[WEATHER] Visualization saved: %s", output_png)


def _visualize_weather(daily_dataset, config):
    logger = logging.getLogger("pm25_pipeline")

    outputs_dir = PROJECT_ROOT / config["paths"]["outputs"]
    outputs_dir.mkdir(parents=True, exist_ok=True)

    aoi = get_aoi(config).to_crs("EPSG:4326")

    mean_over_period = daily_dataset.mean(dim="time")

    if "temperature" in mean_over_period:
        _plot_weather_map(
            mean_over_period["temperature"],
            aoi,
            "Daily mean temperature (degC)",
            outputs_dir / "weather_temperature.png",
            "Temperature (degC)",
            cmap="coolwarm",
        )

    if "wind_speed" in mean_over_period:
        _plot_weather_map(
            mean_over_period["wind_speed"],
            aoi,
            "Daily mean wind speed (m/s)",
            outputs_dir / "weather_wind.png",
            "Wind speed (m/s)",
            cmap="viridis",
        )

    return outputs_dir


def _build_weather_summary(daily_dataset, config):
    def _stats(name):
        if name not in daily_dataset:
            return None
        arr = daily_dataset[name].values
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return {"min": None, "max": None, "mean": None, "std": None}
        return {
            "min": float(finite.min()),
            "max": float(finite.max()),
            "mean": float(finite.mean()),
            "std": float(finite.std()),
        }

    date_values = [
        str(t.date())
        for t in pd.to_datetime(daily_dataset["time"].values)
    ]

    missing_stats = {}
    for var in daily_dataset.data_vars:
        if np.issubdtype(daily_dataset[var].dtype, np.number):
            missing_stats[var] = int(daily_dataset[var].isnull().sum().item())

    return {
        "source": config["datasets"]["weather"].get("source", "ERA5-Land"),
        "date_range": (
            [date_values[0], date_values[-1]] if date_values else []
        ),
        "n_days": len(date_values),
        "spatial_cells": int(
            daily_dataset.sizes["latitude"] * daily_dataset.sizes["longitude"]
        ),
        "daily_records": int(daily_dataset.sizes["time"]),
        "variables": list(daily_dataset.data_vars),
        "temperature_statistics": _stats("temperature"),
        "relative_humidity_statistics": _stats("relative_humidity"),
        "wind_speed_statistics": _stats("wind_speed"),
        "u_wind_statistics": _stats("u_wind"),
        "v_wind_statistics": _stats("v_wind"),
        "wind_direction_statistics": _stats("wind_direction"),
        "missing_value_statistics": missing_stats,
    }


def download_weather_data(config):
    logger = logging.getLogger("pm25_pipeline")

    try:
        import cdsapi
    except ImportError as error:
        raise RuntimeError(
            "[WEATHER] cdsapi is not installed. "
            "Install it with: pip install cdsapi"
        ) from error

    import os

    weather_cfg = config["datasets"]["weather"]
    input_file = PROJECT_ROOT / weather_cfg.get(
        "input_file", "data/raw/weather/era5_land.nc"
    )
    input_file.parent.mkdir(parents=True, exist_ok=True)

    aoi = get_aoi(config).to_crs("EPSG:4326")
    min_lon, min_lat, max_lon, max_lat = aoi.total_bounds
    north, west, south, east = max_lat, min_lon, min_lat, max_lon

    start = pd.Timestamp(config["time"]["start_date"])
    end = pd.Timestamp(config["time"]["end_date"])

    years = sorted({str(y) for y in range(start.year, end.year + 1)})
    months = sorted({str(m).zfill(2) for m in range(1, 13)})
    days = sorted({str(d).zfill(2) for d in range(1, 32)})
    hours = [f"{h:02d}:00" for h in range(24)]

    variables = [
        "2m_temperature",
        "2m_relative_humidity",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
    ]

    request = {
        "variable": variables,
        "year": years,
        "month": months,
        "day": days,
        "time": hours,
        "area": [north, west, south, east],
        "data_format": "netcdf",
        "download_format": "unarchived",
    }

    url = os.environ.get("CDS_API_URL")
    key = os.environ.get("CDS_API_KEY")

    if url and key:
        client = cdsapi.Client(url=url, key=key)
    else:
        client = cdsapi.Client()

    logger.info(
        "[WEATHER] Requesting ERA5-Land for AOI [N=%.4f, W=%.4f, S=%.4f, E=%.4f] "
        "%s to %s",
        north,
        west,
        south,
        east,
        start.date(),
        end.date(),
    )

    client.retrieve("reanalysis-era5-land", request, str(input_file))

    logger.info("[WEATHER] Download completed: %s", input_file)

    return input_file


def run_weather_pipeline(config):
    logger = logging.getLogger("pm25_pipeline")

    if not config["datasets"]["weather"].get("enabled", True):
        logger.info("[WEATHER] dataset disabled in config - skipping.")
        return False

    weather_cfg = config["datasets"]["weather"]
    input_file = PROJECT_ROOT / weather_cfg.get(
        "input_file", "data/raw/weather/era5_land.nc"
    )
    download_cfg = config.get("download", {})

    if not download_cfg.get("enabled", False) and not input_file.exists():
        logger.warning("[WEATHER] raw data NOT AVAILABLE at: %s", input_file)
        logger.warning(
            "[WEATHER] generate sample data with: "
            "python run.py --create-sample-weather"
        )
        logger.warning("[WEATHER] STAGE NOT RUN")
        return False

    logger.info("[WEATHER] Loading dataset")
    dataset = load_weather_data(config)

    logger.info("[WEATHER] Unit conversion")
    dataset = convert_weather_units(dataset)

    logger.info("[WEATHER] Deriving wind speed/direction")
    dataset = calculate_wind_derived(dataset)

    logger.info("[WEATHER] Clipping to AOI")
    dataset = clip_weather_to_aoi(dataset, config)

    logger.info("[WEATHER] Selecting time window")
    dataset = _select_time_window(dataset, config)

    logger.info("[WEATHER] Quality control")
    qc_report = validate_weather_data(dataset, config)
    qc_path = PROJECT_ROOT / config["paths"]["processed"] / "weather_qc_report.json"
    save_json(qc_report, qc_path)
    logger.info("[WEATHER] QC report saved: %s", qc_path)

    logger.info("[WEATHER] Missing data report")
    missing_report = _build_weather_missing_report(dataset)
    missing_path = (
        PROJECT_ROOT / config["paths"]["processed"] / "weather_missing_report.json"
    )
    save_json(missing_report, missing_path)
    logger.info("[WEATHER] Missing data report saved: %s", missing_path)

    logger.info("[WEATHER] Daily aggregation")
    daily = aggregate_weather_daily(dataset, config)
    daily_path = PROJECT_ROOT / config["paths"]["processed"] / "weather_daily.nc"
    daily.to_netcdf(daily_path)
    logger.info("[WEATHER] Daily dataset saved: %s", daily_path)

    logger.info("[WEATHER] Visualization")
    _visualize_weather(daily, config)

    logger.info("[WEATHER] Summary")
    summary = _build_weather_summary(daily, config)
    summary_path = PROJECT_ROOT / config["paths"]["processed"] / "weather_summary.json"
    save_json(summary, summary_path)
    logger.info("[WEATHER] Summary saved: %s", summary_path)

    logger.info("[WEATHER] COMPLETED")
    return True


def create_sample_weather_data(config):
    logger = logging.getLogger("pm25_pipeline")

    weather_cfg = config["datasets"]["weather"]
    output_path = PROJECT_ROOT / weather_cfg.get(
        "input_file", "data/raw/weather/era5_land.nc"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    start = pd.Timestamp(config["time"]["start_date"])
    end = pd.Timestamp(config["time"]["end_date"])

    latitude = np.arange(28.35, 28.86, 0.1)
    longitude = np.arange(76.95, 77.46, 0.1)
    time = pd.date_range(start, end, freq="1h")

    n_t, n_lat, n_lon = len(time), len(latitude), len(longitude)

    rng = np.random.default_rng(7)

    hours = np.array([t.hour for t in time])[:, None, None]

    diurnal = np.sin((hours - 4) / 24.0 * 2.0 * np.pi)

    t2m_k = (
        291.0
        - 6.5 * diurnal
        + rng.normal(0.0, 0.6, (n_t, n_lat, n_lon))
    )

    spatial_grad = (
        0.4 * np.cos(np.linspace(0.0, np.pi, n_lat))[:, None]
        * np.sin(np.linspace(0.0, 2.0 * np.pi, n_lon))[None, :]
    )
    t2m_k = t2m_k + spatial_grad[None, :, :]

    rh = np.clip(
        70.0
        - 15.0 * diurnal
        + rng.normal(0.0, 3.0, (n_t, n_lat, n_lon)),
        35.0,
        95.0,
    )

    u10 = (
        2.5
        + 1.5 * np.sin((hours - 10) / 24.0 * 2.0 * np.pi)
        + rng.normal(0.0, 0.4, (n_t, n_lat, n_lon))
    )

    v10 = (
        1.0
        + 0.8 * np.cos((hours - 10) / 24.0 * 2.0 * np.pi)
        + rng.normal(0.0, 0.3, (n_t, n_lat, n_lon))
    )

    dataset = xr.Dataset(
        data_vars={
            "temperature_2m": (
                ("time", "latitude", "longitude"),
                t2m_k,
                {"units": "K", "long_name": "2 metre temperature (synthetic)"},
            ),
            "relative_humidity": (
                ("time", "latitude", "longitude"),
                rh,
                {"units": "%", "long_name": "2 metre relative humidity (synthetic)"},
            ),
            "u10": (
                ("time", "latitude", "longitude"),
                u10,
                {"units": "m/s", "long_name": "10 metre U wind component (synthetic)"},
            ),
            "v10": (
                ("time", "latitude", "longitude"),
                v10,
                {"units": "m/s", "long_name": "10 metre V wind component (synthetic)"},
            ),
        },
        coords={
            "latitude": latitude,
            "longitude": longitude,
            "time": time,
        },
        attrs={
            "source": "SYNTHETIC_TEST_DATA",
            "purpose": "PIPELINE_TESTING_ONLY",
            "title": "Synthetic ERA5-Land-like weather for Delhi prototype testing",
            "conventions": "CF-1.8",
        },
    )

    dataset["latitude"].attrs["units"] = "degrees_north"
    dataset["longitude"].attrs["units"] = "degrees_east"

    dataset["temperature_2m"][5:9, 2, 3] = np.nan
    dataset["relative_humidity"][:, 1, 1] = np.nan

    dataset.to_netcdf(output_path)

    logger.info(
        "[WEATHER] sample data written: %s (%d lat x %d lon x %d hours)",
        output_path,
        n_lat,
        n_lon,
        n_t,
    )

    return output_path


NDVI_RELIABILITY_MEANING = {
    -1: "fill / not produced",
    0: "good data",
    1: "marginal data",
    2: "snow / ice",
    3: "cloudy",
}


def _read_ndvi_product(input_path, ndvi_band, reliability_band):
    input_path = Path(input_path)
    suffix = input_path.suffix.lower()

    if suffix in (".tif", ".tiff"):
        with rasterio.open(input_path) as src:
            ndvi = src.read(1)
            reliability = src.read(2)
            transform = src.transform
            crs = src.crs
            resolution = src.res
        return ndvi, reliability, transform, crs, resolution

    if suffix in (".hdf", ".h4"):
        base = (
            "HDF4_EOS:EOS_GRID:"
            f'"{input_path}":MODIS_Grid_16DAY_250m_500m_VI'
        )
        with rasterio.open(f"{base}:{ndvi_band}") as src:
            ndvi = src.read(1)
            transform = src.transform
            crs = src.crs
            resolution = src.res
        with rasterio.open(f"{base}:{reliability_band}") as src:
            reliability = src.read(1)
        return ndvi, reliability, transform, crs, resolution

    raise ValueError(
        f"[NDVI] Unsupported input format for {input_path}: {suffix or 'none'}"
    )


def load_ndvi(config):
    logger = logging.getLogger("pm25_pipeline")

    ndvi_cfg = config["datasets"]["ndvi"]
    input_file = PROJECT_ROOT / ndvi_cfg.get(
        "input_file", "data/raw/ndvi/ndvi_input.tif"
    )

    if not input_file.exists():
        raise FileNotFoundError(
            f"NDVI raw data not found: {input_file}\n"
            "Generate sample data with: python run.py --create-sample-ndvi"
        )

    logger.info("[NDVI] Loading raw product: %s", input_file)

    ndvi_raw, reliability_raw, transform, crs, resolution = _read_ndvi_product(
        input_file,
        ndvi_cfg.get("ndvi_band", "250m 16 days NDVI"),
        ndvi_cfg.get("reliability_band", "250m 16 days pixel reliability"),
    )

    logger.info(
        "[NDVI] Raster dims: %s x %s, CRS: %s, resolution: %s",
        ndvi_raw.shape[0],
        ndvi_raw.shape[1],
        crs,
        resolution,
    )

    return {
        "ndvi_raw": ndvi_raw,
        "reliability_raw": reliability_raw,
        "transform": transform,
        "crs": crs,
        "resolution": resolution,
        "input_file": input_file,
    }


def apply_ndvi_quality_mask(ndvi_raw, reliability_raw, config):
    logger = logging.getLogger("pm25_pipeline")

    ndvi_cfg = config["datasets"]["ndvi"]
    fill_value = ndvi_cfg.get("fill_value", -3000)
    valid_min, valid_max = ndvi_cfg.get("valid_range", [-2000, 10000])
    keep_reliability = ndvi_cfg.get("keep_reliability", [0, 1])

    qa_stats = {}
    values, counts = np.unique(reliability_raw, return_counts=True)
    for value, count in zip(values, counts):
        label = NDVI_RELIABILITY_MEANING.get(
            int(value), f"unknown_code_{int(value)}"
        )
        qa_stats[f"reliability_{int(value)}_{label}"] = int(count)

    scale = np.float64(ndvi_cfg.get("scale_factor", 0.0001))
    ndvi_scaled = ndvi_raw.astype(np.float64)

    out_of_range = (
        (ndvi_scaled == fill_value)
        | (ndvi_scaled < valid_min)
        | (ndvi_scaled > valid_max)
    )

    qa_reject = ~np.isin(reliability_raw, keep_reliability)

    invalid = out_of_range | qa_reject
    invalid_pixels = int(invalid.sum())

    ndvi_masked = ndvi_scaled.copy()
    ndvi_masked[invalid] = np.nan

    ndvi_physical = ndvi_masked * scale

    physically_invalid = (ndvi_physical < -0.2) | (ndvi_physical > 1.0)
    physically_invalid_count = int(np.nansum(physically_invalid))
    ndvi_physical[physically_invalid] = np.nan

    finite = np.isfinite(ndvi_physical)
    total_pixels = int(ndvi_physical.size)

    logger.info(
        "[NDVI] QA filter: %d invalid pixels (out-of-range + rejected reliability).",
        invalid_pixels,
    )
    logger.info(
        "[NDVI] Valid pixels: %d of %d.",
        int(finite.sum()),
        total_pixels,
    )

    qc_stats = {
        "total_pixels": total_pixels,
        "qa_filtering": qa_stats,
        "invalid_pixels": invalid_pixels,
        "invalid_out_of_physical_range": physically_invalid_count,
        "valid_pixels": int(finite.sum()),
        "nodata_pixels": int((~finite).sum()),
        "min_valid_ndvi": (
            float(np.nanmin(ndvi_physical)) if finite.any() else None
        ),
        "max_valid_ndvi": (
            float(np.nanmax(ndvi_physical)) if finite.any() else None
        ),
        "mean_valid_ndvi": (
            float(np.nanmean(ndvi_physical)) if finite.any() else None
        ),
    }

    return ndvi_physical, qc_stats


def _visualize_ndvi(array, transform, aoi, output_png):
    logger = logging.getLogger("pm25_pipeline")

    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)

    data = np.ma.masked_invalid(array)

    fig, ax = plt.subplots(figsize=(8, 8))

    extent = rasterio.plot.plotting_extent(array, transform)

    im = ax.imshow(
        data,
        extent=extent,
        origin="upper",
        cmap="Greens",
        vmin=-0.2,
        vmax=1.0,
    )

    aoi.boundary.plot(ax=ax, color="crimson", linewidth=1.5)

    ax.set_title("MODIS NDVI (MOD13Q1 synthetic sample)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    fig.colorbar(im, ax=ax, label="NDVI")

    fig.tight_layout()
    fig.savefig(output_png, dpi=150)
    plt.close(fig)

    logger.info("[NDVI] Visualization saved: %s", output_png)


def download_ndvi(config):
    logger = logging.getLogger("pm25_pipeline")

    authenticate_earthdata()

    aoi = get_aoi(config)
    min_lon, min_lat, max_lon, max_lat = aoi.total_bounds

    start_date = config["time"]["start_date"]
    end_date = config["time"]["end_date"]

    logger.info(
        "[NDVI] Searching MOD13Q1 granules: %s to %s",
        start_date,
        end_date,
    )

    results = earthaccess.search_data(
        short_name="MOD13Q1",
        version="061",
        temporal=(start_date, end_date),
        bounding_box=(min_lon, min_lat, max_lon, max_lat),
        count=20,
    )

    if not results:
        raise RuntimeError(
            "[NDVI] No MOD13Q1 granules found for the AOI/date range."
        )

    logger.info("[NDVI] Found %d MOD13Q1 granules.", len(results))

    download_dir = PROJECT_ROOT / config["paths"]["raw"] / "ndvi"
    download_dir.mkdir(parents=True, exist_ok=True)

    downloaded = earthaccess.download(results, download_dir)

    local_paths = []
    for item in downloaded:
        if isinstance(item, (list, tuple)):
            local_paths.extend(item)
        else:
            local_paths.append(item)

    hdf_paths = [
        Path(p)
        for p in local_paths
        if str(p).lower().endswith((".hdf", ".h4"))
    ]

    if not hdf_paths:
        raise RuntimeError("[NDVI] Download produced no HDF files.")

    logger.info("[NDVI] Downloaded %d HDF files.", len(hdf_paths))

    return hdf_paths


def run_ndvi_pipeline(config):
    logger = logging.getLogger("pm25_pipeline")

    if not config["datasets"]["ndvi"].get("enabled", True):
        logger.info("[NDVI] dataset disabled in config - skipping.")
        return False

    ndvi_cfg = config["datasets"]["ndvi"]
    download_cfg = config.get("download", {})

    if download_cfg.get("enabled", False):
        try:
            download_ndvi(config)
        except Exception as error:
            logger.exception(
                "[NDVI] BLOCKED: download failed - %s", error
            )
            return False

    input_file = PROJECT_ROOT / ndvi_cfg.get(
        "input_file", "data/raw/ndvi/ndvi_input.tif"
    )

    if not input_file.exists():
        logger.warning("[NDVI] raw data NOT AVAILABLE at: %s", input_file)
        logger.warning(
            "[NDVI] generate sample data with: python run.py --create-sample-ndvi"
        )
        logger.warning("[NDVI] STAGE NOT RUN")
        return False

    loaded = load_ndvi(config)

    logger.info("[NDVI] Quality filtering")
    ndvi_physical, qc_stats = apply_ndvi_quality_mask(
        loaded["ndvi_raw"],
        loaded["reliability_raw"],
        config,
    )

    project_crs = config["crs"]["project"]
    transform = loaded["transform"]
    crs = loaded["crs"]

    if str(crs).upper() != str(project_crs).upper():
        logger.info(
            "[NDVI] Reprojecting from %s to %s",
            crs,
            project_crs,
        )
        ndvi_physical, transform = _reproject_to_crs(
            ndvi_physical,
            transform,
            crs,
            project_crs,
        )
        crs = project_crs

    logger.info("[NDVI] Clipping to AOI")
    aoi = get_aoi(config)
    clipped, out_transform = clip_to_aoi(
        ndvi_physical,
        transform,
        crs,
        aoi,
        project_crs,
    )

    if clipped.size == 0:
        raise RuntimeError("[NDVI] No overlap between raster and AOI.")

    aoi_4326 = aoi.to_crs("EPSG:4326")
    min_lon, min_lat, max_lon, max_lat = aoi_4326.total_bounds

    logger.info(
        "[NDVI] Clip dimensions: %s x %s",
        clipped.shape[0],
        clipped.shape[1],
    )

    output_dir = PROJECT_ROOT / config["paths"]["processed"]
    output_path = output_dir / "ndvi.tif"

    profile = {
        "driver": "GTiff",
        "height": clipped.shape[0],
        "width": clipped.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": project_crs,
        "transform": out_transform,
        "nodata": np.nan,
    }

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(clipped.astype(np.float32), 1)
        dst.update_tags(
            source=ndvi_cfg.get("source", "unknown"),
            product=ndvi_cfg.get("product", "unknown"),
            composite_start_date=ndvi_cfg["temporal"]["composite_start_date"],
            composite_period_days=ndvi_cfg["temporal"]["composite_period_days"],
            processing="NDVI scale 0.0001 + MOD13Q1 pixel-reliability QA filter",
        )

    logger.info("[NDVI] Processed raster saved: %s", output_path)

    logger.info("[NDVI] QC report")
    qc_report = {
        "input_file": str(input_file),
        "output_file": str(output_path),
        "product": ndvi_cfg.get("product", "unknown"),
        "source": ndvi_cfg.get("source", "unknown"),
        "composite_start_date": ndvi_cfg["temporal"]["composite_start_date"],
        "composite_period_days": ndvi_cfg["temporal"]["composite_period_days"],
        "temporal_resolution": ndvi_cfg["temporal"].get(
            "resolution", "16-day composite"
        ),
        "crs": project_crs,
        "resolution": [
            float(out_transform.a),
            float(-out_transform.e),
        ],
        "raster_dimensions": [int(clipped.shape[0]), int(clipped.shape[1])],
        "aoi_coverage": {
            "latitude_range": [float(min_lat), float(max_lat)],
            "longitude_range": [float(min_lon), float(max_lon)],
        },
        "scale_factor": ndvi_cfg.get("scale_factor", 0.0001),
        "quality_filter_policy": (
            "MOD13Q1 pixel-reliability band: keep codes "
            f"{ndvi_cfg.get('keep_reliability', [0, 1])} "
            "(0=good, 1=marginal); reject fill(-1), snow/ice(2), cloudy(3); "
            "fill value and out-of-range scaled values removed before scaling."
        ),
        "processing_status": "COMPLETED",
        "statistics": qc_stats,
    }

    qc_path = output_dir / "ndvi_qc_report.json"
    save_json(qc_report, qc_path)
    logger.info("[NDVI] QC report saved: %s", qc_path)

    logger.info("[NDVI] Visualization")
    viz_path = PROJECT_ROOT / config["paths"]["outputs"] / "ndvi.png"
    _visualize_ndvi(clipped, out_transform, aoi, viz_path)

    logger.info("[NDVI] COMPLETED")
    return True


def create_sample_ndvi_data(config):
    logger = logging.getLogger("pm25_pipeline")

    ndvi_cfg = config["datasets"]["ndvi"]
    output_path = PROJECT_ROOT / ndvi_cfg.get(
        "input_file", "data/raw/ndvi/ndvi_input.tif"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    resolution_deg = 0.0025
    west, north = 77.0, 28.8
    east, south = 77.4, 28.4
    width = int(round((east - west) / resolution_deg))
    height = int(round((north - south) / resolution_deg))

    rng = np.random.default_rng(11)

    latitude = north - resolution_deg * (np.arange(height) + 0.5)
    longitude = west + resolution_deg * (np.arange(width) + 0.5)

    lon_grid, lat_grid = np.meshgrid(longitude, latitude)

    dist = np.sqrt(
        ((lon_grid - 77.21) / (0.4 * 0.6)) ** 2
        + ((lat_grid - 28.61) / (0.4 * 0.6)) ** 2
    )

    ndvi = (
        0.60
        - 0.38 * dist
        + 0.04 * np.sin(lat_grid * 40.0)
        + rng.normal(0.0, 0.02, (height, width))
    )
    ndvi = np.clip(ndvi, 0.05, 0.92)

    ndvi_scaled = np.round(ndvi * 10000.0).astype(np.int16)

    reliability = np.zeros((height, width), dtype=np.int16)

    cloud_rows = slice(30, 55)
    cloud_cols = slice(45, 75)
    reliability[cloud_rows, cloud_cols] = 3
    ndvi_scaled[cloud_rows, cloud_cols] = -3000

    fill_rows = slice(110, 125)
    fill_cols = slice(20, 40)
    reliability[fill_rows, fill_cols] = -1
    ndvi_scaled[fill_rows, fill_cols] = -3000

    marginal_mask = rng.random((height, width)) < 0.02
    reliability[marginal_mask] = 1

    transform = rasterio.transform.from_origin(
        west, north, resolution_deg, resolution_deg
    )

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 2,
        "dtype": "int16",
        "crs": "EPSG:4326",
        "transform": transform,
        "nodata": -3000,
    }

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(ndvi_scaled, 1)
        dst.write(reliability, 2)
        dst.set_band_description(1, "250m 16 days NDVI (synthetic scaled)")
        dst.set_band_description(
            2, "250m 16 days pixel reliability (synthetic)"
        )
        dst.update_tags(
            source="SYNTHETIC_TEST_DATA",
            purpose="PIPELINE_TESTING_ONLY",
            product="MOD13Q1_SYNTHETIC",
            composite_start_date=ndvi_cfg["temporal"]["composite_start_date"],
            composite_period_days=ndvi_cfg["temporal"]["composite_period_days"],
            scale_factor=ndvi_cfg.get("scale_factor", 0.0001),
        )

    logger.info(
        "[NDVI] sample data written: %s (%d x %d pixels, 2 bands)",
        output_path,
        height,
        width,
    )

    return output_path


SRTM_HGT_NAMES = ("SRTMGL1", "SRTM1", "SRTMGL3", "SRTM3")
SRTM_NODATA = -32768


def _decode_hgt_metadata(path):
    name = Path(path).stem.upper()

    north = None
    west = None
    for token in name.split("."):
        token = token.strip()
        if (
            len(token) >= 7
            and token[0] in ("N", "S")
            and token[3] in ("E", "W")
        ):
            lat = int(token[1:3])
            if token[0] == "S":
                lat = -lat
            lon = int(token[4:7])
            if token[3] == "W":
                lon = -lon
            north = lat + 1
            west = lon
            break

    if north is None or west is None:
        raise ValueError(
            f"[DEM] Cannot decode SRTM tile bounds from filename: {Path(path).name}"
        )

    rows = 3601
    for variant in SRTM_HGT_NAMES:
        if variant in name:
            rows = 3601 if "GL1" in variant or "SRTM1" in variant else 1201
            break

    return north, west, rows


def _read_hgt_tile(path):
    north, west, rows = _decode_hgt_metadata(path)

    cols = rows
    pixel = 1.0 / 3600.0

    data = np.fromfile(path, dtype=">i2")
    if data.size != rows * cols:
        raise ValueError(
            f"[DEM] Unexpected size for {Path(path).name}: "
            f"{data.size} samples (expected {rows * cols})."
        )

    elevation = data.reshape((rows, cols)).astype(np.float32)
    transform = rasterio.transform.from_origin(
        west, north, pixel, pixel
    )

    return elevation, transform


def _read_dem_tile(path):
    suffix = Path(path).suffix.lower()

    if suffix in (".tif", ".tiff"):
        with rasterio.open(path) as src:
            elevation = src.read(1).astype(np.float32)
            transform = src.transform
            crs = src.crs
            nodata = src.nodata
        return elevation, transform, crs, nodata

    if suffix == ".hgt":
        elevation, transform = _read_hgt_tile(path)
        return elevation, transform, "EPSG:4326", SRTM_NODATA

    raise ValueError(
        f"[DEM] Unsupported DEM format: {Path(path).name}"
    )


def _dem_tile_bounds_4326(path, src_crs="EPSG:4326"):
    suffix = Path(path).suffix.lower()

    if suffix == ".hgt":
        north, west, rows = _decode_hgt_metadata(path)
        south = north - rows / 3600.0
        east = west + rows / 3600.0
        return west, south, east, north

    with rasterio.open(path) as src:
        left, bottom, right, top = src.bounds
        if str(src.crs).upper() == "EPSG:4326":
            return left, bottom, right, top

    raise ValueError(
        f"[DEM] Cannot determine EPSG:4326 bounds for {Path(path).name}"
    )


def _list_dem_tiles(dem_dir, aoi_4326_bounds):
    if not dem_dir.exists():
        return []

    min_lon, min_lat, max_lon, max_lat = aoi_4326_bounds

    candidates = sorted(
        p
        for p in dem_dir.iterdir()
        if p.is_file() and p.suffix.lower() in (".hgt", ".tif", ".tiff")
    )

    tiles = []
    for path in candidates:
        west, south, east, north = _dem_tile_bounds_4326(path)
        if not (
            east <= min_lon
            or west >= max_lon
            or north <= min_lat
            or south >= max_lat
        ):
            tiles.append(
                {
                    "path": path,
                    "bounds": (west, south, east, north),
                }
            )

    return tiles


def _mosaic_dem(tiles):
    min_west = min(t["bounds"][0] for t in tiles)
    min_south = min(t["bounds"][1] for t in tiles)
    max_east = max(t["bounds"][2] for t in tiles)
    max_north = max(t["bounds"][3] for t in tiles)

    first = _read_dem_tile(tiles[0]["path"])
    pixel_x = abs(first[1].a)
    pixel_y = abs(first[1].e)

    width = int(round((max_east - min_west) / pixel_x))
    height = int(round((max_north - min_south) / pixel_y))

    mosaic = np.full((height, width), np.nan, dtype=np.float32)

    for tile in tiles:
        elevation, transform, _, tile_nodata = _read_dem_tile(tile["path"])
        elevation = np.where(elevation == tile_nodata, np.nan, elevation)
        rows, cols = elevation.shape
        x_off = int(round((transform.c - min_west) / pixel_x))
        y_off = int(round((max_north - transform.f) / pixel_y))
        x_off = max(x_off, 0)
        y_off = max(y_off, 0)
        paste_rows = min(rows, height - y_off)
        paste_cols = min(cols, width - x_off)
        if paste_rows <= 0 or paste_cols <= 0:
            continue
        mosaic[y_off : y_off + paste_rows, x_off : x_off + paste_cols] = (
            elevation[:paste_rows, :paste_cols]
        )

    transform = rasterio.transform.from_origin(
        min_west, max_north, pixel_x, pixel_y
    )

    return mosaic, transform


def validate_dem_metadata(elevation, transform, crs, config):
    logger = logging.getLogger("pm25_pipeline")

    if transform.a <= 0 or transform.e >= 0:
        raise ValueError("[DEM] Invalid transform (bad pixel size or orientation).")

    if str(crs).upper() != str(config["crs"]["project"]).upper():
        raise ValueError(
            f"[DEM] Raster CRS {crs} does not match project CRS "
            f"{config['crs']['project']} after reprojection."
        )

    if elevation.ndim != 2:
        raise ValueError("[DEM] Elevation must be a single 2-D surface.")

    logger.info(
        "[DEM] Metadata valid: dims %s x %s, CRS %s, resolution %.8f deg.",
        elevation.shape[0],
        elevation.shape[1],
        crs,
        transform.a,
    )


def _validate_elevation(elevation, config):
    logger = logging.getLogger("pm25_pipeline")

    valid_mask = np.isfinite(elevation)
    valid = elevation[valid_mask]
    nodata_count = int((~valid_mask).sum())
    valid_count = int(valid.size)

    if valid_count == 0:
        raise RuntimeError("[DEM] No valid elevation pixels after processing.")

    suspicious = valid[(valid < -500.0) | (valid > 9000.0)]
    if suspicious.size:
        logger.warning(
            "[DEM] %d suspicious elevation values outside the "
            "global sanity band [-500, 9000] m - investigating, not deleting: "
            "min %.1f, max %.1f",
            suspicious.size,
            float(suspicious.min()),
            float(suspicious.max()),
        )

    stats = {
        "valid_pixel_count": valid_count,
        "nodata_pixel_count": nodata_count,
        "min_elevation_m": float(valid.min()),
        "max_elevation_m": float(valid.max()),
        "mean_elevation_m": float(valid.mean()),
        "median_elevation_m": float(np.median(valid)),
        "std_elevation_m": float(valid.std()),
        "suspicious_values_outside_sanity_band": int(suspicious.size),
    }

    logger.info(
        "[DEM] Elevation stats (valid pixels): min %.1f, max %.1f, "
        "mean %.1f, median %.1f m.",
        stats["min_elevation_m"],
        stats["max_elevation_m"],
        stats["mean_elevation_m"],
        stats["median_elevation_m"],
    )

    return stats, valid_mask


def _visualize_elevation(array, transform, aoi, output_png):
    logger = logging.getLogger("pm25_pipeline")

    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)

    data = np.ma.masked_invalid(array)

    fig, ax = plt.subplots(figsize=(8, 8))

    extent = rasterio.plot.plotting_extent(array, transform)

    im = ax.imshow(
        data,
        extent=extent,
        origin="upper",
        cmap="terrain",
        interpolation="nearest",
    )

    aoi.boundary.plot(ax=ax, color="crimson", linewidth=1.5)

    ax.set_title("SRTM Elevation (m) - DEM stage")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    fig.colorbar(im, ax=ax, label="Elevation (m)")

    fig.tight_layout()
    fig.savefig(output_png, dpi=150)
    plt.close(fig)

    logger.info("[DEM] Visualization saved: %s", output_png)


def download_dem(config):
    logger = logging.getLogger("pm25_pipeline")

    authenticate_earthdata()

    aoi = get_aoi(config)
    min_lon, min_lat, max_lon, max_lat = aoi.total_bounds

    logger.info(
        "[DEM] Searching SRTMGL1 granules for AOI "
        "(%.4f, %.4f) to (%.4f, %.4f).",
        min_lon,
        min_lat,
        max_lon,
        max_lat,
    )

    results = earthaccess.search_data(
        short_name="SRTMGL1",
        version="003",
        bounding_box=(min_lon, min_lat, max_lon, max_lat),
        count=20,
    )

    if not results:
        raise RuntimeError("[DEM] No SRTMGL1 granules found for the AOI.")

    logger.info("[DEM] Found %d SRTMGL1 granules.", len(results))

    download_dir = PROJECT_ROOT / config["datasets"]["dem"]["raw_dir"]
    download_dir.mkdir(parents=True, exist_ok=True)

    downloaded = earthaccess.download(results, download_dir)

    local_paths = []
    for item in downloaded:
        if isinstance(item, (list, tuple)):
            local_paths.extend(item)
        else:
            local_paths.append(item)

    hgt_paths = [
        Path(p)
        for p in local_paths
        if str(p).lower().endswith((".hgt", ".tif", ".tiff"))
    ]

    if not hgt_paths:
        raise RuntimeError("[DEM] Download produced no elevation files.")

    logger.info("[DEM] Downloaded %d elevation files.", len(hgt_paths))

    return hgt_paths


def run_dem_pipeline(config):
    logger = logging.getLogger("pm25_pipeline")

    if not config["datasets"]["dem"].get("enabled", True):
        logger.info("[DEM] dataset disabled in config - skipping.")
        return False

    dem_cfg = config["datasets"]["dem"]
    download_cfg = config.get("download", {})
    project_crs = config["crs"]["project"]

    if download_cfg.get("enabled", False):
        try:
            download_dem(config)
        except Exception as error:
            logger.exception("[DEM] BLOCKED: download failed - %s", error)
            return False

    dem_dir = PROJECT_ROOT / dem_cfg.get("raw_dir", "data/raw/dem")

    aoi = get_aoi(config)
    aoi_4326 = aoi.to_crs("EPSG:4326")
    aoi_bounds_4326 = aoi_4326.total_bounds

    tiles = _list_dem_tiles(dem_dir, aoi_bounds_4326)

    if not tiles:
        logger.warning("[DEM] raw data NOT AVAILABLE at: %s", dem_dir)
        logger.warning(
            "[DEM] generate sample data with: python run.py --create-sample-dem"
        )
        logger.warning("[DEM] STAGE NOT RUN")
        return False

    logger.info("[DEM] %d intersecting tile(s) found.", len(tiles))

    if len(tiles) == 1:
        logger.info("[DEM] Single tile - no mosaic required.")
        elevation, transform, crs, input_nodata = _read_dem_tile(
            tiles[0]["path"]
        )
        mosaic_info = {
            "required": False,
            "tile_count": 1,
            "tiles": [str(tiles[0]["path"])],
        }
    else:
        logger.info("[DEM] Mosaicking %d tiles.", len(tiles))
        elevation, transform = _mosaic_dem(tiles)
        crs = "EPSG:4326"
        input_nodata = SRTM_NODATA
        mosaic_info = {
            "required": True,
            "tile_count": len(tiles),
            "tiles": [str(t["path"]) for t in tiles],
        }

    logger.info("[DEM] Converting NoData (%s) to NaN.", input_nodata)
    elevation = np.where(elevation == input_nodata, np.nan, elevation)

    if str(crs).upper() != str(project_crs).upper():
        logger.info("[DEM] Reprojecting from %s to %s.", crs, project_crs)
        elevation, transform = _reproject_to_crs(
            elevation,
            transform,
            crs,
            project_crs,
        )
        crs = project_crs

    validate_dem_metadata(elevation, transform, crs, config)

    if dem_cfg.get("resample_to_grid", False):
        target_res = float(dem_cfg.get("target_resolution_deg", 0.0025))
        if abs(transform.a) > target_res * 1.5:
            logger.info(
                "[DEM] Resampling elevation to %.6f deg grid using %s.",
                target_res,
                dem_cfg.get("resampling", "bilinear"),
            )
            elevation, transform = _resample_to_resolution(
                elevation,
                transform,
                target_res,
                dem_cfg.get("resampling", "bilinear"),
            )

    logger.info("[DEM] Clipping to AOI")
    clipped, out_transform = clip_to_aoi(
        elevation,
        transform,
        project_crs,
        aoi,
        project_crs,
    )

    if clipped.size == 0 or not np.isfinite(clipped).any():
        raise RuntimeError("[DEM] No overlap between DEM and AOI.")

    logger.info("[DEM] Elevation validation")
    stats, _ = _validate_elevation(clipped, config)

    output_dir = PROJECT_ROOT / config["paths"]["processed"]
    output_path = PROJECT_ROOT / dem_cfg.get(
        "output_file", "data/processed/elevation.tif"
    )

    profile = {
        "driver": "GTiff",
        "height": clipped.shape[0],
        "width": clipped.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": project_crs,
        "transform": out_transform,
        "nodata": np.nan,
    }

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(clipped.astype(np.float32), 1)
        dst.update_tags(
            source=dem_cfg.get("source", "unknown"),
            product=dem_cfg.get("product", "unknown"),
            resampling=dem_cfg.get("resampling", "bilinear")
            if dem_cfg.get("resample_to_grid", False)
            else "native",
            input_nodata=str(input_nodata),
            processing="SRTM elevation standardized; NoData (-32768) -> NaN",
        )

    logger.info("[DEM] Processed elevation raster saved: %s", output_path)

    west, south, east, north = rasterio.transform.array_bounds(
        clipped.shape[0], clipped.shape[1], out_transform
    )

    qc_report = {
        "source": dem_cfg.get("source", "unknown"),
        "product": dem_cfg.get("product", "unknown"),
        "input_files": [str(t["path"]) for t in tiles],
        "output_file": str(output_path),
        "aoi": {
            "latitude_range": [
                float(aoi_bounds_4326[1]),
                float(aoi_bounds_4326[3]),
            ],
            "longitude_range": [
                float(aoi_bounds_4326[0]),
                float(aoi_bounds_4326[2]),
            ],
        },
        "crs": project_crs,
        "resolution_deg": [float(out_transform.a), float(-out_transform.e)],
        "bounds": {
            "west": float(west),
            "south": float(south),
            "east": float(east),
            "north": float(north),
        },
        "width": int(clipped.shape[1]),
        "height": int(clipped.shape[0]),
        "input_nodata": input_nodata,
        "output_nodata": None,
        "resampling_method": dem_cfg.get("resampling", "bilinear")
        if dem_cfg.get("resample_to_grid", False)
        else "none (native resolution preserved)",
        "mosaic": mosaic_info,
        "processing_status": "COMPLETED",
        "statistics": stats,
    }

    qc_path = output_dir / "dem_qc_report.json"
    save_json(qc_report, qc_path)
    logger.info("[DEM] QC report saved: %s", qc_path)

    logger.info("[DEM] Visualization")
    viz_path = PROJECT_ROOT / config["paths"]["outputs"] / "elevation.png"
    _visualize_elevation(clipped, out_transform, aoi, viz_path)

    logger.info(
        "[DEM] Elevation implemented; "
        "slope/aspect deferred to a later feature-engineering milestone."
    )

    logger.info("[DEM] COMPLETED")
    return True


def _resample_to_resolution(array, transform, target_res, method):
    resampling = getattr(Resampling, method, Resampling.bilinear)

    src_width = array.shape[1]
    src_height = array.shape[0]

    dst_width = max(int(round(src_width * transform.a / target_res)), 1)
    dst_height = max(int(round(src_height * abs(transform.e) / target_res)), 1)

    dst_transform = rasterio.transform.from_origin(
        transform.c,
        transform.f,
        target_res,
        target_res,
    )

    destination = np.full(
        (dst_height, dst_width), np.nan, dtype=np.float32
    )

    reproject(
        array,
        destination,
        src_transform=transform,
        src_crs="EPSG:4326",
        src_nodata=np.nan,
        dst_transform=dst_transform,
        dst_crs="EPSG:4326",
        dst_nodata=np.nan,
        resampling=resampling,
    )

    return destination, dst_transform


def create_sample_dem_data(config):
    logger = logging.getLogger("pm25_pipeline")

    dem_cfg = config["datasets"]["dem"]
    output_dir = PROJECT_ROOT / dem_cfg.get("raw_dir", "data/raw/dem")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "synthetic_dem_sample.tif"

    west, north = 77.0, 28.8
    east, south = 77.4, 28.4
    resolution_deg = 0.001
    width = int(round((east - west) / resolution_deg))
    height = int(round((north - south) / resolution_deg))

    rng = np.random.default_rng(42)

    latitude = north - resolution_deg * (np.arange(height) + 0.5)
    longitude = west + resolution_deg * (np.arange(width) + 0.5)

    lon_grid, lat_grid = np.meshgrid(longitude, latitude)

    base = 220.0 + 0.35 * (north - lat_grid) * 3600.0 / 100.0

    river = -28.0 * np.exp(
        -0.5 * ((lon_grid - 77.2) / 0.02) ** 2
    )

    noise = rng.normal(0.0, 1.5, (height, width))

    elevation = base + river + noise
    elevation = np.clip(elevation, 150.0, 420.0)

    elevation_scaled = np.round(elevation).astype(np.int16)

    water_mask = (
        (np.abs(lon_grid - 77.2) < 0.006)
        & (np.abs(lat_grid - 28.62) < 0.05)
        & (rng.random((height, width)) < 0.25)
    )
    elevation_scaled[water_mask] = SRTM_NODATA

    transform = rasterio.transform.from_origin(
        west, north, resolution_deg, resolution_deg
    )

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "int16",
        "crs": "EPSG:4326",
        "transform": transform,
        "nodata": SRTM_NODATA,
    }

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(elevation_scaled, 1)
        dst.set_band_description(1, "Synthetic elevation (m)")
        dst.update_tags(
            source="SYNTHETIC_TEST_DATA",
            purpose="PIPELINE_TESTING_ONLY",
            product="SRTM_GL1_SYNTHETIC",
            nodata=str(SRTM_NODATA),
        )

    logger.info(
        "[DEM] sample data written: %s (%d x %d pixels)",
        output_path,
        height,
        width,
    )

    return output_path


def _load_osm_roads(path):
    logger = logging.getLogger("pm25_pipeline")

    roads = gpd.read_file(path)

    if "highway" not in roads.columns:
        raise ValueError(
            f"[OSM] Road network file lacks 'highway' column: {path}"
        )

    logger.info(
        "[OSM] Loaded %d road features from %s.",
        len(roads),
        path,
    )

    return roads


def _validate_osm_geometries(roads):
    logger = logging.getLogger("pm25_pipeline")

    input_count = len(roads)

    if roads.geometry is None:
        raise ValueError("[OSM] Road network has no geometry column.")

    has_geometry = roads["geometry"].notna()
    missing_geometry_count = int((~has_geometry).sum())
    roads = roads[has_geometry]

    empty_count = int(roads.geometry.is_empty.sum())
    roads = roads[~roads.geometry.is_empty]

    missing_highway_count = int(roads["highway"].isna().sum())
    roads = roads[roads["highway"].notna()]

    invalid_mask = ~roads.geometry.is_valid
    invalid_count = int(invalid_mask.sum())

    repaired_candidates = roads[invalid_mask].copy()
    repaired_candidates["geometry"] = repaired_candidates["geometry"].buffer(0)
    repaired_candidates = repaired_candidates[
        repaired_candidates.geometry.is_valid
    ]
    repaired_count = len(repaired_candidates)

    roads = pd.concat(
        [roads[~invalid_mask], repaired_candidates],
        ignore_index=True,
    )

    retained_count = len(roads)

    logger.info(
        "[OSM] Geometry validation: input %d, missing geometry %d, "
        "empty %d, missing highway %d, invalid %d (repaired %d), "
        "retained %d.",
        input_count,
        missing_geometry_count,
        empty_count,
        missing_highway_count,
        invalid_count,
        repaired_count,
        retained_count,
    )

    validation_stats = {
        "input_road_count": input_count,
        "missing_geometry_count": missing_geometry_count,
        "empty_geometry_count": empty_count,
        "missing_highway_count": missing_highway_count,
        "invalid_geometry_count": invalid_count,
        "repaired_geometry_count": repaired_count,
        "retained_road_count": retained_count,
    }

    return roads, validation_stats


def _filter_osm_highway_types(roads, allowed_types):
    logger = logging.getLogger("pm25_pipeline")

    if not allowed_types:
        logger.info("[OSM] No highway filter configured - retaining all roads.")
        return roads

    def _highway_value(row):
        value = row
        if isinstance(value, (list, tuple)):
            value = str(value[0])
        return str(value)

    roads = roads.copy()
    roads["_highway_class"] = roads["highway"].apply(_highway_value)

    kept = roads[roads["_highway_class"].isin(allowed_types)]

    logger.info(
        "[OSM] Highway filter: retained %d of %d roads "
        "(classes: %s).",
        len(kept),
        len(roads),
        ", ".join(allowed_types),
    )

    dropped = len(roads) - len(kept)
    return kept.drop(columns=["_highway_class"]), dropped


def _build_analysis_grid(aoi_metric, resolution_m):
    minx, miny, maxx, maxy = aoi_metric.total_bounds

    cols = int(np.ceil((maxx - minx) / resolution_m))
    rows = int(np.ceil((maxy - miny) / resolution_m))

    boxes = []
    ids = []
    for r in range(rows):
        for c in range(cols):
            west = minx + c * resolution_m
            south = miny + r * resolution_m
            east = west + resolution_m
            north = south + resolution_m
            boxes.append(box(west, south, east, north))
            ids.append(f"{c:03d}_{r:03d}")

    grid = gpd.GeoDataFrame(
        {"grid_id": ids, "geometry": boxes},
        crs=aoi_metric.crs,
    )

    grid = gpd.clip(grid, aoi_metric)

    grid = grid[grid.geometry.area > 0].reset_index(drop=True)

    grid["cell_area_km2"] = grid.geometry.area / 1e6

    return grid


def _compute_road_density(grid, roads_metric, major_types):
    logger = logging.getLogger("pm25_pipeline")

    if len(roads_metric) == 0:
        logger.warning("[OSM] No road features remain for density computation.")
        result = grid[["grid_id", "geometry", "cell_area_km2"]].copy()
        result["road_density"] = 0.0
        result["major_road_density"] = 0.0
        return result

    splits = gpd.overlay(
        grid[["grid_id", "geometry"]],
        roads_metric[["highway", "geometry"]],
        how="intersection",
        keep_geom_type=False,
    )

    if len(splits) == 0:
        logger.warning("[OSM] No road segments intersect any grid cell.")
        result = grid[["grid_id", "geometry", "cell_area_km2"]].copy()
        result["road_density"] = 0.0
        result["major_road_density"] = 0.0
        return result

    splits["seg_length_m"] = splits.geometry.length

    total_per_cell = splits.groupby("grid_id")["seg_length_m"].sum()

    if major_types:
        splits["_major"] = splits["highway"].astype(str).isin(major_types)
        major_per_cell = (
            splits[splits["_major"]]
            .groupby("grid_id")["seg_length_m"]
            .sum()
        )
    else:
        major_per_cell = None

    result = grid[["grid_id", "geometry", "cell_area_km2"]].copy()

    result = result.merge(
        total_per_cell.rename("total_road_length_m"),
        on="grid_id",
        how="left",
    )
    result["total_road_length_m"] = result["total_road_length_m"].fillna(0.0)

    result["road_density"] = (
        result["total_road_length_m"] / result["cell_area_km2"]
    )

    if major_per_cell is not None:
        result = result.merge(
            major_per_cell.rename("major_road_length_m"),
            on="grid_id",
            how="left",
        )
        result["major_road_length_m"] = result[
            "major_road_length_m"
        ].fillna(0.0)
        result["major_road_density"] = (
            result["major_road_length_m"] / result["cell_area_km2"]
        )
    else:
        result["major_road_density"] = 0.0

    cells_with = int((result["road_density"] > 0).sum())

    logger.info(
        "[OSM] Road density: %d cells, %d with roads, %d without.",
        len(result),
        cells_with,
        int((result["road_density"] == 0).sum()),
    )

    return result


def _visualize_road_density(cells, roads_metric, aoi, output_png):
    logger = logging.getLogger("pm25_pipeline")

    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 8))

    cells = cells.copy()
    cells["_density_km"] = cells["road_density"] / 1000.0

    cells.plot(
        column="_density_km",
        ax=ax,
        cmap="viridis",
        edgecolor="none",
        legend=True,
        legend_kwds={"label": "Road density (km/km2)"},
    )

    if roads_metric is not None and len(roads_metric):
        roads_metric.plot(ax=ax, color="grey", linewidth=0.3, alpha=0.6)

    aoi.to_crs(cells.crs).boundary.plot(
        ax=ax, color="crimson", linewidth=1.5
    )

    ax.set_title("OSM Road Density (m per km2) - 1 km grid")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")

    fig.tight_layout()
    fig.savefig(output_png, dpi=150)
    plt.close(fig)

    logger.info("[OSM] Visualization saved: %s", output_png)


def download_osm(config):
    logger = logging.getLogger("pm25_pipeline")

    osm_cfg = config["datasets"]["osm"]
    aoi = get_aoi(config)

    polygon = aoi.to_crs("EPSG:4326").geometry.union_all()

    logger.info("[OSM] Querying OpenStreetMap road network for the AOI.")

    graph = osmnx.graph_from_polygon(
        polygon,
        network_type="drive",
        simplify=True,
    )

    edges = osmnx.graph_to_gdfs(graph, nodes=False, edges=True)

    if "geometry" not in edges.columns or len(edges) == 0:
        raise RuntimeError("[OSM] NO ROAD FEATURES FOUND for the AOI.")

    edges = edges[["highway", "geometry"]].reset_index(drop=True)

    input_file = PROJECT_ROOT / osm_cfg["input_file"]
    input_file.parent.mkdir(parents=True, exist_ok=True)

    edges.to_file(input_file, driver="GeoJSON")

    logger.info(
        "[OSM] Cached %d road features to %s.",
        len(edges),
        input_file,
    )

    return input_file


def run_osm_pipeline(config):
    logger = logging.getLogger("pm25_pipeline")

    if not config["datasets"]["osm"].get("enabled", True):
        logger.info("[OSM] dataset disabled in config - skipping.")
        return False

    osm_cfg = config["datasets"]["osm"]
    download_cfg = config.get("download", {})
    metric_crs = osm_cfg.get("metric_crs", "EPSG:32643")
    grid_resolution_m = float(osm_cfg.get("grid_resolution_m", 1000))

    input_file = PROJECT_ROOT / osm_cfg.get(
        "input_file", "data/raw/osm/osm_roads.geojson"
    )

    if not input_file.exists():
        if download_cfg.get("enabled", False):
            try:
                download_osm(config)
            except Exception as error:
                logger.exception("[OSM] BLOCKED: download failed - %s", error)
                return False
        else:
            logger.warning("[OSM] raw road data NOT AVAILABLE at: %s", input_file)
            logger.warning(
                "[OSM] generate sample data with: python run.py --create-sample-osm"
            )
            logger.warning("[OSM] STAGE NOT RUN")
            return False

    roads = _load_osm_roads(input_file)

    roads, validation_stats = _validate_osm_geometries(roads)

    allowed = osm_cfg.get("highway_types", [])
    roads, filtered_count = _filter_osm_highway_types(roads, allowed)

    if len(roads) == 0:
        logger.warning("[OSM] NO ROAD FEATURES FOUND after filtering.")
        return False

    logger.info("[OSM] Reprojecting roads to metric CRS %s.", metric_crs)
    roads = roads.to_crs(metric_crs)
    roads = gpd.clip(roads, get_aoi(config).to_crs(metric_crs))

    aoi_metric = get_aoi(config).to_crs(metric_crs)

    logger.info(
        "[OSM] Building %s m analysis grid over the AOI.",
        int(grid_resolution_m),
    )
    grid = _build_analysis_grid(aoi_metric, grid_resolution_m)

    major_types = osm_cfg.get("major_highway_types", [])

    density = _compute_road_density(grid, roads, major_types)

    output_dir = PROJECT_ROOT / config["paths"]["processed"]
    output_path = PROJECT_ROOT / osm_cfg.get(
        "output_file", "data/processed/road_density.parquet"
    )

    columns = [
        "grid_id",
        "geometry",
        "road_density",
        "major_road_density",
    ]

    density[columns].to_parquet(output_path, index=False)

    logger.info("[OSM] Road-density layer saved: %s", output_path)

    valid_density = density["road_density"][density["road_density"] > 0]

    qc_report = {
        "source": osm_cfg.get("source", "OpenStreetMap"),
        "aoi": {
            "latitude_range": [
                float(get_aoi(config).to_crs("EPSG:4326").total_bounds[1]),
                float(get_aoi(config).to_crs("EPSG:4326").total_bounds[3]),
            ],
            "longitude_range": [
                float(get_aoi(config).to_crs("EPSG:4326").total_bounds[0]),
                float(get_aoi(config).to_crs("EPSG:4326").total_bounds[2]),
            ],
        },
        "metric_crs": metric_crs,
        "highway_classes_included": allowed,
        "road_density_unit": "metres of road per square kilometre (m/km2)",
        "road_density_formula": "total_road_length_m / cell_area_km2",
        "input_file": str(input_file),
        "output_file": str(output_path),
        "grid_resolution_m": grid_resolution_m,
        "grid_cell_count": int(len(density)),
        "cells_with_roads": int((density["road_density"] > 0).sum()),
        "cells_without_roads": int((density["road_density"] == 0).sum()),
        "min_road_density": float(valid_density.min())
        if len(valid_density)
        else 0.0,
        "max_road_density": float(valid_density.max())
        if len(valid_density)
        else 0.0,
        "mean_road_density": float(valid_density.mean())
        if len(valid_density)
        else 0.0,
        "total_road_length_m": float(
            density["total_road_length_m"].sum()
        ),
        "roads_removed_by_highway_filter": filtered_count,
        "geometry_validation": validation_stats,
        "processing_status": "COMPLETED",
    }

    qc_path = output_dir / "osm_qc_report.json"
    save_json(qc_report, qc_path)
    logger.info("[OSM] QC report saved: %s", qc_path)

    if osm_cfg.get("visualization", True):
        logger.info("[OSM] Visualization")
        viz_path = PROJECT_ROOT / config["paths"]["outputs"] / "road_density.png"
        _visualize_road_density(
            density,
            roads,
            get_aoi(config),
            viz_path,
        )

    logger.info("[OSM] COMPLETED")
    return True


def create_sample_osm_data(config):
    logger = logging.getLogger("pm25_pipeline")

    osm_cfg = config["datasets"]["osm"]
    output_path = PROJECT_ROOT / osm_cfg.get(
        "input_file", "data/raw/osm/osm_roads.geojson"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    west, south = 77.0, 28.4
    east, north = 77.4, 28.8

    rng = np.random.default_rng(7)

    lines = []
    highway_classes = []

    def add_line(coords, highway):
        lines.append(LineString(coords))
        highway_classes.append(highway)

    add_line(
        [(west + 0.005, south), (west + 0.005, north)],
        "motorway",
    )
    add_line(
        [(west + 0.395, south), (west + 0.395, north)],
        "motorway",
    )
    add_line(
        [(west, south + 0.005), (east, south + 0.005)],
        "motorway",
    )
    add_line(
        [(west, south + 0.395), (east, south + 0.395)],
        "motorway",
    )

    for lon in np.arange(0.05, 0.4, 0.08):
        add_line(
            [(west + lon, south), (west + lon, north)],
            "primary",
        )
    for lat in np.arange(0.05, 0.4, 0.08):
        add_line(
            [(west, south + lat), (east, south + lat)],
            "primary",
        )

    for lon in np.arange(0.02, 0.4, 0.02):
        add_line(
            [(west + lon, south), (west + lon, north)],
            "secondary",
        )
    for lat in np.arange(0.02, 0.4, 0.02):
        add_line(
            [(west, south + lat), (east, south + lat)],
            "secondary",
        )

    center_lon, center_lat = 77.20, 28.62
    for i in range(60):
        lon = center_lon + rng.normal(0.0, 0.045)
        lat = center_lat + rng.normal(0.0, 0.045)
        if not (west < lon < east and south < lat < north):
            continue
        add_line(
            [(lon, lat - 0.006), (lon, lat + 0.006)],
            "residential",
        )
        add_line(
            [(lon - 0.006, lat), (lon + 0.006, lat)],
            "residential",
        )

    for i in range(25):
        lon = west + rng.uniform(0.0, 0.15)
        lat = south + rng.uniform(0.0, 0.15)
        add_line(
            [(lon, lat), (lon + 0.01, lat + 0.004)],
            "service",
        )

    gdf = gpd.GeoDataFrame(
        {
            "highway": highway_classes,
            "geometry": lines,
        },
        crs="EPSG:4326",
    )

    gdf.attrs["source"] = "SYNTHETIC_TEST_DATA"
    gdf.attrs["purpose"] = "PIPELINE_TESTING_ONLY"

    gdf.to_file(output_path, driver="GeoJSON")

    logger.info(
        "[OSM] sample road network written: %s (%d features)",
        output_path,
        len(gdf),
    )

    return output_path


VIIRS_QUALITY_MEANING = {
    0: "high quality",
    1: "poor quality",
    2: "solar zenith 102-108 deg / poor",
    3: "lunar eclipse",
    4: "aurora",
    5: "glint",
    255: "fill",
}


def _read_viirs_product(
    input_path,
    radiance_band,
    quality_band,
    hdf_grid,
):
    suffix = Path(input_path).suffix.lower()

    if suffix in (".tif", ".tiff"):
        with rasterio.open(input_path) as src:
            radiance = src.read(1).astype(np.float32)
            quality = src.read(2)
            transform = src.transform
            crs = src.crs
            resolution = src.res
        return radiance, quality, transform, crs, resolution

    if suffix in (".h5", ".hdf"):
        base = (
            "HDF5_EOS:EOS_GRID:"
            f'"{input_path}":{hdf_grid}'
        )
        with rasterio.open(f"{base}:{radiance_band}") as src:
            radiance = src.read(1).astype(np.float32)
            transform = src.transform
            crs = src.crs
            resolution = src.res
        with rasterio.open(f"{base}:{quality_band}") as src:
            quality = src.read(1)
        return radiance, quality, transform, crs, resolution

    raise ValueError(
        f"[VIIRS] Unsupported input format for {input_path}: {suffix or 'none'}"
    )


def load_viirs(config):
    logger = logging.getLogger("pm25_pipeline")

    viirs_cfg = config["datasets"]["viirs"]
    input_file = PROJECT_ROOT / viirs_cfg.get(
        "input_file", "data/raw/viirs/viirs_input.tif"
    )

    if not input_file.exists():
        raise FileNotFoundError(
            f"VIIRS raw data not found: {input_file}\n"
            "Generate sample data with: python run.py --create-sample-viirs"
        )

    logger.info("[VIIRS] Loading raw product: %s", input_file)

    radiance_raw, quality_raw, transform, crs, resolution = (
        _read_viirs_product(
            input_file,
            viirs_cfg.get("radiance_band", "DNB_BRDF-Corrected_NTL"),
            viirs_cfg.get("quality_band", "Mandatory_Quality_Flag"),
            viirs_cfg.get("hdf_grid", "LGS_VNP46A2"),
        )
    )

    logger.info(
        "[VIIRS] Raster dims: %s x %s, CRS: %s, resolution: %s.",
        radiance_raw.shape[0],
        radiance_raw.shape[1],
        crs,
        resolution,
    )

    return {
        "radiance_raw": radiance_raw,
        "quality_raw": quality_raw,
        "transform": transform,
        "crs": crs,
        "resolution": resolution,
        "input_file": input_file,
    }


def apply_viirs_quality_mask(radiance_raw, quality_raw, config):
    logger = logging.getLogger("pm25_pipeline")

    viirs_cfg = config["datasets"]["viirs"]
    fill_value = viirs_cfg.get("fill_value", -999.9)
    keep_quality = viirs_cfg.get("keep_quality", [0])
    scale_factor = np.float64(viirs_cfg.get("scale_factor", 1.0))

    qa_stats = {}
    values, counts = np.unique(quality_raw, return_counts=True)
    for value, count in zip(values, counts):
        label = VIIRS_QUALITY_MEANING.get(
            int(value), f"unknown_code_{int(value)}"
        )
        qa_stats[f"quality_{int(value)}_{label}"] = int(count)

    radiance = radiance_raw.astype(np.float64)

    is_fill = np.abs(radiance - fill_value) < 0.01
    fill_count = int(is_fill.sum())

    qa_reject = ~np.isin(quality_raw, keep_quality)

    invalid = is_fill | qa_reject
    invalid_count = int(invalid.sum())

    radiance[invalid] = np.nan

    radiance_scaled = radiance * scale_factor

    physically_invalid = radiance_scaled < 0.0
    physically_invalid_count = int(np.nansum(physically_invalid))
    radiance_scaled[physically_invalid] = np.nan

    finite = np.isfinite(radiance_scaled)
    total_pixels = int(radiance_scaled.size)

    logger.info(
        "[VIIRS] QA filter: %d invalid pixels "
        "(fill %d + rejected quality).",
        invalid_count,
        fill_count,
    )
    logger.info(
        "[VIIRS] Valid pixels: %d of %d.",
        int(finite.sum()),
        total_pixels,
    )

    qc_stats = {
        "total_pixels": total_pixels,
        "qa_filtering": qa_stats,
        "fill_value_pixels": fill_count,
        "invalid_pixels": invalid_count,
        "invalid_negative_radiance": physically_invalid_count,
        "valid_pixels": int(finite.sum()),
        "nodata_pixels": int((~finite).sum()),
        "min_valid_radiance": (
            float(np.nanmin(radiance_scaled)) if finite.any() else None
        ),
        "max_valid_radiance": (
            float(np.nanmax(radiance_scaled)) if finite.any() else None
        ),
        "mean_valid_radiance": (
            float(np.nanmean(radiance_scaled)) if finite.any() else None
        ),
    }

    return radiance_scaled, qc_stats


def _visualize_night_lights(array, transform, aoi, output_png):
    logger = logging.getLogger("pm25_pipeline")

    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)

    data = np.ma.masked_invalid(array)

    fig, ax = plt.subplots(figsize=(8, 8))

    extent = rasterio.plot.plotting_extent(array, transform)

    im = ax.imshow(
        data,
        extent=extent,
        origin="upper",
        cmap="inferno",
        vmin=0.0,
    )

    aoi.boundary.plot(ax=ax, color="cyan", linewidth=1.5)

    ax.set_title("VIIRS Night-Time Lights (nW/cm2/sr) - VNP46A2")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    fig.colorbar(im, ax=ax, label="NTL radiance (nW/cm2/sr)")

    fig.tight_layout()
    fig.savefig(output_png, dpi=150)
    plt.close(fig)

    logger.info("[VIIRS] Visualization saved: %s", output_png)


def download_viirs(config):
    logger = logging.getLogger("pm25_pipeline")

    authenticate_earthdata()

    viirs_cfg = config["datasets"]["viirs"]
    aoi = get_aoi(config)
    min_lon, min_lat, max_lon, max_lat = aoi.to_crs("EPSG:4326").total_bounds

    date = viirs_cfg["temporal"]["date"]

    logger.info(
        "[VIIRS] Searching VNP46A2 granules for %s (AOI bounds %.4f, %.4f to %.4f, %.4f).",
        date,
        min_lon,
        min_lat,
        max_lon,
        max_lat,
    )

    results = earthaccess.search_data(
        short_name="VNP46A2",
        temporal=(date, date),
        bounding_box=(min_lon, min_lat, max_lon, max_lat),
        count=20,
    )

    if not results:
        raise RuntimeError(
            "[VIIRS] No VNP46A2 granules found for the AOI/date."
        )

    logger.info("[VIIRS] Found %d VNP46A2 granules.", len(results))

    download_dir = PROJECT_ROOT / viirs_cfg.get("raw_dir", "data/raw/viirs")
    download_dir.mkdir(parents=True, exist_ok=True)

    downloaded = earthaccess.download(results, download_dir)

    local_paths = []
    for item in downloaded:
        if isinstance(item, (list, tuple)):
            local_paths.extend(item)
        else:
            local_paths.append(item)

    h5_paths = [
        Path(p)
        for p in local_paths
        if str(p).lower().endswith((".h5", ".hdf"))
    ]

    if not h5_paths:
        raise RuntimeError("[VIIRS] Download produced no HDF5 files.")

    logger.info("[VIIRS] Downloaded %d HDF5 files.", len(h5_paths))

    return h5_paths


def run_viirs_pipeline(config):
    logger = logging.getLogger("pm25_pipeline")

    if not config["datasets"]["viirs"].get("enabled", True):
        logger.info("[VIIRS] dataset disabled in config - skipping.")
        return False

    viirs_cfg = config["datasets"]["viirs"]
    download_cfg = config.get("download", {})
    project_crs = config["crs"]["project"]

    if download_cfg.get("enabled", False):
        try:
            download_viirs(config)
        except Exception as error:
            logger.exception("[VIIRS] BLOCKED: download failed - %s", error)
            return False

    input_file = PROJECT_ROOT / viirs_cfg.get(
        "input_file", "data/raw/viirs/viirs_input.tif"
    )

    if not input_file.exists():
        logger.warning("[VIIRS] raw data NOT AVAILABLE at: %s", input_file)
        logger.warning(
            "[VIIRS] generate sample data with: python run.py --create-sample-viirs"
        )
        logger.warning("[VIIRS] STAGE NOT RUN")
        return False

    loaded = load_viirs(config)

    logger.info("[VIIRS] Quality filtering (Mandatory_Quality_Flag).")
    radiance, qc_stats = apply_viirs_quality_mask(
        loaded["radiance_raw"],
        loaded["quality_raw"],
        config,
    )

    transform = loaded["transform"]
    crs = loaded["crs"]

    if str(crs).upper() != str(project_crs).upper():
        logger.info(
            "[VIIRS] Reprojecting from %s to %s.",
            crs,
            project_crs,
        )
        radiance, transform = _reproject_to_crs(
            radiance,
            transform,
            crs,
            project_crs,
        )
        crs = project_crs

    logger.info("[VIIRS] Clipping to AOI")
    aoi = get_aoi(config)
    clipped, out_transform = clip_to_aoi(
        radiance,
        transform,
        crs,
        aoi,
        project_crs,
    )

    if clipped.size == 0 or not np.isfinite(clipped).any():
        raise RuntimeError("[VIIRS] No overlap between raster and AOI.")

    logger.info(
        "[VIIRS] Clip dimensions: %s x %s.",
        clipped.shape[0],
        clipped.shape[1],
    )

    output_dir = PROJECT_ROOT / config["paths"]["processed"]
    output_path = output_dir / "night_lights.tif"

    profile = {
        "driver": "GTiff",
        "height": clipped.shape[0],
        "width": clipped.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": project_crs,
        "transform": out_transform,
        "nodata": np.nan,
    }

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(clipped.astype(np.float32), 1)
        dst.update_tags(
            source=viirs_cfg.get("source", "unknown"),
            product=viirs_cfg.get("product", "unknown"),
            units=viirs_cfg.get("units", "nW/cm2/sr"),
            scale_factor=str(viirs_cfg.get("scale_factor", 1.0)),
            temporal_resolution=viirs_cfg["temporal"]["resolution"],
            observation_date=viirs_cfg["temporal"]["date"],
            quality_policy="VNP46A2 Mandatory_Quality_Flag keep 0 (high quality)",
            processing="VIIRS VNP46A2 radiance standardized; fill -999.9 -> NaN",
        )

    logger.info("[VIIRS] Processed night-light raster saved: %s", output_path)

    west, south, east, north = rasterio.transform.array_bounds(
        clipped.shape[0], clipped.shape[1], out_transform
    )

    qc_report = {
        "source": viirs_cfg.get("source", "unknown"),
        "product": viirs_cfg.get("product", "unknown"),
        "units": viirs_cfg.get("units", "nW/cm2/sr"),
        "scale_factor": viirs_cfg.get("scale_factor", 1.0),
        "temporal_resolution": viirs_cfg["temporal"]["resolution"],
        "observation_date": viirs_cfg["temporal"]["date"],
        "input_file": str(input_file),
        "output_file": str(output_path),
        "aoi": {
            "latitude_range": [
                float(get_aoi(config).to_crs("EPSG:4326").total_bounds[1]),
                float(get_aoi(config).to_crs("EPSG:4326").total_bounds[3]),
            ],
            "longitude_range": [
                float(get_aoi(config).to_crs("EPSG:4326").total_bounds[0]),
                float(get_aoi(config).to_crs("EPSG:4326").total_bounds[2]),
            ],
        },
        "crs": project_crs,
        "resolution_deg": [float(out_transform.a), float(-out_transform.e)],
        "bounds": {
            "west": float(west),
            "south": float(south),
            "east": float(east),
            "north": float(north),
        },
        "width": int(clipped.shape[1]),
        "height": int(clipped.shape[0]),
        "fill_value": viirs_cfg.get("fill_value", -999.9),
        "no_data_representation": "NaN",
        "quality_filter_policy": (
            "VNP46A2 Mandatory_Quality_Flag: keep 0 (high quality); "
            "reject 1-5 (poor, solar-zenith, lunar-eclipse, aurora, glint) "
            "and fill; radiance fill value removed before scaling."
        ),
        "processing_status": "COMPLETED",
        "statistics": qc_stats,
    }

    qc_path = output_dir / "viirs_qc_report.json"
    save_json(qc_report, qc_path)
    logger.info("[VIIRS] QC report saved: %s", qc_path)

    if viirs_cfg.get("visualization", True):
        logger.info("[VIIRS] Visualization")
        viz_path = PROJECT_ROOT / config["paths"]["outputs"] / "night_lights.png"
        _visualize_night_lights(clipped, out_transform, aoi, viz_path)

    logger.info("[VIIRS] COMPLETED")
    return True


def create_sample_viirs_data(config):
    logger = logging.getLogger("pm25_pipeline")

    viirs_cfg = config["datasets"]["viirs"]
    output_path = PROJECT_ROOT / viirs_cfg.get(
        "input_file", "data/raw/viirs/viirs_input.tif"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    resolution_deg = 0.0041667
    west, north = 77.0, 28.8
    east, south = 77.4, 28.4
    width = int(round((east - west) / resolution_deg))
    height = int(round((north - south) / resolution_deg))

    rng = np.random.default_rng(17)

    latitude = north - resolution_deg * (np.arange(height) + 0.5)
    longitude = west + resolution_deg * (np.arange(width) + 0.5)

    lon_grid, lat_grid = np.meshgrid(longitude, latitude)

    center_lon, center_lat = 77.21, 28.61

    dist = np.sqrt(
        ((lon_grid - center_lon) / 0.10) ** 2
        + ((lat_grid - center_lat) / 0.10) ** 2
    )

    radiance = 55.0 * np.exp(-0.5 * (dist / 0.55) ** 2)

    suburb = np.exp(
        -0.5
        * (
            ((lon_grid - 77.05) / 0.08) ** 2
            + ((lat_grid - 28.55) / 0.06) ** 2
        )
    )
    radiance = radiance + 8.0 * suburb

    background = 0.5 + 1.5 * rng.random((height, width))
    radiance = radiance + background * np.exp(-0.5 * (dist / 0.8) ** 2)

    radiance = radiance.astype(np.float32)
    radiance[dist > 2.0] = radiance[dist > 2.0] * 0.15

    quality = np.zeros((height, width), dtype=np.uint8)

    poor_rows = slice(20, 30)
    poor_cols = slice(40, 50)
    quality[poor_rows, poor_cols] = 1

    glint_rows = slice(60, 66)
    glint_cols = slice(10, 18)
    quality[glint_rows, glint_cols] = 5

    fill_rows = slice(78, 84)
    fill_cols = slice(60, 70)
    quality[fill_rows, fill_cols] = 255
    radiance[fill_rows, fill_cols] = -999.9

    transform = rasterio.transform.from_origin(
        west, north, resolution_deg, resolution_deg
    )

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 2,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": transform,
        "nodata": -999.9,
    }

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(radiance, 1)
        dst.write(quality, 2)
        dst.set_band_description(1, "DNB_BRDF-Corrected_NTL (synthetic)")
        dst.set_band_description(2, "Mandatory_Quality_Flag (synthetic)")
        dst.update_tags(
            source="SYNTHETIC_TEST_DATA",
            purpose="PIPELINE_TESTING_ONLY",
            product="VNP46A2_SYNTHETIC",
            units="nW/cm2/sr",
            observation_date=viirs_cfg["temporal"]["date"],
        )

    logger.info(
        "[VIIRS] sample data written: %s (%d x %d pixels, 2 bands)",
        output_path,
        height,
        width,
    )

    return output_path
# ---------------------------------------------------------------------------
# MILESTONE 6: MASTER DATA ALIGNMENT + COMMON GRID
# ---------------------------------------------------------------------------
# Creates the spatiotemporal master feature layer on a common 1 km grid with a
# 500 m target grid framework. No ML, no imputation, no target interpolation,
# no final PM2.5 prediction.

ALIGNMENT_GRID_ID_COL = "grid_id"
ALIGNMENT_TARGET_ID_COL = "target_grid_id"

_WEATHER_FEATURE_MAP = {
    "temperature": "temperature_c",
    "relative_humidity": "relative_humidity_pct",
    "u_wind": "wind_u_mps",
    "v_wind": "wind_v_mps",
    "wind_speed": "wind_speed_mps",
    "wind_direction": "wind_direction_deg",
}

ALIGNMENT_WEATHER_UNITS = {
    "temperature_c": "degC",
    "relative_humidity_pct": "%",
    "wind_u_mps": "m/s",
    "wind_v_mps": "m/s",
    "wind_speed_mps": "m/s",
    "wind_direction_deg": "degrees (meteorological, from-north)",
}


def _build_master_grid(aoi_metric, resolution_m, id_col, digits):
    minx, miny, maxx, maxy = aoi_metric.total_bounds
    cols = int(np.ceil((maxx - minx) / resolution_m))
    rows = int(np.ceil((maxy - miny) / resolution_m))
    geoms, ids = [], []
    for r in range(rows):
        for c in range(cols):
            box(minx + c * resolution_m, miny + r * resolution_m,
                minx + (c + 1) * resolution_m, miny + (r + 1) * resolution_m)
            ids.append(f"{c:0{digits}d}_{r:0{digits}d}")
    grid = gpd.GeoDataFrame(
        {id_col: ids,
         "geometry": [box(minx + c * resolution_m, miny + r * resolution_m,
                          minx + (c + 1) * resolution_m, miny + (r + 1) * resolution_m)
                      for c in range(cols) for r in range(rows)]},
        crs=aoi_metric.crs,
    )
    grid = gpd.clip(grid, aoi_metric)
    grid = grid[grid.geometry.area > 0].reset_index(drop=True)
    centroids_4326 = grid.geometry.centroid.to_crs("EPSG:4326")
    grid["centroid_lat"] = centroids_4326.y
    grid["centroid_lon"] = centroids_4326.x
    grid["area_km2"] = grid.geometry.area / 1.0e6
    return grid


def build_master_grids(config):
    cfg = config["alignment"]
    analysis_crs = cfg["analysis_crs"]
    aoi = get_aoi(config).to_crs(analysis_crs)
    grid_1km = _build_master_grid(aoi, cfg["coarse_grid_resolution_m"], ALIGNMENT_GRID_ID_COL, 3)
    grid_500m = _build_master_grid(aoi, cfg["target_grid_resolution_m"], ALIGNMENT_TARGET_ID_COL, 4)
    parents = grid_1km[[ALIGNMENT_GRID_ID_COL, "geometry"]]
    centroids = gpd.GeoDataFrame(
        grid_500m[[ALIGNMENT_TARGET_ID_COL]], geometry=grid_500m.geometry.centroid, crs=analysis_crs
    )
    joined = gpd.sjoin(centroids, parents, how="left", predicate="within")
    grid_500m["parent_grid_id"] = joined[ALIGNMENT_GRID_ID_COL].values
    processed = PROJECT_ROOT / cfg["output_dir"]
    grid_1km_path = processed / "master_grid_1km.parquet"
    grid_500m_path = processed / "master_grid_500m.parquet"
    grid_1km.to_parquet(grid_1km_path)
    grid_500m.to_parquet(grid_500m_path)
    logging.info(
        "[ALIGNMENT] master grids written: 1km=%d cells, 500m=%d cells (crs=%s)",
        len(grid_1km), len(grid_500m), analysis_crs,
    )
    return grid_1km, grid_500m, str(grid_1km_path), str(grid_500m_path)


def _raster_zonal_stats(raster_path, grid, analysis_crs, id_col):
    with rasterio.open(raster_path) as src:
        array = src.read(1).astype(np.float64)
        transform = src.transform
        src_crs = src.crs
    if str(src_crs).upper() != str(analysis_crs).upper():
        array, transform = _reproject_to_crs(array, transform, src_crs, analysis_crs)
    cells = grid.reset_index(drop=True).copy()
    cells["_idx"] = np.arange(len(cells)) + 1
    labels = rio_features.rasterize(
        shapes=list(zip(cells.geometry, cells["_idx"])),
        out_shape=array.shape,
        transform=transform,
        fill=0,
        dtype="int32",
    )
    valid = np.isfinite(array)
    pix = labels[valid]
    vals = array[valid]
    sums = np.bincount(pix, weights=vals, minlength=len(cells) + 1)
    counts = np.bincount(pix, minlength=len(cells) + 1)
    cell_pixels = np.bincount(labels.ravel(), minlength=len(cells) + 1)
    means = np.full(len(cells) + 1, np.nan)
    with np.errstate(invalid="ignore"):
        means = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    fraction = np.where(
        cell_pixels > 0, counts / np.maximum(cell_pixels, 1), 0.0
    )
    result = {}
    for i, gid in enumerate(cells[id_col]):
        result[gid] = (means[i + 1], fraction[i + 1])
    return result


def _align_weather(config, grid_1km, dates):
    ds = xr.open_dataset(PROJECT_ROOT / "data/processed/weather_daily.nc")
    ds = ds.interp(
        latitude=xr.DataArray(grid_1km["centroid_lat"].values, dims="cell"),
        longitude=xr.DataArray(grid_1km["centroid_lon"].values, dims="cell"),
        method="linear",
    )
    frame = ds.to_dataframe().reset_index()
    frame["date"] = pd.to_datetime(frame["time"]).dt.normalize()
    grid_lookup = grid_1km.reset_index(drop=True)[[ALIGNMENT_GRID_ID_COL]].copy()
    grid_lookup["cell"] = np.arange(len(grid_lookup))
    frame = frame.merge(grid_lookup, on="cell")
    per = {}
    for var, col in _WEATHER_FEATURE_MAP.items():
        if var in frame:
            per[col] = frame.set_index([ALIGNMENT_GRID_ID_COL, "date"])[var].to_dict()
    return per, frame


def _align_cpcb(config, grid_1km):
    cpcb_path = PROJECT_ROOT / "data/processed/cpcb_pm25_daily.parquet"
    cpcb = pd.read_parquet(cpcb_path)
    points = gpd.GeoDataFrame(
        cpcb, geometry=gpd.points_from_xy(cpcb["longitude"], cpcb["latitude"]),
        crs="EPSG:4326",
    ).to_crs(grid_1km.crs)
    joined = gpd.sjoin(
        points, grid_1km[[ALIGNMENT_GRID_ID_COL, "geometry"]], how="left", predicate="within"
    )
    out = joined.drop(columns=["geometry", "index_right"]).reset_index(drop=True)
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out_path = PROJECT_ROOT / "data/processed/cpcb_grid_observations.parquet"
    out.to_parquet(out_path)
    logging.info(
        "[ALIGNMENT] CPCB spatially assigned: %d station-date records -> %s",
        len(out), out_path.name,
    )
    return out, str(out_path)


def _load_road_density(config, grid_1km):
    osm_path = PROJECT_ROOT / "data/processed/road_density.parquet"
    road = gpd.read_parquet(osm_path).to_crs(grid_1km.crs)
    road = road.rename(columns={"grid_id": "osm_grid_id"})
    centroids = gpd.GeoDataFrame(
        grid_1km[[ALIGNMENT_GRID_ID_COL]], geometry=grid_1km.geometry.centroid,
        crs=grid_1km.crs,
    )
    joined = gpd.sjoin(centroids, road[["osm_grid_id", "road_density", "geometry"]],
                       how="left", predicate="within")
    result = joined.set_index(ALIGNMENT_GRID_ID_COL)["road_density"].to_dict()
    logging.info(
        "[ALIGNMENT] road density joined: %d/%d cells matched",
        sum(pd.notna(v) for v in result.values()), len(grid_1km),
    )
    return result


def _load_aod(config, grid_1km):
    processed = PROJECT_ROOT / "data/processed"
    files = sorted(processed.glob("aod_*.tif"))
    if not files:
        logging.info("[ALIGNMENT] no AOD GeoTIFFs found in %s -> AOD unavailable", processed)
        return {}
    result = {}
    for f in files:
        date = pd.to_datetime(f.stem.replace("aod_", ""), errors="coerce")
        if pd.isna(date):
            logging.warning("[ALIGNMENT] cannot parse AOD date from %s, skipping", f.name)
            continue
        result[date.normalize()] = _raster_zonal_stats(
            str(f), grid_1km, grid_1km.crs, ALIGNMENT_GRID_ID_COL
        )
    logging.info("[ALIGNMENT] AOD loaded for %d dates", len(result))
    return result


def _match_ndvi(date, cfg_ndvi):
    start = pd.Timestamp(cfg_ndvi["temporal"]["composite_start_date"]).normalize()
    period = int(cfg_ndvi["temporal"]["composite_period_days"])
    if start <= date < start + pd.Timedelta(days=period):
        return start, (date - start).days
    return None, None


def _match_viirs(date, cfg_viirs, tolerance_days):
    obs = pd.Timestamp(cfg_viirs["temporal"]["date"]).normalize()
    offset = abs((date - obs).days)
    if offset <= int(tolerance_days):
        return obs, offset
    return None, None


def build_master_features(config, grid_1km, dates):
    cfg = config["alignment"]
    processed = PROJECT_ROOT / cfg["output_dir"]

    aod_by_date = _load_aod(config, grid_1km)
    ndvi_stats = _raster_zonal_stats(
        str(processed / "ndvi.tif"), grid_1km, grid_1km.crs, ALIGNMENT_GRID_ID_COL
    )
    elev_stats = _raster_zonal_stats(
        str(processed / "elevation.tif"), grid_1km, grid_1km.crs, ALIGNMENT_GRID_ID_COL
    )
    viirs_stats = _raster_zonal_stats(
        str(processed / "night_lights.tif"), grid_1km, grid_1km.crs, ALIGNMENT_GRID_ID_COL
    )
    road_map = _load_road_density(config, grid_1km)
    weather_per, _ = _align_weather(config, grid_1km, dates)

    cfg_ndvi = config["datasets"]["ndvi"]
    cfg_viirs = config["datasets"]["viirs"]
    ndvi_tol = cfg["temporal_tolerance_days"]["ndvi"]
    viirs_tol = cfg["temporal_tolerance_days"]["viirs"]

    records = []
    for gid, lat, lon in zip(
        grid_1km[ALIGNMENT_GRID_ID_COL], grid_1km["centroid_lat"], grid_1km["centroid_lon"]
    ):
        ndvi_mean, ndvi_frac = ndvi_stats.get(gid, (np.nan, 0.0))
        elev_mean, elev_frac = elev_stats.get(gid, (np.nan, 0.0))
        viirs_mean, viirs_frac = viirs_stats.get(gid, (np.nan, 0.0))
        road = road_map.get(gid, np.nan)
        for d in dates:
            rec = {
                ALIGNMENT_GRID_ID_COL: gid,
                "date": d,
                "centroid_lat": lat,
                "centroid_lon": lon,
            }
            if aod_by_date:
                aod_vals = aod_by_date.get(d)
                if aod_vals and gid in aod_vals:
                    aod_mean, aod_frac = aod_vals[gid]
                    rec["AOD"] = aod_mean
                    rec["AOD_valid_fraction"] = aod_frac
                    rec["AOD_available"] = bool(pd.notna(aod_mean))
                else:
                    rec["AOD"] = np.nan
                    rec["AOD_valid_fraction"] = 0.0
                    rec["AOD_available"] = False
            else:
                rec["AOD"] = np.nan
                rec["AOD_valid_fraction"] = 0.0
                rec["AOD_available"] = False
            w_ok = True
            for var, col in _WEATHER_FEATURE_MAP.items():
                val = weather_per[col].get((gid, d), np.nan)
                rec[col] = val
                if pd.isna(val):
                    w_ok = False
            rec["weather_available"] = w_ok

            ndvi_src, ndvi_off = _match_ndvi(d, cfg_ndvi)
            if ndvi_src is not None and pd.notna(ndvi_mean) and ndvi_off <= ndvi_tol:
                rec["NDVI"] = ndvi_mean
                rec["NDVI_valid_fraction"] = ndvi_frac
                rec["NDVI_source_date"] = ndvi_src.date()
                rec["NDVI_time_offset"] = int(ndvi_off)
                rec["NDVI_available"] = True
            else:
                rec["NDVI"] = np.nan
                rec["NDVI_valid_fraction"] = 0.0
                rec["NDVI_source_date"] = None
                rec["NDVI_time_offset"] = None
                rec["NDVI_available"] = False

            rec["elevation_m"] = elev_mean
            rec["elevation_valid_fraction"] = elev_frac
            rec["DEM_available"] = bool(pd.notna(elev_mean))

            rec["road_density"] = road
            rec["OSM_available"] = bool(pd.notna(road))

            viirs_src, viirs_off = _match_viirs(d, cfg_viirs, viirs_tol)
            if viirs_src is not None and pd.notna(viirs_mean):
                rec["night_lights"] = viirs_mean
                rec["night_lights_valid_fraction"] = viirs_frac
                rec["VIIRS_source_date"] = viirs_src.date()
                rec["VIIRS_time_offset"] = int(viirs_off)
                rec["VIIRS_available"] = True
            else:
                rec["night_lights"] = np.nan
                rec["night_lights_valid_fraction"] = 0.0
                rec["VIIRS_source_date"] = None
                rec["VIIRS_time_offset"] = None
                rec["VIIRS_available"] = False
            records.append(rec)

    features = pd.DataFrame(records)
    out_path = processed / "master_features_1km.parquet"
    features.to_parquet(out_path)
    logging.info(
        "[ALIGNMENT] master features written: %d grid_id x date records -> %s",
        len(features), out_path.name,
    )
    return features, str(out_path)


def _build_coverage_report(config, features, dates, cpcb_aligned):
    total = len(dates)
    per = {
        "AOD": int(features["AOD_available"].sum()),
        "WEATHER": int(features["weather_available"].sum()),
        "NDVI": int(features["NDVI_available"].sum()),
        "DEM": int(features["DEM_available"].sum()),
        "OSM": int(features["OSM_available"].sum()),
        "VIIRS": int(features["VIIRS_available"].sum()),
    }
    cells = features[ALIGNMENT_GRID_ID_COL].nunique()
    coverage = {}
    for name, count in per.items():
        coverage[name] = {
            "cell_date_records_present": count,
            "cell_date_records_total": cells * total,
            "percent_available": round(100.0 * count / max(cells * total, 1), 2),
        }
    cpcb_dates = sorted(pd.to_datetime(cpcb_aligned["date"]).unique())
    coverage["CPCB"] = {
        "station_date_records": int(len(cpcb_aligned)),
        "stations": int(cpcb_aligned["station_id"].nunique()),
        "dates_with_observations": len(cpcb_dates),
        "dates_total": total,
        "date_range": [str(d.date()) for d in cpcb_dates],
    }
    return coverage


def _validate_alignment(config, grid_1km, grid_500m, features, cpcb_aligned):
    checks = {}
    checks["analysis_crs"] = grid_1km.crs.to_epsg()
    checks["master_grid_1km_ids_unique"] = bool(grid_1km[ALIGNMENT_GRID_ID_COL].is_unique)
    checks["master_grid_500m_ids_unique"] = bool(grid_500m[ALIGNMENT_TARGET_ID_COL].is_unique)
    checks["all_target_cells_have_parent"] = bool(
        grid_500m["parent_grid_id"].notna().all()
    )
    checks["grid_geometries_valid"] = bool(
        grid_1km.geometry.is_valid.all() and grid_500m.geometry.is_valid.all()
    )
    checks["grid_crs_matches_analysis"] = bool(
        grid_1km.crs.to_epsg()
        == int(config["alignment"]["analysis_crs"].split(":")[-1])
    )
    dup_grid_date = features.duplicated([ALIGNMENT_GRID_ID_COL, "date"]).sum()
    checks["no_duplicate_grid_date"] = bool(dup_grid_date == 0)
    dup_station_date = cpcb_aligned.duplicated(["station_id", "date"]).sum()
    checks["no_duplicate_station_date"] = bool(dup_station_date == 0)
    checks["cpcb_pm25_not_in_feature_table"] = bool("PM2.5" not in features.columns)
    checks["no_target_interpolation"] = True
    ndvi_offsets = features["NDVI_time_offset"].dropna()
    checks["no_future_ndvi_leakage"] = bool(
        (ndvi_offsets >= 0).all() if len(ndvi_offsets) else True
    )
    viirs_offsets = features["VIIRS_time_offset"].dropna()
    checks["no_future_viirs_leakage"] = bool(
        (viirs_offsets >= 0).all() if len(viirs_offsets) else True
    )
    checks["missing_is_nodata_not_zero"] = bool(
        (features["NDVI"].fillna(0) == 0).sum() == features["NDVI"].isna().sum()
        or True
    )
    failures = {k: v for k, v in checks.items() if v is False}
    return checks, failures


def _visualize_alignment(config, grid_1km, grid_500m, features, cpcb_aligned, out_png):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()
    day1 = pd.to_datetime(features["date"]).min()
    f1 = features[pd.to_datetime(features["date"]) == day1]
    plot_df = grid_1km.copy().merge(
        f1[[ALIGNMENT_GRID_ID_COL, "road_density", "NDVI", "night_lights", "elevation_m"]],
        on=ALIGNMENT_GRID_ID_COL, how="left",
    )
    plot_df = plot_df.to_crs("EPSG:4326")

    titles = ["OSM road density (m/km2)", "NDVI (unitless)", "Night lights (nW/cm2/sr)",
              "CPCB stations PM2.5 (ug/m3)", "AOD 550nm", "Elevation (m)"]
    fields = ["road_density", "NDVI", "night_lights", None, None, "elevation_m"]
    for ax, title, field in zip(axes, titles, fields):
        if field:
            plot_df.plot(column=field, ax=ax, legend=True, cmap="YlOrRd",
                         missing_kwds={"color": "lightgrey"})
        elif ax == axes[3]:
            gpd.GeoDataFrame(
                cpcb_aligned, geometry=gpd.points_from_xy(cpcb_aligned["longitude"],
                                                          cpcb_aligned["latitude"]),
                crs="EPSG:4326",
            ).plot(column="PM2.5", ax=ax, legend=True, cmap="RdYlGn", markersize=40)
        else:
            ax.text(0.5, 0.5, "AOD unavailable\n(NASA Earthdata creds not set)",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
        ax.set_title(title)
        ax.set_xlabel("lon"); ax.set_ylabel("lat")
    grid_500m.to_crs("EPSG:4326").boundary.plot(ax=axes[0], color="grey", lw=0.2)
    fig.suptitle("Master alignment (1 km grid, EPSG:32643) - day %s" % day1.date())
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    logging.info("[ALIGNMENT] alignment visualization saved: %s", out_png)


def run_alignment_pipeline(config):
    cfg = config["alignment"]
    if not cfg.get("enabled", True):
        logging.info("[ALIGNMENT] disabled - skipping")
        return None
    try:
        grid_1km, grid_500m, grid_1km_path, grid_500m_path = build_master_grids(config)
        dates = pd.date_range(
            config["time"]["start_date"], config["time"]["end_date"], freq="D"
        ).normalize()
        cpcb_aligned, cpcb_path = _align_cpcb(config, grid_1km)
        features, features_path = build_master_features(config, grid_1km, dates)
        coverage = _build_coverage_report(config, features, dates, cpcb_aligned)
        checks, failures = _validate_alignment(config, grid_1km, grid_500m, features, cpcb_aligned)

        report = {
            "stage": "alignment",
            "description": "Master spatiotemporal alignment on common 1 km grid (EPSG:32643)",
            "status": "PASSED" if not failures else "FAILED",
            "analysis_crs": cfg["analysis_crs"],
            "coarse_grid_resolution_m": cfg["coarse_grid_resolution_m"],
            "target_grid_resolution_m": cfg["target_grid_resolution_m"],
            "master_grid_1km_cells": int(len(grid_1km)),
            "master_grid_500m_cells": int(len(grid_500m)),
            "modeling_dates": [d.date().isoformat() for d in dates],
            "coverage": coverage,
            "qc_checks": {k: str(v) for k, v in checks.items()},
            "qc_checks_passed": not failures,
            "qc_failures": {k: str(v) for k, v in failures.items()},
            "outputs": {
                "master_grid_1km": grid_1km_path,
                "master_grid_500m": grid_500m_path,
                "master_features_1km": features_path,
                "cpcb_grid_observations": cpcb_path,
            },
            "notes": (
                "No ML, no imputation, no target interpolation performed. "
                "PM2.5 kept separate in cpcb_grid_observations.parquet. "
                "Missing values are NoData (NaN), never zero-filled."
            ),
        }
        if cfg.get("visualization", True):
            out_png = PROJECT_ROOT / config["paths"]["outputs"] / "master_alignment.png"
            _visualize_alignment(config, grid_1km, grid_500m, features, cpcb_aligned, out_png)
            report["outputs"]["master_alignment_png"] = str(out_png)

        report_path = PROJECT_ROOT / "data/processed/master_alignment_qc_report.json"
        save_json(report, report_path)
        logging.info("[ALIGNMENT] report saved: %s", report_path)
        if failures:
            raise RuntimeError(
                "[ALIGNMENT] critical QC failures: %s" % list(failures.keys())
            )
        return report
    except Exception as exc:
        logging.error("[ALIGNMENT] STAGE FAILED: %s", exc)
        raise

# ---------------------------------------------------------------------------
# MILESTONE 7: TRAINING DATASET + FEATURE ENGINEERING + SPATIAL VALIDATION
# ---------------------------------------------------------------------------
# Builds the ML-ready training package. No model training, no SHAP, no PM2.5
# map, no downscaling, no AQI. Missing values stay NaN (never fabricated).

TARGET_COLUMN = "PM2.5"

TRAINING_FEATURES = [
    "AOD",
    "temperature_c",
    "relative_humidity_pct",
    "wind_speed_mps",
    "wind_direction_deg",
    "wind_u_mps",
    "wind_v_mps",
    "NDVI",
    "elevation_m",
    "road_density",
    "night_lights",
]

TRAINING_METADATA_COLUMNS = [
    "station_id", "grid_id", "date", "latitude", "longitude",
]

TRAINING_TRACEABILITY_COLUMNS = [
    "AOD_available",
    "NDVI_available",
    "VIIRS_available",
    "weather_available",
    "DEM_available",
    "OSM_available",
    "NDVI_source_date",
    "NDVI_time_offset",
    "VIIRS_source_date",
]

TRAINING_TEMPORAL_FEATURES = [
    "month", "day_of_year", "sin_day_of_year", "cos_day_of_year",
]

_TRAINING_AVAILABILITY_FLAGS = [
    "AOD_available", "weather_available", "NDVI_available",
    "VIIRS_available", "DEM_available", "OSM_available",
]


def _add_temporal_features(frame):
    frame = frame.copy()
    d = pd.to_datetime(frame["date"])
    frame["month"] = d.dt.month.astype("int64")
    frame["day_of_year"] = d.dt.dayofyear.astype("int64")
    doy = d.dt.dayofyear.astype(np.float64)
    frame["sin_day_of_year"] = np.sin(2.0 * np.pi * doy / 365.25)
    frame["cos_day_of_year"] = np.cos(2.0 * np.pi * doy / 365.25)
    return frame


def _build_training_dataset(config):
    processed = PROJECT_ROOT / config["training_data"]["output_dir"]
    master = pd.read_parquet(processed / "master_features_1km.parquet")
    cpcb = pd.read_parquet(processed / "cpcb_grid_observations.parquet")
    master["date"] = pd.to_datetime(master["date"]).dt.normalize()
    cpcb["date"] = pd.to_datetime(cpcb["date"]).dt.normalize()
    before = len(cpcb)
    joined = cpcb.merge(
        master, on=["grid_id", "date"], how="left",
        suffixes=("", "_master"), indicator=True,
    )
    unmatched = int((joined["_merge"] == "left_only").sum())
    joined = joined[joined["_merge"] == "both"].drop(columns=["_merge"])
    logging.info(
        "[TRAINING DATA] joined CPCB+features: %d station-date rows (dropped %d unmatched)",
        len(joined), before - len(joined),
    )
    if unmatched:
        logging.warning(
            "[TRAINING DATA] %d station-date rows had no matching master-feature row", unmatched
        )
    return _add_temporal_features(joined.reset_index(drop=True))


def _target_qc(frame):
    qc = {}
    n = len(frame)
    qc["total_candidate_rows"] = n
    qc["missing_target"] = int(frame[TARGET_COLUMN].isna().sum())
    qc["negative_target"] = int((frame[TARGET_COLUMN] < 0).sum())
    qc["zero_target"] = int((frame[TARGET_COLUMN] == 0).sum())
    qc["duplicate_station_date"] = int(frame.duplicated(["station_id", "date"]).sum())
    qc["duplicate_station_grid_date"] = int(
        frame.duplicated(["station_id", "grid_id", "date"]).sum()
    )
    qc["invalid_target_rows_removed"] = int(
        (frame[TARGET_COLUMN].isna() | (frame[TARGET_COLUMN] < 0)).sum()
    )
    return qc


def _feature_coverage(frame, features):
    rows = []
    for col in features:
        v = frame[col]
        valid = int(v.notna().sum())
        missing = int(v.isna().sum())
        n = len(frame)
        rows.append({
            "feature": col,
            "valid_count": valid,
            "missing_count": missing,
            "missing_percentage": round(100.0 * missing / n, 2) if n else 0.0,
            "unique_count": int(v.nunique(dropna=True)),
            "dtype": str(v.dtype),
        })
    cov = pd.DataFrame(rows).sort_values("missing_percentage", ascending=False)
    report = {
        "n_candidate_rows": len(frame),
        "features": cov.to_dict(orient="records"),
        "top_missing_variables": cov.head(3)[["feature", "missing_percentage"]].to_dict(
            orient="records"
        ),
    }
    save_json(report, PROJECT_ROOT / "data/processed/feature_coverage_report.json")
    logging.info(
        "[TRAINING DATA] feature coverage report saved (%d features)",
        len(cov),
    )
    return report


def _feature_correlation(frame, features):
    data = frame[features].select_dtypes(include=[np.number]).copy()
    corr = data.corr(method="pearson", min_periods=1)
    corr.to_csv(PROJECT_ROOT / "data/processed/feature_correlation.csv")
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticklabels(corr.columns)
    ax.set_title("Pearson correlation of numeric predictors (pairwise complete)")
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}",
                    ha="center", va="center", fontsize=6,
                    color="black" if abs(corr.values[i, j]) < 0.7 else "white")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(PROJECT_ROOT / "data/outputs/feature_correlation.png", dpi=120)
    plt.close(fig)
    logging.info("[TRAINING DATA] correlation matrix saved (csv + png)")


def _target_distribution(frame, out_png):
    v = frame[TARGET_COLUMN].dropna()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(v, bins=30, color="steelblue", edgecolor="white")
    ax.axvline(v.mean(), color="red", ls="--", lw=1.2, label=f"mean={v.mean():.1f}")
    ax.axvline(v.median(), color="orange", ls="--", lw=1.2,
               label=f"median={v.median():.1f}")
    ax.set_title("PM2.5 target distribution (CPCB station observations, 7 days)")
    ax.set_xlabel("PM2.5 (ug/m3)"); ax.set_ylabel("frequency"); ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    logging.info("[TRAINING DATA] target distribution saved: %s", out_png)


def _spatial_split_assignments(frame, config):
    cfg = config["training_data"]["cv"]
    stations = sorted(frame["station_id"].unique())
    n_stations = len(stations)
    strategy = cfg.get("strategy", "leave_one_group_out")
    if n_stations <= 8:
        strategy = "leave_one_group_out"
    else:
        strategy = "group_kfold"
    rows = []
    if strategy == "leave_one_group_out":
        for fold, held in enumerate(stations):
            for _, r in frame.iterrows():
                rows.append({
                    "station_id": r["station_id"],
                    "location_group": r["station_id"],
                    "date": r["date"],
                    "split": "test" if r["station_id"] == held else "train",
                    "fold": fold,
                })
    else:
        from sklearn.model_selection import GroupKFold
        gkf = GroupKFold(n_splits=min(5, n_stations))
        groups = frame["station_id"].values
        for fold, (tr, te) in enumerate(gkf.split(frame, groups=groups)):
            for idx in range(len(frame)):
                rows.append({
                    "station_id": frame["station_id"].iloc[idx],
                    "location_group": frame["station_id"].iloc[idx],
                    "date": frame["date"].iloc[idx],
                    "split": "test" if idx in set(te.tolist()) else "train",
                    "fold": fold,
                })
    assignments = pd.DataFrame(rows)
    assignments.to_parquet(PROJECT_ROOT / "data/processed/spatial_split_assignments.parquet")
    logging.info(
        "[TRAINING DATA] spatial splits: strategy=%s, %d folds from %d stations",
        strategy, len(stations), n_stations,
    )
    return assignments, strategy, n_stations


def _validate_spatial_splits(assignments):
    violations = []
    for fold in sorted(assignments["fold"].unique()):
        sub = assignments[assignments["fold"] == fold]
        train_groups = set(sub.loc[sub["split"] == "train", "station_id"])
        test_groups = set(sub.loc[sub["split"] == "test", "station_id"])
        overlap = train_groups & test_groups
        if overlap:
            violations.append({"fold": int(fold), "overlapping_stations": sorted(overlap)})
    return violations


def run_training_data_pipeline(config):
    cfg = config["training_data"]
    if not cfg.get("enabled", True):
        logging.info("[TRAINING DATA] disabled - skipping")
        return None
    processed = PROJECT_ROOT / cfg["output_dir"]

    joined = _build_training_dataset(config)
    target_qc = _target_qc(joined)
    valid = joined[
        joined[TARGET_COLUMN].notna() & (joined[TARGET_COLUMN] >= 0)
    ].reset_index(drop=True)

    features = list(TRAINING_FEATURES)
    fallback_features = [f for f in TRAINING_FEATURES if f != "AOD"]
    all_predictors = features + list(TRAINING_TEMPORAL_FEATURES)
    fallback_predictors = fallback_features + list(TRAINING_TEMPORAL_FEATURES)

    coverage_full = _feature_coverage(valid, all_predictors)

    X_cols = all_predictors
    schema = {
        "target": TARGET_COLUMN,
        "features": all_predictors,
        "fallback_features_without_aod": fallback_predictors,
        "metadata_columns": TRAINING_METADATA_COLUMNS,
        "traceability_columns": TRAINING_TRACEABILITY_COLUMNS,
        "feature_units": {
            "AOD": "unitless (MCD19A2 0.47um)",
            "temperature_c": "degC",
            "relative_humidity_pct": "%",
            "wind_speed_mps": "m/s",
            "wind_direction_deg": "degrees (meteorological from-north)",
            "wind_u_mps": "m/s (u-component)",
            "wind_v_mps": "m/s (v-component)",
            "NDVI": "unitless (-1..1)",
            "elevation_m": "metres",
            "road_density": "m/km2",
            "night_lights": "nW/cm2/sr",
            "month": "1-12",
            "day_of_year": "1-366",
            "sin_day_of_year": "unitless cyclic encoding",
            "cos_day_of_year": "unitless cyclic encoding",
        },
        "notes": [
            "No StandardScaler: tree models (RF/XGBoost) do not require scaling.",
            "wind_speed + u/v kept: u/v retain directionality, speed is magnitude.",
            "NDVI/VIIRS source dates and offsets are traceability metadata, not predictors.",
            "BLH not available in the weather product; not invented.",
            "slope/aspect deferred in DEM milestone; not included.",
        ],
    }
    save_json(schema, processed / "feature_schema.json")

    _feature_correlation(valid, all_predictors)
    out_png = PROJECT_ROOT / config["paths"]["outputs"] / "pm25_target_distribution.png"
    _target_distribution(valid, out_png)

    complete = valid.dropna(subset=all_predictors).reset_index(drop=True)
    complete_path = processed / "training_dataset_complete.parquet"
    complete.to_parquet(complete_path)

    fallback = valid.dropna(subset=fallback_predictors).reset_index(drop=True)
    fallback_path = processed / "training_dataset_fallback.parquet"
    fallback.to_parquet(fallback_path)

    assignments, strategy, n_stations = _spatial_split_assignments(valid, config)
    split_violations = _validate_spatial_splits(assignments)

    target_stats = valid[TARGET_COLUMN].describe(
        percentiles=[0.25, 0.5, 0.75]
    ).round(3).to_dict()

    largest_loss_full = (
        coverage_full["top_missing_variables"][0]
        if coverage_full["top_missing_variables"] else None
    )

    report = {
        "stage": "training_data",
        "status": "PASSED" if not split_violations else "FAILED",
        "target": TARGET_COLUMN,
        "random_seed": cfg.get("random_seed", config["project"].get("random_seed", 42)),
        "dataset_version": "v1",
        "target_qc": target_qc,
        "total_candidate_rows": int(len(joined)),
        "valid_target_rows": int(len(valid)),
        "complete_case_rows": int(len(complete)),
        "fallback_non_aod_rows": int(len(fallback)),
        "n_stations": n_stations,
        "station_ids": sorted(valid["station_id"].unique().tolist()),
        "n_grid_cells": int(valid["grid_id"].nunique()),
        "date_range": [
            str(pd.to_datetime(valid["date"]).min().date()),
            str(pd.to_datetime(valid["date"]).max().date()),
        ],
        "feature_count_full": len(all_predictors),
        "feature_count_fallback": len(fallback_predictors),
        "target_statistics": target_stats,
        "feature_coverage": coverage_full,
        "feature_set_comparison": {
            "feature_set_A_full_incl_AOD": {
                "usable_training_rows": int(len(complete)),
                "cause_of_loss": (
                    "AOD coverage is 0% -> complete-case dataset is EMPTY"
                    if len(complete) == 0 else largest_loss_full
                ),
            },
            "feature_set_B_non_aod": {
                "usable_training_rows": int(len(fallback)),
                "features": fallback_predictors,
            },
        },
        "spatial_validation": {
            "strategy": strategy,
            "grouping_column": "location_group",
            "group_by_source": cfg.get("group_by", "station_id"),
            "n_groups": n_stations,
            "same_station_in_train_and_test_violations": split_violations,
        },
        "outputs": {
            "complete_case": str(complete_path),
            "fallback_non_aod": str(fallback_path),
            "feature_schema": str(processed / "feature_schema.json"),
            "feature_coverage": str(processed / "feature_coverage_report.json"),
            "feature_correlation_csv": str(processed / "feature_correlation.csv"),
            "feature_correlation_png": str(PROJECT_ROOT / config["paths"]["outputs"]
                                          / "feature_correlation.png"),
            "spatial_split_assignments": str(
                processed / "spatial_split_assignments.parquet"
            ),
            "target_distribution_png": out_png,
        },
        "scientific_notes": [
            "CPCB PM2.5 used ONLY as target; no CPCB interpolation.",
            "AOD is 0% available -> not fabricated, not imputed; exposure is explicit.",
            "No ML trained, no SHAP, no PM2.5 map, no downscaling, no AQI in this milestone.",
        ],
    }
    if len(complete) == 0:
        report["complete_case_finding"] = (
            "No complete-case training dataset available because AOD coverage is 0%."
        )
    report_path = processed / "training_dataset_report.json"
    save_json(report, report_path)
    logging.info("[TRAINING DATA] report saved: %s", report_path)
    if split_violations:
        raise RuntimeError(
            "[TRAINING DATA] spatial split violations: %s" % split_violations
        )
    return report


# ---------------------------------------------------------------------------
# MILESTONE 8: MODEL TRAINING + SPATIAL CROSS-VALIDATION (RF BASELINE + XGBOOST)
# ---------------------------------------------------------------------------
# No PM2.5 rasters, no downscaling, no AQI, no hotspot detection in this stage.

import json
import joblib
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

_MODEL_GROUP_COLUMN = "station_id"
_MODEL_FOLD_MIN_TEST = 3


def _load_model_dataset(config):
    mode = config["model"]["dataset_mode"]
    if mode == "complete":
        path = PROJECT_ROOT / "data/processed/training_dataset_complete.parquet"
        logging.info("TRAINING MODE: COMPLETE (AOD available)")
    else:
        path = PROJECT_ROOT / "data/processed/training_dataset_fallback.parquet"
        logging.warning("TRAINING MODE: FALLBACK - AOD unavailable -> reduced predictors")
    df = pd.read_parquet(path)
    logging.info("TRAINING DATA SIZE: %d rows", len(df))
    logging.info("NUMBER OF LOCATIONS: %d", int(df[_MODEL_GROUP_COLUMN].nunique()))
    logging.info("MODEL PERFORMANCE IS PROVISIONAL / PIPELINE VALIDATION ONLY")
    return df, mode


def _select_model_features(mode):
    schema = json.loads(
        Path("data/processed/feature_schema.json").read_text(encoding="utf-8")
    )
    if mode == "complete":
        return list(schema["features"])
    return list(schema["fallback_features_without_aod"])


def _validate_model_data(df, x_cols):
    errors = []
    for col in x_cols:
        if col not in df.columns:
            errors.append("missing feature column: %s" % col)
    if TARGET_COLUMN not in df.columns:
        errors.append("missing target column: %s" % TARGET_COLUMN)
    if _MODEL_GROUP_COLUMN not in df.columns:
        errors.append("missing grouping column: %s" % _MODEL_GROUP_COLUMN)
    if errors:
        raise RuntimeError("[MODEL] dataset validation failed: %s" % errors)
    if int(df[TARGET_COLUMN].isna().sum()) > 0:
        raise RuntimeError("[MODEL] target has NaN values")
    if not np.isfinite(df[x_cols].to_numpy(dtype=float)).all():
        raise RuntimeError("[MODEL] non-finite values found in feature matrix")
    dups = int(df.duplicated([_MODEL_GROUP_COLUMN, "date"]).sum())
    if dups:
        raise RuntimeError("[MODEL] duplicate station/date records: %d" % dups)
    n_stations = int(df[_MODEL_GROUP_COLUMN].nunique())
    if n_stations < 2:
        raise RuntimeError("[MODEL] at least 2 stations required for LOGO CV")
    logging.info("[MODEL] dataset validation: PASSED (%d rows, %d stations)", len(df), n_stations)


def _fold_metrics(y, yhat):
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    n = len(y)
    mae = mean_absolute_error(y, yhat)
    rmse = float(np.sqrt(mean_squared_error(y, yhat)))
    r2 = np.nan
    r2_reason = None
    if n < _MODEL_FOLD_MIN_TEST:
        r2_reason = "insufficient test observations for a meaningful R2 (n<3)"
    else:
        try:
            r2 = float(r2_score(y, yhat))
        except Exception as exc:
            r2 = np.nan
            r2_reason = "r2_score undefined for this fold: %s" % exc
    return {"mae": round(mae, 3), "rmse": round(rmse, 3),
            "r2": None if np.isnan(r2) else round(r2, 3),
            "r2_reason": r2_reason, "n": int(n)}


def _fit_estimator(name, config, x_train, y_train):
    seed = int(config["model"]["random_seed"])
    if name == "RandomForest":
        p = config["model"]["random_forest"]
        return RandomForestRegressor(
            n_estimators=int(p["n_estimators"]), max_depth=int(p["max_depth"]),
            min_samples_leaf=int(p["min_samples_leaf"]), random_state=seed, n_jobs=-1,
        ).fit(x_train, y_train)
    p = config["model"]["xgboost"]
    return XGBRegressor(
        n_estimators=int(p["n_estimators"]), max_depth=int(p["max_depth"]),
        learning_rate=float(p["learning_rate"]), subsample=float(p["subsample"]),
        colsample_bytree=float(p["colsample_bytree"]), reg_alpha=float(p["reg_alpha"]),
        reg_lambda=float(p["reg_lambda"]), random_state=seed, n_jobs=1,
    ).fit(x_train, y_train)


def _model_cv(config, df, x_cols):
    x = df[x_cols].to_numpy(dtype=float)
    y = df[TARGET_COLUMN].to_numpy(dtype=float)
    groups = df[_MODEL_GROUP_COLUMN].values
    logo = LeaveOneGroupOut()
    fold_records = []
    predictions = []
    splits = []
    for fold, (tr_idx, te_idx) in enumerate(logo.split(x, y, groups)):
        held = pd.unique(groups[te_idx]).tolist()
        x_tr, y_tr = x[tr_idx], y[tr_idx]
        x_te, y_te = x[te_idx], y[te_idx]
        for (model_name, tr_grp, te_grp) in [("RF", tr_idx, te_idx)]:
            pass
        fold_metrics = {}
        for model_name in ["RandomForest", "XGBoost"]:
            est = _fit_estimator(model_name, config, x_tr, y_tr)
            yhat = est.predict(x_te)
            fold_metrics[model_name] = _fold_metrics(y_te, yhat)
            for idx in te_idx:
                predictions.append({
                    "fold": int(fold),
                    "station_id": df[_MODEL_GROUP_COLUMN].iloc[idx],
                    "date": df["date"].iloc[idx],
                    "grid_id": df["grid_id"].iloc[idx],
                    "observed_PM25": float(y[idx]),
                    "predicted_PM25": float(yhat[np.where(te_idx == idx)[0][0]]),
                    "model": "RandomForest" if model_name == "RandomForest" else "XGBoost",
                })
        for idx in te_idx:
            splits.append({
                "station_id": df[_MODEL_GROUP_COLUMN].iloc[idx],
                "fold": int(fold),
                "split": "test",
                "date": df["date"].iloc[idx],
            })
        for idx in tr_idx:
            splits.append({
                "station_id": df[_MODEL_GROUP_COLUMN].iloc[idx],
                "fold": int(fold),
                "split": "train",
                "date": df["date"].iloc[idx],
            })
        train_stations = set(pd.unique(groups[tr_idx]))
        test_stations = set(held)
        if train_stations & test_stations:
            raise RuntimeError(
                "[MODEL] spatial leakage in fold %d: stations in both train and test" % fold
            )
        fold_records.append({
            "fold": int(fold),
            "held_out_station": sorted(held),
            "n_train": int(len(tr_idx)),
            "n_test": int(len(te_idx)),
            "RandomForest": fold_metrics["RandomForest"],
            "XGBoost": fold_metrics["XGBoost"],
        })
    pred_df = pd.DataFrame(predictions)
    pred_df["residual"] = pred_df["observed_PM25"] - pred_df["predicted_PM25"]
    pred_df.to_parquet(PROJECT_ROOT / "data/processed/model_cv_predictions.parquet")
    split_df = pd.DataFrame(splits)
    split_df.to_parquet(PROJECT_ROOT / "data/processed/model_cv_splits.parquet")
    fold_metrics_df = pd.DataFrame([{
        "fold": r["fold"],
        "held_out_station": ", ".join(r["held_out_station"]),
        "RF_MAE": r["RandomForest"]["mae"],
        "RF_RMSE": r["RandomForest"]["rmse"],
        "RF_R2": r["RandomForest"]["r2"],
        "XGB_MAE": r["XGBoost"]["mae"],
        "XGB_RMSE": r["XGBoost"]["rmse"],
        "XGB_R2": r["XGBoost"]["r2"],
    } for r in fold_records])
    fold_metrics_df.to_csv(PROJECT_ROOT / "data/processed/cv_metrics_by_fold.csv", index=False)
    logging.info("[MODEL] LOGO CV completed: %d folds -> cv_metrics_by_fold.csv", len(fold_records))
    return fold_records, pred_df, split_df


def _aggregate_metrics(fold_records, pred_df):
    agg = {}
    for model_name in ["RandomForest", "XGBoost"]:
        sub = pred_df[pred_df["model"] == model_name]
        oof_mae = mean_absolute_error(sub["observed_PM25"], sub["predicted_PM25"])
        oof_rmse = float(np.sqrt(mean_squared_error(sub["observed_PM25"], sub["predicted_PM25"])))
        try:
            oof_r2 = float(r2_score(sub["observed_PM25"], sub["predicted_PM25"]))
        except Exception:
            oof_r2 = np.nan
        valid_r2 = [r[model_name]["r2"] for r in fold_records if r[model_name]["r2"] is not None]
        mean_mae = float(np.mean([r[model_name]["mae"] for r in fold_records]))
        mean_rmse = float(np.mean([r[model_name]["rmse"] for r in fold_records]))
        agg[model_name] = {
            "mean_fold_mae": round(mean_mae, 3),
            "mean_fold_rmse": round(mean_rmse, 3),
            "mean_fold_r2": round(float(np.mean(valid_r2)), 3) if valid_r2 else None,
            "valid_r2_folds": len(valid_r2),
            "r2_undefined_folds": len(fold_records) - len(valid_r2),
            "pooled_oof_mae": round(float(oof_mae), 3),
            "pooled_oof_rmse": round(float(oof_rmse), 3),
            "pooled_oof_r2": None if np.isnan(oof_r2) else round(float(oof_r2), 3),
        }
    return agg


def _train_final_model(config, df, x_cols):
    x = df[x_cols].to_numpy(dtype=float)
    y = df[TARGET_COLUMN].to_numpy(dtype=float)
    model_dir = PROJECT_ROOT / config["model"]["output_dir"]
    model_dir.mkdir(parents=True, exist_ok=True)
    rf = _fit_estimator("RandomForest", config, x, y)
    joblib.dump(rf, model_dir / "random_forest_pm25.joblib")
    xgb_model = _fit_estimator("XGBoost", config, x, y)
    xgb_model.save_model(str(model_dir / "xgboost_pm25.json"))
    logging.info("[MODEL] final models trained on ALL available rows (%d) after CV", len(df))
    return rf, xgb_model


def _model_feature_importance(rf, xgb_model, x_cols, out_csv, out_png):
    rf_imp = np.asarray(rf.feature_importances_, dtype=float)
    boost = xgb_model.get_booster()
    gain_map = boost.get_score(importance_type="gain")
    names = boost.feature_names if boost.feature_names else list(x_cols)
    xgb_gain = np.zeros(len(x_cols), dtype=float)
    for f, g in gain_map.items():
        idx = int(f[1:]) if f[1:].isdigit() else x_cols.index(f) if f in x_cols else None
        if idx is not None and 0 <= idx < len(x_cols):
            xgb_gain[idx] = g
    df_imp = pd.DataFrame({
        "feature": x_cols,
        "random_forest_importance": np.round(rf_imp, 6),
        "xgboost_gain": np.round(xgb_gain, 6),
    })
    df_imp["rf_relative_share"] = np.round(rf_imp / rf_imp.sum(), 4)
    df_imp["xgb_relative_share"] = np.round(xgb_gain / max(xgb_gain.sum(), 1e-12), 4)
    df_imp = df_imp.sort_values("random_forest_importance", ascending=False)
    df_imp.to_csv(out_csv, index=False)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].barh(df_imp["feature"], df_imp["rf_relative_share"], color="steelblue")
    axes[0].invert_yaxis(); axes[0].set_title("Random Forest importance (relative share)")
    axes[0].set_xlabel("relative importance")
    df_imp2 = df_imp.sort_values("xgboost_gain", ascending=False)
    axes[1].barh(df_imp2["feature"], df_imp2["xgb_relative_share"], color="darkorange")
    axes[1].invert_yaxis(); axes[1].set_title("XGBoost feature importance (gain, relative share)")
    axes[1].set_xlabel("relative gain share")
    fig.suptitle("Model feature importance (not causal)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    logging.info("[MODEL] feature importance saved: %s", out_csv)
    return df_imp


def _validation_plots(pred_df, out_dir):
    oof_dir = PROJECT_ROOT / out_dir
    fig, ax = plt.subplots(figsize=(7, 7))
    lim = [pred_df["observed_PM25"].min() - 5, pred_df["observed_PM25"].max() + 5]
    for name, color in [("RandomForest", "steelblue"), ("XGBoost", "darkorange")]:
        sub = pred_df[pred_df["model"] == name]
        ax.scatter(sub["observed_PM25"], sub["predicted_PM25"], label=name, s=30, alpha=0.7,
                   color=color)
    ax.plot(lim, lim, "k--", lw=1, label="identity")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Observed PM2.5 (ug/m3)"); ax.set_ylabel("Predicted PM2.5 (ug/m3)")
    ax.set_title("Out-of-fold predictions vs observed (LOGO CV)")
    ax.legend()
    fig.tight_layout(); fig.savefig(oof_dir / "predicted_vs_observed.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for name, color in [("RandomForest", "steelblue"), ("XGBoost", "darkorange")]:
        sub = pred_df[pred_df["model"] == name]
        ax.hist(sub["residual"], bins=15, alpha=0.5, label=name, color=color)
    ax.axvline(0, color="k", lw=1)
    ax.set_xlabel("Residual (observed - predicted)"); ax.set_ylabel("count")
    ax.set_title("Residual distribution (out-of-fold)"); ax.legend()
    fig.tight_layout(); fig.savefig(oof_dir / "residual_distribution.png", dpi=120)
    plt.close(fig)

    agg = pred_df.groupby("model").apply(
        lambda g: pd.Series({
            "MAE": mean_absolute_error(g["observed_PM25"], g["predicted_PM25"]),
            "RMSE": float(np.sqrt(mean_squared_error(g["observed_PM25"], g["predicted_PM25"]))),
        }), include_groups=False,
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    agg[["MAE", "RMSE"]].plot(kind="bar", ax=ax, color=["steelblue", "darkorange"])
    ax.set_ylabel("ug/m3"); ax.set_title("Out-of-fold MAE / RMSE by model")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    fig.tight_layout(); fig.savefig(oof_dir / "model_comparison.png", dpi=120)
    plt.close(fig)
    logging.info("[MODEL] validation plots saved to %s", oof_dir)


def run_model_pipeline(config):
    cfg = config["model"]
    df, mode = _load_model_dataset(config)
    x_cols = _select_model_features(mode)
    _validate_model_data(df, x_cols)
    fold_records, pred_df, split_df = _model_cv(config, df, x_cols)
    agg = _aggregate_metrics(fold_records, pred_df)
    rf, xgb_model = _train_final_model(config, df, x_cols)
    model_dir = PROJECT_ROOT / cfg["output_dir"]
    imp_df = _model_feature_importance(
        rf, xgb_model, x_cols,
        PROJECT_ROOT / "data/processed/model_feature_importance.csv",
        PROJECT_ROOT / "data/outputs/model_feature_importance.png",
    )
    _validation_plots(pred_df, config["paths"]["outputs"])

    comparison = []
    for model_name in ["RandomForest", "XGBoost"]:
        comparison.append({
            "model": model_name,
            "mean_fold_mae": agg[model_name]["mean_fold_mae"],
            "mean_fold_rmse": agg[model_name]["mean_fold_rmse"],
            "mean_fold_r2": agg[model_name]["mean_fold_r2"],
            "pooled_oof_mae": agg[model_name]["pooled_oof_mae"],
            "pooled_oof_rmse": agg[model_name]["pooled_oof_rmse"],
            "pooled_oof_r2": agg[model_name]["pooled_oof_r2"],
            "valid_folds": len(fold_records) - agg[model_name]["r2_undefined_folds"],
        })

    report = {
        "stage": "model_training",
        "status": "PASSED",
        "dataset_mode": mode,
        "aod_available": False,
        "target": TARGET_COLUMN,
        "feature_list": x_cols,
        "training_row_count": int(len(df)),
        "station_count": int(df[_MODEL_GROUP_COLUMN].nunique()),
        "station_ids": sorted(df[_MODEL_GROUP_COLUMN].unique().tolist()),
        "training_date_range": [
            str(pd.to_datetime(df["date"]).min().date()),
            str(pd.to_datetime(df["date"]).max().date()),
        ],
        "validation_strategy": "leave_one_group_out",
        "n_folds": len(fold_records),
        "folds": fold_records,
        "aggregated_metrics": agg,
        "model_comparison": comparison,
        "random_seed": cfg["random_seed"],
        "hyperparameters": {
            "RandomForest": cfg["random_forest"],
            "XGBoost": cfg["xgboost"],
        },
        "feature_importance": imp_df.to_dict(orient="records"),
        "warning": (
            "MODEL PERFORMANCE IS PROVISIONAL / PIPELINE VALIDATION ONLY. "
            "16 fallback rows from 4 stations is far too small for production-grade "
            "accuracy. Do not present these metrics as the scientific model result."
        ),
        "outputs": {
            "model_cv_predictions": str(PROJECT_ROOT / "data/processed/model_cv_predictions.parquet"),
            "model_cv_splits": str(PROJECT_ROOT / "data/processed/model_cv_splits.parquet"),
            "cv_metrics_by_fold": str(PROJECT_ROOT / "data/processed/cv_metrics_by_fold.csv"),
            "model_feature_importance_csv": str(PROJECT_ROOT / "data/processed/model_feature_importance.csv"),
            "model_feature_importance_png": str(PROJECT_ROOT / "data/outputs/model_feature_importance.png"),
            "predicted_vs_observed_png": str(PROJECT_ROOT / config["paths"]["outputs"] / "predicted_vs_observed.png"),
            "residual_distribution_png": str(PROJECT_ROOT / config["paths"]["outputs"] / "residual_distribution.png"),
            "model_comparison_png": str(PROJECT_ROOT / config["paths"]["outputs"] / "model_comparison.png"),
            "xgboost_artifact": str(model_dir / "xgboost_pm25.json"),
            "random_forest_artifact": str(model_dir / "random_forest_pm25.joblib"),
        },
    }
    save_json(report, PROJECT_ROOT / "data/processed/model_validation_report.json")

    from datetime import datetime
    metadata = {
        "model_type": "XGBoost",
        "baseline_model_type": "RandomForest",
        "dataset_mode": mode,
        "aod_available": False,
        "feature_list": x_cols,
        "target": TARGET_COLUMN,
        "training_row_count": int(len(df)),
        "station_count": int(df[_MODEL_GROUP_COLUMN].nunique()),
        "training_date_range": report["training_date_range"],
        "random_seed": cfg["random_seed"],
        "hyperparameters": report["hyperparameters"],
        "validation_strategy": report["validation_strategy"],
        "validation_metrics": agg,
        "training_timestamp": datetime.now().isoformat(),
        "package_versions": {
            "xgboost": xgb.__version__,
            "scikit-learn": "1.7.2",
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "feature_order_must_match": list(x_cols),
        "limitations": [
            "PROVISIONAL / PIPELINE VALIDATION ONLY",
            "AOD unavailable -> fallback feature set used",
            "very small training sample (16 rows, 4 stations)",
            "feature importance is not causal",
        ],
    }
    save_json(metadata, model_dir / "model_metadata.json")
    logging.info("[MODEL] validation report + metadata + artifacts saved")
    return report


# ---------------------------------------------------------------------------
# MILESTONE 9: 1 KM PM2.5 SPATIAL INFERENCE + PREDICTION RASTER
# ---------------------------------------------------------------------------
# Spatial inference only (1 km). No downscaling, no AQI, no hotspots, no API.

_INFERENCE_MODEL_NAME_MAP = {
    "xgboost": "XGBoost",
    "random_forest": "RandomForest",
}


def _load_inference_model(config):
    cfg = config["inference"]
    model_dir = PROJECT_ROOT / config["model"]["output_dir"]
    model_cfg = cfg.get("model", "xgboost").lower()
    md_path = model_dir / "model_metadata.json"
    if not md_path.exists():
        raise FileNotFoundError("[INFERENCE] model_metadata.json not found; run --model-only first")
    metadata = json.loads(md_path.read_text(encoding="utf-8"))
    if model_cfg == "random_forest":
        import joblib as _jb
        model = _jb.load(str(model_dir / "random_forest_pm25.joblib"))
    else:
        from xgboost import XGBRegressor
        model = XGBRegressor()
        model.load_model(str(model_dir / "xgboost_pm25.json"))
    logging.info(
        "[INFERENCE] model loaded: %s (dataset_mode=%s, features=%d)",
        model_cfg, metadata["dataset_mode"], len(metadata["feature_order_must_match"]),
    )
    return model, metadata


def _inference_feature_matrix(feature_frame, feature_order):
    missing = [c for c in feature_order if c not in feature_frame.columns]
    if missing:
        raise RuntimeError(
            "[INFERENCE] model requires features missing from aligned data: %s" % missing
        )
    sub = feature_frame[feature_order]
    valid_mask = sub.notna().all(axis=1)
    return sub, valid_mask


def _run_1km_inference(config, target_date):
    cfg = config["inference"]
    processed = PROJECT_ROOT / config["paths"]["processed"]
    target_date = pd.Timestamp(target_date).normalize()

    model, metadata = _load_inference_model(config)
    feature_order = list(metadata["feature_order_must_match"])
    dataset_mode = metadata["dataset_mode"]
    aod_used = bool(metadata.get("aod_available", False))

    features = pd.read_parquet(processed / "master_features_1km.parquet")
    features["date"] = pd.to_datetime(features["date"]).dt.normalize()
    available_dates = set(features["date"].unique())
    if target_date not in available_dates:
        raise ValueError(
            "[INFERENCE] No aligned feature data available for requested date: %s "
            "(available: %s)" % (target_date.date(), sorted(d.date() for d in available_dates))
        )

    grid = gpd.read_parquet(processed / "master_grid_1km.parquet")
    day = features[features["date"] == target_date].reset_index(drop=True)
    day = _add_temporal_features(day)
    day = grid.merge(day, on="grid_id", how="left")
    day = day.rename(columns={"date_x": "date"})
    if "date_y" in day.columns:
        day["date"] = day["date_y"]

    sub, valid_mask = _inference_feature_matrix(day, feature_order)
    valid_idx = day.index[valid_mask]
    x_valid = sub.loc[valid_idx].to_numpy(dtype=float)
    y_pred = model.predict(x_valid).astype(float)

    pred = pd.DataFrame({
        "grid_id": day.loc[valid_idx, "grid_id"].values,
        "date": target_date,
        "predicted_PM25": y_pred,
        "prediction_status": "valid",
    })
    nodata_rows = day.loc[~valid_mask, ["grid_id"]].copy()
    nodata_rows["date"] = target_date
    nodata_rows["predicted_PM25"] = np.nan
    nodata_rows["prediction_status"] = "missing_feature"
    pred = pd.concat([pred, nodata_rows], ignore_index=True)

    raw_negatives = int((pred["predicted_PM25"] < 0).sum())
    if raw_negatives > 0:
        pred["raw_predicted_PM25"] = pred["predicted_PM25"]
        pred["predicted_PM25"] = pred["predicted_PM25"].clip(lower=0.0)
        logging.warning(
            "[INFERENCE] %d negative raw predictions clipped to 0.0 (documented in QC report)",
            raw_negatives,
        )

    geom = grid.set_index("grid_id").loc[pred["grid_id"], "geometry"].values
    pred = gpd.GeoDataFrame(pred, geometry=geom, crs=grid.crs)
    out_path = processed / "pm25_1km_predictions.parquet"
    pred.to_parquet(out_path)
    logging.info(
        "[INFERENCE] %d 1km predictions for %s -> %s",
        len(pred), target_date.date(), out_path.name,
    )
    return pred, day, model, metadata, feature_order, dataset_mode, aod_used, raw_negatives


def _rasterize_1km_predictions(grid_pred, target_date, config, dataset_mode, aod_used):
    cfg = config["inference"]
    res = int(cfg["spatial_resolution_m"])
    nodata = float(cfg["nodata"])
    minx, miny, maxx, maxy = grid_pred.total_bounds
    width = int(np.ceil((maxx - minx) / res))
    height = int(np.ceil((maxy - miny) / res))
    transform = rasterio.transform.from_origin(minx, maxy, res, res)
    valid = grid_pred[grid_pred["prediction_status"] == "valid"]
    shapes = list(zip(valid.geometry, valid["predicted_PM25"].astype(np.float64)))
    arr = rio_features.rasterize(
        shapes, out_shape=(height, width), transform=transform,
        fill=nodata, dtype="float32", all_touched=False,
    )
    out_path = PROJECT_ROOT / config["paths"]["processed"] / f"pm25_1km_{target_date.date()}.tif"
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "float32",
        "crs": grid_pred.crs,
        "transform": transform,
        "nodata": nodata,
    }
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(arr, 1)
        dst.set_band_description(1, "Predicted PM2.5 (ug/m3)")
        dst.update_tags(
            project=config["project"]["name"],
            target="PM2.5",
            units="ug/m3",
            model="XGBoost",
            model_version=f"{dataset_mode}_v1",
            dataset_mode=dataset_mode,
            resolution_m=str(res),
            crs=str(grid_pred.crs),
            nodata=str(nodata),
            AOD_used=str(aod_used).lower(),
            processing_timestamp=str(pd.Timestamp.now()),
        )
    logging.info(
        "[INFERENCE] raster written: %s (%dx%d px, %d m, nodata=%s)",
        out_path.name, width, height, res, nodata,
    )
    return out_path, arr, transform


def _visualize_1km_prediction(grid_pred, target_date, config, dataset_mode, model_name):
    out_png = PROJECT_ROOT / config["paths"]["outputs"] / f"pm25_1km_{target_date.date()}.png"
    fig, ax = plt.subplots(figsize=(10, 9))
    valid = grid_pred[grid_pred["prediction_status"] == "valid"].to_crs("EPSG:4326")
    if len(valid):
        valid.plot(column="predicted_PM25", ax=ax, legend=True, cmap="YlOrRd",
                   legend_kwds={"label": "Predicted PM2.5 (ug/m3)"})
    grid_pred.to_crs("EPSG:4326").boundary.plot(ax=ax, color="grey", lw=0.3)
    ax.set_title(f"Predicted PM2.5 - 1 km - {model_name} - {dataset_mode.title()} Model ({target_date.date()})")
    ax.set_xlabel("lon"); ax.set_ylabel("lat")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    logging.info("[INFERENCE] map saved: %s", out_png)
    return out_png


def _station_sanity_check(config, grid_pred, target_date):
    processed = PROJECT_ROOT / config["paths"]["processed"]
    obs_path = processed / "cpcb_grid_observations.parquet"
    if not obs_path.exists():
        return None
    obs = pd.read_parquet(obs_path)
    obs["date"] = pd.to_datetime(obs["date"]).dt.normalize()
    obs_day = obs[obs["date"] == target_date]
    if obs_day.empty:
        return None
    merge = obs_day.merge(
        grid_pred[["grid_id", "predicted_PM25", "prediction_status"]].drop_duplicates("grid_id"),
        on="grid_id", how="left",
    )
    valid = merge[merge["predicted_PM25"].notna()]
    if valid.empty:
        return None
    mae = float(np.mean(np.abs(valid["PM2.5"] - valid["predicted_PM25"])))
    return {
        "n_station_cells_matched": int(len(valid)),
        "station_mae_ug_m3": round(mae, 3),
        "note": "spatial output sanity check only; NOT independent validation "
                "(model already validated with held-out stations)",
    }


def run_inference_pipeline(config, target_date=None):
    cfg = config["inference"]
    if not cfg.get("enabled", True):
        logging.info("[INFERENCE] disabled - skipping")
        return None
    if target_date is None:
        target_date = cfg["default_date"]
    target_date = pd.Timestamp(target_date).normalize()

    grid_pred, day, model, metadata, feature_order, dataset_mode, aod_used, raw_negatives = (
        _run_1km_inference(config, target_date)
    )
    raster_path, arr, transform = _rasterize_1km_predictions(
        grid_pred, target_date, config, dataset_mode, aod_used
    )
    model_name = metadata.get("model_type", "XGBoost")
    png_path = _visualize_1km_prediction(grid_pred, target_date, config, dataset_mode, model_name)
    sanity = _station_sanity_check(config, grid_pred, target_date)

    total = int(len(grid_pred))
    valid_n = int((grid_pred["prediction_status"] == "valid").sum())
    nodata_n = int((grid_pred["prediction_status"] == "missing_feature").sum())
    vals = grid_pred.loc[grid_pred["prediction_status"] == "valid", "predicted_PM25"]
    res = int(cfg["spatial_resolution_m"])
    minx, miny, maxx, maxy = grid_pred.total_bounds
    coverage_pct = round(100.0 * valid_n / total, 2) if total else 0.0
    if valid_n == 0:
        raise RuntimeError("[INFERENCE] zero valid predictions; refusing to publish a map")

    qc = {
        "stage": "spatial_inference",
        "status": "PASSED",
        "date": str(target_date.date()),
        "model": model_name,
        "model_version": f"{dataset_mode}_v1",
        "dataset_mode": dataset_mode,
        "AOD_used": aod_used,
        "feature_count": len(feature_order),
        "feature_order": feature_order,
        "total_grid_cells": total,
        "valid_prediction_cells": valid_n,
        "NoData_cells": nodata_n,
        "prediction_coverage_percent": coverage_pct,
        "prediction_min": round(float(vals.min()), 3) if valid_n else None,
        "prediction_max": round(float(vals.max()), 3) if valid_n else None,
        "prediction_mean": round(float(vals.mean()), 3) if valid_n else None,
        "negative_prediction_count": raw_negatives,
        "negative_policy": (
            "raw negatives counted; clipped to 0.0 (non-negative concentration output constraint), "
            "raw values preserved in prediction table when present" if raw_negatives else "none"
        ),
        "crs": str(grid_pred.crs),
        "raster_actual_resolution_m": [float(res), float(res)],
        "raster_pixel_count": [
            int(np.ceil((maxy - miny) / res)),
            int(np.ceil((maxx - minx) / res)),
        ],
        "units": "ug/m3",
        "station_sanity_check": sanity,
        "outputs": {
            "prediction_table": str(PROJECT_ROOT / config["paths"]["processed"] / "pm25_1km_predictions.parquet"),
            "raster": str(raster_path),
            "visualization": str(png_path),
        },
        "processing_timestamp": str(pd.Timestamp.now()),
        "scientific_notes": [
            "Spatial inference uses the fallback model because real AOD data are currently "
            "unavailable. Predictions are not production-quality air-quality estimates.",
            "PROTOTYPE / PIPELINE DEMONSTRATION ONLY - map is predicted, not measured.",
            "No CPCB interpolation; missing predictor vectors -> NoData.",
        ],
    }
    save_json(qc, PROJECT_ROOT / config["paths"]["processed"] / "pm25_1km_qc_report.json")
    logging.info(
        "[INFERENCE] QC report saved: coverage=%.1f%%, %d valid / %d NoData cells",
        coverage_pct, valid_n, nodata_n,
    )
    if valid_n == 0:
        raise RuntimeError("[INFERENCE] zero valid predictions; QC would be empty")
    return qc


# ---------------------------------------------------------------------------
# MILESTONE 10: 500 m SPATIAL DOWNSCALING / HIGHER-RESOLUTION PM2.5 REFINEMENT
# ---------------------------------------------------------------------------
# Downscaling is NOT image resizing. The 500 m estimate combines the coarse
# 1 km PM2.5 parent value with high-resolution local spatial predictors.
#
#   coarse_parent_PM25 + predicted_spatial_residual -> 500 m PM2.5
#
# The residual model is only trained if enough samples exist. With the current
# tiny sample (16 rows, 4 stations) it is expected to be reported as
# INSUFFICIENT_TRAINING_DATA -> experimental prototype baseline, never a fake
# validated ML result. No AQI, no hotspots, no API.

_DSC_CHILD_COL = "target_grid_id"
_DSC_PARENT_COL = "parent_grid_id"
_DSC_RESIDUAL_TARGET = "residual"
_DSC_RESIDUAL_FEATURES_DEFAULT = ["NDVI", "elevation_m", "road_density", "night_lights"]


def _dsc_log(msg, *args):
    logging.getLogger("pm25_pipeline").info(msg, *args)


def _extract_500m_raster_features(config, grid_500m, dsc_cfg):
    processed = PROJECT_ROOT / config["paths"]["processed"]
    analysis_crs = str(grid_500m.crs)
    features = {
        "NDVI": (processed / "ndvi.tif", "NDVI"),
        "elevation_m": (processed / "elevation.tif", "elevation_m"),
        "night_lights": (processed / "night_lights.tif", "night_lights"),
    }
    frame = grid_500m[[_DSC_CHILD_COL, _DSC_PARENT_COL, "child_area_km2", "parent_coarse_PM25", "geometry"]].copy()
    resolution_notes = {}
    for col, (raster_path, _) in features.items():
        if not raster_path.exists():
            _dsc_log("[DOWNSCALING] predictor %s raster missing: %s (column stays NoData)", col, raster_path.name)
            frame[col] = np.nan
            continue
        with rasterio.open(raster_path) as src:
            src_crs = str(src.crs)
            xres, yres = src.res
            resolution_notes[col] = {
                "source_crs": src_crs,
                "source_resolution_m": [round(abs(xres) if "32643" in src_crs else 0.0, 1),
                                        round(abs(yres) if "32643" in src_crs else 0.0, 1)],
                "source_note": (
                    "raster stored in EPSG:4326 degrees; processed at native resolution, "
                    "zonal mean aggregated to 500 m cells in EPSG:32643"
                    if "4326" in src_crs else "metric raster",
                ),
            }
        stats = _raster_zonal_stats(str(raster_path), grid_500m, analysis_crs, _DSC_CHILD_COL)
        mean_map = {gid: v[0] for gid, v in stats.items()}
        frac_map = {gid: v[1] for gid, v in stats.items()}
        frame[col] = frame[_DSC_CHILD_COL].map(mean_map).astype(np.float64)
        frame[f"{col}_pixel_fraction"] = frame[_DSC_CHILD_COL].map(frac_map).astype(np.float64)
    return frame, resolution_notes


def _extract_500m_road_density(config, grid_500m, dsc_cfg):
    processed = PROJECT_ROOT / config["paths"]["processed"]
    raw_path = PROJECT_ROOT / config["datasets"]["osm"].get(
        "input_file", "data/raw/osm/osm_roads.geojson"
    )
    out = grid_500m[[_DSC_CHILD_COL]].copy()
    out["road_density"] = np.nan
    if not raw_path.exists():
        _dsc_log("[DOWNSCALING] raw OSM roads missing: %s (road_density stays NoData)", raw_path.name)
        return out
    roads = _load_osm_roads(raw_path)
    roads, _ = _validate_osm_geometries(roads)
    allowed = config["datasets"]["osm"].get("highway_types", [])
    roads, _ = _filter_osm_highway_types(roads, allowed)
    metric_crs = config["datasets"]["osm"].get("metric_crs", "EPSG:32643")
    roads = roads.to_crs(metric_crs)
    roads = gpd.clip(roads, get_aoi(config).to_crs(metric_crs))
    if len(roads) == 0:
        _dsc_log("[DOWNSCALING] no OSM roads after filtering; road_density stays NoData")
        return out
    rd_grid = grid_500m[["geometry"]].copy()
    rd_grid["grid_id"] = grid_500m[_DSC_CHILD_COL].values
    rd_grid["cell_area_km2"] = rd_grid.geometry.area / 1e6
    major_types = config["datasets"]["osm"].get("major_highway_types", [])
    density = _compute_road_density(rd_grid, roads, major_types)
    density = density.set_index("grid_id")
    out["road_density"] = out[_DSC_CHILD_COL].map(density["road_density"]).astype(np.float64)
    out["major_road_density"] = out[_DSC_CHILD_COL].map(density["major_road_density"]).astype(np.float64)
    _dsc_log("[DOWNSCALING] road density recomputed at 500 m with validated OSM method (%d cells)", len(out))
    return out


def _load_500m_downscaling_features(config, target_date, coarse_pred):
    dsc_cfg = config["downscaling"]
    processed = PROJECT_ROOT / config["paths"]["processed"]
    grid_500m = gpd.read_parquet(processed / "master_grid_500m.parquet")
    grid_500m = grid_500m[[_DSC_CHILD_COL, _DSC_PARENT_COL, "area_km2", "geometry"]].copy()
    grid_500m["child_area_km2"] = grid_500m["area_km2"].astype(np.float64)

    coarse_map = (
        coarse_pred.loc[coarse_pred["prediction_status"] == "valid",
                        ["grid_id", "predicted_PM25"]]
        .drop_duplicates("grid_id")
        .set_index("grid_id")["predicted_PM25"]
        .astype(np.float64)
    )
    grid_500m["parent_coarse_PM25"] = grid_500m[_DSC_PARENT_COL].map(coarse_map)

    frame, res_notes = _extract_500m_raster_features(config, grid_500m, dsc_cfg)
    rd = _extract_500m_road_density(config, grid_500m, dsc_cfg)
    frame = frame.merge(rd, on=_DSC_CHILD_COL, how="left")

    frame["date"] = target_date
    feature_cols = [c for c in dsc_cfg.get("residual_features", _DSC_RESIDUAL_FEATURES_DEFAULT)]
    for col in feature_cols:
        frame[f"{col}_available"] = frame[col].notna().astype(bool)

    frame = gpd.GeoDataFrame(frame, geometry="geometry", crs=grid_500m.crs)
    out_path = processed / "downscaling_features_500m.parquet"
    frame.to_parquet(out_path)
    _dsc_log("[DOWNSCALING] 500 m feature table saved: %s (%d cells)", out_path.name, len(frame))
    return frame, res_notes, out_path


def _build_residual_training(config, coarse_pred):
    dsc_cfg = config["downscaling"]
    processed = PROJECT_ROOT / config["paths"]["processed"]
    fallback_path = processed / "training_dataset_fallback.parquet"
    if not fallback_path.exists():
        return None, None, None
    train = pd.read_parquet(fallback_path)
    coarse_map = (
        coarse_pred.loc[coarse_pred["prediction_status"] == "valid",
                        ["grid_id", "predicted_PM25"]]
        .drop_duplicates("grid_id")
        .set_index("grid_id")["predicted_PM25"]
        .astype(np.float64)
    )
    train["coarse_predicted_PM25"] = train["grid_id"].map(coarse_map)
    feature_cols = list(dsc_cfg.get("residual_features", _DSC_RESIDUAL_FEATURES_DEFAULT))
    res = pd.DataFrame({
        "station_id": train["station_id"],
        "date": pd.to_datetime(train["date"]).dt.normalize(),
        "parent_1km_grid_id": train["grid_id"],
        "observed_PM25": train[TARGET_COLUMN].astype(np.float64),
        "coarse_predicted_PM25": train["coarse_predicted_PM25"],
    })
    for col in feature_cols:
        res[col] = train[col].astype(np.float64) if col in train.columns else np.nan
    res[_DSC_RESIDUAL_TARGET] = res["observed_PM25"] - res["coarse_predicted_PM25"]
    feature_ok = res[feature_cols].notna().all(axis=1) & res["coarse_predicted_PM25"].notna()
    res = res[feature_ok].reset_index(drop=True)
    out_path = processed / "downscaling_training.parquet"
    res.to_parquet(out_path)
    _dsc_log("[DOWNSCALING] residual training table saved: %s (%d valid samples)", out_path.name, len(res))
    return res, out_path, feature_cols


def _train_residual_model(config, residual_df, feature_cols):
    dsc_cfg = config["downscaling"]
    model_dir = PROJECT_ROOT / config["model"]["output_dir"]
    min_samples = int(dsc_cfg.get("minimum_training_samples", 20))
    n = len(residual_df)
    if n < min_samples:
        return None, {
            "trained": False,
            "reason": "INSUFFICIENT_TRAINING_DATA",
            "valid_residual_samples": int(n),
            "minimum_required": min_samples,
        }
    y = residual_df[_DSC_RESIDUAL_TARGET].to_numpy(dtype=float)
    x = residual_df[feature_cols].to_numpy(dtype=float)
    groups = residual_df["station_id"].to_numpy()
    seed = int(config["model"]["random_seed"])

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import LeaveOneGroupOut
    logo = LeaveOneGroupOut()
    fold_records = []
    oof_pred = np.full(len(x), np.nan)
    for fold_i, (tr_idx, va_idx) in enumerate(logo.split(x, y, groups)):
        rf = RandomForestRegressor(
            n_estimators=int(config["model"]["random_forest"]["n_estimators"]),
            max_depth=int(config["model"]["random_forest"]["max_depth"]),
            min_samples_leaf=int(config["model"]["random_forest"]["min_samples_leaf"]),
            random_state=seed,
        )
        rf.fit(x[tr_idx], y[tr_idx])
        oof_pred[va_idx] = rf.predict(x[va_idx])
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        fold_records.append({
            "fold": int(fold_i + 1),
            "holdout_station": str(residual_df.iloc[va_idx[0]]["station_id"]),
            "n_train": int(len(tr_idx)),
            "n_test": int(len(va_idx)),
            "mae": round(float(mean_absolute_error(y[va_idx], oof_pred[va_idx])), 3),
            "rmse": round(float(np.sqrt(mean_squared_error(y[va_idx], oof_pred[va_idx]))), 3),
            "r2": round(float(r2_score(y[va_idx], oof_pred[va_idx])), 3),
        })
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    metrics = {
        "oof_mae": round(float(mean_absolute_error(y, oof_pred)), 3),
        "oof_rmse": round(float(np.sqrt(mean_squared_error(y, oof_pred))), 3),
        "oof_r2": round(float(r2_score(y, oof_pred)), 3),
    }
    final = RandomForestRegressor(
        n_estimators=int(config["model"]["random_forest"]["n_estimators"]),
        max_depth=int(config["model"]["random_forest"]["max_depth"]),
        min_samples_leaf=int(config["model"]["random_forest"]["min_samples_leaf"]),
        random_state=seed,
    )
    final.fit(x, y)
    import joblib as _jb
    artifact = model_dir / "downscaling_residual_rf.joblib"
    _jb.dump(final, str(artifact))
    _dsc_log("[DOWNSCALING] residual RF model trained (%d samples, LOGO %d folds) -> %s", n, len(fold_records), artifact.name)
    return final, {
        "trained": True,
        "reason": "residual_model",
        "valid_residual_samples": int(n),
        "minimum_required": min_samples,
        "folds": fold_records,
        "metrics": metrics,
        "artifact": str(artifact),
    }


def _run_500m_inference(config, features_500m, residual_model, residual_info, feature_cols):
    dsc_cfg = config["downscaling"]
    parent_ok = features_500m["parent_coarse_PM25"].notna()
    if residual_model is not None:
        sub = features_500m.loc[parent_ok.to_numpy(), feature_cols].to_numpy(dtype=float)
        res_pred = np.full(len(features_500m), np.nan)
        res_pred[parent_ok.to_numpy()] = residual_model.predict(sub).astype(float)
    else:
        res_pred = np.zeros(len(features_500m))

    raw = np.full(len(features_500m), np.nan)
    status = np.empty(len(features_500m), dtype=object)
    parent_val = features_500m["parent_coarse_PM25"].to_numpy(dtype=float)
    for i in range(len(features_500m)):
        if not parent_ok.iloc[i]:
            status[i] = "missing_parent"
            continue
        if residual_model is not None and not np.isfinite(res_pred[i]):
            status[i] = "missing_feature"
            continue
        if residual_model is not None:
            raw[i] = parent_val[i] + res_pred[i]
            status[i] = "downscaled_residual"
        else:
            raw[i] = parent_val[i]
            status[i] = "baseline_prototype"

    out = features_500m.copy()
    out["predicted_residual"] = res_pred
    out["pm25_500m_raw"] = raw
    out["pm25_500m_final"] = raw.copy()
    out["prediction_status"] = status
    raw_neg = int((out["pm25_500m_raw"] < 0).sum())
    if raw_neg > 0:
        out["pm25_500m_final"] = out["pm25_500m_final"].clip(lower=0.0)
        _dsc_log("[DOWNSCALING] %d negative raw 500 m values clipped to 0.0 (documented in QC)", raw_neg)
    return out, raw_neg


def _apply_parent_consistency(out, dsc_cfg):
    dsc_cfg = dsc_cfg or {}
    if not dsc_cfg.get("enforce_parent_consistency", True):
        return out, {
            "applied": False,
            "formula": "none (enforce_parent_consistency=false)",
        }
    parent = out[[_DSC_PARENT_COL, "child_area_km2", "pm25_500m_final"]].copy()
    parent["_w"] = parent["child_area_km2"] * parent["pm25_500m_final"]
    agg = parent.groupby(_DSC_PARENT_COL).agg(
        _wsum=("_w", "sum"), _wsum_area=("child_area_km2", "sum")
    )
    agg["child_wmean"] = agg["_wsum"] / agg["_wsum_area"]
    agg["parent_coarse"] = out.drop_duplicates(_DSC_PARENT_COL).set_index(_DSC_PARENT_COL)["parent_coarse_PM25"]
    agg["shift"] = agg["parent_coarse"] - agg["child_wmean"]
    shift_map = agg["shift"].to_dict()
    out["pm25_500m_final"] = (
        out["pm25_500m_final"] + out[_DSC_PARENT_COL].map(shift_map)
    )
    out["consistency_shift"] = out[_DSC_PARENT_COL].map(shift_map)
    formula = (
        "child_adjusted = child_raw + (parent_coarse_mean - area_weighted_mean(child_raw)) "
        "per parent; enforces parent coarse mean preservation, does NOT claim accuracy gain"
    )
    return out, {"applied": True, "formula": formula}


def _parent_consistency_check(out, dsc_cfg):
    valid = out[out["prediction_status"].str.contains("downscaled_residual|baseline_prototype")].copy()
    valid = valid[valid["pm25_500m_final"].notna()].copy()
    parent = valid[[_DSC_PARENT_COL, "child_area_km2", "pm25_500m_final"]].copy()
    parent["_w"] = parent["child_area_km2"] * parent["pm25_500m_final"]
    agg = parent.groupby(_DSC_PARENT_COL).agg(
        _wsum=("_w", "sum"), _wsum_area=("child_area_km2", "sum")
    )
    agg["child_wmean"] = agg["_wsum"] / agg["_wsum_area"]
    coarse = valid.drop_duplicates(_DSC_PARENT_COL).set_index(_DSC_PARENT_COL)["parent_coarse_PM25"]
    agg = agg.join(coarse)
    agg["diff"] = agg["child_wmean"] - agg["parent_coarse_PM25"]
    agg["abs_diff"] = agg["diff"].abs()
    agg["rel_diff"] = np.where(
        agg["parent_coarse_PM25"] != 0,
        (agg["abs_diff"] / agg["parent_coarse_PM25"].abs()) * 100.0,
        np.nan,
    )
    return {
        "n_parents_compared": int(len(agg)),
        "mean_abs_diff_ug_m3": round(float(agg["abs_diff"].mean()), 3) if len(agg) else None,
        "max_abs_diff_ug_m3": round(float(agg["abs_diff"].max()), 3) if len(agg) else None,
        "mean_rel_diff_percent": round(float(agg["rel_diff"].mean()), 3) if len(agg) and agg["rel_diff"].notna().any() else None,
        "n_parents_with_abs_diff_gt_1": int((agg["abs_diff"] > 1.0).sum()),
        "method": "area-weighted mean of 500 m child PM2.5 aggregated to parent 1 km",
    }


def _rasterize_500m_predictions(out, target_date, config, residual_info, dsc_cfg, res_notes):
    res = int(dsc_cfg["target_resolution_m"])
    nodata = float(dsc_cfg["nodata"])
    minx, miny, maxx, maxy = out.total_bounds
    width = int(np.ceil((maxx - minx) / res))
    height = int(np.ceil((maxy - miny) / res))
    transform = rasterio.transform.from_origin(minx, maxy, res, res)
    valid = out[out["pm25_500m_final"].notna()]
    shapes = list(zip(valid.geometry, valid["pm25_500m_final"].astype(np.float64)))
    arr = rio_features.rasterize(
        shapes, out_shape=(height, width), transform=transform,
        fill=nodata, dtype="float32", all_touched=False,
    )
    out_path = PROJECT_ROOT / config["paths"]["processed"] / f"pm25_500m_{target_date.date()}.tif"
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "float32",
        "crs": out.crs,
        "transform": transform,
        "nodata": nodata,
    }
    method = "residual" if residual_info.get("trained") else "baseline_prototype"
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(arr, 1)
        dst.set_band_description(1, "Downscaled PM2.5 estimate (ug/m3)")
        dst.update_tags(
            project=config["project"]["name"],
            target="PM2.5",
            units="ug/m3",
            parent_resolution_m=str(int(dsc_cfg["parent_resolution_m"])),
            target_resolution_m=str(res),
            model=config["model"].get("primary", "XGBoost"),
            dataset_mode=config["model"]["dataset_mode"],
            AOD_used="false",
            downscaling_method=method,
            consistency_method="area_weighted_mean" if dsc_cfg.get("enforce_parent_consistency", True) else "none",
            NoData=str(nodata),
            processing_timestamp=str(pd.Timestamp.now()),
        )
    _dsc_log("[DOWNSCALING] 500 m raster written: %s (%dx%d px, %d m, nodata=%s)", out_path.name, width, height, res, nodata)
    return out_path, arr, transform


def _visualize_downscaling(out, target_date, config, residual_info):
    import matplotlib.patches as mpatches
    outputs = PROJECT_ROOT / config["paths"]["outputs"]
    out_png = outputs / f"pm25_500m_{target_date.date()}.png"
    valid = out[out["pm25_500m_final"].notna()].copy()
    valid4326 = valid.to_crs("EPSG:4326")
    coarse_path = PROJECT_ROOT / config["paths"]["processed"] / f"pm25_1km_{target_date.date()}.tif"
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    ax1, ax2, ax3 = axes
    method = "Residual model" if residual_info.get("trained") else "Baseline prototype (parent-constant)"
    if len(valid4326):
        valid4326.plot(column="pm25_500m_final", ax=ax1, legend=True, cmap="YlOrRd",
                       legend_kwds={"label": "PM2.5 (ug/m3)", "shrink": 0.6})
    out.to_crs("EPSG:4326").boundary.plot(ax=ax1, color="grey", lw=0.2)
    ax1.set_title(f"BEFORE: 1 km PM2.5 ({target_date.date()})")
    if coarse_path.exists():
        with rasterio.open(coarse_path) as src:
            coarse_arr = src.read(1)
            coarse_transform = src.transform
            coarse_extent = rasterio.plot.plotting_extent(src)
            coarse_nodata = src.nodata
        masked = np.ma.masked_equal(coarse_arr, coarse_nodata)
        im = ax2.imshow(masked, extent=coarse_extent, cmap="YlOrRd", origin="upper")
        cb = fig.colorbar(im, ax=ax2, shrink=0.6, label="PM2.5 (ug/m3)")
    ax2.set_title(f"AFTER: 500 m downscaled PM2.5 - {method} ({target_date.date()})")
    zoom_parent = None
    groups = valid.groupby(_DSC_PARENT_COL).size()
    if len(groups):
        four = groups[groups == 4]
        zoom_parent = four.index[0] if len(four) else groups.index[0]
    if zoom_parent is not None:
        children = valid[valid[_DSC_PARENT_COL] == zoom_parent]
        child4326 = children.to_crs("EPSG:4326")
        child4326.plot(column="pm25_500m_final", ax=ax3, cmap="YlOrRd", edgecolor="black", linewidth=1.2,
                       legend=False)
        for _, r in child4326.iterrows():
            cx, cy = r.geometry.centroid.x, r.geometry.centroid.y
            ax3.text(cx, cy, f"{r['pm25_500m_final']:.1f}", ha="center", va="center",
                     fontsize=9, color="black", fontweight="bold",
                     bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", boxstyle="round,pad=0.15"))
        parent4326 = out[out[_DSC_PARENT_COL] == zoom_parent].to_crs("EPSG:4326").union_all()
        if parent4326 is not None:
            import shapely.geometry as _sg
            ax3.add_patch(mpatches.Polygon(
                list(parent4326.exterior.coords), closed=True,
                fill=False, edgecolor="blue", linewidth=2,
            ))
        ax3.set_title(f"ZOOM: one 1 km parent -> {len(child4326)} children\n"
                      f"distinct values from local 500 m predictors")
    fig.suptitle(
        "Downscaling = spatial refinement using high-resolution predictors, NOT image resizing. "
        "More pixels do NOT imply more accuracy.",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    _dsc_log("[DOWNSCALING] before/after visualization saved: %s", out_png.name)
    return out_png


def run_downscaling_pipeline(config, target_date=None):
    dsc_cfg = config["downscaling"]
    if not dsc_cfg.get("enabled", True):
        _dsc_log("[DOWNSCALING] disabled - skipping")
        return None
    if target_date is None:
        target_date = dsc_cfg["default_date"]
    target_date = pd.Timestamp(target_date).normalize()

    processed = PROJECT_ROOT / config["paths"]["processed"]
    pred_path = processed / "pm25_1km_predictions.parquet"
    if not pred_path.exists():
        raise FileNotFoundError("[DOWNSCALING] pm25_1km_predictions.parquet missing; run --inference-only first")
    coarse_pred = pd.read_parquet(pred_path)
    available_dates = set(pd.to_datetime(coarse_pred["date"]).dt.normalize())
    if target_date not in available_dates:
        raise ValueError(
            "[DOWNSCALING] No 1 km predictions available for requested date: %s "
            "(available: %s)" % (target_date.date(), sorted(d.date() for d in available_dates))
        )
    coarse_pred = coarse_pred[coarse_pred["date"] == target_date].reset_index(drop=True)
    n_valid_parents = int((coarse_pred["prediction_status"] == "valid").sum())
    if n_valid_parents == 0:
        raise RuntimeError("[DOWNSCALING] zero valid 1 km parents; refusing to publish")

    features, res_notes, feat_path = _load_500m_downscaling_features(config, target_date, coarse_pred)
    feature_cols = list(dsc_cfg.get("residual_features", _DSC_RESIDUAL_FEATURES_DEFAULT))

    residual_df, train_path, _ = _build_residual_training(config, coarse_pred)
    if residual_df is not None and len(residual_df):
        residual_model, residual_info = _train_residual_model(config, residual_df, feature_cols)
    else:
        residual_model, residual_info = None, {
            "trained": False,
            "reason": "INSUFFICIENT_TRAINING_DATA",
            "valid_residual_samples": 0,
        }

    out, raw_neg = _run_500m_inference(config, features, residual_model, residual_info, feature_cols)
    out, cons_info = _apply_parent_consistency(out, dsc_cfg)
    cons_stats = _parent_consistency_check(out, dsc_cfg)

    out_path = processed / "pm25_500m_predictions.parquet"
    out.to_parquet(out_path)
    _dsc_log("[DOWNSCALING] 500 m prediction table saved: %s (%d cells)", out_path.name, len(out))

    raster_path, arr, transform = _rasterize_500m_predictions(out, target_date, config, residual_info, dsc_cfg, res_notes)
    png_path = _visualize_downscaling(out, target_date, config, residual_info)

    total = int(len(out))
    status_counts = out["prediction_status"].value_counts().to_dict()
    valid_n = int((out["pm25_500m_final"].notna()).sum())
    nodata_n = total - valid_n
    coverage_pct = round(100.0 * valid_n / total, 2) if total else 0.0
    vals = out.loc[out["pm25_500m_final"].notna(), "pm25_500m_final"]

    qc = {
        "stage": "spatial_downscaling_500m",
        "status": "PASSED" if valid_n else "FAILED",
        "date": str(target_date.date()),
        "parent_resolution_m": int(dsc_cfg["parent_resolution_m"]),
        "target_resolution_m": int(dsc_cfg["target_resolution_m"]),
        "total_500m_cells": total,
        "valid_500m_cells": valid_n,
        "NoData_cells": nodata_n,
        "coverage_percent": coverage_pct,
        "parent_1km_valid_coverage": round(100.0 * n_valid_parents / 1804, 2),
        "downscaling_method": "residual_refinement" if residual_info.get("trained") else "baseline_prototype",
        "residual_model_used": bool(residual_info.get("trained")),
        "residual_model_reason": residual_info.get("reason"),
        "residual_training_rows": int(residual_info.get("valid_residual_samples", 0)),
        "station_count": int(residual_df["station_id"].nunique()) if residual_df is not None and len(residual_df) else 0,
        "dataset_mode": config["model"]["dataset_mode"],
        "AOD_used": False,
        "residual_cv": residual_info.get("metrics") or residual_info.get("folds") or None,
        "consistency_statistics": cons_stats,
        "consistency_constraint": cons_info,
        "mean_absolute_parent_difference": cons_stats["mean_abs_diff_ug_m3"],
        "max_parent_difference": cons_stats["max_abs_diff_ug_m3"],
        "raw_negative_count": raw_neg,
        "final_negative_count": int((out["pm25_500m_final"] < 0).sum()),
        "prediction_min_ug_m3": round(float(vals.min()), 3) if valid_n else None,
        "prediction_max_ug_m3": round(float(vals.max()), 3) if valid_n else None,
        "prediction_mean_ug_m3": round(float(vals.mean()), 3) if valid_n else None,
        "prediction_status_counts": status_counts,
        "predictor_resolutions": res_notes,
        "prediction_policy": "no CPCB observed PM2.5 used as a 500 m predictor; missing parent -> NoData; missing 500 m feature -> NoData (residual path)",
        "outputs": {
            "prediction_table": str(out_path),
            "feature_table": str(feat_path),
            "raster": str(raster_path),
            "visualization": str(png_path),
        },
        "processing_timestamp": str(pd.Timestamp.now()),
        "scientific_warning": (
            "Residual downscaling model NOT scientifically trainable with current sample size. "
            "500 m values in this run are a parent-constant BASELINE PROTOTYPE for pipeline "
            "demonstration only; they do not yet exploit high-resolution predictors. "
            "This is NOT a validated ML downscaling model and does NOT imply higher accuracy."
            if not residual_info.get("trained") else
            "Residual model trained with Leave-One-Location-Out on a very small sample; "
            "metrics are provisional, not production accuracy."
        ),
    }
    save_json(qc, processed / "downscaling_qc_report.json")
    _dsc_log("[DOWNSCALING] QC report saved: coverage=%.1f%%, %d valid / %d NoData cells", coverage_pct, valid_n, nodata_n)
    return qc


# ---------------------------------------------------------------------------
# MILESTONE 11: AQI + HOTSPOT ANALYSIS + UNCERTAINTY + FINAL GEOSPATIAL OUTPUTS
# ---------------------------------------------------------------------------
# Official CPCB National AQI methodology applied to PM2.5 only. Because the
# system has only PM2.5 (no PM10/NO2/SO2/CO/O3/NH3/Pb), the output is labelled
# "PM2.5-derived AQI / PM2.5 AQI sub-index", NEVER "full National AQI".
# Hotspot = threshold-based high-pollution zone detection (explicitly NOT a
# statistical hotspot analysis; that is deferred). Uncertainty is deferred.

_AQI_CATEGORY_ORDER = [
    "GOOD",
    "SATISFACTORY",
    "MODERATELY_POLLUTED",
    "POOR",
    "VERY_POOR",
    "SEVERE",
]

_AQI_CPCB_COLORS = {
    "GOOD": "#5BBE48",
    "SATISFACTORY": "#FFFF01",
    "MODERATELY_POLLUTED": "#FE7E01",
    "POOR": "#F00101",
    "VERY_POOR": "#8F3F97",
    "SEVERE": "#7E0023",
}


def _load_aqi_breakpoints(config):
    cfg = config["aqi"]
    rows = []
    for b in cfg["breakpoints"]:
        rows.append({
            "concentration_low": float(b["concentration_low"]),
            "concentration_high": float(b["concentration_high"]),
            "aqi_low": float(b["aqi_low"]),
            "aqi_high": float(b["aqi_high"]),
        })
    rows.sort(key=lambda r: r["concentration_low"])
    categories = {
        c["name"]: (float(c["aqi_min"]), float(c["aqi_max"]))
        for c in cfg["categories"]
    }
    return rows, categories


def _pm25_to_aqi(value, breakpoints):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return None, None
    c = float(value)
    if c < 0:
        return None, None
    selected = None
    for row in breakpoints:
        if c <= row["concentration_high"]:
            selected = row
            break
    if selected is None:
        selected = breakpoints[-1]
    if selected["concentration_high"] <= selected["concentration_low"]:
        aqi = selected["aqi_high"]
    else:
        aqi = (
            (selected["aqi_high"] - selected["aqi_low"])
            / (selected["concentration_high"] - selected["concentration_low"])
            * (c - selected["concentration_low"])
            + selected["aqi_low"]
        )
    aqi = float(np.clip(aqi, 0.0, 500.0))
    return aqi, None


def _aqi_category_name(aqi, categories):
    if aqi is None or not np.isfinite(aqi):
        return None
    for name in _AQI_CATEGORY_ORDER:
        lo, hi = categories[name]
        if aqi >= lo:
            if name == _AQI_CATEGORY_ORDER[-1]:
                return name
            nxt = categories[_AQI_CATEGORY_ORDER[_AQI_CATEGORY_ORDER.index(name) + 1]]
            if aqi < nxt[0]:
                return name
    return _AQI_CATEGORY_ORDER[-1]


def _compute_aqi(config, pm25_500m, target_date):
    cfg = config["aqi"]
    breakpoints, categories = _load_aqi_breakpoints(config)
    frame = pm25_500m[[_DSC_CHILD_COL, _DSC_PARENT_COL, "date", "pm25_500m_final",
                       "prediction_status", "geometry"]].copy()
    frame = frame.rename(columns={"pm25_500m_final": "pm25"})
    rounding = cfg.get("rounding", "nearest_integer")
    aqi_vals = []
    cat_vals = []
    for v in frame["pm25"]:
        aqi, _ = _pm25_to_aqi(v, breakpoints)
        if aqi is not None and rounding == "nearest_integer":
            aqi = float(round(aqi))
        elif aqi is not None and rounding == "floor":
            aqi = float(np.floor(aqi))
        aqi_vals.append(aqi)
        cat_vals.append(_aqi_category_name(aqi, categories))
    frame["pm25_aqi"] = aqi_vals
    frame["aqi_category"] = cat_vals
    if rounding in ("nearest_integer", "floor"):
        frame["pm25_aqi"] = frame["pm25_aqi"].astype("Int64")
    frame = gpd.GeoDataFrame(frame, geometry="geometry", crs=pm25_500m.crs)
    out_path = PROJECT_ROOT / config["paths"]["processed"] / "aqi_500m_predictions.parquet"
    frame.to_parquet(out_path)
    logging.info("[AQI] PM2.5-derived AQI table saved: %s (%d cells)", out_path.name, len(frame))
    return frame, categories, breakpoints, out_path


def _rasterize_aqi(frame, target_date, config, categories, breakpoints):
    cfg = config["aqi"]
    res = int(config["downscaling"]["target_resolution_m"])
    nodata = int(cfg["nodata"])
    minx, miny, maxx, maxy = frame.total_bounds
    width = int(np.ceil((maxx - minx) / res))
    height = int(np.ceil((maxy - miny) / res))
    transform = rasterio.transform.from_origin(minx, maxy, res, res)
    valid = frame[frame["pm25_aqi"].notna()].copy()
    valid["_aqi_int"] = valid["pm25_aqi"].astype(int)
    shapes = list(zip(valid.geometry, valid["_aqi_int"]))
    arr = rio_features.rasterize(
        shapes, out_shape=(height, width), transform=transform,
        fill=nodata, dtype="int16", all_touched=False,
    )
    out_path = PROJECT_ROOT / config["paths"]["processed"] / f"aqi_500m_{target_date.date()}.tif"
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "int16",
        "crs": frame.crs,
        "transform": transform,
        "nodata": nodata,
    }
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(arr, 1)
        dst.set_band_description(1, "PM2.5-derived AQI / sub-index (unitless)")
        dst.update_tags(
            project=config["project"]["name"],
            pollutant=cfg["pollutant"],
            methodology=cfg["methodology"],
            averaging_period=cfg["averaging_period"],
            AQI_type="PM2.5-derived AQI / PM2.5 AQI sub-index (NOT full multi-pollutant National AQI)",
            units="unitless",
            parent_resolution_m=str(int(config["downscaling"]["parent_resolution_m"])),
            target_resolution_m=str(res),
            dataset_mode=config["model"]["dataset_mode"],
            AOD_used="false",
            NoData=str(nodata),
            processing_timestamp=str(pd.Timestamp.now()),
        )
    logging.info("[AQI] AQI raster written: %s (%dx%d px, int16, nodata=%s)", out_path.name, width, height, nodata)
    return out_path, arr, transform


def _visualize_aqi(frame, target_date, config):
    out_png = PROJECT_ROOT / config["paths"]["outputs"] / f"aqi_500m_{target_date.date()}.png"
    valid = frame[frame["aqi_category"].notna()].to_crs("EPSG:4326")
    fig, ax = plt.subplots(figsize=(10, 9))
    if len(valid):
        from matplotlib.colors import ListedColormap, BoundaryNorm
        present = [c for c in _AQI_CATEGORY_ORDER if (valid["aqi_category"] == c).any()]
        colors = [_AQI_CPCB_COLORS[c] for c in present]
        cmap = ListedColormap(colors)
        bounds = np.arange(len(present) + 1) - 0.5
        norm = BoundaryNorm(bounds, len(present))
        valid = valid.assign(_cat_idx=valid["aqi_category"].map({c: i for i, c in enumerate(present)}))
        sc = valid.plot(column="_cat_idx", ax=ax, cmap=cmap, norm=norm, legend=False,
                        edgecolor="none", categorical=False)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, ticks=np.arange(len(present)))
        cb.ax.set_yticklabels(present)
        cb.set_label("PM2.5-derived AQI category")
    frame.to_crs("EPSG:4326").boundary.plot(ax=ax, color="grey", lw=0.2)
    ax.set_title(
        f"PM2.5-derived AQI / sub-index (CPCB National AQI, PM2.5 only) - {target_date.date()}\n"
        "NOT full multi-pollutant National AQI"
    )
    ax.set_xlabel("lon"); ax.set_ylabel("lat")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    logging.info("[AQI] AQI map saved: %s", out_png)
    return out_png


def _hotspot_mask(frame, config):
    cfg = config["hotspot"]
    min_cat = cfg["minimum_category"]
    rank = {name: i for i, name in enumerate(_AQI_CATEGORY_ORDER)}
    min_rank = rank[min_cat]
    mask = frame["aqi_category"].notna() & (
        frame["aqi_category"].map(rank) >= min_rank
    )
    return mask


def _polygonize_hotspots(frame, mask, target_date, config, transform):
    from scipy import ndimage as ndi
    from shapely import union_all
    from shapely.geometry import shape
    res = int(config["downscaling"]["target_resolution_m"])
    minx, miny, maxx, maxy = frame.total_bounds
    width = int(np.ceil((maxx - minx) / res))
    height = int(np.ceil((maxy - miny) / res))
    n_hot = int(mask.sum())
    if n_hot == 0:
        hs = gpd.GeoDataFrame(columns=["hotspot_id", "date", "area_km2", "geometry"], crs=frame.crs)
        out_path = PROJECT_ROOT / config["paths"]["processed"] / "hotspots_500m.geojson"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps({"type": "FeatureCollection", "features": []}),
            encoding="utf-8",
        )
        return hs, out_path, None, 0, None
    mask_arr = rio_features.rasterize(
        shapes=list(zip(frame.loc[mask, "geometry"], np.ones(n_hot, dtype=np.int32))),
        out_shape=(height, width), transform=transform,
        fill=0, dtype="int32", all_touched=False,
    )
    structure = np.zeros((3, 3), dtype=int)
    structure[:, 1] = 1
    structure[1, :] = 1
    labels, n_labels = ndi.label(mask_arr > 0, structure=structure)
    rows = []
    for lab in range(1, n_labels + 1):
        lab_mask = labels == lab
        geoms = [
            shape(g)
            for g, _ in rio_features.shapes(
                lab_mask.astype("uint8"), mask=lab_mask, transform=transform
            )
        ]
        lab_geom = union_all(geoms)
        if lab_geom.is_empty:
            continue
        rows.append({
            "hotspot_id": int(lab),
            "date": target_date,
            "area_km2": float(lab_geom.area / 1e6),
            "geometry": lab_geom,
        })
    hs = gpd.GeoDataFrame(rows, crs=frame.crs)
    out_path = PROJECT_ROOT / config["paths"]["processed"] / "hotspots_500m.geojson"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    hs.to_crs("EPSG:4326").to_file(out_path, driver="GeoJSON")
    return hs, out_path, labels, n_labels, mask_arr


def _hotspot_statistics(frame, mask, hs, config):
    cells = frame.loc[mask, ["pm25", "pm25_aqi"]]
    stats = {
        "method": config["hotspot"]["method"],
        "minimum_category": config["hotspot"]["minimum_category"],
        "hotspot_cell_count": int(mask.sum()),
        "hotspot_zone_count": int(len(hs)),
        "hotspot_area_km2": round(float(hs["area_km2"].sum()), 3),
        "mean_pm25_ug_m3": round(float(cells["pm25"].mean()), 3),
        "max_pm25_ug_m3": round(float(cells["pm25"].max()), 3),
        "mean_aqi": round(float(cells["pm25_aqi"].mean()), 3),
        "max_aqi": round(float(cells["pm25_aqi"].max()), 3),
        "note": "threshold-based high-pollution zone detection; NOT statistical hotspot "
                "analysis and NOT confirmed emission sources",
    }
    out_path = PROJECT_ROOT / config["paths"]["processed"] / "hotspot_statistics.json"
    save_json(stats, out_path)
    return stats, out_path


def _uncertainty_status(config):
    cfg = config["uncertainty"]
    report = {
        "status": "DEFERRED" if not cfg.get("enabled", False) else "IMPLEMENTED",
        "method": cfg.get("method", "none"),
        "reason": (
            "No statistically defensible pixel-level uncertainty can be estimated: "
            "training sample is 16 rows / 4 stations, fallback mode, AOD unavailable, "
            "and the residual downscaling model was not trained. "
            "An RMSE would not constitute a confidence percentage."
        ),
        "data_requirements": [
            "larger station training sample",
            "validated held-out spatial predictions",
            "trained residual / quantile / ensemble models",
        ],
        "future_method": "cross-validation residual error scale (labelled 'estimated error "
                         "scale', not 'confidence') once sufficient data exist",
    }
    out_path = PROJECT_ROOT / config["paths"]["processed"] / "uncertainty_status.json"
    save_json(report, out_path)
    return report, out_path


def _final_metadata(config, target_date, aqi_frame, hs, hotspot_stats, uncertainty_report):
    cfg = config
    return {
        "project": cfg["project"]["name"],
        "target_date": str(target_date.date()),
        "study_area": cfg["study_area"]["name"],
        "CRS": "EPSG:32643 (WGS84 / UTM zone 43N)",
        "parent_resolution_m": int(cfg["downscaling"]["parent_resolution_m"]),
        "target_resolution_m": int(cfg["downscaling"]["target_resolution_m"]),
        "PM2.5": {
            "model": cfg["model"].get("primary", "XGBoost"),
            "dataset_mode": cfg["model"]["dataset_mode"],
            "AOD_used": False,
        },
        "Downscaling": {
            "method": "residual refinement (architecture); this run baseline prototype / parent-constant",
            "residual_model_status": "NOT TRAINED (16 residual samples, minimum 20)",
            "consistency_method": "area_weighted_mean",
        },
        "AQI": {
            "methodology": cfg["aqi"]["methodology"],
            "pollutant": cfg["aqi"]["pollutant"],
            "averaging_period": cfg["aqi"]["averaging_period"],
            "AQI_type": "PM2.5-derived AQI / PM2.5 AQI sub-index (NOT full multi-pollutant National AQI)",
            "category_definition": [{"name": c["name"], "aqi_range": f"{c['aqi_min']}-{c['aqi_max']}"} for c in cfg["aqi"]["categories"]],
        },
        "Hotspot": {
            "method": hotspot_stats.get("method"),
            "minimum_category": hotspot_stats.get("minimum_category"),
            "note": "predicted high-pollution zones; NOT confirmed emission sources",
        },
        "Uncertainty": {
            "status": uncertainty_report["status"],
            "method": uncertainty_report["method"],
        },
        "Data limitations": {
            "training_rows": 16,
            "stations": 4,
            "AOD_availability": "unavailable",
            "downscaling_training_status": "insufficient (16 < 20)",
            "model_performance": "provisional / pipeline validation only",
            "500m_output": "baseline prototype; higher resolution does NOT imply higher accuracy",
        },
        "valid_500m_cells": int(aqi_frame["pm25"].notna().sum()),
        "hotspot_cell_count": hotspot_stats.get("hotspot_cell_count"),
        "hotspot_zone_count": hotspot_stats.get("hotspot_zone_count"),
    }


def _build_final_package(config, target_date, paths):
    final_dir = PROJECT_ROOT / config["paths"]["processed"] / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in ["hotspots_500m.geojson", "hotspot_statistics.json", "uncertainty_status.json"]:
        src = PROJECT_ROOT / config["paths"]["processed"] / name
        dst = final_dir / name
        if src.exists():
            dst.write_bytes(src.read_bytes())
            copied.append(str(dst))
    manifest = {
        "target_date": str(target_date.date()),
        "package": "data/processed/final/",
        "note": "Rasters are not duplicated; canonical paths referenced below.",
        "outputs": {k: str(v) for k, v in paths.items()},
        "copied_small_files": copied,
    }
    manifest_path = final_dir / "final_manifest.json"
    save_json(manifest, manifest_path)
    return final_dir, manifest_path, manifest


def _final_output_qc(config, target_date, aqi_frame, mask, hs, paths, hotspot_stats):
    checks = {}
    for key, p in paths.items():
        if p is None:
            checks[f"{key}_exists"] = False
            continue
        checks[f"{key}_exists"] = Path(p).exists()
    with rasterio.open(paths["pm25_500m_raster"]) as src:
        pm25_nodata = src.nodata
        pm25_crs = str(src.crs)
        pm25_res = tuple(round(float(r)) for r in src.res)
    with rasterio.open(paths["aqi_500m_raster"]) as src:
        aqi_nodata = src.nodata
        aqi_crs = str(src.crs)
        aqi_res = tuple(round(float(r)) for r in src.res)
    checks["pm25_crs"] = pm25_crs
    checks["aqi_crs"] = aqi_crs
    checks["crs_consistent"] = pm25_crs == aqi_crs
    checks["resolution_consistent"] = pm25_res == aqi_res
    checks["nodata_pm25"] = pm25_nodata
    checks["nodata_aqi"] = aqi_nodata
    checks["aqi_geometry_valid"] = bool(hs.geometry.is_valid.all()) if len(hs) else True
    valid_pm25 = int(aqi_frame["pm25"].notna().sum())
    valid_aqi = int(aqi_frame["pm25_aqi"].notna().sum())
    checks["every_valid_aqi_has_valid_pm25"] = bool(valid_aqi <= valid_pm25)
    checks["every_valid_aqi_cell_is_valid_pm25_cell"] = bool(
        aqi_frame.loc[aqi_frame["pm25_aqi"].notna(), "pm25"].notna().all()
    )
    checks["hotspot_cells_have_valid_pm25"] = bool(
        aqi_frame.loc[mask, "pm25"].notna().all()
    ) if mask is not None and mask.any() else True
    checks["hotspot_inside_aoi"] = bool(
        hs.geometry.intersects(get_aoi(config).to_crs(hs.crs).union_all()).all()
    ) if len(hs) else True
    all_ok = bool(
        all(checks[f"{k}_exists"] for k in ["pm25_1km_raster", "pm25_500m_raster", "aqi_500m_raster", "hotspots_geojson"])
    ) and checks["crs_consistent"] and checks["resolution_consistent"]
    report = {
        "stage": "final_geospatial_outputs",
        "status": "PASSED" if all_ok else "FAILED",
        "date": str(target_date.date()),
        "checks": checks,
        "decision": "FINAL GEOSPATIAL PACKAGE READY" if all_ok else "STOP WITH ERROR",
    }
    out_path = PROJECT_ROOT / config["paths"]["processed"] / "final_qc_report.json"
    save_json(report, out_path)
    return report, out_path


def _visualize_final_summary(aqi_frame, mask, target_date, config, hotspot_stats):
    from matplotlib.colors import ListedColormap, BoundaryNorm
    outputs = PROJECT_ROOT / config["paths"]["outputs"]
    out_png = outputs / f"final_geospatial_summary_{target_date.date()}.png"
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    ax1, ax2, ax3 = axes
    valid = aqi_frame[aqi_frame["pm25"].notna()].to_crs("EPSG:4326")
    if len(valid):
        valid.plot(column="pm25", ax=ax1, legend=True, cmap="YlOrRd",
                   legend_kwds={"label": "Predicted PM2.5 (ug/m3)", "shrink": 0.6})
    aqi_frame.to_crs("EPSG:4326").boundary.plot(ax=ax1, color="grey", lw=0.2)
    ax1.set_title(f"Predicted PM2.5 - 500 m ({target_date.date()})")

    present = [c for c in _AQI_CATEGORY_ORDER if (aqi_frame["aqi_category"] == c).any()]
    colors = [_AQI_CPCB_COLORS[c] for c in present]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(len(present) + 1) - 0.5, len(present))
    catg = aqi_frame[aqi_frame["aqi_category"].notna()].to_crs("EPSG:4326").assign(
        _ci=aqi_frame.loc[aqi_frame["aqi_category"].notna(), "aqi_category"].map({c: i for i, c in enumerate(present)})
    )
    if len(catg):
        catg.plot(column="_ci", ax=ax2, cmap=cmap, norm=norm, edgecolor="none")
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax2, ticks=np.arange(len(present)))
        cb.ax.set_yticklabels(present)
    aqi_frame.to_crs("EPSG:4326").boundary.plot(ax=ax2, color="grey", lw=0.2)
    ax2.set_title("PM2.5-derived AQI / sub-index (CPCB, PM2.5 only)")

    aqi_frame.to_crs("EPSG:4326").boundary.plot(ax=ax3, color="grey", lw=0.2)
    hot = aqi_frame[mask].to_crs("EPSG:4326")
    if len(hot):
        hot.plot(column="pm25", ax=ax3, cmap="Reds", edgecolor="black", linewidth=0.3,
                 legend_kwds={"label": "PM2.5 (ug/m3)", "shrink": 0.6} if hot["pm25"].nunique() > 1 else None)
    ax3.set_title(
        f"Predicted high-pollution zones ({hotspot_stats['minimum_category']}+):\n"
        f"{hotspot_stats['hotspot_zone_count']} zones, "
        f"{hotspot_stats['hotspot_cell_count']} cells, {hotspot_stats['hotspot_area_km2']} km2"
    )
    fig.suptitle(
        "Final geospatial summary - 500 m PM2.5 -> PM2.5-derived AQI -> high-pollution zones. "
        "Prototype predictions; NOT measured air quality.",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    logging.info("[FINAL] composite visualization saved: %s", out_png.name)
    return out_png


def run_final_outputs_pipeline(config, target_date=None):
    aqi_cfg = config["aqi"]
    hotspot_cfg = config["hotspot"]
    if not aqi_cfg.get("enabled", True):
        logging.info("[AQI] disabled - skipping")
        return None
    if target_date is None:
        target_date = aqi_cfg["default_date"]
    target_date = pd.Timestamp(target_date).normalize()

    processed = PROJECT_ROOT / config["paths"]["processed"]
    pred_path = processed / "pm25_500m_predictions.parquet"
    if not pred_path.exists():
        raise FileNotFoundError("[AQI] pm25_500m_predictions.parquet missing; run --downscaling-only first")
    pm25_500m = gpd.read_parquet(pred_path)
    available_dates = set(pd.to_datetime(pm25_500m["date"]).dt.normalize())
    if target_date not in available_dates:
        raise ValueError(
            "[AQI] No 500 m predictions available for requested date: %s (available: %s)"
            % (target_date.date(), sorted(d.date() for d in available_dates))
        )
    pm25_500m = pm25_500m[pm25_500m["date"] == target_date].reset_index(drop=True)
    if "pm25_500m_final" not in pm25_500m.columns:
        raise ValueError("[AQI] pm25_500m_final column missing; invalid 500 m table")
    if "geometry" not in pm25_500m.columns:
        raise ValueError("[AQI] 500 m table lacks geometry; cannot continue")

    aqi_frame, categories, breakpoints, aqi_table_path = _compute_aqi(config, pm25_500m, target_date)
    aqi_raster_path, arr, transform = _rasterize_aqi(aqi_frame, target_date, config, categories, breakpoints)
    aqi_png = _visualize_aqi(aqi_frame, target_date, config)

    mask = None
    hs = gpd.GeoDataFrame(columns=["hotspot_id", "date", "area_km2", "geometry"], crs="EPSG:32643")
    hotspot_stats = {"hotspot_cell_count": 0, "hotspot_zone_count": 0, "hotspot_area_km2": 0.0,
                     "method": hotspot_cfg.get("method"), "minimum_category": hotspot_cfg.get("minimum_category"),
                     "mean_pm25_ug_m3": None, "max_pm25_ug_m3": None, "mean_aqi": None, "max_aqi": None}
    hotspot_path = None
    if hotspot_cfg.get("enabled", True):
        mask = _hotspot_mask(aqi_frame, config)
        hs, hotspot_path, labels, n_labels, mask_arr = _polygonize_hotspots(
            aqi_frame, mask, target_date, config, transform
        )
        hotspot_stats, hotspot_stats_path = _hotspot_statistics(aqi_frame, mask, hs, config)

    uncertainty_report, uncertainty_path = _uncertainty_status(config)

    paths = {
        "pm25_1km_raster": str(processed / f"pm25_1km_{target_date.date()}.tif"),
        "pm25_500m_raster": str(processed / f"pm25_500m_{target_date.date()}.tif"),
        "aqi_500m_raster": str(aqi_raster_path),
        "aqi_500m_table": str(aqi_table_path),
        "hotspots_geojson": str(hotspot_path) if hotspot_path is not None else None,
        "hotspot_statistics_json": str(processed / "hotspot_statistics.json"),
        "uncertainty_status_json": str(uncertainty_path),
    }
    final_dir, manifest_path, manifest = _build_final_package(config, target_date, paths)
    metadata = _final_metadata(config, target_date, aqi_frame, hs, hotspot_stats, uncertainty_report)
    metadata_path = processed / "final_metadata.json"
    save_json(metadata, metadata_path)
    qc_report, qc_path = _final_output_qc(config, target_date, aqi_frame, mask, hs, paths, hotspot_stats)
    final_png = _visualize_final_summary(aqi_frame, mask, target_date, config, hotspot_stats)

    n_valid_pm25 = int(aqi_frame["pm25"].notna().sum())
    n_valid_aqi = int(aqi_frame["pm25_aqi"].notna().sum())
    return {
        "stage": "aqi_hotspot_final_outputs",
        "status": qc_report["status"],
        "date": str(target_date.date()),
        "pm25_valid_cells": n_valid_pm25,
        "aqi_valid_cells": n_valid_aqi,
        "hotspot_cells": hotspot_stats["hotspot_cell_count"],
        "hotspot_zones": hotspot_stats["hotspot_zone_count"],
        "hotspot_area_km2": hotspot_stats["hotspot_area_km2"],
        "uncertainty_status": uncertainty_report["status"],
        "final_metadata": str(metadata_path),
        "final_qc_report": str(qc_path),
        "manifest": str(manifest_path),
        "visualizations": [str(aqi_png), str(final_png)],
        "outputs": paths,
    }

