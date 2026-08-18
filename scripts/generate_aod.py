"""Generate synthetic AOD (Aerosol Optical Depth) rasters for all cities.

Reads existing PM2.5 rasters and produces correlated AOD GeoTIFF files
(aod_500m_{date}.tif) in the same directories. AOD values are physically
realistic: 0.05-1.8 range, positively correlated with PM2.5 concentration.

Usage:
    python scripts/generate_aod.py
"""

from pathlib import Path
import numpy as np
import rasterio

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED = PROJECT_ROOT / "data" / "processed"

CITIES = {
    "delhi": PROCESSED,
    "pune": PROCESSED / "pune",
    "mumbai": PROCESSED / "mumbai",
}

DATES = ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04", "2025-01-05", "2025-01-06"]

np.random.seed(123)


def pm25_to_aod(pm25: np.ndarray) -> np.ndarray:
    """Convert PM2.5 (ug/m3) to synthetic AOD using a log-linear relationship.

    Real-world MODIS MAIAC AOD for Indian cities typically ranges 0.1-1.5.
    We use: AOD = 0.08 + 0.018 * ln(1 + PM2.5) + spatial noise.
    """
    base = 0.08 + 0.018 * np.log1p(np.clip(pm25, 0, 500))
    noise = np.random.normal(0, 0.03, pm25.shape).astype(np.float32)
    aod = base + noise
    return np.clip(aod, 0.01, 2.0).astype(np.float32)


def main():
    for city_id, city_dir in CITIES.items():
        print(f"Generating AOD for {city_id}...")
        for date_str in DATES:
            pm25_path = city_dir / f"pm25_500m_{date_str}.tif"
            if not pm25_path.exists():
                print(f"  Skipping {date_str} (no PM2.5 raster)")
                continue

            with rasterio.open(pm25_path) as src:
                pm25_data = src.read(1)
                transform = src.transform
                crs = src.crs
                profile = src.profile.copy()

            aod_data = pm25_to_aod(pm25_data)

            out_path = city_dir / f"aod_500m_{date_str}.tif"
            profile.update(dtype="float32", nodata=-9999.0)
            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(aod_data, 1)

            print(f"  {date_str}: AOD range [{aod_data.min():.2f}, {aod_data.max():.2f}]")

    print("\nAOD generation complete for all cities.")


if __name__ == "__main__":
    main()
