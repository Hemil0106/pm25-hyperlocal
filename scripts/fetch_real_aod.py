"""Fetch real NASA MAIAC AOD (MCD19A2 V061) data for all cities.

Uses the NASA Harmony API to subset MCD19A2 data by bounding box and date,
returning GeoTIFF subsets that are reprojected and clipped to match the
existing PM2.5 raster grid.

Authentication: Set EARTHDATA_USERNAME and EARTHDATA_PASSWORD env vars,
or configure ~/.netrc with: machine urs.earthdata.nasa.gov login <user> password <pass>

Usage:
    set EARTHDATA_USERNAME=your_username
    set EARTHDATA_PASSWORD=your_password
    python scripts/fetch_real_aod.py
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import requests
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import calculate_default_transform, reproject
from pyproj import Transformer

try:
    import earthaccess
except ImportError:
    sys.exit("earthaccess is required: pip install earthaccess")


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED = PROJECT_ROOT / "data" / "processed"

CITIES = {
    "delhi": {
        "bbox_wgs84": (77.0, 28.4, 77.4, 28.8),
        "utm_epsg": 32643,
    },
    "pune": {
        "bbox_wgs84": (73.7, 18.4, 74.1, 18.7),
        "utm_epsg": 32643,
    },
    "mumbai": {
        "bbox_wgs84": (72.7, 18.8, 73.0, 19.3),
        "utm_epsg": 32643,
    },
}

DATES = [
    "2025-01-01",
    "2025-01-02",
    "2025-01-03",
    "2025-01-04",
    "2025-01-05",
    "2025-01-06",
]

MCD19A2_COLLECTION = "C2324689816-LPCLOUD"
HARMONY_BASE = "https://harmony.earthdata.nasa.gov"

NODATA = -9999.0


def authenticate():
    """Authenticate with NASA Earthdata."""
    username = os.environ.get("EARTHDATA_USERNAME")
    password = os.environ.get("EARTHDATA_PASSWORD")

    if username and password:
        auth = earthaccess.login(strategy="environment")
    else:
        print("No EARTHDATA_USERNAME/PASSWORD env vars found.")
        print("Trying ~/.netrc or stored credentials...")
        auth = earthaccess.login()

    if not auth.authenticated:
        sys.exit(
            "Authentication failed. Set EARTHDATA_USERNAME and EARTHDATA_PASSWORD\n"
            "or configure ~/.netrc:\n"
            "  machine urs.earthdata.nasa.gov login YOUR_USER password YOUR_PASS"
        )

    print(f"Authenticated as: {auth.username}")
    return auth


def fetch_aod_harmony(
    session: requests.Session,
    bbox_wgs84: tuple,
    date_str: str,
    output_path: Path,
) -> bool:
    """Fetch AOD via Harmony API and save as GeoTIFF.

    Uses the OGC Coverages API to subset MCD19A2 by spatial/temporal bounds
    and variable (Optical_Depth_055).
    """
    west, south, east, north = bbox_wgs84
    dt_start = f"{date_str}T00:00:00.000Z"
    dt_end = f"{date_str}T23:59:59.999Z"

    url = (
        f"{HARMONY_BASE}/{MCD19A2_COLLECTION}"
        f"/ogc-api-coverages/1.0.0/collections/parameter_vars/coverage/rangeset"
        f"?subset=lat({south}:{north})"
        f"&subset=lon({west}:{east})"
        f'&subset=time("{dt_start}":"{dt_end}")'
        f"&variable=Optical_Depth_055"
        f"&format=application/geo+tiff"
        f"&outputCrs=EPSG:4326"
        f"&maxResults=1"
    )

    print(f"  Harmony request for {date_str}...")
    try:
        resp = session.get(url, timeout=120, stream=True)
    except requests.RequestException as exc:
        print(f"  ERROR: request failed: {exc}")
        return False

    if resp.status_code != 200:
        print(f"  ERROR: Harmony returned {resp.status_code}")
        try:
            body = resp.text[:500]
            print(f"  Response: {body}")
        except Exception:
            pass
        return False

    content_type = resp.headers.get("Content-Type", "")
    if "application/json" in content_type:
        data = resp.json()
        if "status" in data:
            status_url = data["status"]
            print(f"  Async job, polling {status_url}...")
            return _poll_and_download(session, status_url, output_path)
        print(f"  Unexpected JSON response: {json.dumps(data)[:300]}")
        return False

    if "image/tiff" in content_type or "application/geo+tiff" in content_type:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 256):
                f.write(chunk)
        size_kb = output_path.stat().st_size / 1024
        print(f"  Saved: {output_path.name} ({size_kb:.0f} KB)")
        return True

    print(f"  ERROR: unexpected Content-Type: {content_type}")
    return False


def _poll_and_download(
    session: requests.Session, status_url: str, output_path: Path, max_wait: int = 600
) -> bool:
    """Poll an async Harmony job until completion, then download the result."""
    deadline = time.time() + max_wait
    while time.time() < deadline:
        try:
            resp = session.get(status_url, timeout=30)
        except requests.RequestException:
            time.sleep(5)
            continue

        if resp.status_code != 200:
            time.sleep(5)
            continue

        data = resp.json()
        status = data.get("status")
        if status == "successful":
            for item in data.get("data", []):
                href = item.get("href", "")
                if href.endswith(".tif") or "tiff" in href.lower():
                    print(f"  Downloading result: {href[:80]}...")
                    dl = session.get(href, timeout=120, stream=True)
                    if dl.status_code == 200:
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(output_path, "wb") as f:
                            for chunk in dl.iter_content(chunk_size=1024 * 256):
                                f.write(chunk)
                        size_kb = output_path.stat().st_size / 1024
                        print(f"  Saved: {output_path.name} ({size_kb:.0f} KB)")
                        return True
                    print(f"  ERROR: download failed ({dl.status_code})")
                    return False
            print("  ERROR: no GeoTIFF in successful response")
            return False
        elif status == "failed":
            print(f"  ERROR: Harmony job failed: {data.get('message', 'unknown')}")
            return False
        else:
            print(f"  Status: {status}...")
            time.sleep(10)

    print(f"  ERROR: timed out after {max_wait}s")
    return False


def reproject_to_utm(
    src_path: Path, dst_path: Path, target_crs: str, target_bounds_utm: tuple,
    target_shape: tuple, target_transform
):
    """Reproject a WGS84 GeoTIFF to UTM, matching the existing PM2.5 grid."""
    with rasterio.open(src_path) as src:
        src_data = src.read(1)
        src_nodata = src.nodata if src.nodata is not None else -9999.0

        profile = src.profile.copy()
        profile.update(
            crs=target_crs,
            transform=target_transform,
            width=target_shape[1],
            height=target_shape[0],
            dtype="float32",
            nodata=NODATA,
        )

        dst_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(dst_path, "w", **profile) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=src_nodata,
                dst_transform=target_transform,
                dst_crs=target_crs,
                dst_nodata=NODATA,
                resampling=Resampling.bilinear,
            )

    with rasterio.open(dst_path) as check:
        data = check.read(1)
        valid = data[data != NODATA]
        if len(valid) > 0:
            print(f"  Reprojected: shape={data.shape}, AOD range=[{valid.min():.3f}, {valid.max():.3f}]")
        else:
            print(f"  Reprojected: all nodata (no valid AOD retrieved)")


def load_reference_grid(city_id: str, date_str: str):
    """Load the PM2.5 raster grid as reference for output dimensions."""
    if city_id == "delhi":
        ref_path = PROCESSED / f"pm25_500m_{date_str}.tif"
    else:
        ref_path = PROCESSED / city_id / f"pm25_500m_{date_str}.tif"

    if not ref_path.exists():
        print(f"  WARNING: reference grid not found: {ref_path}")
        return None

    with rasterio.open(ref_path) as src:
        return {
            "crs": src.crs,
            "transform": src.transform,
            "shape": (src.height, src.width),
            "bounds": src.bounds,
        }


def main():
    print("=" * 60)
    print("NASA MAIAC AOD Fetcher (MCD19A2 V061)")
    print("=" * 60)

    auth = authenticate()
    session = earthaccess.get_requests_https_session()

    for city_id, city in CITIES.items():
        print(f"\n--- {city_id.upper()} ---")
        city_dir = PROCESSED / city_id if city_id != "delhi" else PROCESSED

        for date_str in DATES:
            print(f"\n  Date: {date_str}")

            out_dir = city_dir
            raw_tif = out_dir / f"aod_raw_{date_str}.tif"
            final_tif = out_dir / f"aod_500m_{date_str}.tif"

            if final_tif.exists():
                print(f"  Already exists: {final_tif.name}, skipping")
                continue

            ok = fetch_aod_harmony(session, city["bbox_wgs84"], date_str, raw_tif)
            if not ok:
                print(f"  FAILED for {date_str}, falling back to synthetic")
                continue

            ref = load_reference_grid(city_id, date_str)
            if ref is None:
                print("  Cannot reproject without reference grid, keeping raw file")
                continue

            reproject_to_utm(
                raw_tif, final_tif,
                target_crs=str(ref["crs"]),
                target_bounds_utm=ref["bounds"],
                target_shape=ref["shape"],
                target_transform=ref["transform"],
            )

            raw_tif.unlink(missing_ok=True)

    print("\n" + "=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
