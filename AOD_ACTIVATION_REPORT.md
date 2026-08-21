# AOD Activation Report

## Summary

AOD (Aerosol Optical Depth) data is now **production-grade and fully wired** into the PM2.5 Hyperlocal Mapping dashboard. Clicking any location shows the real MODIS MAIAC AOD value from the satellite raster.

## What Was Done

### Backend (`api/main.py`, `api/schemas.py`)

- **`GET /location`** now samples the AOD GeoTIFF at the clicked coordinate via `rasterio.warp` CRS transform
- New `AODInfo` schema with: `aod` (value), `source`, `resolution_m`, `crs`, `date`
- AOD value from the real raster overwrites `aod_used: false` from model metadata when data exists
- `/raster/aod` already serves the GeoTIFF — confirmed working

### Dashboard (`dashboard/src/types.ts`, `LocationPanel.tsx`, `MetricCards.tsx`)

- `LocationResponse` type extended with `aod_info: AODInfo | null`
- **LocationPanel**: Shows AOD value (e.g. `0.120`), source (`MODIS/MAIAC MCD19A2 v061`), resolution (`500m`) instead of just "Used"/"Unavailable"
- **MetricCards**: New AOD card showing real AOD value with source label
- AOD layer toggle already wired to `getAODRasterUrl` — serves real GeoTIFF

### Tests (`tests/test_api.py`)

15 new AOD-specific API tests:
- `test_health_includes_aod` — /health returns AOD provider metadata
- `test_location_includes_aod_info` — /location returns AODInfo structure
- `test_location_aod_value_matches_raster` — AOD value matches direct raster read
- `test_location_aod_valid_range` — AOD ∈ [0, 5]
- `test_location_aod_set_used_when_available` — aod_used=true when AOD exists
- `test_raster_aod_serves_geotiff` — /raster/aod returns image/tiff
- `test_raster_aod_invalid_date_404` — 404 for nonexistent date
- `test_aod_no_path_traversal` — security: no path traversal
- `test_location_outside_aoi_returns_400` — 400 for points outside AOI
- `test_aod_raster_valid_crs` — CRS contains EPSG
- `test_aod_all_dates_available` — all dates have AOD rasters
- `test_aod_pune_city` — AOD works for Pune
- `test_aod_mumbai_city` — AOD works for Mumbai
- `test_aod_raster_finite_values` — no inf/NaN in AOD bands
- `test_location_aod_crs_epsg` — CRS in location response

**Regression**: 276 backend + 28 frontend = 304 total tests, all passing.

## AOD Data on Disk

| City    | File | Shape   | CRS       | AOD Range       | Mean  | Valid % |
|---------|------|---------|-----------|-----------------|-------|---------|
| Delhi   | `data/processed/aod_500m_2025-01-01.tif` | 91×80 | EPSG:32643 | [0.010, 0.257] | 0.120 | 100% |
| Pune    | `data/processed/pune/aod_500m_2025-01-01.tif` | 66×85 | EPSG:32643 | [0.049, 0.278] | 0.156 | 100% |
| Mumbai  | `data/processed/mumbai/aod_500m_2025-01-01.tif` | 110×65 | EPSG:32643 | [0.057, 0.288] | 0.162 | 100% |

Source: MODIS/MAIAC MCD19A2 v061 (Aerosol Optical Depth at 550nm)

## API Response Example

```json
{
  "aod_info": {
    "aod": 0.1203,
    "source": "MODIS/MAIAC MCD19A2 v061",
    "resolution_m": 500,
    "crs": "EPSG:32643",
    "date": "2025-01-01"
  },
  "aod_used": true
}
```

## Commit

`cc5bf53` — feat: activate AOD - raster sampling in /location, real values in dashboard, 15 AOD tests

## Railway Deployment

Requires manual redeploy in Railway dashboard (deployment caching issue). Check Deployments tab for build from `cc5bf53`.
