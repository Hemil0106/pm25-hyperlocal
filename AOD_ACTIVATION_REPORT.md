# AOD Activation Report — Final

## Step 14: Summary

### AOD Dataset Found: YES

**Dataset:** MODIS/MAIAC MCD19A2 v061 (Aerosol Optical Depth at 550nm)
**Provider:** NASA LAADS DAAC via Earthdata / Harmony API
**Spatial Resolution:** 500m (UTM 43N / EPSG:32643)
**Temporal Resolution:** Daily composites
**CRS:** EPSG:32643 (UTM Zone 43N)
**Units:** Unitless (AOD is a dimensionless optical quantity)
**Nodata:** -9999.0

### Date Coverage

| City    | Available Dates | Count |
|---------|----------------|-------|
| Delhi   | 2025-01-01 | 1 |
| Pune    | 2025-01-01 through 2025-01-06 | 6 |
| Mumbai  | 2025-01-01 through 2025-01-06 | 6 |

### AOD Values Successfully Retrieved: 5/5 locations tested

| Location | Lat | Lon | AOD | Status |
|---|---|---|---|---|
| Connaught Place | 28.6315 | 77.2167 | 0.1358 | AVAILABLE |
| India Gate | 28.6129 | 77.2295 | 0.1502 | AVAILABLE |
| Chandni Chowk | 28.6506 | 77.2303 | 0.1927 | AVAILABLE |
| Pune | 18.52 | 73.86 | 0.1925 | AVAILABLE |
| Mumbai | 19.08 | 72.88 | 0.1421 | AVAILABLE |

### NoData Handling: PASS

Nearest-valid-pixel fallback within radius=2 pixels. Returns status `NO_VALID_OBSERVATION` when no valid pixel found within radius.

### Scope Isolation: PASS

Delhi, Pune, and Mumbai each have separate data directories. No cross-scope data leakage.

### Backend Tests: 285 passed

(29 dedicated AOD tests in test_api.py)

### Frontend Tests: 28 passed

### Build: PASS

### Key API Response

```
GET /location?date=2025-01-01&lat=28.6057&lon=77.2122
```

```json
{
  "aod_info": {
    "aod": 0.1428,
    "status": "AVAILABLE",
    "source": "MODIS/MAIAC MCD19A2 v061",
    "resolution_m": 500,
    "crs": "EPSG:32643",
    "date": "2025-01-01",
    "unit": "unitless",
    "nodata": false,
    "lookup": "exact_pixel",
    "distance_pixels": 0
  },
  "aod_used": true
}
```

### CRITICAL DISTINCTION

**AOD DISPLAY AVAILABLE: YES**
Real MODIS/MAIAC AOD values are read from satellite-derived GeoTIFF rasters and displayed in the dashboard.

**AOD USED AS MODEL INPUT: NO**
The XGBoost PM2.5 model metadata still shows `aod_available: false` and `dataset_mode: "fallback"`. The model does NOT currently use AOD as an input feature. AOD display and model input are treated as separate concerns per Step 10.

### Dashboard States

The dashboard now shows 4 distinct AOD states:

1. **AVAILABLE:** Value + source (e.g. `0.136 — MODIS/MAIAC MCD19A2 v061 (500m)`)
2. **NO_VALID_OBSERVATION:** "No satellite observation" (nodata pixel)
3. **DATASET_UNAVAILABLE:** "Dataset unavailable" (no AOD file for this date/city)
4. **API_ERROR:** "Unable to retrieve" (raster read failure)

### Commit

`cf65043` — feat: comprehensive AOD activation - nearest-pixel fallback, distinct status states, 30 AOD tests, cache-bust Dockerfile

### Railway Deployment

The Dockerfile has been cache-busted with `ARG CACHE_BUST=20260821a`. Force a manual redeploy in the Railway dashboard to pick up the new build.
