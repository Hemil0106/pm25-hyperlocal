# Architecture

## Data layer

Inputs (in `data/raw/`): CPCB PM2.5 CSV, ERA5-Land NetCDF, MODIS NDVI GeoTIFF,
SRTM elevation GeoTIFF, OSM road GeoJSON, VIIRS night-lights GeoTIFF, AOI
GeoJSON. Download stages (NASA Earthdata via `earthaccess`, ERA5 via `cdsapi`,
OSM via `osmnx`) are present but disabled (`download.enabled: false`); the demo
uses pre-downloaded files.

## Processing layer

Per-source QC and preprocessing (quality masks, unit conversion, clipping to
AOI, daily aggregation) in `src/data_pipeline.py`, then spatial + temporal
alignment onto a common 1 km grid (and 500 m target grid) with coverage/QC
reports.

## ML layer

`src/ml_pipeline.py` builds the training dataset, trains a RandomForest
baseline and an XGBoost primary model under leave-one-location-out spatial
cross-validation, records metrics and feature importance. Fallback mode drops
AOD predictors when AOD is unavailable. Results are labelled provisional.

## Geospatial layer

`src/downscaling.py` produces the 500 m refinement (currently baseline
parent-constant prototype, parent-consistency enforced) and `src/outputs.py`
writes PM2.5 / AQI GeoTIFFs, predicted high-pollution-zone GeoJSON, hotspot
statistics, uncertainty status and final metadata/QC JSON.

## API layer

FastAPI (`api/main.py`, `api/schemas.py`) serves the canonical generated
products only — no training or map regeneration on request. Controlled raster
serving for known dates/resolutions, validation of lat/lon/date, CORS for the
dashboard origin.

## Dashboard layer

React + TypeScript + Vite + MapLibre GL (`dashboard/`). The browser decodes
GeoTIFFs (geotiff + proj4), colorizes them on a canvas, and overlays them as a
MapLibre image source, alongside hotspots and CPCB station markers.

## Data flow

```
CPCB / ERA5 / MODIS / SRTM / OSM / VIIRS  →  QC  →  alignment
   →  training data  →  RandomForest + XGBoost (LOGO CV)
   →  1 km PM2.5  →  500 m refinement  →  AQI + hotspots
   →  final outputs  →  FastAPI  →  dashboard
```
