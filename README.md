# PM2.5 Hyperlocal Downscaling — Smart India Hackathon Prototype

AI/ML-assisted fusion of satellite, ground, meteorological and geographic data
to estimate **higher-resolution surface PM2.5** over Delhi (1 km → 500 m),
delivered through a FastAPI backend and a React + MapLibre GIS dashboard.

> **Scientific disclaimer**
> This project is a Smart India Hackathon prototype. Current results are
> pipeline-demonstration outputs and should not be interpreted as operational
> air-quality measurements.
>
> - AOD is currently **unavailable** in the demonstration environment
>   (NASA Earthdata credentials not configured). The pipeline runs in
>   **fallback mode**.
> - The current 500 m output is a **baseline spatial-refinement prototype**
>   (parent-constant; the residual model was not trained).
> - Uncertainty estimation is **deferred**.
> - The **PM2.5-derived AQI** is a PM2.5 sub-index, **not** the complete
>   multi-pollutant National AQI.

---

## Problem statement

CPCB ground monitors are sparse (a handful of stations around Delhi), while
satellite data is coarse (typically 1 km+). The goal is to produce a
hyperlocal (500 m) predicted PM2.5 map for the Delhi prototype area by fusing
available data sources with machine learning, and to make the results
explorable through a clean interactive map.

## What the system does

1. Ingests and quality-controls multiple geospatial/atmospheric inputs
   (CPCB PM2.5, ERA5-Land weather, MODIS NDVI, SRTM elevation, OSM road
   density, VIIRS night lights, and — when credentials are configured — MODIS
   MAIAC AOD).
2. Aligns all sources to a common 1 km grid and builds a training dataset.
3. Trains a **Random Forest baseline** and an **XGBoost primary model** with
   leave-one-location-out spatial cross-validation.
4. Predicts **1 km PM2.5**, then applies a **500 m spatial refinement**
   (currently the baseline parent-constant prototype).
5. Computes the **PM2.5-derived AQI** and **predicted high-pollution zones**.
6. Writes final geospatial outputs (GeoTIFF, GeoJSON, JSON metadata).
7. Serves everything through a **FastAPI** API consumed by a **React +
   MapLibre** dashboard.

## Core architecture

```
Satellite / Ground / Weather / GIS
↓
Data QC
↓
Spatial + Temporal Alignment
↓
Feature Engineering
↓
ML (RandomForest baseline + XGBoost primary, leave-one-location-out CV)
↓
1 km PM2.5
↓
500 m Spatial Refinement
↓
AQI + Hotspots
↓
FastAPI
↓
React + GIS Dashboard
```

## Technology stack

| Layer        | Technology                                              |
|--------------|---------------------------------------------------------|
| Python       | 3.10, pandas, geopandas, xarray, rasterio, scikit-learn |
| ML           | XGBoost, SHAP                                           |
| Geospatial   | pyproj, shapely, fiona, pyogrio, rioxarray              |
| Data sources | earthaccess (NASA), cdsapi (ERA5), osmnx (OSM)          |
| API          | FastAPI, uvicorn, pydantic                              |
| Frontend     | React 18, TypeScript, Vite 5, MapLibre GL, geotiff, proj4 |
| Tests        | pytest, Vitest + React Testing Library                  |

## Repository structure

```
project/
│
├── data/
│   ├── raw/          # input data (CPCB CSV, ERA5-Land NC, NDVI/DEM/VIIRS GeoTIFF, OSM GeoJSON, AOI)
│   ├── processed/    # QC reports, aligned grids, training data, predictions, final outputs
│   └── outputs/      # visualization PNGs
│
├── src/              # pipeline modules (data_pipeline, ml_pipeline, downscaling, outputs, config, utils)
│   └── geospatial/   # AOI handling
├── api/              # FastAPI app + Pydantic schemas
├── dashboard/        # React + MapLibre GIS dashboard
├── models/           # trained artifacts + model metadata
├── tests/            # pytest suite (pipeline, AQI, hotspots, API)
├── notebooks/        # experimentation notebook (stub)
├── config.yaml       # single source of configuration
├── run.py            # CLI entry point (full pipeline + stage-only flags)
├── requirements.txt  # pinned Python dependencies
├── .env.example      # environment variable template (copy to .env)
└── .gitignore
```

## Installation

### Python environment

```
py -3.10 -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

### Frontend setup

```
cd dashboard
npm install
```

### Environment variables

Copy the template and fill in what you need:

```
cp .env.example .env
```

`.env` is gitignored. Keys:

- `NASA_EARTHDATA_USERNAME` / `NASA_EARTHDATA_PASSWORD` — required **only** if
  you enable downloads (`download.enabled: true` in `config.yaml`) for
  AOD/NDVI/DEM/VIIRS. Without them the AOD stage fails gracefully and the
  pipeline continues in fallback mode.
- `CDS_API_URL` / `CDS_API_KEY` — optional, only for ERA5-Land downloads.
- `CPCB_API_URL` — optional, not required when the raw CPCB CSV is provided.

The demonstration environment uses **pre-downloaded input files** in
`data/raw/`, so no credentials are required to reproduce the demo.

## How to run the pipeline

```
python run.py
```

Expected: every stage runs in order; the **AOD stage fails gracefully**
(credentials unavailable) and the pipeline continues to CPCB, weather, NDVI,
DEM, OSM, VIIRS, alignment, training data, model training, inference, 500 m
downscaling, and final outputs.

### Individual stages

```
python run.py --cpcb-only
python run.py --weather-only
python run.py --ndvi-only
python run.py --dem-only
python run.py --osm-only
python run.py --viirs-only
python run.py --alignment-only
python run.py --training-data-only
python run.py --model-only
python run.py --inference-only --date 2025-01-01
python run.py --downscaling-only --date 2025-01-01
python run.py --aqi-hotspot-only --date 2025-01-01
```

### Global data acquisition (Milestone 16)

Downloads and QC's tile-based global feature data and global PM2.5
observations for the `global` / `india` / `delhi` scopes. This is a
**data-engineering layer only** — it trains no global ML model and produces no
global PM2.5/AQI/hotspot prediction (both explicitly NOT implemented in M16).

```
python run.py --global-data-only
python run.py --india-data-only
python run.py --delhi-data-only
```

- Credentials are read from the environment only (`OPENAQ_API_KEY`,
  `NASA_EARTHDATA_USERNAME`/`PASSWORD`, `CDS_API_URL`/`CDS_API_KEY`); missing
  credentials yield honest `UNAVAILABLE` per source, never fabricated or
  substituted data.
- Artifacts are scope-scoped: `data/processed/global_data_availability_report_<scope>.json`,
  `global_data_manifest_<scope>.json`, and per-run
  `data/processed/global/availability/global_failed_downloads_<scope>.json`.
- See `global_data_acquisition_report.md` for full details.

### Global training dataset (Milestone 17)

Builds the schema-correct global training table + feature engineering +
spatial-validation setup (LOSO/LOCO groups, leakage checks, data-derived
readiness gate) from the real Milestone 16 outputs. This is a
**data-engineering layer only** — it trains no global ML model and produces no
global prediction (both explicitly NOT implemented in M17).

```
python run.py --global-training-only
```

- Real data only: unavailable features are `NaN` with per-feature
  `<feature>_available` flags; nothing is fabricated or substituted.
- Leakage checks stop the pipeline before writing outputs on any FAIL/PRESENT
  (target leakage, duplicate station-dates, future information, synthetic
  markers, Delhi→Global contamination).
- Readiness is fully data-derived and never auto-YES
  (`reports/global_training_readiness.json`).
- Outputs: `data/processed/global/training/global_training_dataset{,_complete,_incomplete}.parquet`,
  `reports/global_training_report.json`, validation assignments, metadata,
  diagnostics.
- See `global_training_report.md` for full details.

## How to run tests

```
python -m pytest -q            # backend (pipeline, AQI, hotspots, API)
cd dashboard && npm test       # frontend (vitest)
cd dashboard && npm run build  # frontend build + typecheck
```

## How to start the API

```
python -m uvicorn api.main:app --reload
```

- API root: `http://127.0.0.1:8000/`
- Interactive docs (Swagger): `http://127.0.0.1:8000/docs`
- Endpoints: `/health`, `/available-dates`, `/metadata`, `/pm25`,
  `/pm25/grid`, `/aqi`, `/location`, `/hotspots`, `/hotspots/statistics`,
  `/stations`, `/stations/{id}`, `/feature-importance`, `/uncertainty`,
  `/raster/pm25`, `/raster/aqi`.
- M16 global data status: `/data/sources`, `/data/availability`,
  `/data/coverage`, `/data/status` (all take `?scope=global|india|delhi`;
  scope-scoped and refuse cross-scope reads).

## How to start the dashboard

```
cd dashboard
npm run dev
```

Open **`http://localhost:5173/`** (note: `localhost`, not `127.0.0.1`, on this
machine).

Set `VITE_API_BASE_URL` in `dashboard/.env` only if the API is not at
`http://127.0.0.1:8000`.

### One-shot demo startup

`start_demo.bat` (Windows) launches backend + dashboard in two terminals.
`start_demo.sh` (Linux/macOS) does the same.

## Demo workflow

1. Open the dashboard.
2. Select date **2025-01-01** (the available demonstration date).
3. Toggle the **PM2.5 map**.
4. Enable **CPCB stations**.
5. Enable **predicted high-pollution zones**.
6. Switch **1 km → 500 m** and explain: *"spatial detail increases; this does
   not automatically imply higher accuracy."*
7. Switch **PM2.5 → PM2.5-derived AQI**.
8. **Click a location** to see predicted PM2.5, PM2.5-derived AQI, model,
   dataset mode, resolution and limitations.
9. **Click a hotspot** to see area, mean/max PM2.5 and mean/max AQI.
10. Open **model/data information** and **feature importance**, explaining:
    *"these are model feature contributions, not causal proof."*

## Available output date(s)

- **2025-01-01** (the current generated demonstration date).

## Current model mode

- **Model:** XGBoost (primary artifact), RandomForest (baseline).
- **Mode:** FALLBACK (AOD unavailable → reduced predictor set).
- **Training sample:** 16 rows / 4 stations — **provisional / pipeline
  validation only**.
- **500 m:** baseline parent-constant spatial refinement prototype (residual
  model not trained: 16 residual samples < minimum 20).
- **Uncertainty:** DEFERRED.
- **AQI:** PM2.5-derived AQI / sub-index (not full National AQI).
- **Hotspots:** threshold-based **predicted high-pollution zones** (not
  statistical hotspot analysis, not confirmed emission sources).

## Deploying

### Frontend (Netlify)

1. Push this repo to GitHub.
2. On [Netlify](https://app.netlify.com), import the repo. Build settings are
   pre-configured via `netlify.toml` (`dashboard/` → `dist/`).
3. Set environment variable `VITE_API_BASE_URL` to your backend's public URL
   (e.g. `https://your-backend.up.railway.app`).

### Backend (any Python host)

The backend is a standard FastAPI app. Deploy to Railway, Render, Fly.io, or a
VPS. Start with:

```bash
CORS_ORIGINS=https://your-site.netlify.app \
  python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### Local development

```bash
# Terminal 1 — backend
python -m uvicorn api.main:app --reload

# Terminal 2 — dashboard (Vite proxies API to localhost:8000)
cd dashboard && npm install && npm run dev
```

## Current limitations

1. AOD unavailable (NASA credentials not configured) → fallback mode.
2. Very small training dataset (16 rows, 4 stations).
3. Model performance is provisional / pipeline validation only.
4. 500 m output is a baseline prototype; **higher resolution does not imply
   higher accuracy**.
5. Uncertainty estimation is deferred; no fabricated confidence values.
6. Hotspots are predicted zones, not confirmed emission sources.
7. PM2.5-derived AQI is not the complete multi-pollutant National AQI.
8. Basemap tiles require internet connectivity.
9. Not production ready.

## Scientific interpretation

- The system performs **AI-assisted satellite + ground + meteorological +
  geographic data fusion for higher-resolution PM2.5 estimation**. It does
  **not** measure surface PM2.5 directly from satellite.
- Downscaling is spatial refinement, **not image resizing**.
- Feature importance is **model contribution, not causal attribution**.
- All output labels in the API, metadata, QC reports and dashboard are kept
  consistent (model, mode, AOD status, training size, downscaling status,
  uncertainty status, AQI type, hotspot meaning).
