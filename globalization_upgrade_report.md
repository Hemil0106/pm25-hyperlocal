# Globalization Upgrade — Final Report (Milestone 15, Phases 1–28)

Date: 2026-08-15
Scope: Turn the locked Delhi-only SIH prototype into a global-ready, AOI-configurable,
tile-based PM2.5 platform while preserving the scientifically honest Delhi prototype
(Milestones 1–14) unchanged.

---

## 1. Executive summary

- The Delhi prototype pipeline (`src/data_pipeline.py`, 6,502 lines) is **untouched** —
  verified line count before and after; all 54 pre-existing backend tests still pass.
- A new globalization layer (`src/geospatial/`, `src/globalization/`, `api/globalization.py`,
  dashboard extensions) makes the platform AOI-configurable (named regions, bbox, global)
  and tile-based, with dataset registry, ground-truth normalization, model scopes,
  grouped spatial validation, generalized inference/downscaling/AQI/hotspots, output
  metadata, and honest API + dashboard messaging.
- **Explicit statement:** no scientifically valid global or regional PM2.5 model exists.
  Global/regional scopes stay `unavailable`; the Delhi model is structurally prevented
  from being presented as global. The dashboard and API surface this honestly.

## 2. Files created

| File | Purpose |
|---|---|
| `src/geospatial/aoi.py` | AOI dataclass, `resolve_aoi(config, region, bbox)`, `area_km2()`, `centroid_crs()`; modes: named_region / bbox / geojson / geometry_file / global |
| `src/geospatial/crs.py` | GEOGRAPHIC_CRS, EQUAL_AREA_GLOBAL_CRS, `utm_zone_for`, `get_metric_crs` (Delhi→EPSG:32643), transforms, geodesic distance |
| `src/geospatial/tiles.py` | `generate_tiles` (global 10° = 648 tiles), `tile_for_point`, tile cache (marker `COMPLETE.marker`), `process_tiles`, `merge_tile_outputs` |
| `src/geospatial/grid.py` | `generate_grid`, `generate_coarse_fine_grids` (max 2M cells guard), metric-CRS centroids → 4326, parent-grid linkage |
| `src/globalization/__init__.py` | Package init |
| `src/globalization/datasets.py` | `build_dataset_registry` (aod/cpcb/weather/ndvi/osm/dem/viirs/ground_truth) |
| `src/globalization/ground_truth.py` | `normalize_ground_truth` (cpcb+openaq), GROUND_TRUTH_SCHEMA, summary; drops physically invalid coords (1 CPCB row lat=91) |
| `src/globalization/availability.py` | `build_data_availability` (honest manifest; global=`data_limited`) |
| `src/globalization/ingestion.py` | `ingest_raster/netcdf/tabular/vector` interfaces |
| `src/globalization/cache.py` | `get_or_compute`, `cache_stats` |
| `src/globalization/model_scopes.py` | `resolve_model_scope`, `assert_scope_allows_inference`, scope metadata; Delhi-only gating |
| `src/globalization/validation.py` | `grouped_spatial_cv` (LeaveOneGroupOut; station/city/region/country; R²≥0.5 = validated) |
| `src/globalization/training.py` | global training schema (13 features + spatial groups), `validate_training_schema`, `build_training_frame` |
| `src/globalization/aqi.py` | `pm25_to_aqi` (locked formula), categories, `aqi_scheme_for`, `compute_aqi_frame` |
| `src/globalization/inference.py` | `inference_plan`, `predict_for_aoi` (scope-gated) |
| `src/globalization/downscaling.py` | `downscaling_plan`, `downscale_for_aoi` (honest caveats) |
| `src/globalization/hotspots.py` | `hotspot_plan`, `hotspots_for_aoi` ("predicted high-pollution zone", never "emission source") |
| `api/globalization.py` | Router `/global`: regions, countries, aoi, datasets, data-availability, model-scopes, output-metadata, pm25/bbox |
| `tests/test_globalization.py` | 45 globalization tests (23 required cases + coverage) |
| `dashboard/src/components/GlobalStatusPanel.tsx` | Honest AOI scope / availability / limitation panel |
| `dashboard/src/App.test.tsx` additions | Region selector + global-mode + Delhi-mode dashboard tests |
| `globalization_audit.json` | Phase 1 audit (root) |

## 3. Files modified

| File | Change |
|---|---|
| `config.yaml` | Added `aoi` (regions delhi/india/global), `tiles`, `cache`, `ground_truth`, `aqi_scheme`, `model_scopes` blocks; legacy `study_area` preserved |
| `api/main.py` | `from api.globalization import router as globalization_router` + `app.include_router(...)` — locked endpoints untouched |
| `dashboard/src/types.ts` | Globalization response types (RegionInfo, ModelScopes, DataAvailability, OutputMetadata, AOIInfo) |
| `dashboard/src/api.ts` | `getRegions`, `getModelScopes`, `getAOIInfo`, `getDataAvailability`, `getOutputMetadata` |
| `dashboard/src/components/Header.tsx` | AOI region selector + honest scope badge |
| `dashboard/src/App.tsx` | Region/scope state, availability fetching, honest "prediction unavailable" banner, GlobalStatusPanel, region-bound map fitting |
| `dashboard/src/map/MapView.tsx` | `regionBounds` prop for non-Delhi region recentering |
| `dashboard/src/styles/index.css` | `badge-muted`, `panel-warning` |

## 4. Delhi prototype preserved?

**Yes.**
- `src/data_pipeline.py` unchanged (6,502 lines before and after; no edits made).
- All 54 original backend tests pass unchanged.
- Legacy `study_area` config and `/pm25`, `/aqi`, `/location`, `/hotspots`, `/stations`,
  `/raster/*`, `/metadata`, `/feature-importance`, `/uncertainty` endpoints unchanged.
- Delhi AOI still resolves to EPSG:32643 (backward compatible) and `prototype_local`
  scope remains `available` for Delhi and any bbox contained within Delhi bounds.

## 5. Global AOI support

- `resolve_aoi` handles `global` (world bbox, no single metric CRS), `india` (bbox,
  EPSG:32644), `delhi` (geometry file, EPSG:32643), custom bbox, and GeoJSON/geometry files.
- Tiling: global at 10° = 648 tiles; tile cache with provenance + `COMPLETE.marker`.
- Grid generation guarded by a 2,000,000-cell limit with a tile-based recommendation
  (India at 500 m = 55M cells → correctly refused; global whole-grid → rejected).

## 6. Datasets

- Registry (`data/processed/dataset_registry.json`) covers AOD, CPCB, weather (ERA5),
  NDVI, OSM, DEM, VIIRS, and normalized ground truth, each with `local_data_present`
  honesty (download disabled → reflects local data only).

## 7. Ground truth

- Normalized to `data/processed/ground_pm25.parquet` (246 rows, 5 stations, schema
  station_id/source/country/latitude/longitude/timestamp/PM2.5/units/quality_flag).
- One CPCB row with lat=91.0 (physically invalid) dropped with a warning.
- OpenAQ source disabled in config; countries = ["India"].

## 8. Model scopes

- `prototype_local` (Delhi): `available` only for Delhi (+ contained bboxes); model
  files must exist.
- `regional`, `global`: permanently `unavailable` with honest reasons until a validated
  model trained on that scope's observations exists.
- `assert_scope_allows_inference` structurally prevents out-of-scope prediction.
- All prediction outputs tagged with `model_scope` metadata.

## 9. Validation

- `grouped_spatial_cv` = LeaveOneGroupOut at station/city/region/country levels;
  "validated" requires R² ≥ 0.5 on held-out spatial groups.

## 10. Generalized inference / downscaling / AQI / hotspots

- `predict_for_aoi`: Delhi → 7,094-cell 500 m predictions; global/regional → honest
  `unavailable`, never fabricated.
- Downscaling plan: Delhi available; caveats "downscaling ≠ resizing" and "500 m does
  NOT imply higher accuracy".
- AQI schemes: `india_cpcb` (Delhi, India), `none` (global — concentration only, AQI
  refused). Formula locked to the Delhi pipeline.
- Hotspots: Delhi → 3 predicted high-pollution zones (defined as predicted high
  pollution, not emission sources).

## 11. API

`/global/*` endpoints (all validated, no filesystem injection):
`/regions`, `/countries`, `/aoi`, `/datasets`, `/data-availability`,
`/model-scopes`, `/output-metadata`, `/pm25/bbox` (max 10°/side).

## 12. Dashboard

- Region selector (Delhi / India / Global); world basemap recenters to the AOI.
- "Globalization status" panel shows AOI, area, metric CRS, model scope, ground-truth
  coverage, prediction availability.
- Switching to a non-Delhi AOI shows an honest banner: "Prediction unavailable — no
  validated model trained on observations for this scope exists. The Delhi prototype is
  not applied outside Delhi."

## 13. Tests run / passed / failed

| Suite | Count | Result |
|---|---|---|
| Backend `pytest -q` | 99 (54 original + 45 globalization) | all passed |
| Dashboard `npm test` (vitest) | 25 (22 original + 3 new) | all passed |
| Dashboard `npm run build` | — | PASS (chunk-size warning only, cosmetic) |

## 14. Data / model limitations

- Only India (Delhi) ground-truth observations exist; no global/multi-country training data.
- AOD/NDVI/OSM/etc. raw inputs are not downloaded (`download.enabled: false`); ERA5
  NetCDF is synthetic pipeline test data.
- The Delhi model has no defensible pixel-level uncertainty (status DEFERRED).
- Global/regional AQI is not computed (scheme `none`).

## 15. No scientifically valid global model exists

There is no trained global or regional PM2.5 model in this repository. Global/regional
model scopes are reported `unavailable` everywhere (backend, API, dashboard). The Delhi
prototype is never reused for other geographies. Any future global model must be trained
on observations covering that scope and pass grouped spatial validation before the scope
label flips to `available`.

## 16. Exact commands

```powershell
# Backend tests (from project root)
.venv\Scripts\python.exe -m pytest -q

# Dashboard tests + build
cd dashboard
npm test
npm run build
cd ..

# Run
python run.py
uvicorn api.main:app --reload
cd dashboard; npm run dev

# Manual checks
# http://127.0.0.1:8000/global/regions
# http://127.0.0.1:8000/global/model-scopes
# http://127.0.0.1:8000/global/data-availability
# http://127.0.0.1:8000/global/output-metadata?region=global
# http://127.0.0.1:8000/global/pm25/bbox?date=2025-01-01&west=77.05&south=28.45&east=77.25&north=28.7
```

## 17. Demo workflow

1. Start backend (`uvicorn api.main:app --reload`) and dashboard (`npm run dev`).
2. Default view = Delhi: map, 500 m PM2.5, AQI, hotspots, stations all work as before.
3. Switch Region → Global/India: the map recenters to that AOI and an honest banner
   explains that prediction is unavailable (no validated model for that scope).
4. The "Globalization status" panel explains why and lists model-scope availability.
