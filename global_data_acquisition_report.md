# Global Data Acquisition — Final Report (Milestone 16)

Date: 2026-08-15
Scope: A real, tile-based, QC'd global data-acquisition layer for the
`global` / `india` / `delhi` scopes — data engineering ONLY. No ML training,
no global PM2.5/AQI/hotspot prediction, no dashboard redesign, no fabricated
or synthetic-as-real data. The locked Delhi prototype (Milestones 1–15) is
preserved unchanged.

---

## 1. Executive summary

- The Delhi prototype pipeline (`src/data_pipeline.py`, 6,502 lines) is
  **untouched** (last modified 2026-08-14, before M16 work). All pre-existing
  backend tests still pass.
- A new `src/global_data/` package acquires and normalizes global PM2.5
  observations and tile-based feature data with honest per-source
  `AVAILABLE / PARTIAL / UNAVAILABLE` status. Credentials are read from the
  environment only; missing credentials produce a graceful `UNAVAILABLE`,
  never fabricated data and never silent substitution of Delhi data.
- **Explicit statement (spec section 43):**
  - **Global ML is NOT implemented** — deliberately deferred to a future
    milestone. No global model is trained, and no global prediction artifacts
    are produced.
  - **Global PM2.5 prediction is NOT implemented** — no global PM2.5/AQI/
    hotspot raster is produced from this layer. The M15 model-scope gates
    (`prototype_local` only, Delhi) remain the sole prediction path.
- Scope isolation is enforced end-to-end: artifact reports and manifests are
  scope-scoped (`..._global.json`, `..._india.json`, `..._delhi.json`), and
  `assert_scope_isolated` / `verify_artifact_scope` refuse to serve an
  artifact built for one scope under another.

## 2. Files created

| File | Purpose |
|---|---|
| `src/global_data/__init__.py` | Package init + public API |
| `src/global_data/scope.py` | `SUPPORTED_SCOPES`, `SCOPE_BOUNDS`, `validate_scope`, `infer_scope`, `assert_scope_isolated` (requested == artifact) |
| `src/global_data/sources.py` | `DATA_SOURCE_REGISTRY`, `credential_available` (env only), `build_data_source_registry` |
| `src/global_data/units.py` | `TARGET_UNIT=pm25_ug_m3`, conversion factors, `normalize_pm25_units`; unknown units never guessed |
| `src/global_data/qc.py` | `apply_qc`: removes duplicates/invalid coords/bad timestamps/missing/negative/above-max; FLAGS outliers (never deletes) |
| `src/global_data/temporal.py` | `aggregate_daily`: daily mean + count; insufficient days skipped, never zero-filled |
| `src/global_data/stations.py` | `build_station_registry`, `build_station_summary`, `write_station_outputs` |
| `src/global_data/integrity.py` | `sha256_file`, `validate_checksum`, `verify_artifact_scope`, `has_synthetic_rows`, `synthetic_leakage_report` |
| `src/global_data/cache.py` | `fetch_with_retry` (retry/backoff), `record_failed_download`, `GlobalCache` (validate-before-redownload + checksums) |
| `src/global_data/pm25_global.py` | OpenAQ v3 adapter (`_normalize_row`, paginated `fetch_openaq_pm25`, `acquire_pm25`); v2 retired (410) → v3 key from env only |
| `src/global_data/satellite.py` | `SatelliteSource` base: creds from env, `check_availability`, `probe_connectivity`, `attempt_acquire` (tile-based, `max_tiles` cap, failed-download manifest) |
| `src/global_data/aod_global.py` | MODIS MAIAC MCD19A2 adapter (tile-based) |
| `src/global_data/weather_global.py` | ERA5-Land CDS adapter (chunked, job+poll) |
| `src/global_data/ndvi_global.py` | MOD13Q1 adapter + pure `composite_dates`, `nearest_valid_composite` (no-future default), `composite_metadata` |
| `src/global_data/dem_global.py` | SRTM GL1 adapter; `SRTM_NODATA=-32768` preserved, never zero-filled; `srtm_tile_name` |
| `src/global_data/osm_global.py` | Overpass road-density proxy (spatial proxy only); no creds required; `road_density_query` |
| `src/global_data/viirs_global.py` | VNP46A2 adapter with per-tile QA (`qa_pass_mask`, keep quality 0) |
| `src/global_data/grid.py` | `grid_for_tile`, `common_grid` (per-tile for global; never a whole-globe raster in RAM); reuses locked `src/geospatial/*` |
| `src/global_data/normalize.py` | `sample_raster_at_points` (bilinear/nearest via rasterio `Resampling`), `normalize_feature_to_grid` |
| `src/global_data/temporal_join.py` | `spatial_join_to_grid`, `add_feature_to_observations`, `temporal_join` with `<feature>_available` flags |
| `src/global_data/coverage.py` | `build_coverage_report` → `..._availability_report_<scope>.json` (PARTIAL when any tile failed) |
| `src/global_data/manifest.py` | `build_manifest` → `global_data_manifest_<scope>.json` (checksums, scope isolation, ml_not_implemented) |
| `src/global_data/ingest.py` | `ensure_global_dirs`, `_adapter_acquisition`, `run_global_data_pipeline` (fresh per-run failed-download log) |
| `api/data.py` | Router `/data`: `/sources`, `/availability`, `/coverage`, `/status` (scope-scoped, refuse cross-scope reads) |
| `tests/test_global_data.py` | 69 M16 tests (scope, units, QC, daily agg, stations, integrity, NDVI, adapters, registry, grid, temporal join, coverage, manifest, cache, OpenAQ, `/data/*` API) |

## 3. Files modified

| File | Change |
|---|---|
| `config.yaml` | Added `global_data:` block (scope, common_grid, temporal, qc, pm25 v3, fetch retries/backoff/`max_tiles`, cache, storage, sources with env-only credential var names) |
| `run.py` | Added `--global-data-only`, `--india-data-only`, `--delhi-data-only` stage-only flags + branches |
| `api/main.py` | `from api.data import router as global_data_router` + `app.include_router(...)` |
| `dashboard/src/types.ts` | M16 types (`GlobalDataStatusResponse`, `GlobalDataAvailabilityResponse`) |
| `dashboard/src/api.ts` | `getGlobalDataStatus`, `getGlobalDataAvailability` |
| `dashboard/src/App.tsx` | Fetch M16 status per region (mapped to scope); pass to GlobalStatusPanel |
| `dashboard/src/components/GlobalStatusPanel.tsx` | Honest "Global data acquisition" panel (scope, overall, acquired/unavailable sources) |
| `dashboard/src/App.test.tsx` | Mock + default for `getGlobalDataStatus` |

## 4. Delhi prototype preserved?

**Yes.**
- `src/data_pipeline.py` unchanged (verified 6,502 lines; no edits in M16).
- All 54 original backend tests pass unchanged (168 total backend tests pass).
- Legacy endpoints, config, and dashboard behavior unchanged.

## 5. Acquisition scopes

- `global` (world), `india` (65–100 E, 5–40 N), `delhi` (77.0–77.4 E, 28.4–28.8 N).
- Scope isolation: `infer_scope` returns the *smallest* configured scope
  containing artifact bounds (Delhi artifact is never reported as India or
  Global); `assert_scope_isolated` / `verify_artifact_scope` refuse a Delhi
  artifact under `india`/`global` and an India artifact under `global`.
- Reports, manifests, and failed-download logs are scope-scoped; the API
  refuses to serve a report built for a different scope (`not_run` instead).

## 6. PM2.5 ground observations (OpenAQ v3)

- OpenAQ v2 is retired (HTTP 410); v3 is the only real path and requires an
  API key from the environment (`OPENAQ_API_KEY`).
- No key is present in this environment → source honestly reported
  `UNAVAILABLE`; `acquire_pm25` writes no observation artifacts. No fabricated
  or substituted data.
- When a key is configured, the pipeline is: paginated fetch (retried with
  backoff) → schema normalization → unit normalization to `pm25_ug_m3` → QC →
  daily aggregation → station registry/summary parquet with SHA-256 checksums.

## 7. Tile-based feature acquisition

- All satellite/vector adapters process per tile (`src/geospatial/tiles`
  10° grid, 648 global tiles); a full global raster/grid is never allocated in
  memory. `fetch.max_tiles` (default 3) bounds acceptance runs.
- Cache: validate-before-redownload with SHA-256 provenance; corrupted files
  are detected and redownloaded. Failures are logged to the scope-scoped
  `global_failed_downloads_<scope>.json` (fresh per run).
- Missing credentials → graceful `UNAVAILABLE` for aod/weather/ndvi/dem/viirs.
- OSM (OpenStreetMap Overpass) needs no credentials and is used as a
  **spatial proxy only** (`road_segments` count per tile; clearly labeled
  "not a measurement").

## 8. Normalization rules honored

- Units → `pm25_ug_m3` (µg/m³, mg/m³, g/m³, ng/m³); unknown units marked
  invalid and excluded, never guessed.
- QC removes duplicates, physically invalid coordinates, unparseable
  timestamps, missing/negative/above-max PM2.5; outliers are FLAGGED
  (`outlier_flag`) and retained, never deleted.
- Daily aggregation skips station-days with fewer than
  `min_observations_per_day`; missing days are never zero-filled.
- NDVI picks the nearest valid composite with `no_future_match` default and
  records `ndvi_date`/`target_date`/`temporal_offset_days`.
- DEM SRTM NoData (`-32768`) is preserved; elevation is never zero-filled.
- VIIRS per-pixel QA keeps only quality flag `0`.
- Temporal join attaches `grid_id` + feature values with explicit
  `<feature>_available` flags; missing features are NaN, never silently filled.
- Synthetic test rows (tag `SYNTHETIC_TEST_DATA`) are detected; the report
  records `synthetic_data_leakage: NONE` and refuses leakage into real data.

## 9. Acceptance run (2026-08-15)

`python run.py --global-data-only` (and `--india-data-only`, `--delhi-data-only`):

| Scope  | pm25/aod/weather/ndvi/dem/viirs | osm (Overpass) | Report |
|---|---|---|---|
| global | UNAVAILABLE (no creds) | AVAILABLE — 3/3 tiles, real road-density files with SHA-256 | `..._report_global.json` |
| india  | UNAVAILABLE (no creds) | PARTIAL — some tiles rate-limited (429), recorded | `..._report_india.json` |
| delhi  | UNAVAILABLE (no creds) | UNAVAILABLE — 1/1 tile rate-limited (429), recorded | `..._report_delhi.json` |

- Real artifacts: `data/raw/global/osm/global/roads_tile_000_00{0,1,2}.json`
  (global scope) with checksums in `data/processed/global_data_manifest_global.json`.
- Failed downloads are recorded in scope-scoped
  `global_failed_downloads_<scope>.json` (e.g. delhi 429 Too Many Requests),
  never silently ignored.
- Coverage report + manifest written for each scope; `ml_not_implemented` and
  `scope_isolation.verified=OK` recorded.

## 10. API (`/data/*`)

- `/data/sources?scope=` — registry with honest per-source credential status.
- `/data/availability?scope=` and `/data/coverage?scope=` — scope-scoped
  coverage report; `not_run` when no run exists for that scope or when the
  on-disk report was built for a different scope (scope isolation).
- `/data/status?scope=` — compact status (`available`/`unavailable`/`not_run`),
  available/unavailable source lists, `synthetic_data_leakage`,
  `ml_not_implemented: true`, `prediction_not_implemented: true`.
- Invalid scopes → 400.

## 11. Dashboard (minimal, honest, no redesign)

- `GlobalStatusPanel` gains a "Global data acquisition" section: scope,
  overall status, acquired sources, unavailable sources, and the note that
  missing credentials are reported, never substituted, and that no global
  prediction/AQI is produced from this layer.
- The M16 layer never draws a global PM2.5 map or hotspots; prediction
  availability continues to be governed by M15 model scopes (Delhi only).

## 12. Tests run / passed / failed

| Suite | Count | Result |
|---|---|---|
| Backend `pytest -q` | 168 (99 M15 baseline + 69 M16) | all passed |
| Dashboard `npm test` (vitest) | 25 | all passed |
| Dashboard `npm run build` | — | PASS (chunk-size warning only, cosmetic) |

## 13. Data / model limitations

- No credentials (OpenAQ v3 key, NASA Earthdata, CDS) are configured in this
  environment → PM2.5 observations and AOD/weather/NDVI/DEM/VIIRS remain
  UNAVAILABLE. Only OSM road-density data was actually acquired.
- Overpass is subject to rate limiting (429s observed); failures are honest
  and recorded.
- No global ML model and no global PM2.5/AQI/hotspot prediction exists; the
  M16 layer is data engineering only.

## 14. Explicit: Global ML and prediction NOT implemented

- **Global ML training is NOT part of Milestone 16** (deferred). The manifest
  records this statement in `ml_not_implemented`.
- **Global PM2.5 prediction is NOT implemented** (deferred). The `/data/*`
  endpoints report `prediction_not_implemented: true`; the dashboard shows no
  global predictions, and M15 scope gates keep prediction Delhi-only.

## 15. Exact commands

```powershell
# Backend tests (from project root)
.venv\Scripts\python.exe -m pytest -q

# Dashboard tests + build
cd dashboard
npm test
npm run build
cd ..

# Acquisition (stage-only; scope variants)
python run.py --global-data-only
python run.py --india-data-only
python run.py --delhi-data-only

# API endpoints
# GET /data/sources?scope=global
# GET /data/availability?scope=global
# GET /data/coverage?scope=india
# GET /data/status?scope=delhi
```

## 16. Demo workflow (M16 additions)

1. Start backend + dashboard as in M15.
2. The "Global data acquisition" panel in `GlobalStatusPanel` shows the
   per-scope data status (e.g. global: OSM acquired, everything else
   unavailable for lack of credentials).
3. Switch Region → India/Global: the panel reflects that scope's report
   (scope-scoped) and repeats the honest disclaimer that no global prediction
   or AQI is produced from the data layer.
