# Global Training Dataset — Final Report (Milestone 17)

Date: 2026-08-15
Scope: Build the global training dataset + feature-engineering + spatial
validation setup (LOSO/LOCO groups, leakage checks, readiness gate) from the
real Milestone 16 outputs — data engineering ONLY. No global ML training and
no global PM2.5/AQI/hotspot prediction in M17. The locked Delhi prototype
(Milestones 1–15) is preserved unchanged.

---

## 1. Executive summary

- The locked pipeline (`src/data_pipeline.py`) is **untouched**. All M15/M16
  tests continue to pass unchanged.
- A new `src/global_training/` package assembles a schema-correct training
  table from **real data only**: M16 daily PM2.5 observations joined with real
  OSM road-density tile artifacts. Unavailable features are `NaN` with
  explicit per-feature `<feature>_available` flags. Nothing is fabricated or
  substituted.
- **Explicit statement (M17 spec):**
  - **Global ML is NOT implemented** — intentionally deferred to Milestone 18.
  - **Global PM2.5 prediction is NOT implemented** — no global model/prediction
    artifact is produced.
- The readiness gate is **fully data-derived and never auto-YES**. With no
  real observations on disk (M16 produced none for lack of credentials), the
  run produces a **schema-correct empty table** + honest **NOT READY /
  INSUFFICIENT_DATA** reports. This is a valid, successful milestone: the
  engineering is in place and proven by synthetic in-memory tests.

## 2. Files created

| File | Purpose |
|---|---|
| `src/global_training/__init__.py` | Public API `run_global_training_pipeline` |
| `src/global_training/schema.py` | `SCHEMA_VERSION=1`, target/identity/feature/availability/meta columns, `FEATURE_SOURCE_MAP` |
| `src/global_training/osm_density.py` | `load_osm_road_segments`, `road_density_for` (per-tile OSM `road_segments` via `tile_for_point`), `osm_tiles_present` |
| `src/global_training/features.py` | Temporal cyclics (row's own date) + wind u/v components; never from PM2.5 |
| `src/global_training/builder.py` | `build_training_table`, `split_complete_cases`, `write_training_outputs` (3 parquet files) |
| `src/global_training/qc.py` | Schema/dtypes/duplicate-station-date/missingness; flags outliers, never deletes |
| `src/global_training/coverage.py` | Country/month/station coverage + geographic-bias flags + per-country CSV |
| `src/global_training/correlation.py` | Pearson + Spearman on complete-case rows only (honest `no_data`) |
| `src/global_training/target.py` | Target distribution + `completeness_flag` (`sufficient`/`insufficient`), no invented thresholds |
| `src/global_training/representativeness.py` | Spatial/temporal representativeness; PNGs only when data supports |
| `src/global_training/groups.py` | Station→country→region hierarchy, fold assignments, LOSO/LOCO status |
| `src/global_training/leakage.py` | 7 leakage checks; any FAIL/PRESENT stops the pipeline before writing outputs |
| `src/global_training/readiness.py` | Data-derived `MODEL TRAINING READY YES/NO` gate |
| `src/global_training/metadata.py` | Versioned metadata + input manifest (SHA-256) + limitations |
| `src/global_training/report.py` | Final `global_training_report.json` (this report's machine form) |
| `src/global_training/pipeline.py` | Orchestrator: inputs → build → QC → leakage gate → analysis → readiness → outputs |
| `tests/test_global_training.py` | 41 M17 tests (schema, features, OSM, builder, QC, coverage, correlation, target, groups, leakage, readiness, pipeline) |

## 3. Files modified

| File | Change |
|---|---|
| `run.py` | Added `--global-training-only` stage flag + branch |
| `README.md` | Added M17 section |

## 4. Delhi prototype preserved?

**Yes.** `src/data_pipeline.py` is unchanged; all M15/M16 backend tests pass
unchanged; legacy endpoints, config, and dashboard behavior unchanged.

## 5. Dataset summary (real run, scope=global)

| Metric | Value |
|---|---|
| Rows | 0 |
| Stations | 0 |
| Countries | 0 |
| Date range | — (no real observations on disk) |
| Complete-case rows | 0 |
| Feature coverage | every source feature 100% missing (nothing acquired) |
| OSM road-density artifacts | 3 real tile files joined when observations exist |

Real M16 daily observations (`data/processed/global/pm25/global_pm25_daily.parquet`)
do not exist in this environment because no credentials were configured. The
builder therefore emits the schema-correct empty table and every report runs
honestly on it. With real daily data present, the identical code produces the
populated table (proven by `tests/test_global_training.py::TestPipeline`).

## 6. Feature engineering (strictly limited)

- **Identity**: `station_id`, `station_name` (NaN — no display name acquired),
  `country`, `city` (NaN — no reverse-geocoding performed, never fabricated),
  `latitude`, `longitude`, `date`.
- **Target**: real normalized daily PM2.5 (`PM2.5`, µg/m³). Never a feature.
- **Gridded features** (AOD, temperature, humidity, wind speed/direction, NDVI,
  elevation, night lights): `NaN` + `_available=False` unless a real M16
  artifact is joined.
- **`road_density`**: real per-tile OSM `road_segments` count (spatial proxy,
  labeled "not a measurement"); `road_density_available` = tile artifact exists.
- **Derived**: `month`, `day_of_year`, `sin/cos(day_of_year)` from the row's own
  date; `wind_u/v = −V sin/cos(θ)` only when speed and direction are both real.
- **Availability**: 11 per-feature `_available` flags + `n_features_available`.
- **NDVI provenance**: `NDVI_source_date`/`NDVI_offset_days` (NaN when no NDVI).
- **Complete case**: valid target AND all features present; incomplete rows are
  preserved in `global_training_dataset_incomplete.parquet` — never deleted.

## 7. Spatial validation setup

| Group | Status | Reason |
|---|---|---|
| LOSO (Leave-One-Station-Out) | **INSUFFICIENT_DATA** | 0 stations / 0 complete-case rows; folds are never manufactured |
| LOCO (Leave-One-Country-Out) | **INSUFFICIENT_DATA** | 0 countries / 0 complete-case rows |
| Hierarchy | stations 0 · countries 0 · regions 0 | built from real data only |

`validation/global_spatial_split_assignments.parquet` + `global_validation_groups.json`
are written; they describe the real data and never invent folds. With synthetic
real-shaped data (20+ stations, 100+ complete rows) LOSO reports READY
(covered by tests).

## 8. Leakage checks

| Check | Status |
|---|---|
| Target leakage (features generated from PM2.5) | PASS |
| Duplicate station-date keys | PASS |
| Future information (features from other/future rows) | PASS |
| Same station/country in train & test (no split in M17) | N/A |
| Features generated from PM2.5 | PASS |
| Synthetic contamination (SYNTHETIC_TEST_DATA markers from M16) | NONE |
| Delhi→Global artifact contamination (rows inside Delhi bounds) | NONE |

**Overall: PASS.** Any FAIL/PRESENT stops the pipeline before writing training
outputs. Delhi rows are additionally detected and the run halts for the
`global`/`india` scopes (delhi scope is exempt by definition).

## 9. Readiness gate

- **MODEL TRAINING READY: NO** (real run). Reason: complete-case rows 0 < 1000;
  stations 0 < 20; countries 0 < 5; temporal days 0 < 30; every source feature
  coverage insufficient.
- The gate is **fully data-derived and never auto-YES** (thresholds recorded in
  `reports/global_training_readiness.json`).
- Proven both ways in tests: an all-features-present synthetic frame reports
  YES; the same frame with AOD dropped reports NO.

## 10. Global ML and prediction: NOT implemented

- **Global ML training is NOT part of Milestone 17** — intentionally deferred
  to Milestone 18. No global model is trained, saved, or served.
- **Global PM2.5 prediction is NOT implemented.** No global PM2.5/AQI/hotspot
  prediction artifact is produced. The M15 model-scope gates (`prototype_local`
  only, Delhi) remain the sole prediction path.
- Recorded in `global_training_metadata.json`, `global_training_report.json`,
  and `reports/global_training_readiness.json` (`global_ml_not_implemented`).

## 11. Acceptance run (2026-08-15)

`python run.py --global-training-only`:

```
Daily observations: status=missing  OSM tile artifacts=3
leakage target_leakage PASS
leakage duplicate_station_date PASS
leakage future_information PASS
leakage same_station_or_country_in_train_test N/A
leakage features_generated_from_pm25 PASS
leakage synthetic_contamination NONE
leakage delhi_artifact_contamination PASS
M17 SUMMARY (scope=global): rows=0 complete=0 stations=0 countries=0 readiness=NO
  LOSO INSUFFICIENT_DATA | LOCO INSUFFICIENT_DATA
  Global ML: NOT IMPLEMENTED - intentionally deferred to M18.
```

Artifacts written:

| Artifact | Path |
|---|---|
| Training table ×3 | `data/processed/global/training/global_training_dataset{,_complete,_incomplete}.parquet` |
| Final report | `data/processed/global/reports/global_training_report.json` |
| Readiness | `data/processed/global/reports/global_training_readiness.json` |
| Coverage | `data/processed/global/reports/global_feature_coverage_report.json` |
| Target | `data/processed/global/reports/global_target_report.json` |
| Correlation | `data/processed/global/diagnostics/global_feature_correlation.csv` |
| Coverage by country | `data/processed/global/diagnostics/global_coverage_by_country.csv` |
| Split assignments + groups | `data/processed/global/validation/global_spatial_split_assignments.parquet`, `global_validation_groups.json` |
| Metadata | `data/processed/global/metadata/global_training_metadata.json` |

## 12. Tests run / passed / failed

| Suite | Count | Result |
|---|---|---|
| Backend `pytest -q` | 209 (168 M15/M16 + 41 M17) | all passed |
| Dashboard `npm test` (vitest) | 28 (24 App + 4 api) | all passed |
| Dashboard `npm run build` | — | PASS (chunk-size warning only, cosmetic) |

## 13. Data / model limitations

- No credentials (OpenAQ v3, NASA Earthdata, CDS) in this environment → real
  M16 daily observations absent, so the real M17 table is empty. The full
  engineering path is proven by synthetic in-memory tests (never written as
  real artifacts).
- `road_density` is a per-tile road-segment count — a spatial proxy, not a
  measurement.
- `station_name`/`city` are NaN (no reverse-geocoding; never fabricated).
- Temporal features use the row's own date only — no future information.
- LOSO/LOCO report INSUFFICIENT_DATA rather than manufacturing fold sets.
- Correlation output is descriptive (complete-case only) and never used to
  auto-remove features (correlation ≠ causation).
- No global ML model and no global prediction exist; readiness is NOT READY.

## 14. Next stage

**Milestone 18 — do not implement yet.** Global model training + evaluation
may proceed only after real M16 observations/features exist AND the readiness
gate reports `MODEL TRAINING READY: YES`.

## 15. Exact commands

```powershell
# Backend tests (from project root)
.venv\Scripts\python.exe -m pytest -q

# Dashboard tests + build
cd dashboard
npm test
npm run build
cd ..

# Milestone 17 stage (from real M16 outputs)
python run.py --global-training-only
```
