export interface HealthResponse {
  status: string;
  service: string;
  version: string;
  timestamp: string;
}

export interface DateResponse {
  dates: string[];
}

export interface GridResponse {
  date: string;
  resolution_m: number;
  crs: string;
  n_rows: number;
  n_cols: number;
  bounds: {
    left: number;
    bottom: number;
    right: number;
    top: number;
  };
  raster_url: string;
  note: string;
}

export interface PM25Response {
  date: string;
  latitude: number;
  longitude: number;
  resolution_m: number;
  pm25: number | null;
  units: string;
  status: "valid" | "NoData";
}

export interface AQIResponse {
  date: string;
  latitude: number;
  longitude: number;
  pm25_aqi: number | null;
  category: string | null;
  type: string;
}

export interface AODInfo {
  aod: number | null;
  source: string;
  resolution_m: number;
  crs: string | null;
  date: string | null;
}

export interface LocationResponse {
  date: string;
  location: { latitude: number; longitude: number };
  pm25: number | null;
  pm25_units: string;
  pm25_derived_aqi: number | null;
  aqi_category: string | null;
  aqi_type: string;
  uncertainty: unknown;
  uncertainty_status: string;
  model: string;
  dataset_mode: string;
  aod_used: boolean;
  aod_info: AODInfo | null;
}

export interface HotspotProperties {
  hotspot_id: number;
  date: string;
  area_km2: number;
}

export interface HotspotFeature {
  type: "Feature";
  properties: HotspotProperties;
  geometry: {
    type: string;
    coordinates: number[][][];
  };
}

export interface HotspotCollection {
  type: "FeatureCollection";
  features: HotspotFeature[];
}

export interface HotspotStatisticsResponse {
  method: string | null;
  minimum_category: string | null;
  hotspot_cell_count: number | null;
  hotspot_zone_count: number | null;
  hotspot_area_km2: number | null;
  mean_pm25_ug_m3: number | null;
  max_pm25_ug_m3: number | null;
  mean_aqi: number | null;
  max_aqi: number | null;
  note: string | null;
}

export interface Station {
  station_id: string;
  latitude: number;
  longitude: number;
  observation_count: number | null;
  first_timestamp: string | null;
  last_timestamp: string | null;
  latest_pm25: number | null;
}

export interface StationObservation {
  date: string;
  pm25: number;
  observation_count: number;
}

export interface StationDetail {
  station_id: string;
  latitude: number;
  longitude: number;
  available_dates: string[];
  observations: StationObservation[];
}

export interface FeatureImportanceItem {
  feature: string;
  random_forest_importance: number | null;
  xgboost_gain: number | null;
  rf_relative_share: number | null;
  xgb_relative_share: number | null;
}

export interface FeatureImportanceResponse {
  model: string | null;
  dataset_mode: string | null;
  features: FeatureImportanceItem[];
  interpretation: string;
}

export interface UncertaintyResponse {
  status: string;
  method: string | null;
  reason: string | null;
  data_requirements: string[] | null;
  future_method: string | null;
  uncertainty_score?: number | null;
  sources?: Record<string, unknown>;
}

export type MapQuadCoordinates = [
  [number, number],
  [number, number],
  [number, number],
  [number, number],
];

export interface RasterLayerData {
  /** RGBA canvas of colorized raster pixels. */
  canvas: HTMLCanvasElement;
  /** Four WGS84 corner coordinates: [TL, TR, BR, BL] */
  coordinates: MapQuadCoordinates;
  bounds: { west: number; south: number; east: number; north: number };
  /** In-bounds min/max values actually drawn. */
  valueRange: { min: number; max: number } | null;
}

export interface ApiError extends Error {
  status?: number;
}

export type LayerId = "pm25" | "aqi" | "hotspots" | "stations" | "pm25_1km" | "aod";

export interface LayerVisibility {
  pm25: boolean;
  aqi: boolean;
  hotspots: boolean;
  stations: boolean;
  pm25_1km: boolean;
  aod: boolean;
}

// ---------------------------------------------------------------------------
// Globalization (Milestone 15): AOI-configurable platform
// ---------------------------------------------------------------------------

export interface RegionInfo {
  name: string;
  mode: string | null;
  aqi_scheme: string | null;
  bounds: {
    west: number;
    south: number;
    east: number;
    north: number;
  } | null;
  area_km2: number | null;
}

export interface RegionCatalogResponse {
  regions: Record<string, RegionInfo>;
  default: string;
}

export interface ModelScopeInfo {
  label: string;
  status: string;
  reason: string;
  training_region?: string | null;
}

export interface ModelScopesResponse {
  default_scope: string;
  scopes: Record<string, ModelScopeInfo>;
}

export interface GroundTruthAvailability {
  n_rows_total: number;
  n_stations_total: number;
  n_rows_in_aoi: number;
  n_stations_in_aoi: number;
  sources: string[];
  countries: string[];
}

export interface DataAvailabilityResponse {
  manifest_version: number;
  date: string | null;
  aoi: { name: string; mode: string; bounds: unknown };
  ground_truth: GroundTruthAvailability;
  model_scopes: Record<string, { label: string; status: string; reason: string }>;
  datasets: Record<string, unknown>;
  overall: string;
  notes: string[];
}

export interface OutputMetadataResponse {
  aoi: { name: string; mode: string };
  date: string | null;
  inference: {
    can_predict: boolean;
    scope_status: string;
    reason: string;
    resolution_500m_output_exists: boolean;
    has_raster_data: boolean;
  };
  downscaling: { available: boolean; reason: string; caveats: string[] };
  hotspots: { available: boolean; reason: string; definition: string };
  note: string;
}

export interface AOIInfoResponse {
  name: string;
  mode: string;
  bounds: { west: number; south: number; east: number; north: number };
  area_km2: number;
  is_global: boolean;
  metric_crs: string | null;
  model_scope: {
    id: string;
    status: string;
    can_predict: boolean;
    reason: string;
  };
}

// ---------------------------------------------------------------------------
// Global data acquisition (Milestone 16): /data/* endpoints
// Honest, read-only metadata about which real datasets were acquired.
// ---------------------------------------------------------------------------

export interface GlobalDataSourceSpec {
  name: string;
  status: "AVAILABLE" | "PARTIAL" | "UNAVAILABLE" | string;
  details: {
    reason?: string;
    tiles_completed?: number;
    tiles_failed?: number;
    artifacts?: string[];
    composite_metadata?: {
      target_date?: string;
      ndvi_date?: string;
      temporal_offset_days?: number;
      no_future_match?: boolean;
    };
  };
}

export interface GlobalDataAvailabilityResponse {
  report_version: number;
  built_for_scope: string;
  timestamp: string | null;
  status?: "not_run";
  note?: string;
  observations: {
    n_observations: number | null;
    n_daily_rows: number | null;
    n_stations: number | null;
    countries: string[];
    date_range: string[] | null;
    qc: Record<string, unknown>;
  } | null;
  sources: Record<string, GlobalDataSourceSpec> | null;
  synthetic_data_leakage: string;
}

export interface GlobalDataStatusResponse {
  scope: string;
  overall: "available" | "unavailable" | "not_run";
  available_sources: string[];
  unavailable_sources: string[];
  observations: {
    n_observations: number | null;
    n_stations: number | null;
  } | null;
  synthetic_data_leakage: string;
  ml_not_implemented: boolean;
  prediction_not_implemented: boolean;
}

// ---------------------------------------------------------------------------
// Production-grade availability registry (Stage 1)
// ---------------------------------------------------------------------------

export interface SourceFreshness {
  last_fetch_timestamp: string | null;
  window_s: number | null;
  fresh: boolean | null;
  stale_reason: string | null;
  age_s: number | null;
}

export interface SourceConfidence {
  level: "HIGH" | "MEDIUM" | "LOW" | "NONE";
  reason: string | null;
}

export interface AvailabilitySourceEntry {
  id: string;
  name: string;
  type: string;
  coverage: string;
  enabled: boolean;
  status: "AVAILABLE" | "PARTIAL" | "UNAVAILABLE" | "FAILED" | "STALE";
  reason: string | null;
  credentials_required: string[];
  credentials_present: Record<string, boolean>;
  freshness: SourceFreshness;
  confidence: SourceConfidence;
  artifact_checksums: Record<string, string> | null;
  notes: string;
}

export interface ReadinessSummary {
  available_source_count: number;
  total_source_count: number;
  pm25_ground_truth_available: boolean;
  has_any_real_data: boolean;
  all_sources_unavailable: boolean;
}

export interface AvailabilityRegistryResponse {
  registry_version: number;
  built_for_scope: string;
  timestamp: string;
  overall_status: "AVAILABLE" | "PARTIAL" | "UNAVAILABLE" | "FAILED" | "STALE";
  sources: Record<string, AvailabilitySourceEntry>;
  readiness_summary: ReadinessSummary;
  freshness_windows_s: Record<string, number>;
  note: string;
}

export interface StaleSourceResponse {
  scope: string;
  overall: string;
  available_sources: string[];
  stale_sources: string[];
  unavailable_sources: string[];
  readiness_summary: ReadinessSummary | null;
}
