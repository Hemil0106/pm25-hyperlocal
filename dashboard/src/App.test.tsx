import { beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import App from "./App";
import * as api from "./api";
import * as raster from "./map/raster";
import type {
  FeatureImportanceResponse,
  HotspotCollection,
  HotspotStatisticsResponse,
  LocationResponse,
  Station,
  UncertaintyResponse,
} from "./types";

vi.mock("./api", () => ({
  getHealth: vi.fn(),
  getAvailableDates: vi.fn(),
  getMetadata: vi.fn(),
  getStations: vi.fn(),
  getFeatureImportance: vi.fn(),
  getUncertainty: vi.fn(),
  getHotspots: vi.fn(),
  getHotspotStatistics: vi.fn(),
  getLocation: vi.fn(),
  getStationDetail: vi.fn(),
  getRegions: vi.fn(),
  getModelScopes: vi.fn(),
  getAOIInfo: vi.fn(),
  getDataAvailability: vi.fn(),
  getOutputMetadata: vi.fn(),
  getGlobalDataStatus: vi.fn(),
  getPM25RasterUrl: vi.fn((_date: string, resolution: string) => `/raster/pm25?resolution=${resolution}`),
  getAQIRasterUrl: vi.fn(() => "/raster/aqi"),
}));

vi.mock("./map/raster", () => ({
  loadRasterLayer: vi.fn(async () => ({
    canvas: document.createElement("canvas"),
    coordinates: [
      [77.0, 28.8],
      [77.4, 28.8],
      [77.4, 28.4],
      [77.0, 28.4],
    ],
    bounds: { west: 77.0, south: 28.4, east: 77.4, north: 28.8 },
    valueRange: { min: 100, max: 143 },
  })),
}));

vi.mock("./map/MapView", () => ({
  MapView: (props: {
    onMapClick: (lat: number, lon: number) => void;
    onHotspotClick: (p: { hotspot_id: number; area_km2: number; date: string }) => void;
    onStationClick: (s: Station) => void;
  }) => (
    <>
      <button
        type="button"
        data-testid="map"
        onClick={() => props.onMapClick(28.6, 77.2)}
      >
        map
      </button>
      <button
        type="button"
        data-testid="hotspot"
        onClick={() => {
          props.onMapClick(28.6, 77.2);
          props.onHotspotClick({ hotspot_id: 1, area_km2: 641, date: "2025-01-01" });
        }}
      >
        hotspot
      </button>
      <button
        type="button"
        data-testid="station"
        onClick={() =>
          props.onStationClick({
            station_id: "ST_01",
            latitude: 28.628386,
            longitude: 77.217111,
            observation_count: 49,
            first_timestamp: "2025-01-01 00:00:00+05:30",
            last_timestamp: "2025-01-07 00:00:00+05:30",
            latest_pm25: 94.85,
          })
        }
      >
        station
      </button>
    </>
  ),
}));

const mockedApi = vi.mocked(api);

const LOCATION: LocationResponse = {
  date: "2025-01-01",
  location: { latitude: 28.6, longitude: 77.2 },
  pm25: 123.4,
  pm25_units: "µg/m³",
  pm25_derived_aqi: 303,
  aqi_category: "VERY_POOR",
  aqi_type: "PM2.5-derived AQI/sub-index",
  uncertainty: null,
  uncertainty_status: "DEFERRED",
  model: "XGBoost",
  dataset_mode: "fallback",
  aod_used: false,
};

const STATIONS: Station[] = [
  {
    station_id: "ST_01",
    latitude: 28.628,
    longitude: 77.217,
    observation_count: 49,
    first_timestamp: "2025-01-01",
    last_timestamp: "2025-01-07",
    latest_pm25: 94.85,
  },
];

const HOTSPOTS: HotspotCollection = {
  type: "FeatureCollection",
  features: [
    {
      type: "Feature",
      properties: { hotspot_id: 1, date: "2025-01-01", area_km2: 641 },
      geometry: { type: "Polygon", coordinates: [] },
    },
  ],
};

const STATS: HotspotStatisticsResponse = {
  method: "aqi_category",
  minimum_category: "VERY_POOR",
  hotspot_cell_count: 2576,
  hotspot_zone_count: 3,
  hotspot_area_km2: 644,
  mean_pm25_ug_m3: 126.9,
  max_pm25_ug_m3: 143.3,
  mean_aqi: 305.5,
  max_aqi: 318,
  note: null,
};

const FEATURE_IMPORTANCE: FeatureImportanceResponse = {
  model: "XGBoost",
  dataset_mode: "fallback",
  features: [{ feature: "road_density", random_forest_importance: 0.1275, xgboost_gain: 46.1, rf_relative_share: 0.1275, xgb_relative_share: 0.1008 }],
  interpretation: "Model feature importance; not causal attribution.",
};

const UNCERTAINTY: UncertaintyResponse = {
  status: "DEFERRED",
  method: "none",
  reason: "Insufficient data",
  data_requirements: ["more stations"],
  future_method: "cross-validation",
};

const NODATA_LOCATION: LocationResponse = {
  date: "2025-01-01",
  location: { latitude: 28.6, longitude: 77.2 },
  pm25: null,
  pm25_units: "µg/m³",
  pm25_derived_aqi: null,
  aqi_category: null,
  aqi_type: "PM2.5-derived AQI/sub-index",
  uncertainty: null,
  uncertainty_status: "DEFERRED",
  model: "XGBoost",
  dataset_mode: "fallback",
  aod_used: false,
};

function installDefaultMocks() {
  mockedApi.getHealth.mockResolvedValue({
    status: "ok",
    service: "pm25-mapping-api",
    version: "0.1.0",
    timestamp: "2026-01-01T00:00:00Z",
  });
  mockedApi.getAvailableDates.mockResolvedValue({ dates: ["2025-01-01"] });
  mockedApi.getMetadata.mockResolvedValue({
    project: "PM2.5 Hyperlocal Mapping",
    "PM2.5": { model: "XGBoost", dataset_mode: "fallback", AOD_used: false },
    "Data limitations": {
      training_rows: 16,
      stations: 4,
      AOD_availability: "unavailable",
      downscaling_training_status: "insufficient (16 < 20)",
      model_performance: "provisional / pipeline validation only",
      "500m_output": "baseline prototype; higher resolution does NOT imply higher accuracy",
    },
    Downscaling: {
      method: "residual refinement (architecture); this run baseline prototype / parent-constant",
    },
    Uncertainty: { status: "DEFERRED", method: "none" },
  });
  mockedApi.getStations.mockResolvedValue(STATIONS);
  mockedApi.getFeatureImportance.mockResolvedValue(FEATURE_IMPORTANCE);
  mockedApi.getUncertainty.mockResolvedValue(UNCERTAINTY);
  mockedApi.getHotspots.mockResolvedValue(HOTSPOTS);
  mockedApi.getHotspotStatistics.mockResolvedValue(STATS);
  mockedApi.getLocation.mockResolvedValue(LOCATION);
  mockedApi.getStationDetail.mockResolvedValue({
    station_id: "ST_01",
    latitude: 28.628,
    longitude: 77.217,
    available_dates: ["2025-01-01"],
    observations: [],
  });
  mockedApi.getRegions.mockResolvedValue({
    regions: {
      delhi: {
        name: "Delhi",
        mode: "named_region",
        aqi_scheme: "india_cpcb",
        bounds: { west: 77.0, south: 28.4, east: 77.4, north: 28.8 },
        area_km2: 1734.5,
      },
      india: {
        name: "India",
        mode: "named_region",
        aqi_scheme: null,
        bounds: { west: 68.0, south: 6.0, east: 98.0, north: 36.0 },
        area_km2: 3287000,
      },
      global: {
        name: "Global",
        mode: "global",
        aqi_scheme: null,
        bounds: null,
        area_km2: 510000000,
      },
    },
    default: "delhi",
  });
  mockedApi.getModelScopes.mockResolvedValue({
    default_scope: "prototype_local",
    scopes: {
      prototype_local: {
        label: "Prototype (Delhi-local)",
        status: "available",
        reason: "Delhi-trained prototype",
      },
      regional: { label: "Regional", status: "unavailable", reason: "No regional model" },
      global: { label: "Global", status: "unavailable", reason: "No global model" },
    },
  });
  mockedApi.getAOIInfo.mockResolvedValue({
    name: "Delhi",
    mode: "named_region",
    bounds: { west: 77.0, south: 28.4, east: 77.4, north: 28.8 },
    area_km2: 1734.5,
    is_global: false,
    metric_crs: "EPSG:32643",
    model_scope: {
      id: "prototype_local",
      status: "available",
      can_predict: true,
      reason: "Delhi-trained prototype",
    },
  });
  mockedApi.getDataAvailability.mockResolvedValue({
    manifest_version: 1,
    date: "2025-01-01",
    aoi: { name: "Delhi", mode: "named_region", bounds: { west: 77.0, south: 28.4, east: 77.4, north: 28.8 } },
    ground_truth: {
      n_rows_total: 246,
      n_stations_total: 5,
      n_rows_in_aoi: 246,
      n_stations_in_aoi: 5,
      sources: ["cpcb"],
      countries: ["India"],
    },
    model_scopes: {
      prototype_local: { label: "Prototype (Delhi-local)", status: "available", reason: "Delhi-trained prototype" },
      regional: { label: "Regional", status: "unavailable", reason: "No regional model" },
      global: { label: "Global", status: "unavailable", reason: "No global model" },
    },
    datasets: {},
    overall: "available",
    notes: [],
  });
  mockedApi.getGlobalDataStatus.mockResolvedValue({
    scope: "delhi",
    overall: "unavailable",
    available_sources: [],
    unavailable_sources: ["aod", "dem", "ndvi", "osm", "pm25", "viirs", "weather"],
    observations: { n_observations: null, n_stations: null },
    synthetic_data_leakage: "NONE",
    ml_not_implemented: true,
    prediction_not_implemented: true,
  });
  mockedApi.getOutputMetadata.mockResolvedValue({
    aoi: { name: "Delhi", mode: "named_region" },
    date: "2025-01-01",
    inference: {
      can_predict: true,
      scope_status: "available",
      reason: "Delhi-trained prototype",
      resolution_500m_output_exists: true,
      has_raster_data: true,
    },
    downscaling: { available: true, reason: "Delhi prototype", caveats: [] },
    hotspots: { available: true, reason: "Delhi prototype", definition: "predicted high-pollution zone" },
    note: "",
  });
}

describe("App dashboard", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(raster.loadRasterLayer).mockResolvedValue({
      canvas: document.createElement("canvas"),
      coordinates: [
        [77.0, 28.8],
        [77.4, 28.8],
        [77.4, 28.4],
        [77.0, 28.4],
      ],
      bounds: { west: 77.0, south: 28.4, east: 77.4, north: 28.8 },
      valueRange: { min: 100, max: 143 },
    });
    installDefaultMocks();
  });

  it("renders the dashboard with real API data", async () => {
    render(<App />);
    await waitFor(() => {
      expect(screen.getByText("Hyperlocal PM2.5 Intelligence")).toBeInTheDocument();
    });
    expect(screen.getByText("SIH Prototype")).toBeInTheDocument();
    await waitFor(() => {
      expect(screen.getAllByText("XGBoost").length).toBeGreaterThan(0);
    });
    await waitFor(() => {
      expect(screen.getAllByText("3").length).toBeGreaterThan(0);
    });
  });

  it("shows backend unavailable state on API failure", async () => {
    mockedApi.getHealth.mockRejectedValue(new Error("network"));
    mockedApi.getAvailableDates.mockRejectedValue(new Error("network"));
    render(<App />);
    await waitFor(() => {
      expect(
        screen.getByText(/Backend unavailable — start the FastAPI server/i),
      ).toBeInTheDocument();
    });
  });

  it("populates the date selector from /available-dates", async () => {
    render(<App />);
    const select = await screen.findByLabelText("Selected date");
    expect(select).toHaveValue("2025-01-01");
    expect(screen.getByText(/01 Jan 2025/)).toBeInTheDocument();
  });

  it("toggles layer visibility", async () => {
    render(<App />);
    const aqiToggle = await screen.findByLabelText("PM2.5-derived AQI");
    expect(aqiToggle).not.toBeChecked();
    fireEvent.click(aqiToggle);
    expect(aqiToggle).toBeChecked();
  });

  it("shows PM2.5 and AQI for a clicked location", async () => {
    render(<App />);
    await waitFor(() => {
      expect(screen.getByTestId("map")).toBeInTheDocument();
    });
    fireEvent.click(screen.getByTestId("map"));
    await waitFor(() => {
      expect(screen.getAllByText("VERY_POOR").length).toBeGreaterThan(0);
    });
    expect(screen.getAllByText(/123\.4 µg\/m³/).length).toBeGreaterThan(0);
    expect(screen.getAllByText("303").length).toBeGreaterThan(0);
    expect(
      screen.getByText(/model-derived spatial estimates, not direct measurements/i),
    ).toBeInTheDocument();
  });

  it("never labels the AQI as National AQI", async () => {
    render(<App />);
    await waitFor(() => {
      expect(screen.getByTestId("map")).toBeInTheDocument();
    });
    fireEvent.click(screen.getByTestId("map"));
    await waitFor(() => {
      expect(screen.getAllByText("VERY_POOR").length).toBeGreaterThan(0);
    });
    expect(screen.queryByText(/National AQI/i)).not.toBeInTheDocument();
    expect(screen.getAllByText(/PM2\.5-derived AQI/).length).toBeGreaterThan(0);
  });

  it("displays DEFERRED uncertainty with no fabricated confidence", async () => {
    render(<App />);
    await waitFor(() => {
      expect(screen.getAllByText("DEFERRED").length).toBeGreaterThan(0);
    });
    expect(screen.queryByText(/95% confidence/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/87% confidence/i)).not.toBeInTheDocument();
  });

  it("shows hotspot statistics from the API", async () => {
    render(<App />);
    await waitFor(() => {
      expect(screen.getAllByText("3").length).toBeGreaterThan(0);
    });
    expect(screen.getByText("km² total area")).toBeInTheDocument();
    expect(screen.getByText("126.9")).toBeInTheDocument();
  });

  it("renders the Model & data panel from /metadata without em-dashes", async () => {
    render(<App />);
    const panel = await screen.findByText("Model & data");
    const section = panel.closest("section");
    expect(section).not.toBeNull();
    const scope = within(section as HTMLElement);
    await waitFor(() => {
      expect(scope.getByText("XGBoost")).toBeInTheDocument();
    });
    expect(scope.getByText("Fallback")).toBeInTheDocument();
    expect(scope.getByText("Unavailable")).toBeInTheDocument();
    expect(scope.getByText("16")).toBeInTheDocument();
    expect(scope.getByText("4")).toBeInTheDocument();
  });

  it("shows Select a location in metric cards before any selection", async () => {
    render(<App />);
    await waitFor(() => {
      expect(screen.getAllByText("Select a location").length).toBe(2);
    });
  });

  it("updates the top metric cards for a selected location", async () => {
    render(<App />);
    await waitFor(() => {
      expect(screen.getByTestId("map")).toBeInTheDocument();
    });
    fireEvent.click(screen.getByTestId("map"));
    await waitFor(() => {
      expect(screen.getAllByText(/123\.4 µg\/m³/).length).toBeGreaterThan(0);
    });
    expect(screen.getAllByText("303").length).toBeGreaterThan(0);
    expect(screen.queryByText("Select a location")).not.toBeInTheDocument();
  });

  it("shows NoData for a selected cell with no valid prediction", async () => {
    mockedApi.getLocation.mockResolvedValue(NODATA_LOCATION);
    render(<App />);
    await waitFor(() => {
      expect(screen.getByTestId("map")).toBeInTheDocument();
    });
    fireEvent.click(screen.getByTestId("map"));
    await waitFor(() => {
      expect(screen.getByText("NoData")).toBeInTheDocument();
    });
    expect(screen.getAllByText("Unavailable").length).toBeGreaterThan(0);
    expect(screen.getAllByText("No data").length).toBeGreaterThan(0);
  });

  it("shows a friendly message for an outside-AOI click without crashing", async () => {
    mockedApi.getLocation.mockRejectedValue(
      new Error("Point is outside the study area (AOI)."),
    );
    render(<App />);
    await waitFor(() => {
      expect(screen.getByTestId("map")).toBeInTheDocument();
    });
    fireEvent.click(screen.getByTestId("map"));
    await waitFor(() => {
      expect(
        screen.getByText(/Location outside the study area \(AOI\)/),
      ).toBeInTheDocument();
    });
    expect(screen.getByText("Hyperlocal PM2.5 Intelligence")).toBeInTheDocument();
  });

  it("shows the selected hotspot and updates the location on a hotspot click", async () => {
    render(<App />);
    await waitFor(() => {
      expect(screen.getByTestId("hotspot")).toBeInTheDocument();
    });
    fireEvent.click(screen.getByTestId("hotspot"));
    await waitFor(() => {
      expect(screen.getByText("Selected hotspot")).toBeInTheDocument();
    });
    expect(screen.getByText("641.0 km²")).toBeInTheDocument();
    await waitFor(() => {
      expect(screen.getAllByText(/123\.4 µg\/m³/).length).toBeGreaterThan(0);
    });
  });

  it("shows the selected station and fetches its detail", async () => {
    render(<App />);
    await waitFor(() => {
      expect(screen.getByTestId("station")).toBeInTheDocument();
    });
    fireEvent.click(screen.getByTestId("station"));
    await waitFor(() => {
      expect(screen.getByText("Selected station")).toBeInTheDocument();
    });
    expect(screen.getAllByText("ST_01").length).toBeGreaterThan(0);
    expect(mockedApi.getStationDetail).toHaveBeenCalledWith("ST_01", undefined);
  });

  it("renders human-readable feature importance labels", async () => {
    render(<App />);
    await waitFor(() => {
      expect(screen.getByText("Model feature importance")).toBeInTheDocument();
    });
    expect(screen.getByText("Road Density")).toBeInTheDocument();
    expect(screen.queryByText("road_density")).not.toBeInTheDocument();
    expect(screen.getByText("Not causal attribution.")).toBeInTheDocument();
  });

  it("shows prototype limitations when expanded", async () => {
    render(<App />);
    await waitFor(() => {
      expect(
        screen.getByRole("button", { name: "Show limitations" }),
      ).toBeInTheDocument();
    });
    fireEvent.click(screen.getByRole("button", { name: "Show limitations" }));
    expect(screen.getByText("AOD currently unavailable")).toBeInTheDocument();
    expect(screen.getByText("500 m residual model not trained")).toBeInTheDocument();
    expect(
      screen.getByText("PM2.5-derived AQI is not the full multi-pollutant National AQI"),
    ).toBeInTheDocument();
  });

  it("propagates the selected date to the hotspots request", async () => {
    render(<App />);
    await waitFor(() => {
      expect(mockedApi.getHotspots).toHaveBeenCalledWith("2025-01-01", undefined);
    });
  });

  it("shows the AOI region selector with Delhi, India and Global", async () => {
    render(<App />);
    const select = await screen.findByLabelText("Selected region");
    expect(select).toHaveValue("delhi");
    const options = Array.from(select.querySelectorAll("option")).map((o) => o.value);
    expect(options).toEqual(["delhi", "india", "global"]);
  });

  it("keeps the Delhi scope available and labelled SIH Prototype", async () => {
    render(<App />);
    await waitFor(() => {
      expect(screen.getByText("SIH Prototype")).toBeInTheDocument();
    });
    await waitFor(() => {
      expect(screen.getByText("Globalization status")).toBeInTheDocument();
    });
    expect(screen.getAllByText("1,734.5 km²").length).toBeGreaterThan(0);
  });

  it("shows honest unavailable status when switching to the Global AOI", async () => {
    mockedApi.getAOIInfo.mockResolvedValue({
      name: "Global",
      mode: "global",
      bounds: { west: -180, south: -90, east: 180, north: 90 },
      area_km2: 510065621.72,
      is_global: true,
      metric_crs: null,
      model_scope: {
        id: "global",
        status: "unavailable_for_aoi",
        can_predict: false,
        reason: "No global model trained; Delhi prototype is never applied outside Delhi.",
      },
    });
    mockedApi.getDataAvailability.mockResolvedValue({
      manifest_version: 1,
      date: "2025-01-01",
      aoi: { name: "Global", mode: "global", bounds: null },
      ground_truth: {
        n_rows_total: 246,
        n_stations_total: 5,
        n_rows_in_aoi: 0,
        n_stations_in_aoi: 0,
        sources: ["cpcb"],
        countries: ["India"],
      },
      model_scopes: {},
      datasets: {},
      overall: "data_limited",
      notes: ["Ground truth covers only India; global PM2.5 prediction is not scientifically supported."],
    });
    mockedApi.getOutputMetadata.mockResolvedValue({
      aoi: { name: "Global", mode: "global" },
      date: "2025-01-01",
      inference: {
        can_predict: false,
        scope_status: "unavailable_for_aoi",
        reason: "No global model trained.",
        resolution_500m_output_exists: false,
        has_raster_data: false,
      },
      downscaling: { available: false, reason: "No model", caveats: [] },
      hotspots: { available: false, reason: "No model", definition: "predicted high-pollution zone" },
      note: "",
    });
    vi.mocked(raster.loadRasterLayer).mockRejectedValue(new Error("no raster for global"));
    render(<App />);
    const select = await screen.findByLabelText("Selected region");
    fireEvent.change(select, { target: { value: "global" } });
    await waitFor(() => {
      expect(
        screen.getByText(/no validated model exists for this scope/i),
      ).toBeInTheDocument();
    });
    expect(
      screen.getByText(/The Delhi prototype is not applied outside Delhi/i),
    ).toBeInTheDocument();
    await waitFor(() => {
      expect(screen.getByText("Why is prediction unavailable?")).toBeInTheDocument();
    });
    expect(
      screen.getByText(/no validated model trained on observations covering this AOI exists/i),
    ).toBeInTheDocument();
    expect(screen.queryByText(/National AQI/i)).not.toBeInTheDocument();
  });

  it("shows honest prediction-unavailable state for the India AOI", async () => {
    mockedApi.getAOIInfo.mockResolvedValue({
      name: "India",
      mode: "named_region",
      bounds: { west: 68, south: 6, east: 98, north: 36 },
      area_km2: 3287000,
      is_global: false,
      metric_crs: "EPSG:32643",
      model_scope: {
        id: "regional",
        status: "unavailable",
        can_predict: false,
        reason: "No regional model trained; Delhi prototype is never applied outside Delhi.",
      },
    });
    mockedApi.getOutputMetadata.mockResolvedValue({
      aoi: { name: "India", mode: "named_region" },
      date: "2025-01-01",
      inference: {
        can_predict: false,
        scope_status: "unavailable",
        reason: "No regional model trained.",
        resolution_500m_output_exists: false,
        has_raster_data: false,
      },
      downscaling: { available: false, reason: "No model", caveats: [] },
      hotspots: { available: false, reason: "No model", definition: "predicted high-pollution zone" },
      note: "",
    });
    mockedApi.getDataAvailability.mockResolvedValue({
      manifest_version: 1,
      date: "2025-01-01",
      aoi: { name: "India", mode: "named_region", bounds: { west: 68, south: 6, east: 98, north: 36 } },
      ground_truth: {
        n_rows_total: 246,
        n_stations_total: 5,
        n_rows_in_aoi: 246,
        n_stations_in_aoi: 5,
        sources: ["cpcb"],
        countries: ["India"],
      },
      model_scopes: {},
      datasets: {},
      overall: "available",
      notes: [],
    });
    vi.mocked(raster.loadRasterLayer).mockRejectedValue(new Error("no raster for india"));
    render(<App />);
    const select = await screen.findByLabelText("Selected region");
    fireEvent.change(select, { target: { value: "india" } });
    await waitFor(() => {
      expect(
        screen.getByText(/no validated model trained on observations for this scope exists/i),
      ).toBeInTheDocument();
    });
    expect(
      screen.getByText(/The Delhi prototype is not applied outside Delhi/i),
    ).toBeInTheDocument();
  });

  it("derives data-source statuses from API metadata instead of hard-coding", async () => {
    mockedApi.getGlobalDataStatus.mockResolvedValue({
      scope: "delhi",
      overall: "available",
      available_sources: ["osm"],
      unavailable_sources: ["aod", "dem", "ndvi", "pm25", "viirs", "weather"],
      observations: { n_observations: 42, n_stations: 4 },
      synthetic_data_leakage: "NONE",
      ml_not_implemented: true,
      prediction_not_implemented: true,
    });
    render(<App />);
    const title = await screen.findByText("Data sources");
    const panel = title.closest("section");
    expect(panel).not.toBeNull();
    const scope = within(panel as HTMLElement);
    await waitFor(() => {
      expect(scope.getByText("OpenStreetMap")).toBeInTheDocument();
    });
    expect(scope.getByText("Available")).toBeInTheDocument();
    expect(scope.getAllByText("Unavailable").length).toBeGreaterThan(0);
    expect(scope.queryByText(/unavailable in current run/i)).not.toBeInTheDocument();
  });

  it("shows a loading state while the map layer is being fetched", async () => {
    let resolveRaster: ((value: unknown) => void) | undefined;
    const pending = new Promise((resolve) => {
      resolveRaster = resolve;
    });
    vi.mocked(raster.loadRasterLayer).mockImplementation(() => pending as never);
    render(<App />);
    await waitFor(() => {
      expect(screen.getByText("Loading PM2.5 layer…")).toBeInTheDocument();
    });
    resolveRaster?.({
      canvas: document.createElement("canvas"),
      coordinates: [
        [77.0, 28.8],
        [77.4, 28.8],
        [77.4, 28.4],
        [77.0, 28.4],
      ],
      bounds: { west: 77.0, south: 28.4, east: 77.4, north: 28.8 },
      valueRange: { min: 100, max: 143 },
    });
    await waitFor(() => {
      expect(screen.queryByText("Loading PM2.5 layer…")).not.toBeInTheDocument();
    });
  });
});
