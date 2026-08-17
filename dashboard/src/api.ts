import type {
  AOIInfoResponse,
  AQIResponse,
  DataAvailabilityResponse,
  DateResponse,
  FeatureImportanceResponse,
  GlobalDataAvailabilityResponse,
  GlobalDataStatusResponse,
  GridResponse,
  HealthResponse,
  HotspotCollection,
  HotspotStatisticsResponse,
  LocationResponse,
  ModelScopesResponse,
  OutputMetadataResponse,
  PM25Response,
  RegionCatalogResponse,
  Station,
  StationDetail,
  UncertaintyResponse,
} from "./types";

const API_BASE_URL: string =
  import.meta.env.VITE_API_BASE_URL ?? "";

export class ApiRequestError extends Error {
  status?: number;

  constructor(message: string, status?: number) {
    super(message);
    this.name = "ApiRequestError";
    this.status = status;
  }
}

async function request<T>(path: string): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`);
  } catch {
    throw new ApiRequestError("Backend unavailable");
  }
  if (!response.ok) {
    let detail = `HTTP ${response.status}`;
    try {
      const body = await response.json();
      if (body && typeof body.detail === "string") {
        detail = body.detail;
      }
    } catch {
      /* ignore malformed error bodies */
    }
    throw new ApiRequestError(detail, response.status);
  }
  return (await response.json()) as T;
}

export function getApiBaseUrl(): string {
  return API_BASE_URL;
}

export const getHealth = () => request<HealthResponse>("/health");
export const getAvailableDates = () => request<DateResponse>("/available-dates");
export const getMetadata = () => request<Record<string, unknown>>("/metadata");

export const getPM25 = (date: string, lat: number, lon: number, resolution = "500m") =>
  request<PM25Response>(`/pm25?date=${date}&lat=${lat}&lon=${lon}&resolution=${resolution}`);

export const getAQI = (date: string, lat: number, lon: number) =>
  request<AQIResponse>(`/aqi?date=${date}&lat=${lat}&lon=${lon}`);

export const getLocation = (date: string, lat: number, lon: number) =>
  request<LocationResponse>(`/location?date=${date}&lat=${lat}&lon=${lon}`);

export const getHotspots = (date?: string) =>
  request<HotspotCollection>(`/hotspots${date ? `?date=${date}` : ""}`);

export const getHotspotStatistics = () =>
  request<HotspotStatisticsResponse>("/hotspots/statistics");

export const getStations = () => request<Station[]>("/stations");

export const getStationDetail = (stationId: string) =>
  request<StationDetail>(`/stations/${stationId}`);

export const getFeatureImportance = () =>
  request<FeatureImportanceResponse>("/feature-importance");

export const getUncertainty = () => request<UncertaintyResponse>("/uncertainty");

export const getPM25Grid = (date: string, resolution = "500m") =>
  request<GridResponse>(`/pm25/grid?date=${date}&resolution=${resolution}`);

export const getPM25RasterUrl = (date: string, resolution = "500m") =>
  `${API_BASE_URL}/raster/pm25?date=${date}&resolution=${resolution}`;

export const getAQIRasterUrl = (date: string) =>
  `${API_BASE_URL}/raster/aqi?date=${date}`;

export const getRegions = () => request<RegionCatalogResponse>("/global/regions");

export const getModelScopes = () => request<ModelScopesResponse>("/global/model-scopes");

export const getAOIInfo = (region?: string) =>
  request<AOIInfoResponse>(`/global/aoi${region ? `?region=${region}` : ""}`);

export const getDataAvailability = (region?: string, date?: string) => {
  const params = new URLSearchParams();
  if (region) params.set("region", region);
  if (date) params.set("date", date);
  const qs = params.toString();
  return request<DataAvailabilityResponse>(`/global/data-availability${qs ? `?${qs}` : ""}`);
};

export const getOutputMetadata = (region?: string, date?: string) => {
  const params = new URLSearchParams();
  if (region) params.set("region", region);
  if (date) params.set("date", date);
  const qs = params.toString();
  return request<OutputMetadataResponse>(`/global/output-metadata${qs ? `?${qs}` : ""}`);
};

export const getGlobalDataStatus = (scope: string = "global") =>
  request<GlobalDataStatusResponse>(`/data/status?scope=${encodeURIComponent(scope)}`);

export const getGlobalDataAvailability = (scope: string = "global") =>
  request<GlobalDataAvailabilityResponse>(`/data/availability?scope=${encodeURIComponent(scope)}`);
