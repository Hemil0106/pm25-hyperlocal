import type {
  AOIInfoResponse,
  AQIResponse,
  AvailabilityRegistryResponse,
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

function cityQs(city?: string): string {
  return city ? `city=${encodeURIComponent(city)}` : "";
}

export function getApiBaseUrl(): string {
  return API_BASE_URL;
}

export const getHealth = () => request<HealthResponse>("/health");

export const getAvailableDates = (city?: string) =>
  request<DateResponse>(`/available-dates${cityQs(city) ? `?${cityQs(city)}` : ""}`);

export const getMetadata = (city?: string) =>
  request<Record<string, unknown>>(`/metadata${cityQs(city) ? `?${cityQs(city)}` : ""}`);

export const getPM25 = (date: string, lat: number, lon: number, resolution = "500m", city?: string) => {
  const params = `date=${date}&lat=${lat}&lon=${lon}&resolution=${resolution}${city ? `&city=${encodeURIComponent(city)}` : ""}`;
  return request<PM25Response>(`/pm25?${params}`);
};

export const getAQI = (date: string, lat: number, lon: number, city?: string) => {
  const params = `date=${date}&lat=${lat}&lon=${lon}${city ? `&city=${encodeURIComponent(city)}` : ""}`;
  return request<AQIResponse>(`/aqi?${params}`);
};

export const getLocation = (date: string, lat: number, lon: number, city?: string) => {
  const params = `date=${date}&lat=${lat}&lon=${lon}${city ? `&city=${encodeURIComponent(city)}` : ""}`;
  return request<LocationResponse>(`/location?${params}`);
};

export const getHotspots = (date?: string, city?: string) => {
  const params: string[] = [];
  if (date) params.push(`date=${date}`);
  if (city) params.push(`city=${encodeURIComponent(city)}`);
  return request<HotspotCollection>(`/hotspots${params.length ? `?${params.join("&")}` : ""}`);
};

export const getHotspotStatistics = (city?: string) =>
  request<HotspotStatisticsResponse>(`/hotspots/statistics${cityQs(city) ? `?${cityQs(city)}` : ""}`);

export const getStations = (city?: string) =>
  request<Station[]>(`/stations${cityQs(city) ? `?${cityQs(city)}` : ""}`);

export const getStationDetail = (stationId: string, city?: string) => {
  const params = city ? `?city=${encodeURIComponent(city)}` : "";
  return request<StationDetail>(`/stations/${stationId}${params}`);
};

export const getFeatureImportance = (city?: string) =>
  request<FeatureImportanceResponse>(`/feature-importance${cityQs(city) ? `?${cityQs(city)}` : ""}`);

export const getUncertainty = (city?: string) =>
  request<UncertaintyResponse>(`/uncertainty${cityQs(city) ? `?${cityQs(city)}` : ""}`);

export const getPM25Grid = (date: string, resolution = "500m", city?: string) => {
  const params = `date=${date}&resolution=${resolution}${city ? `&city=${encodeURIComponent(city)}` : ""}`;
  return request<GridResponse>(`/pm25/grid?${params}`);
};

export const getPM25RasterUrl = (date: string, resolution = "500m", city?: string) => {
  const params = `date=${date}&resolution=${resolution}${city ? `&city=${encodeURIComponent(city)}` : ""}`;
  return `${API_BASE_URL}/raster/pm25?${params}`;
};

export const getAQIRasterUrl = (date: string, city?: string) => {
  const params = `date=${date}${city ? `&city=${encodeURIComponent(city)}` : ""}`;
  return `${API_BASE_URL}/raster/aqi?${params}`;
};

export const getAODRasterUrl = (date: string, city?: string) => {
  const params = `date=${date}${city ? `&city=${encodeURIComponent(city)}` : ""}`;
  return `${API_BASE_URL}/raster/aod?${params}`;
};

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

export const getAvailabilityRegistry = (scope: string = "global") =>
  request<AvailabilityRegistryResponse>(`/data/availability?scope=${encodeURIComponent(scope)}`);
