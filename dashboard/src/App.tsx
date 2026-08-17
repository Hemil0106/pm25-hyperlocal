import { useEffect, useState } from "react";
import {
  getAOIInfo,
  getAQIRasterUrl,
  getAvailableDates,
  getDataAvailability,
  getFeatureImportance,
  getGlobalDataStatus,
  getHealth,
  getHotspots,
  getHotspotStatistics,
  getLocation,
  getMetadata,
  getModelScopes,
  getOutputMetadata,
  getPM25RasterUrl,
  getRegions,
  getStationDetail,
  getStations,
  getUncertainty,
} from "./api";
import { Header } from "./components/Header";
import { MetricCards } from "./components/MetricCards";
import { LocationPanel } from "./components/LocationPanel";
import { ModelInfoPanel } from "./components/ModelInfoPanel";
import { DataSourcesPanel } from "./components/DataSourcesPanel";
import { LimitationsPanel } from "./components/LimitationsPanel";
import { GlobalStatusPanel } from "./components/GlobalStatusPanel";
import { Footer } from "./components/Footer";
import { LayerControl, HotspotSummary } from "./components/LayerControl";
import { Legend } from "./components/Legend";
import { ResolutionCompare } from "./components/ResolutionCompare";
import { Panel, ErrorBox } from "./components/Panel";
import { MapView } from "./map/MapView";
import { loadRasterLayer } from "./map/raster";
import type {
  AOIInfoResponse,
  DataAvailabilityResponse,
  FeatureImportanceResponse,
  GlobalDataStatusResponse,
  HotspotCollection,
  HotspotProperties,
  HotspotStatisticsResponse,
  LayerVisibility,
  LocationResponse,
  ModelScopesResponse,
  OutputMetadataResponse,
  RasterLayerData,
  RegionInfo,
  Station,
  StationDetail,
  UncertaintyResponse,
} from "./types";

const DEFAULT_VISIBILITY: LayerVisibility = {
  pm25: true,
  aqi: false,
  hotspots: true,
  stations: true,
  pm25_1km: false,
};

interface RasterState {
  pm25: RasterLayerData | null;
  pm25_1km: RasterLayerData | null;
  aqi: RasterLayerData | null;
}

function distanceKm(lat1: number, lon1: number, lat2: number, lon2: number): number {
  const toRad = (deg: number) => (deg * Math.PI) / 180;
  const earthKm = 6371;
  const dLat = toRad(lat2 - lat1);
  const dLon = toRad(lon2 - lon1);
  const a =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.sin(dLon / 2) ** 2;
  return 2 * earthKm * Math.asin(Math.sqrt(a));
}

/** Map an M15 region id to the M16 acquisition scope (global | india | delhi). */
function scopeForRegion(region: string): string {
  if (region === "india" || region === "global") return region;
  return "delhi";
}

/** Map region id to the backend city param (None for delhi, which is the default). */
function cityParam(region: string): string | undefined {
  if (region === "delhi") return undefined;
  return region;
}

export default function App() {
  const [apiStatus, setApiStatus] = useState<"loading" | "ok" | "unavailable">("loading");
  const [dates, setDates] = useState<string[]>([]);
  const [selectedDate, setSelectedDate] = useState("");
  const [metadata, setMetadata] = useState<Record<string, unknown> | null>(null);
  const [stations, setStations] = useState<Station[] | null>(null);
  const [hotspots, setHotspots] = useState<HotspotCollection | null>(null);
  const [statistics, setStatistics] = useState<HotspotStatisticsResponse | null>(null);
  const [featureImportance, setFeatureImportance] =
    useState<FeatureImportanceResponse | null>(null);
  const [uncertainty, setUncertainty] = useState<UncertaintyResponse | null>(null);

  const [regions, setRegions] = useState<Record<string, RegionInfo> | null>(null);
  const [selectedRegion, setSelectedRegion] = useState("delhi");
  const [scopes, setScopes] = useState<ModelScopesResponse | null>(null);
  const [availability, setAvailability] = useState<DataAvailabilityResponse | null>(null);
  const [outputMetadata, setOutputMetadata] = useState<OutputMetadataResponse | null>(null);
  const [aoiInfo, setAoiInfo] = useState<AOIInfoResponse | null>(null);
  const [globalStatusLoading, setGlobalStatusLoading] = useState(true);
  const [globalDataStatus, setGlobalDataStatus] = useState<GlobalDataStatusResponse | null>(null);
  const [globalDataLoading, setGlobalDataLoading] = useState(true);

  const [visibility, setVisibility] = useState<LayerVisibility>(DEFAULT_VISIBILITY);
  const [resolution, setResolution] = useState<"500m" | "1000m">("500m");
  const [pm25Opacity, setPm25Opacity] = useState(0.8);
  const [rasters, setRasters] = useState<RasterState>({
    pm25: null,
    pm25_1km: null,
    aqi: null,
  });
  const [rasterError, setRasterError] = useState<string | null>(null);

  const [selectedLocation, setSelectedLocation] = useState<{
    latitude: number;
    longitude: number;
  } | null>(null);
  const [location, setLocation] = useState<LocationResponse | null>(null);
  const [locationLoading, setLocationLoading] = useState(false);
  const [locationError, setLocationError] = useState<string | null>(null);

  const [selectedHotspot, setSelectedHotspot] = useState<HotspotProperties | null>(null);
  const [selectedStation, setSelectedStation] = useState<Station | null>(null);
  const [stationDetail, setStationDetail] = useState<StationDetail | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const health = await getHealth();
        if (!cancelled && health.status === "ok") setApiStatus("ok");
      } catch {
        if (cancelled) return;
      }
      try {
        const { dates: found } = await getAvailableDates(cityParam(selectedRegion));
        if (cancelled) return;
        setDates(found);
        if (found.length > 0) setSelectedDate(found[0]);
        setApiStatus("ok");
      } catch {
        if (!cancelled) setApiStatus("unavailable");
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [selectedRegion]);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      const [regionList, scopeList] = await Promise.allSettled([
        getRegions(),
        getModelScopes(),
      ]);
      if (cancelled) return;
      if (regionList.status === "fulfilled") setRegions(regionList.value.regions);
      if (scopeList.status === "fulfilled") setScopes(scopeList.value);
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    setGlobalStatusLoading(true);
    (async () => {
      const [availResult, metaResult, aoiResult] = await Promise.allSettled([
        getDataAvailability(selectedRegion, selectedDate || undefined),
        getOutputMetadata(selectedRegion, selectedDate || undefined),
        getAOIInfo(selectedRegion),
      ]);
      if (cancelled) return;
      if (availResult.status === "fulfilled") setAvailability(availResult.value);
      if (metaResult.status === "fulfilled") setOutputMetadata(metaResult.value);
      if (aoiResult.status === "fulfilled") setAoiInfo(aoiResult.value);
      setGlobalStatusLoading(false);
    })();
    return () => {
      cancelled = true;
    };
  }, [selectedRegion, selectedDate]);

  useEffect(() => {
    let cancelled = false;
    setGlobalDataLoading(true);
    setGlobalDataStatus(null);
    (async () => {
      const result = await getGlobalDataStatus(scopeForRegion(selectedRegion));
      if (cancelled) return;
      setGlobalDataStatus(result);
      setGlobalDataLoading(false);
    })();
    return () => {
      cancelled = true;
    };
  }, [selectedRegion]);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      const city = cityParam(selectedRegion);
      const [meta, stationList, importance, uncertaintyData] = await Promise.allSettled([
        getMetadata(city),
        getStations(city),
        getFeatureImportance(city),
        getUncertainty(city),
      ]);
      if (cancelled) return;
      if (meta.status === "fulfilled") setMetadata(meta.value);
      if (stationList.status === "fulfilled") setStations(stationList.value);
      if (importance.status === "fulfilled") setFeatureImportance(importance.value);
      if (uncertaintyData.status === "fulfilled") setUncertainty(uncertaintyData.value);
    })();
    return () => {
      cancelled = true;
    };
  }, [selectedRegion]);

  useEffect(() => {
    if (!selectedDate) return;
    let cancelled = false;
    (async () => {
      const city = cityParam(selectedRegion);
      const [hotspotData, statsData] = await Promise.allSettled([
        getHotspots(selectedDate, city),
        getHotspotStatistics(city),
      ]);
      if (cancelled) return;
      if (hotspotData.status === "fulfilled") setHotspots(hotspotData.value);
      if (statsData.status === "fulfilled") setStatistics(statsData.value);
    })();
    return () => {
      cancelled = true;
    };
  }, [selectedDate, selectedRegion]);

  useEffect(() => {
    if (!selectedDate) return;
    let cancelled = false;
    setRasterError(null);
    setRasters({ pm25: null, pm25_1km: null, aqi: null });
    (async () => {
      const city = cityParam(selectedRegion);
      const [pm25Result, pm25_1kmResult, aqiResult] = await Promise.allSettled([
        loadRasterLayer(getPM25RasterUrl(selectedDate, "500m", city), "pm25"),
        loadRasterLayer(getPM25RasterUrl(selectedDate, "1000m", city), "pm25"),
        loadRasterLayer(getAQIRasterUrl(selectedDate, city), "aqi"),
      ]);
      if (cancelled) return;
      setRasters({
        pm25: pm25Result.status === "fulfilled" ? pm25Result.value : null,
        pm25_1km: pm25_1kmResult.status === "fulfilled" ? pm25_1kmResult.value : null,
        aqi: aqiResult.status === "fulfilled" ? aqiResult.value : null,
      });
      if (
        pm25Result.status === "rejected" &&
        pm25_1kmResult.status === "rejected" &&
        aqiResult.status === "rejected"
      ) {
        setRasterError("Map layer unavailable for this date.");
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [selectedDate, selectedRegion]);

  async function handleMapClick(latitude: number, longitude: number) {
    if (!selectedDate) return;
    setSelectedLocation({ latitude, longitude });
    setLocationLoading(true);
    setLocationError(null);
    setSelectedHotspot(null);
    setSelectedStation(null);
    setStationDetail(null);
    try {
      const result = await getLocation(selectedDate, latitude, longitude, cityParam(selectedRegion));
      setLocation(result);
    } catch (error) {
      setLocation(null);
      const message = error instanceof Error ? error.message : "";
      setLocationError(
        message.includes("outside the study area")
          ? "Location outside the study area (AOI). Click inside the map to inspect a location."
          : message || "Unable to retrieve location information.",
      );
    } finally {
      setLocationLoading(false);
    }
  }

  function handleHotspotClick(properties: HotspotProperties) {
    setSelectedHotspot(properties);
  }

  async function handleStationClick(station: Station) {
    setSelectedStation(station);
    setLocation(null);
    setSelectedHotspot(null);
    try {
      const detail = await getStationDetail(station.station_id, cityParam(selectedRegion));
      setStationDetail(detail);
    } catch {
      setStationDetail(null);
    }
  }

  function handleLayerToggle(layer: keyof LayerVisibility) {
    setVisibility((current) => ({ ...current, [layer]: !current[layer] }));
  }

  function handleResolutionChange(next: "500m" | "1000m") {
    setResolution(next);
    setVisibility((current) =>
      next === "500m"
        ? { ...current, pm25: true, pm25_1km: false }
        : { ...current, pm25: false, pm25_1km: true },
    );
  }

  function handleDateChange(date: string) {
    setSelectedDate(date);
    setSelectedLocation(null);
    setLocation(null);
    setLocationError(null);
    setSelectedHotspot(null);
    setSelectedStation(null);
    setStationDetail(null);
  }

  function handleRegionChange(region: string) {
    setSelectedRegion(region);
    setSelectedDate("");
    setDates([]);
    setSelectedLocation(null);
    setLocation(null);
    setLocationError(null);
    setSelectedHotspot(null);
    setSelectedStation(null);
    setStationDetail(null);
    setRasters({ pm25: null, pm25_1km: null, aqi: null });
    setMetadata(null);
    setStations(null);
    setFeatureImportance(null);
    setUncertainty(null);
    setStatistics(null);
    setHotspots(null);
  }

  const nearbyStation: Station | null = (() => {
    if (!selectedLocation || !stations || stations.length === 0) return null;
    let nearest: Station | null = null;
    let nearestDistance = Number.POSITIVE_INFINITY;
    for (const station of stations) {
      const d = distanceKm(
        selectedLocation.latitude,
        selectedLocation.longitude,
        station.latitude,
        station.longitude,
      );
      if (d < nearestDistance) {
        nearestDistance = d;
        nearest = station;
      }
    }
    return nearestDistance <= 6 && nearest ? nearest : null;
  })();

  const metadataPm25 = metadata
    ? ((metadata as Record<string, unknown>)["PM2.5"] as {
        model?: string;
        dataset_mode?: string;
        AOD_used?: boolean;
      })
    : null;
  const model = metadataPm25?.model ?? null;
  const canPredict = outputMetadata?.inference?.can_predict ?? true;
  const hasRasterData = rasters.pm25 !== null || rasters.pm25_1km !== null || rasters.aqi !== null;
  const regionScopeStatus = aoiInfo?.model_scope?.status === "available" ? "available" : "unavailable";
  const regionBounds =
    aoiInfo?.bounds ??
    regions?.[selectedRegion]?.bounds ??
    (regions && "delhi" in regions ? regions.delhi.bounds : null);
  const nonDelhiRegion = selectedRegion !== "delhi" && regions != null;
  const mapLoading =
    !rasterError && rasters.pm25 === null && rasters.pm25_1km === null && rasters.aqi === null;

  return (
    <div className="app">
      <Header
        dates={dates}
        model={model ?? "XGBoost"}
        datasetMode={metadataPm25?.dataset_mode ?? "fallback"}
        aodUsed={metadataPm25?.AOD_used ?? false}
        apiStatus={apiStatus}
        selectedDate={selectedDate}
        onDateChange={handleDateChange}
        regions={regions}
        selectedRegion={selectedRegion}
        onRegionChange={handleRegionChange}
        regionScopeStatus={regionScopeStatus}
        dataStatus={globalDataStatus}
        dataLoading={globalDataLoading}
      />

      <MetricCards
        location={location}
        locationLoading={locationLoading}
        hasSelection={selectedLocation != null && selectedStation == null}
        hotspotCount={statistics?.hotspot_zone_count ?? null}
        resolution={resolution === "500m" ? "500 m" : "1 km"}
        model={model ?? "XGBoost"}
      />

      {!canPredict && !hasRasterData && outputMetadata !== null && (
        <div className="api-banner" role="alert">
          Prediction unavailable for the {outputMetadata.aoi.name} AOI — no validated
          model trained on observations for this scope exists. The Delhi prototype is
          not applied outside Delhi.
        </div>
      )}

      {apiStatus === "unavailable" && (
        <div className="api-banner" role="alert">
          Backend unavailable — start the FastAPI server to load live project data.
        </div>
      )}

      <main className="main-grid">
        <section className="map-col" aria-label="Interactive map">
          <div className="map-shell">
            <MapView
              rasters={rasters}
              visibility={visibility}
              hotspots={hotspots}
              stations={stations}
              bounds={rasters.pm25?.bounds ?? null}
              regionBounds={nonDelhiRegion ? regionBounds : null}
              selectedLocation={selectedLocation}
              pm25Opacity={pm25Opacity}
              onMapClick={handleMapClick}
              onHotspotClick={handleHotspotClick}
              onStationClick={handleStationClick}
            />
            {nonDelhiRegion && !canPredict && !hasRasterData && (
              <div className="map-error">
                <ErrorBox>
                  Global PM2.5 map unavailable for the {regions?.[selectedRegion]?.name ?? selectedRegion}{" "}
                  AOI on this date — no validated model exists for this scope.
                </ErrorBox>
              </div>
            )}
            {rasterError && (
              <div className="map-error">
                <ErrorBox>{rasterError}</ErrorBox>
              </div>
            )}
            {mapLoading && (
              <div className="map-loading" role="status">
                <span
                  className="skeleton"
                  style={{ width: 12, height: 12, borderRadius: 999, flex: "0 0 auto" }}
                />
                Loading PM2.5 layer…
              </div>
            )}
            <div className="map-floating-controls">
              <LayerControl
                visibility={visibility}
                onToggle={handleLayerToggle}
                pm25Opacity={pm25Opacity}
                onOpacityChange={setPm25Opacity}
              />
              <ResolutionCompare
                resolution={resolution}
                onResolutionChange={handleResolutionChange}
                pm25Loading={!rasters.pm25 && !rasterError}
              />
              <Legend visibility={visibility} />
            </div>
          </div>
        </section>

        <aside className="side-col">
          <LocationPanel
            location={location}
            loading={locationLoading}
            error={locationError}
            nearbyStation={nearbyStation}
            nearbyStationLoading={false}
          />

          {selectedHotspot && (
            <Panel title="Selected hotspot">
              <div className="kv-grid">
                <div>
                  <span className="loc-label">Hotspot ID</span>
                  <span className="loc-value">{selectedHotspot.hotspot_id}</span>
                </div>
                <div>
                  <span className="loc-label">Area</span>
                  <span className="loc-value">
                    {selectedHotspot.area_km2 != null
                      ? `${selectedHotspot.area_km2.toFixed(1)} km²`
                      : "—"}
                  </span>
                </div>
                <div>
                  <span className="loc-label">Date</span>
                  <span className="loc-value">{String(selectedHotspot.date).slice(0, 10)}</span>
                </div>
              </div>
              <p className="body-text muted">
                Predicted high-pollution zone — not a confirmed emission source.
              </p>
            </Panel>
          )}

          {selectedStation && (
            <Panel title="Selected station">
              <div className="kv-grid">
                <div>
                  <span className="loc-label">Station ID</span>
                  <span className="loc-value">{selectedStation.station_id}</span>
                </div>
                <div>
                  <span className="loc-label">Latitude</span>
                  <span className="loc-value">{selectedStation.latitude.toFixed(4)}</span>
                </div>
                <div>
                  <span className="loc-label">Longitude</span>
                  <span className="loc-value">{selectedStation.longitude.toFixed(4)}</span>
                </div>
                <div>
                  <span className="loc-label">Latest PM2.5</span>
                  <span className="loc-value">
                    {selectedStation.latest_pm25 != null
                      ? `${selectedStation.latest_pm25.toFixed(1)} µg/m³`
                      : "—"}
                  </span>
                </div>
                <div>
                  <span className="loc-label">Available dates</span>
                  <span className="loc-value">
                    {stationDetail ? stationDetail.available_dates.length : "…"}
                  </span>
                </div>
              </div>
            </Panel>
          )}

          <HotspotSummary stats={statistics} />
          <GlobalStatusPanel
            availability={availability}
            scopes={scopes}
            outputMetadata={outputMetadata}
            aoi={aoiInfo}
            loading={globalStatusLoading}
            globalDataStatus={globalDataStatus}
            globalDataLoading={globalDataLoading}
          />
          <ModelInfoPanel
            metadata={metadata}
            featureImportance={featureImportance}
            uncertainty={uncertainty}
          />
          <DataSourcesPanel
            globalDataStatus={globalDataStatus}
            loading={globalDataLoading}
          />
          <LimitationsPanel />
        </aside>
      </main>

      <Footer />
    </div>
  );
}
