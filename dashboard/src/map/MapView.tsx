import { useEffect, useRef } from "react";
import maplibregl from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import {
  DARK_STYLE,
  LIGHT_STYLE,
  addHotspotLayer,
  addImageLayer,
  addStationsLayer,
  setLayerVisible,
} from "./layers";
import type {
  HotspotCollection,
  HotspotProperties,
  LayerVisibility,
  RasterLayerData,
  Station,
} from "../types";

interface RasterInput {
  pm25: RasterLayerData | null;
  pm25_1km: RasterLayerData | null;
  aqi: RasterLayerData | null;
  aod: RasterLayerData | null;
}

interface Bounds {
  west: number;
  south: number;
  east: number;
  north: number;
}

interface MapViewProps {
  rasters: RasterInput;
  visibility: LayerVisibility;
  hotspots: HotspotCollection | null;
  stations: Station[] | null;
  bounds: Bounds | null;
  regionBounds: Bounds | null;
  selectedRegion: string;
  selectedLocation: { latitude: number; longitude: number } | null;
  pm25Opacity: number;
  mapTheme: "dark" | "light";
  onMapClick: (latitude: number, longitude: number) => void;
  onHotspotClick: (properties: HotspotProperties) => void;
  onStationClick: (station: Station) => void;
}

const REGION_CENTERS: Record<
  string,
  { center: [number, number]; zoom: number }
> = {
  delhi: { center: [77.2, 28.6], zoom: 11 },
  pune: { center: [73.85, 18.55], zoom: 11 },
  mumbai: { center: [72.85, 19.05], zoom: 11 },
};

function createSelectionMarker(): HTMLElement {
  const el = document.createElement("div");
  el.style.cssText = `
    width: 24px;
    height: 24px;
    position: relative;
    cursor: pointer;
  `;
  el.innerHTML = `
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
      <circle cx="12" cy="12" r="10" stroke="#3da5f4" stroke-width="2" fill="rgba(61,165,244,0.15)"/>
      <circle cx="12" cy="12" r="5" fill="#3da5f4"/>
      <circle cx="12" cy="12" r="10" stroke="#3da5f4" stroke-width="1" opacity="0.4">
        <animate attributeName="r" from="10" to="16" dur="1.5s" repeatCount="indefinite"/>
        <animate attributeName="opacity" from="0.5" to="0" dur="1.5s" repeatCount="indefinite"/>
      </circle>
      <line x1="12" y1="0" x2="12" y2="6" stroke="#3da5f4" stroke-width="1.5"/>
      <line x1="12" y1="18" x2="12" y2="24" stroke="#3da5f4" stroke-width="1.5"/>
      <line x1="0" y1="12" x2="6" y2="12" stroke="#3da5f4" stroke-width="1.5"/>
      <line x1="18" y1="12" x2="24" y2="12" stroke="#3da5f4" stroke-width="1.5"/>
    </svg>
  `;
  return el;
}

export function MapView({
  rasters,
  visibility,
  hotspots,
  stations,
  bounds,
  regionBounds,
  selectedRegion,
  selectedLocation,
  pm25Opacity,
  mapTheme,
  onMapClick,
  onHotspotClick,
  onStationClick,
}: MapViewProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<maplibregl.Map | null>(null);
  const markerRef = useRef<maplibregl.Marker | null>(null);
  const fittedBoundsRef = useRef<string | null>(null);
  const onMapClickRef = useRef(onMapClick);
  const onHotspotClickRef = useRef(onHotspotClick);
  const onStationClickRef = useRef(onStationClick);

  useEffect(() => {
    onMapClickRef.current = onMapClick;
    onHotspotClickRef.current = onHotspotClick;
    onStationClickRef.current = onStationClick;
  });

  function whenReady(
    map: maplibregl.Map,
    callback: () => void,
  ) {
    if (map.isStyleLoaded()) {
      callback();
    } else {
      map.once("load", callback);
    }
  }

  useEffect(() => {
    if (!containerRef.current || mapRef.current) return;
    const map = new maplibregl.Map({
      container: containerRef.current,
      style: DARK_STYLE,
      center: [77.2, 28.6],
      zoom: 10,
      attributionControl: { compact: true },
    });
    map.addControl(
      new maplibregl.NavigationControl({ showCompass: false }),
      "top-left",
    );
    mapRef.current = map;

    map.on("click", (event) => {
      const point = event.point;
      const stationFeature = map.queryRenderedFeatures(point, {
        layers: ["stations-circle"],
      })[0];
      if (stationFeature && stationFeature.properties) {
        const props =
          stationFeature.properties as unknown as Station;
        onStationClickRef.current(props);
        return;
      }
      const hotspotFeature = map.queryRenderedFeatures(point, {
        layers: ["hotspots-fill", "hotspots-outline"],
      })[0];
      onMapClickRef.current(
        event.lngLat.lat,
        event.lngLat.lng,
      );
      if (
        hotspotFeature &&
        hotspotFeature.properties
      ) {
        onHotspotClickRef.current(
          hotspotFeature.properties as unknown as HotspotProperties,
        );
      }
    });

    map.on("mousemove", (event) => {
      const features = map.queryRenderedFeatures(
        event.point,
        {
          layers: [
            "hotspots-fill",
            "hotspots-outline",
            "stations-circle",
          ],
        },
      );
      map.getCanvas().style.cursor = features.length
        ? "pointer"
        : "";
    });

    return () => {
      map.remove();
      mapRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !rasters.pm25) return;
    whenReady(map, () =>
      addImageLayer(map, "pm25", {
        data: rasters.pm25!,
        opacity: pm25Opacity,
      }),
    );
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [rasters.pm25]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !rasters.pm25_1km) return;
    whenReady(map, () =>
      addImageLayer(map, "pm25_1km", {
        data: rasters.pm25_1km!,
        opacity: pm25Opacity,
      }),
    );
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [rasters.pm25_1km]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !rasters.aqi) return;
    whenReady(map, () =>
      addImageLayer(map, "aqi", {
        data: rasters.aqi!,
        opacity: 0.8,
      }),
    );
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [rasters.aqi]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !rasters.aod) return;
    whenReady(map, () =>
      addImageLayer(map, "aod", {
        data: rasters.aod!,
        opacity: 0.75,
      }),
    );
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [rasters.aod]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    whenReady(map, () => {
      setLayerVisible(map, "pm25", visibility.pm25);
      setLayerVisible(
        map,
        "pm25_1km",
        visibility.pm25_1km,
      );
      setLayerVisible(map, "aqi", visibility.aqi);
      setLayerVisible(map, "aod", visibility.aod);
      setLayerVisible(
        map,
        "hotspots",
        visibility.hotspots,
      );
      setLayerVisible(
        map,
        "stations",
        visibility.stations,
      );
    });
  }, [visibility]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !map.getLayer("pm25")) return;
    map.setPaintProperty(
      "pm25",
      "raster-opacity",
      pm25Opacity,
    );
  }, [pm25Opacity]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    whenReady(map, () => {
      if (hotspots) {
        addHotspotLayer(map, "hotspots", hotspots);
        setLayerVisible(
          map,
          "hotspots",
          visibility.hotspots,
        );
      } else if (map.getSource("hotspots")) {
        (
          map.getSource("hotspots") as maplibregl.GeoJSONSource
        ).setData({
          type: "FeatureCollection",
          features: [],
        });
      }
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [hotspots]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    whenReady(map, () => {
      if (stations) {
        addStationsLayer(map, "stations", stations);
        setLayerVisible(
          map,
          "stations",
          visibility.stations,
        );
      } else if (map.getSource("stations")) {
        (
          map.getSource("stations") as maplibregl.GeoJSONSource
        ).setData({
          type: "FeatureCollection",
          features: [],
        });
      }
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [stations]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    if (!markerRef.current) {
      markerRef.current = new maplibregl.Marker({
        element: createSelectionMarker(),
        anchor: "center",
      })
        .setLngLat([77.2, 28.6])
        .addTo(map);
    }
    if (selectedLocation) {
      markerRef.current.setLngLat([
        selectedLocation.longitude,
        selectedLocation.latitude,
      ]);
      markerRef.current.getElement().style.display = "";
    } else {
      markerRef.current.getElement().style.display =
        "none";
    }
  }, [selectedLocation]);

  const prevRegionRef = useRef(selectedRegion);
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    const prev = prevRegionRef.current;
    prevRegionRef.current = selectedRegion;
    if (prev === selectedRegion) return;
    const target =
      REGION_CENTERS[selectedRegion] ??
      REGION_CENTERS.delhi;
    map.flyTo({
      center: target.center,
      zoom: target.zoom,
      duration: 800,
      curve: 1.5,
    });
  }, [selectedRegion]);

  const rastersRef = useRef(rasters);
  const visibilityRef = useRef(visibility);
  const hotspotsRef = useRef(hotspots);
  const stationsRef = useRef(stations);
  useEffect(() => { rastersRef.current = rasters; });
  useEffect(() => { visibilityRef.current = visibility; });
  useEffect(() => { hotspotsRef.current = hotspots; });
  useEffect(() => { stationsRef.current = stations; });

  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    const target = mapTheme === "light" ? LIGHT_STYLE : DARK_STYLE;
    map.setStyle(target);
    map.once("style.load", () => {
      const cur = rastersRef.current;
      if (cur.pm25) addImageLayer(map, "pm25", { data: cur.pm25, opacity: 0.8 });
      if (cur.pm25_1km) addImageLayer(map, "pm25_1km", { data: cur.pm25_1km, opacity: 0.8 });
      if (cur.aqi) addImageLayer(map, "aqi", { data: cur.aqi, opacity: 0.8 });
      if (cur.aod) addImageLayer(map, "aod", { data: cur.aod, opacity: 0.75 });
      if (hotspotsRef.current) addHotspotLayer(map, "hotspots", hotspotsRef.current);
      if (stationsRef.current) addStationsLayer(map, "stations", stationsRef.current);
      const vis = visibilityRef.current;
      setLayerVisible(map, "pm25", vis.pm25);
      setLayerVisible(map, "pm25_1km", vis.pm25_1km);
      setLayerVisible(map, "aqi", vis.aqi);
      setLayerVisible(map, "aod", vis.aod);
      setLayerVisible(map, "hotspots", vis.hotspots);
      setLayerVisible(map, "stations", vis.stations);
    });
  }, [mapTheme]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !bounds) return;
    const key = JSON.stringify(bounds);
    if (fittedBoundsRef.current === key) return;
    fittedBoundsRef.current = key;
    const fit = () => {
      map.fitBounds(
        [
          [bounds.west, bounds.south],
          [bounds.east, bounds.north],
        ],
        { padding: 40, duration: 0 },
      );
    };
    if (map.loaded()) fit();
    else map.on("load", fit);
  }, [bounds]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !regionBounds || bounds) return;
    const key = `region:${JSON.stringify(regionBounds)}`;
    if (fittedBoundsRef.current === key) return;
    fittedBoundsRef.current = key;
    const fit = () => {
      map.fitBounds(
        [
          [regionBounds.west, regionBounds.south],
          [regionBounds.east, regionBounds.north],
        ],
        { padding: 40, duration: 0 },
      );
    };
    if (map.loaded()) fit();
    else map.on("load", fit);
  }, [regionBounds, bounds]);

  return (
    <div
      ref={containerRef}
      className="map-container"
      aria-label="GIS map"
    />
  );
}
