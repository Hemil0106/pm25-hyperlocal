import { useEffect, useRef } from "react";
import maplibregl from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import {
  BASE_STYLE,
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
  selectedLocation: { latitude: number; longitude: number } | null;
  pm25Opacity: number;
  onMapClick: (latitude: number, longitude: number) => void;
  onHotspotClick: (properties: HotspotProperties) => void;
  onStationClick: (station: Station) => void;
}

export function MapView({
  rasters,
  visibility,
  hotspots,
  stations,
  bounds,
  regionBounds,
  selectedLocation,
  pm25Opacity,
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

  function whenReady(map: maplibregl.Map, callback: () => void) {
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
      style: BASE_STYLE,
      center: [77.2, 28.6],
      zoom: 10,
      attributionControl: { compact: true },
    });
    map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-left");
    mapRef.current = map;

    map.on("click", (event) => {
      const point = event.point;
      const stationFeature = map.queryRenderedFeatures(point, {
        layers: ["stations-circle"],
      })[0];
      if (stationFeature && stationFeature.properties) {
        const props = stationFeature.properties as unknown as Station;
        onStationClickRef.current(props);
        return;
      }
      const hotspotFeature = map.queryRenderedFeatures(point, {
        layers: ["hotspots-fill", "hotspots-outline"],
      })[0];
      onMapClickRef.current(event.lngLat.lat, event.lngLat.lng);
      if (hotspotFeature && hotspotFeature.properties) {
        onHotspotClickRef.current(hotspotFeature.properties as unknown as HotspotProperties);
      }
    });

    map.on("mousemove", (event) => {
      const features = map.queryRenderedFeatures(event.point, {
        layers: ["hotspots-fill", "hotspots-outline", "stations-circle"],
      });
      map.getCanvas().style.cursor = features.length ? "pointer" : "";
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
      addImageLayer(map, "aqi", { data: rasters.aqi!, opacity: 0.8 }),
    );
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [rasters.aqi]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    whenReady(map, () => {
      setLayerVisible(map, "pm25", visibility.pm25);
      setLayerVisible(map, "pm25_1km", visibility.pm25_1km);
      setLayerVisible(map, "aqi", visibility.aqi);
      setLayerVisible(map, "hotspots", visibility.hotspots);
      setLayerVisible(map, "stations", visibility.stations);
    });
  }, [visibility]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !map.getLayer("pm25")) return;
    map.setPaintProperty("pm25", "raster-opacity", pm25Opacity);
  }, [pm25Opacity]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    whenReady(map, () => {
      if (hotspots) {
        addHotspotLayer(map, "hotspots", hotspots);
        setLayerVisible(map, "hotspots", visibility.hotspots);
      } else if (map.getSource("hotspots")) {
        (map.getSource("hotspots") as maplibregl.GeoJSONSource).setData({
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
        setLayerVisible(map, "stations", visibility.stations);
      } else if (map.getSource("stations")) {
        (map.getSource("stations") as maplibregl.GeoJSONSource).setData({
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
        color: "#0B3D91",
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
      markerRef.current.getElement().style.display = "none";
    }
  }, [selectedLocation]);

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

  return <div ref={containerRef} className="map-container" aria-label="GIS map" />;
}
