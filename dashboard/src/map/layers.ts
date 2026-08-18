import type { RasterLayerData, Station } from "../types";
import type { GeoJSONSource, Map } from "maplibre-gl";

/** Dark basemap (CARTO, no API key required). */
export const DARK_STYLE: maplibregl.StyleSpecification = {
  version: 8,
  sources: {
    basemap: {
      type: "raster",
      tiles: [
        "https://a.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
        "https://b.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
      ],
      tileSize: 256,
      attribution: "© OpenStreetMap contributors © CARTO",
    },
  },
  layers: [
    {
      id: "basemap",
      type: "raster",
      source: "basemap",
    },
  ],
};

/** Light basemap (CARTO, no API key required). */
export const LIGHT_STYLE: maplibregl.StyleSpecification = {
  version: 8,
  sources: {
    basemap: {
      type: "raster",
      tiles: [
        "https://a.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png",
        "https://b.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png",
      ],
      tileSize: 256,
      attribution: "© OpenStreetMap contributors © CARTO",
    },
  },
  layers: [
    {
      id: "basemap",
      type: "raster",
      source: "basemap",
    },
  ],
};

export const BASE_STYLE = DARK_STYLE;

export interface ImageLayerOptions {
  id: string;
  data: RasterLayerData;
  opacity: number;
}

/** Add (or refresh) a colorized raster as an image source + layer. */
export function addImageLayer(
  map: Map,
  id: string,
  { data, opacity }: Omit<ImageLayerOptions, "id">,
): void {
  const url = data.canvas.toDataURL("image/png");
  const existing = map.getSource(id);
  if (existing && existing.type === "image") {
    (existing as maplibregl.ImageSource).updateImage({
      url,
      coordinates: data.coordinates,
    });
  } else {
    map.addSource(id, {
      type: "image",
      url,
      coordinates: data.coordinates,
    });
  }
  if (!map.getLayer(id)) {
    map.addLayer({
      id,
      type: "raster",
      source: id,
      paint: { "raster-opacity": opacity, "raster-fade-duration": 0 },
    });
  }
}

/** Add hotspots as a filled GeoJSON layer with glow effect. */
export function addHotspotLayer(
  map: Map,
  id: string,
  geojson: unknown,
): void {
  if (map.getSource(id)) {
    (map.getSource(id) as GeoJSONSource).setData(geojson as never);
  } else {
    map.addSource(id, {
      type: "geojson",
      data: geojson as never,
    });
  }
  if (!map.getLayer(`${id}-fill`)) {
    map.addLayer({
      id: `${id}-fill`,
      type: "fill",
      source: id,
      paint: {
        "fill-color": "#a855f7",
        "fill-opacity": 0.35,
      },
    });
    map.addLayer({
      id: `${id}-glow`,
      type: "fill",
      source: id,
      paint: {
        "fill-color": "#c084fc",
        "fill-opacity": 0.12,
      },
    });
    map.addLayer({
      id: `${id}-outline`,
      type: "line",
      source: id,
      paint: {
        "line-color": "#c084fc",
        "line-width": 1.5,
        "line-opacity": 0.7,
      },
    });
  }
}

/** Add CPCB stations as a circle + label layer with radar pulse. */
export function addStationsLayer(
  map: Map,
  id: string,
  stations: Station[],
): void {
  const geojson = {
    type: "FeatureCollection",
    features: stations.map((station) => ({
      type: "Feature",
      properties: { ...station },
      geometry: {
        type: "Point",
        coordinates: [station.longitude, station.latitude],
      },
    })),
  };
  if (map.getSource(id)) {
    (map.getSource(id) as GeoJSONSource).setData(geojson as never);
  } else {
    map.addSource(id, { type: "geojson", data: geojson as never });
  }
  if (!map.getLayer(`${id}-pulse`)) {
    map.addLayer({
      id: `${id}-pulse`,
      type: "circle",
      source: id,
      paint: {
        "circle-radius": 12,
        "circle-color": "#3da5f4",
        "circle-opacity": 0.15,
        "circle-stroke-width": 0,
      },
    });
    map.addLayer({
      id: `${id}-circle`,
      type: "circle",
      source: id,
      paint: {
        "circle-radius": 6,
        "circle-color": "#3da5f4",
        "circle-stroke-color": "#ffffff",
        "circle-stroke-width": 2,
      },
    });
    map.addLayer({
      id: `${id}-label`,
      type: "symbol",
      source: id,
      layout: {
        "text-field": ["get", "station_id"],
        "text-size": 10,
        "text-offset": [0, 1.4],
        "text-anchor": "top",
      },
      paint: {
        "text-color": "#93c5fd",
        "text-halo-color": "rgba(0, 0, 0, 0.7)",
        "text-halo-width": 1.5,
      },
    });
  }
}

/** Set a layer's visibility by id. */
export function setLayerVisible(
  map: Map,
  id: string,
  visible: boolean,
): void {
  const layerIds = map
    .getStyle()
    .layers.filter(
      (layer) => (layer as { source?: string }).source === id,
    )
    .map((layer) => layer.id);
  for (const layerId of layerIds) {
    map.setLayoutProperty(
      layerId,
      "visibility",
      visible ? "visible" : "none",
    );
  }
}
