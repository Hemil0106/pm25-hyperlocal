import { fromUrl } from "geotiff";
import proj4 from "proj4";
import { binColor, PM25_BINS, CPCB_CATEGORIES, AOD_BINS } from "../colormaps";
import type { MapQuadCoordinates, RasterLayerData } from "../types";

function hexToRgb(hex: string): [number, number, number] {
  const cleaned = hex.replace("#", "");
  const int = parseInt(cleaned, 16);
  return [(int >> 16) & 255, (int >> 8) & 255, int & 255];
}

/**
 * Colorize raster pixels into a canvas using CPCB-style classification bins.
 * NoData pixels are left transparent. Returns the canvas and the value range
 * of the valid pixels actually drawn.
 */
function colorizeCanvas(
  data: ArrayLike<number>,
  width: number,
  height: number,
  kind: "pm25" | "aqi" | "aod",
): { canvas: HTMLCanvasElement; valueRange: { min: number; max: number } | null } {
  const bins = kind === "pm25" ? PM25_BINS : kind === "aqi" ? CPCB_CATEGORIES : AOD_BINS;
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d")!;
  const image = context.createImageData(width, height);

  let min: number | null = null;
  let max: number | null = null;

  for (let i = 0; i < data.length && i < width * height; i++) {
    const value = Number(data[i]);
    const index = i * 4;
    const color = binColor(value, bins);
    if (color === null) {
      image.data[index + 3] = 0;
      continue;
    }
    const [r, g, b] = hexToRgb(color);
    image.data[index] = r;
    image.data[index + 1] = g;
    image.data[index + 2] = b;
    image.data[index + 3] = 200;
    if (min === null || value < min) min = value;
    if (max === null || value > max) max = value;
  }

  context.putImageData(image, 0, 0);
  return {
    canvas,
    valueRange: min === null || max === null ? null : { min, max },
  };
}

function projectedCrs(image: { geoKeys?: () => Record<string, number> }): string {
  if (typeof image.geoKeys === "function") {
    try {
      const keys = image.geoKeys();
      if (keys && keys.ProjectedCSTypeGeoKey) {
        return `EPSG:${keys.ProjectedCSTypeGeoKey}`;
      }
      if (keys && keys.GeographicTypeGeoKey) {
        return `EPSG:${keys.GeographicTypeGeoKey}`;
      }
    } catch {
      /* fall through */
    }
  }
  return "EPSG:32643";
}

/**
 * Fetch a GeoTIFF raster from the backend, decode it in the browser, colorize
 * it, and return a MapLibre-compatible image overlay (canvas + WGS84 corners).
 */
export async function loadRasterLayer(
  url: string,
  kind: "pm25" | "aqi" | "aod",
): Promise<RasterLayerData> {
  const tiff = await fromUrl(url);
  const image = await tiff.getImage();

  const width = image.getWidth();
  const height = image.getHeight();
  const origin = image.getOrigin();
  const resolution = image.getResolution();

  const west = origin[0];
  const north = origin[1];
  const east = west + width * resolution[0];
  const south = north - height * Math.abs(resolution[1]);

  const sourceCrs = projectedCrs(image);

  const corners = (
    [
      [west, north],
      [east, north],
      [east, south],
      [west, south],
    ] as [number, number][]
  ).map(([x, y]) => proj4(sourceCrs, "EPSG:4326", [x, y]).slice(0, 2) as [number, number]) as MapQuadCoordinates;

  const raster = await image.readRasters({ interleave: false });
  const data = raster[0] as ArrayLike<number>;

  const { canvas, valueRange } = colorizeCanvas(data, width, height, kind);

  return {
    canvas,
    coordinates: corners,
    bounds: {
      west: corners[0][0],
      south: corners[2][1],
      east: corners[1][0],
      north: corners[0][1],
    },
    valueRange,
  };
}
