/** Official CPCB AQI category colors (Milestone 11). */
export const CPCB_CATEGORIES: {
  name: string;
  min: number;
  max: number;
  color: string;
}[] = [
  { name: "GOOD", min: 0, max: 50, color: "#5BBE48" },
  { name: "SATISFACTORY", min: 51, max: 100, color: "#FFFF01" },
  { name: "MODERATELY_POLLUTED", min: 101, max: 200, color: "#FE7E01" },
  { name: "POOR", min: 201, max: 300, color: "#F00101" },
  { name: "VERY_POOR", min: 301, max: 400, color: "#8F3F97" },
  { name: "SEVERE", min: 401, max: 500, color: "#7E0023" },
];

/** CPCB PM2.5 24-hour concentration bins for the PM2.5 layer colouring. */
export const PM25_BINS: { min: number; max: number; color: string }[] = [
  { min: 0, max: 30, color: "#5BBE48" },
  { min: 31, max: 60, color: "#FFFF01" },
  { min: 61, max: 90, color: "#FE7E01" },
  { min: 91, max: 120, color: "#F00101" },
  { min: 121, max: 250, color: "#8F3F97" },
  { min: 250, max: 350, color: "#7E0023" },
];

export const AQI_LEGEND = CPCB_CATEGORIES;
export const PM25_LEGEND = PM25_BINS;

export interface Bin {
  min: number;
  max: number;
  color: string;
}

/** Return the bin color for a value, or null when below/above the binning range. */
export function binColor(value: number, bins: Bin[]): string | null {
  for (const bin of bins) {
    if (value >= bin.min && value <= bin.max) {
      return bin.color;
    }
  }
  return null;
}

/** Human readable label for a bin, e.g. "31 – 60 µg/m³". */
export function binLabel(bin: Bin, unit: string): string {
  return `${bin.min} – ${bin.max} ${unit}`;
}
