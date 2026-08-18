import type { LocationResponse } from "../types";

function aqiCategoryColor(category: string | null): string {
  const map: Record<string, string> = {
    GOOD: "#5BBE48",
    SATISFACTORY: "#FFFF01",
    MODERATELY_POLLUTED: "#FE7E01",
    POOR: "#F00101",
    VERY_POOR: "#8F3F97",
    SEVERE: "#7E0023",
  };
  return category ? map[category] ?? "#888" : "#888";
}

interface MetricCardsProps {
  location: LocationResponse | null;
  locationLoading: boolean;
  hasSelection: boolean;
  hotspotCount: number | null;
  resolution: string;
  model: string | null;
}

export function MetricCards({
  location,
  locationLoading,
  hasSelection,
  hotspotCount,
  resolution,
  model,
}: MetricCardsProps) {
  const aqiCategory = hasSelection ? location?.aqi_category ?? null : null;
  return (
    <div className="metrics">
      <div className={`metric-card ${hasSelection ? "has-selection" : ""}`}>
        <div className="metric-head">
          <span className="metric-label">Predicted PM2.5</span>
        </div>
        <span className="metric-value">
          {locationLoading
            ? "…"
            : location?.pm25 != null
              ? `${location.pm25.toFixed(1)} µg/m³`
              : hasSelection
                ? "NoData"
                : "Select a location"}
        </span>
        <span className="metric-sub">
          {locationLoading
            ? "reading location…"
            : location?.pm25 != null
              ? "model-derived estimate"
              : hasSelection
                ? "valid location, no prediction"
                : "click the map to inspect"}
        </span>
      </div>

      <div className={`metric-card ${hasSelection ? "has-selection" : ""}`}>
        <div className="metric-head">
          <span className="metric-label">PM2.5-derived AQI</span>
          <span className="metric-unit">sub-index</span>
        </div>
        <span className="metric-value">
          {locationLoading
            ? "…"
            : location?.pm25_derived_aqi != null
              ? `${location.pm25_derived_aqi}`
              : hasSelection
                ? "Unavailable"
                : "Select a location"}
        </span>
        <span
          className="metric-sub"
          style={{ color: aqiCategoryColor(aqiCategory) }}
        >
          {hasSelection ? (location?.aqi_category ?? "Unavailable") : "—"}
        </span>
      </div>

      <div className="metric-card">
        <div className="metric-head">
          <span className="metric-label">High-pollution zones</span>
          <span className="metric-unit">predicted</span>
        </div>
        <span className="metric-value">{hotspotCount ?? "—"}</span>
        <span className="metric-sub">not confirmed sources</span>
      </div>

      <div className="metric-card">
        <div className="metric-head">
          <span className="metric-label">Map resolution</span>
          <span className="metric-unit">spatial output</span>
        </div>
        <span className="metric-value">{resolution}</span>
        <span className="metric-sub">detail ≠ accuracy</span>
      </div>

      <div className="metric-card">
        <div className="metric-head">
          <span className="metric-label">Model</span>
          <span className="metric-unit">primary</span>
        </div>
        <span className="metric-value">{model ?? "—"}</span>
        <span className="metric-sub">no accuracy claim</span>
      </div>
    </div>
  );
}
