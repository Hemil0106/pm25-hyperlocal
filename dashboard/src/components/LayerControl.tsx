import type { LayerVisibility, HotspotStatisticsResponse } from "../types";

interface LayerControlProps {
  visibility: LayerVisibility;
  onToggle: (layer: keyof LayerVisibility) => void;
  pm25Opacity: number;
  onOpacityChange: (value: number) => void;
}

const LAYER_ITEMS: { id: keyof LayerVisibility; label: string }[] = [
  { id: "pm25", label: "PM2.5 — 500 m" },
  { id: "pm25_1km", label: "PM2.5 — 1 km comparison" },
  { id: "aqi", label: "PM2.5-derived AQI" },
  { id: "aod", label: "MODIS / MAIAC AOD" },
  { id: "hotspots", label: "High-pollution zones" },
  { id: "stations", label: "CPCB stations" },
];

export function LayerControl({
  visibility,
  onToggle,
  pm25Opacity,
  onOpacityChange,
}: LayerControlProps) {
  return (
    <div className="layer-control" aria-label="Map layers">
      <h3 className="section-label">Layers</h3>
      {LAYER_ITEMS.map((item) => (
        <label key={item.id} className="layer-item">
          <input
            type="checkbox"
            checked={visibility[item.id]}
            onChange={() => onToggle(item.id)}
            aria-label={item.label}
          />
          <span>{item.label}</span>
        </label>
      ))}
      <div className="opacity-row">
        <label htmlFor="pm25-opacity">PM2.5 opacity</label>
        <input
          id="pm25-opacity"
          type="range"
          min={0.2}
          max={1}
          step={0.1}
          value={pm25Opacity}
          onChange={(event) => onOpacityChange(Number(event.target.value))}
        />
      </div>
    </div>
  );
}

export function HotspotSummary({ stats }: { stats: HotspotStatisticsResponse | null }) {
  if (!stats || stats.hotspot_zone_count == null) {
    return null;
  }
  return (
    <div className="hotspot-summary" aria-label="Hotspot statistics">
      <div className="panel-header">
        <h3 className="panel-title">Predicted high-pollution zones</h3>
      </div>
      <div className="hotspot-summary-grid">
        <div className="hs-cell">
          <span className="hs-value tech-value">{stats.hotspot_zone_count}</span>
          <span className="hs-label">zones</span>
        </div>
        <div className="hs-cell">
          <span className="hs-value tech-value">
            {stats.hotspot_area_km2 != null ? `${Math.round(stats.hotspot_area_km2)}` : "—"}
          </span>
          <span className="hs-label">km² total area</span>
        </div>
        <div className="hs-cell">
          <span className="hs-value tech-value">
            {stats.mean_pm25_ug_m3 != null ? stats.mean_pm25_ug_m3.toFixed(1) : "—"}
          </span>
          <span className="hs-label">mean PM2.5 (µg/m³)</span>
        </div>
        <div className="hs-cell">
          <span className="hs-value tech-value">
            {stats.max_pm25_ug_m3 != null ? stats.max_pm25_ug_m3.toFixed(1) : "—"}
          </span>
          <span className="hs-label">max PM2.5 (µg/m³)</span>
        </div>
        <div className="hs-cell">
          <span className="hs-value tech-value">
            {stats.mean_aqi != null ? stats.mean_aqi.toFixed(1) : "—"}
          </span>
          <span className="hs-label">mean AQI</span>
        </div>
        <div className="hs-cell">
          <span className="hs-value tech-value">
            {stats.max_aqi != null ? stats.max_aqi.toFixed(1) : "—"}
          </span>
          <span className="hs-label">max AQI</span>
        </div>
      </div>
      <p className="disclaimer-note">
        Not confirmed emission sources — threshold-based predicted zones.
      </p>
    </div>
  );
}
