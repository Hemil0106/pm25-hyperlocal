import { AQI_LEGEND, PM25_LEGEND, AOD_LEGEND, binLabel } from "../colormaps";
import type { LayerVisibility } from "../types";

export function Legend({ visibility }: { visibility: LayerVisibility }) {
  const showAqi = visibility.aqi;
  const showPm25 = visibility.pm25 || visibility.pm25_1km;
  const showAod = visibility.aod;
  if (!showAqi && !showPm25 && !showAod) {
    return null;
  }
  return (
    <div className="legend" aria-label="Map legend">
      <h3 className="section-label">Legend</h3>
      {showPm25 && (
        <div className="legend-block">
          <div className="legend-title">Predicted PM2.5 (µg/m³)</div>
          {PM25_LEGEND.map((bin) => (
            <div className="legend-row" key={`${bin.min}-${bin.max}`}>
              <span className="legend-swatch" style={{ background: bin.color }} />
              <span>{binLabel(bin, "µg/m³")}</span>
            </div>
          ))}
        </div>
      )}
      {showAqi && (
        <div className="legend-block">
          <div className="legend-title">PM2.5-derived AQI / sub-index</div>
          {AQI_LEGEND.map((category) => (
            <div className="legend-row" key={category.name}>
              <span className="legend-swatch" style={{ background: category.color }} />
              <span>{category.name}</span>
            </div>
          ))}
        </div>
      )}
      {showAod && (
        <div className="legend-block">
          <div className="legend-title">Aerosol Optical Depth (AOD)</div>
          {AOD_LEGEND.map((bin) => (
            <div className="legend-row" key={`aod-${bin.min}-${bin.max}`}>
              <span className="legend-swatch" style={{ background: bin.color }} />
              <span>{bin.min.toFixed(1)} – {bin.max.toFixed(1)}</span>
            </div>
          ))}
        </div>
      )}
      {(visibility.hotspots || visibility.stations) && (
        <div className="legend-block">
          {visibility.hotspots && (
            <div className="legend-row">
              <span className="legend-swatch hotspot-swatch" />
              <span>Predicted high-pollution zone</span>
            </div>
          )}
          {visibility.stations && (
            <div className="legend-row">
              <span className="legend-swatch station-swatch" />
              <span>CPCB monitoring station</span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
