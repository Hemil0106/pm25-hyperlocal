import { useState } from "react";
import { Panel } from "./Panel";

const PIPELINE_STEPS = [
  {
    id: "1",
    title: "Data Acquisition",
    desc: "Satellite imagery (MODIS/MAIAC AOD, NDVI, VIIRS night lights), ground station observations (CPCB PM2.5), weather reanalysis (ERA5-Land), topography (SRTM DEM), and road networks (OpenStreetMap) are acquired for the study region.",
  },
  {
    id: "2",
    title: "Spatial Alignment",
    desc: "All raster and vector datasets are resampled and reprojected to a common 500 m UTM grid (EPSG:32643). Station point observations are extracted at the coarse 1 km parent-grid cells.",
  },
  {
    id: "3",
    title: "ML Model Training",
    desc: "An XGBoost regression model is trained on aligned station-level PM2.5 observations using satellite-derived, meteorological, and land-use features. A Random Forest baseline is trained for comparison.",
  },
  {
    id: "4",
    title: "Spatial Downscaling",
    desc: "A residual-based downscaling method produces 500 m hyperlocal PM2.5 maps from the 1 km coarse model predictions. High-resolution covariates (NDVI, DEM, road density, night lights) refine the spatial detail.",
  },
  {
    id: "5",
    title: "AQI & Hotspot Analysis",
    desc: "PM2.5 concentrations are converted to CPCB National AQI sub-index values. High-pollution hotspot zones are detected using a threshold-based approach on AQI categories.",
  },
  {
    id: "6",
    title: "Visualization",
    desc: "The interactive map renders 500 m PM2.5, AQI, and AOD layers with station markers and hotspot overlays. A location inspector provides per-pixel estimates with model uncertainty and feature attributions.",
  },
];

const DATA_SOURCES = [
  { name: "MODIS/MAIAC MCD19A2 V061", detail: "AOD — 1 km daily" },
  { name: "MODIS Terra MOD13Q1 V6.1", detail: "NDVI — 250 m 16-day" },
  { name: "VIIRS VNP46A2 V2.0", detail: "Night lights — 500 m daily" },
  { name: "CPCB Ground Stations", detail: "PM2.5 — hourly/daily" },
  { name: "ERA5-Land (ECMWF)", detail: "Meteorology — hourly" },
  { name: "SRTM GL1 V003 (NASA)", detail: "Elevation — 30 m" },
  { name: "OpenStreetMap", detail: "Road networks — vector" },
];

export function MethodologyPanel() {
  const [open, setOpen] = useState(false);
  return (
    <Panel
      title="Methodology"
      right={
        <button
          type="button"
          className="toggle-button"
          onClick={() => setOpen((v) => !v)}
          aria-expanded={open}
        >
          {open ? "Hide" : "Show details"}
        </button>
      }
    >
      <p className="body-text muted" style={{ marginBottom: open ? 10 : 0 }}>
        End-to-end ML pipeline: satellite + ground data → trained XGBoost → residual downscaling → 500 m PM2.5/AQI maps.
      </p>
      {open && (
        <div className="methodology-content">
          <h4 className="section-label" style={{ marginTop: 8 }}>Pipeline stages</h4>
          <ol className="pipeline-list">
            {PIPELINE_STEPS.map((step) => (
              <li key={step.id} className="pipeline-step">
                <div className="pipeline-step-head">
                  <span className="pipeline-step-num">{step.id}</span>
                  <span className="pipeline-step-title">{step.title}</span>
                </div>
                <p className="pipeline-step-desc">{step.desc}</p>
              </li>
            ))}
          </ol>

          <h4 className="section-label" style={{ marginTop: 12 }}>Data sources</h4>
          <ul className="data-source-list">
            {DATA_SOURCES.map((src) => (
              <li key={src.name} className="data-source-item">
                <span className="data-source-name">{src.name}</span>
                <span className="data-source-detail">{src.detail}</span>
              </li>
            ))}
          </ul>

          <p className="disclaimer-note" style={{ marginTop: 12 }}>
            This is an SIH prototype. All rasters are pre-generated; no live inference runs during API requests. Model performance is illustrative, not operationally validated.
          </p>
        </div>
      )}
    </Panel>
  );
}
