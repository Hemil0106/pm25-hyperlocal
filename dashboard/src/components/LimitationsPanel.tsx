import { useState } from "react";
import { Panel } from "./Panel";

const LIMITATIONS = [
  "Delhi uses a trained XGBoost model; Pune/Mumbai use pre-generated PM2.5 rasters",
  "ML model trained on 6 Delhi CPCB stations only — 36 stations across 3 cities, but no cross-city model exists",
  "500 m maps are pre-generated spatial refinements, not from a trained downscaling model",
  "Uncertainty estimation deferred",
  "Hotspots represent predicted high-pollution zones — not confirmed emission sources",
  "PM2.5-derived AQI is not the full multi-pollutant National AQI",
];

export function LimitationsPanel() {
  const [open, setOpen] = useState(false);
  return (
    <Panel
      title="Prototype & Scientific Limitations"
      right={
        <button
          type="button"
          className="toggle-button"
          onClick={() => setOpen((value) => !value)}
          aria-expanded={open}
        >
          {open ? "Hide limitations" : "Show limitations"}
        </button>
      }
    >
      {open && (
        <ul className="limitation-list">
          {LIMITATIONS.map((item) => (
            <li key={item}>{item}</li>
          ))}
        </ul>
      )}
    </Panel>
  );
}
