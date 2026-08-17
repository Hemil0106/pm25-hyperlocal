import { useState } from "react";
import { Panel } from "./Panel";

const LIMITATIONS = [
  "AOD currently unavailable",
  "Fallback model currently active",
  "Training dataset is small",
  "500 m residual model not trained",
  "500 m output is a baseline spatial refinement",
  "Uncertainty estimation deferred",
  "Hotspots represent predicted high-pollution zones",
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
