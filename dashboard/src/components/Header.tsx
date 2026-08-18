import type { GlobalDataStatusResponse } from "../types";
import { DateAnimator } from "./DateAnimator";

interface HeaderProps {
  dates: string[];
  model: string | null;
  datasetMode: string | null;
  aodUsed: boolean;
  apiStatus: "ok" | "unavailable" | "loading";
  selectedDate: string;
  onDateChange: (date: string) => void;
  regions: Record<string, { name: string }> | null;
  selectedRegion: string;
  onRegionChange: (region: string) => void;
  regionScopeStatus: string;
  dataStatus: GlobalDataStatusResponse | null;
  dataLoading: boolean;
}

function formatDate(iso: string): string {
  const d = new Date(`${iso}T00:00:00`);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleDateString("en-GB", {
    day: "2-digit",
    month: "short",
    year: "numeric",
  });
}

function BrandMark() {
  return (
    <span className="brand-mark" aria-hidden="true">
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
        <path d="M3 8h11a3 3 0 0 0 0-6H9" />
        <path d="M3 16h12a3 3 0 0 1 0 6H9" />
        <path d="M3 12h17" />
      </svg>
    </span>
  );
}

const REGION_LABELS: Record<string, string> = {
  delhi: "Delhi",
  pune: "Pune",
  mumbai: "Mumbai",
  india: "India",
  global: "Global",
};

export function Header({
  dates,
  model,
  datasetMode,
  aodUsed,
  apiStatus,
  selectedDate,
  onDateChange,
  regions,
  selectedRegion,
  onRegionChange,
  regionScopeStatus,
  dataStatus,
  dataLoading,
}: HeaderProps) {
  const regionNames = regions ? Object.entries(regions) : [];
  const scopeLabel =
    (REGION_LABELS[selectedRegion] ?? selectedRegion).toUpperCase();

  const dataOverall = dataLoading
    ? "…"
    : dataStatus?.overall === "available"
      ? "Partial"
      : dataStatus?.overall === "not_run"
        ? "Not run"
        : "Unavailable";

  return (
    <header className="header">
      <div className="header-brand">
        <BrandMark />
        <div className="brand-text">
          <h1 className="app-title">Hyperlocal PM2.5 Intelligence</h1>
          <p className="app-descriptor">
            AI/ML-based downscaling of satellite-based air quality maps
          </p>
        </div>
        <span
          className={`badge ${regionScopeStatus === "available" ? "badge-prototype" : "badge-muted"}`}
        >
          {regionScopeStatus === "available" ? "SIH Prototype" : "Prototype"}
        </span>
      </div>

      <div className="header-controls">
        <div className="header-control">
          <span className="header-control-label">Scope</span>
          <select
            className="header-select"
            value={selectedRegion}
            onChange={(event) => onRegionChange(event.target.value)}
            aria-label="Selected region"
          >
            {regionNames.length === 0 && <option value="">—</option>}
            {regionNames.map(([id, info]) => (
              <option key={id} value={id}>
                {info.name}
              </option>
            ))}
          </select>
        </div>
        <div className="header-control header-control-date">
          <span className="header-control-label">Date</span>
          <select
            className="header-select"
            value={selectedDate}
            onChange={(event) => onDateChange(event.target.value)}
            aria-label="Selected date"
          >
            {dates.length === 0 && <option value="">No dates</option>}
            {dates.map((item) => (
              <option key={item} value={item}>
                {formatDate(item)}
              </option>
            ))}
          </select>
          <DateAnimator
            dates={dates}
            selectedDate={selectedDate}
            onDateChange={onDateChange}
          />
        </div>

        <div className="header-status-cluster">
          <div className="header-status" title="API connectivity">
            <span className="status-label">API</span>
            <span
              className={`status-dot ${apiStatus === "ok" ? "status-ok" : "status-down"}`}
              aria-hidden="true"
            />
            {apiStatus === "ok" ? "Online" : apiStatus === "loading" ? "…" : "Offline"}
          </div>
          <div className="header-status" title="Global data acquisition status">
            <span className="status-label">Data</span>
            <span
              className={`status-dot ${
                dataStatus?.overall === "available"
                  ? "status-warn"
                  : dataStatus?.overall === "unavailable"
                    ? "status-down"
                    : "status-warn"
              }`}
              aria-hidden="true"
            />
            {dataOverall}
          </div>
          <div className="header-status" title="Prediction model status">
            <span className="status-label">Model</span>
            <span className="tech-value">
              {model ?? "—"} · {scopeLabel}
            </span>
          </div>
          <div className="header-status" title="Training mode / AOD">
            <span className="status-label">Mode</span>
            {datasetMode ? datasetMode.charAt(0).toUpperCase() + datasetMode.slice(1) : "—"} ·{" "}
            {aodUsed ? "AOD on" : "AOD off"}
          </div>
        </div>
      </div>
    </header>
  );
}
