import { Panel, SkeletonLines } from "./Panel";
import type { GlobalDataStatusResponse } from "../types";

interface SourceDescriptor {
  id: string;
  name: string;
}

const SOURCE_CATALOG: SourceDescriptor[] = [
  { id: "pm25", name: "CPCB / OpenAQ monitoring stations" },
  { id: "aod", name: "MODIS / MAIAC / AOD" },
  { id: "weather", name: "ERA5-Land meteorology" },
  { id: "ndvi", name: "MODIS NDVI" },
  { id: "dem", name: "SRTM / DEM" },
  { id: "osm", name: "OpenStreetMap" },
  { id: "viirs", name: "VIIRS Night Lights" },
];

type Tone = "ok" | "warn" | "na";

function pillFor(
  sourceId: string,
  status: GlobalDataStatusResponse | null,
): { tone: Tone; label: string } {
  if (status) {
    if (status.available_sources.includes(sourceId)) {
      return { tone: "ok", label: "Available" };
    }
    if (status.unavailable_sources.includes(sourceId)) {
      return { tone: "na", label: "Unavailable" };
    }
  }
  return { tone: "na", label: "Not run" };
}

interface DataSourcesPanelProps {
  globalDataStatus: GlobalDataStatusResponse | null;
  loading: boolean;
}

export function DataSourcesPanel({ globalDataStatus, loading }: DataSourcesPanelProps) {
  const observations = globalDataStatus?.observations;
  return (
    <Panel
      title="Data sources"
      right={<span className="micro-note" style={{ margin: 0 }}>live metadata</span>}
    >
      {observations && (
        <p className="micro-note" style={{ marginTop: 0 }}>
          {observations.n_observations != null
            ? `${observations.n_observations} observations`
            : "No observations acquired"}
          {observations.n_stations != null
            ? ` · ${observations.n_stations} stations`
            : ""}
        </p>
      )}
      {loading && <SkeletonLines lines={5} />}
      {!loading && (
        <ul className="source-list">
          {SOURCE_CATALOG.map((source) => {
            const pill = pillFor(source.id, globalDataStatus);
            return (
              <li key={source.id}>
                <span className="source-item">
                  <span>{source.name}</span>
                </span>
                <span className={`source-pill pill-${pill.tone}`}>{pill.label}</span>
              </li>
            );
          })}
        </ul>
      )}
      <p className="micro-note" style={{ marginTop: 8 }}>
        Status is read from live API metadata — sources are never assumed available.
      </p>
    </Panel>
  );
}
