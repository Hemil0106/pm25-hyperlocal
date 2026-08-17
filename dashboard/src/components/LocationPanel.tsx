import { Panel } from "./Panel";
import { SkeletonLines } from "./Panel";
import type { LocationResponse, Station } from "../types";

interface LocationPanelProps {
  location: LocationResponse | null;
  loading: boolean;
  error: string | null;
  nearbyStation: Station | null;
  nearbyStationLoading: boolean;
}

function StateBlock({
  title,
  body,
  kind,
  children,
}: {
  title: string;
  body?: string;
  kind: "nodata" | "unavailable" | "error" | "empty";
  children?: React.ReactNode;
}) {
  return (
    <div className={`state-block state-${kind}`} role={kind === "error" ? "alert" : undefined}>
      <span className="state-title">{title}</span>
      {body && <p className="state-body">{body}</p>}
      {children}
    </div>
  );
}

export function LocationPanel({
  location,
  loading,
  error,
  nearbyStation,
  nearbyStationLoading,
}: LocationPanelProps) {
  return (
    <Panel title="Selected location">
      {loading && (
        <div role="status">
          <SkeletonLines lines={4} />
          <p className="micro-note" style={{ marginTop: 8 }}>
            Loading location data…
          </p>
        </div>
      )}

      {!loading && error && (
        <StateBlock title="Unable to retrieve location data" kind="error">
          <p className="state-body">{error}</p>
          <p className="state-body">Select a location inside the study area.</p>
        </StateBlock>
      )}

      {!loading && !error && !location && (
        <StateBlock title="No location selected" kind="empty">
          <p className="state-body">
            Click the map to inspect a point within the study area.
          </p>
        </StateBlock>
      )}

      {!loading && !error && location && (
        <div className="location-details">
          <div className="loc-coords">
            <span>
              Lat <strong className="tech-value">{location.location.latitude.toFixed(4)}</strong>
            </span>
            <span>
              Lon <strong className="tech-value">{location.location.longitude.toFixed(4)}</strong>
            </span>
          </div>

          <div className="loc-row">
            <span className="loc-label">Predicted PM2.5</span>
            <span className="loc-value">
              {location.pm25 != null ? (
                <span className="tech-value">{location.pm25.toFixed(1)} µg/m³</span>
              ) : (
                <span className="status-pill pill-na">No data</span>
              )}
            </span>
          </div>

          <div className="loc-row">
            <span className="loc-label">PM2.5-derived AQI</span>
            <span className="loc-value">
              {location.pm25_derived_aqi != null ? (
                <span className="tech-value">{location.pm25_derived_aqi}</span>
              ) : (
                <span className="status-pill pill-na">No data</span>
              )}
            </span>
          </div>

          <div className="loc-category">
            {location.aqi_category ? (
              <span className="category-name">{location.aqi_category}</span>
            ) : (
              <span className="category-name">—</span>
            )}
          </div>

          {location.pm25 == null && (
            <StateBlock title="NoData — valid location, no valid prediction" kind="nodata">
              <p className="state-body">
                No valid model output exists for this point and date. This is a
                geographic data condition, not an error.
              </p>
            </StateBlock>
          )}

          <div className="loc-row">
            <span className="loc-label">Spatial output</span>
            <span className="loc-value tech-value">500 m</span>
          </div>
          <div className="loc-row">
            <span className="loc-label">Model</span>
            <span className="loc-value">{location.model}</span>
          </div>
          <div className="loc-row">
            <span className="loc-label">Dataset mode</span>
            <span className="loc-value">
              {location.dataset_mode.charAt(0).toUpperCase() + location.dataset_mode.slice(1)}
            </span>
          </div>
          <div className="loc-row">
            <span className="loc-label">AOD</span>
            <span className="loc-value">{location.aod_used ? "Used" : "Unavailable"}</span>
          </div>
          <div className="loc-row">
            <span className="loc-label">Uncertainty</span>
            <span className="loc-value">{location.uncertainty_status}</span>
          </div>

          <p className="loc-note">
            These values are model-derived spatial estimates, not direct measurements at this
            location.
          </p>

          {nearbyStation && (
            <div className="nearby-station">
              <span className="loc-label">Nearby/selected CPCB station</span>
              <span className="loc-value tech-value">{nearbyStation.station_id}</span>
              {nearbyStation.latest_pm25 != null && (
                <span className="loc-sub">
                  Latest observed PM2.5: {nearbyStation.latest_pm25.toFixed(1)} µg/m³
                  {nearbyStationLoading && " (loading…)"}
                </span>
              )}
            </div>
          )}
        </div>
      )}
    </Panel>
  );
}
