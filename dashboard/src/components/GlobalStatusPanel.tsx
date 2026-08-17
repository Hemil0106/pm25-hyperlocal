import { Panel } from "./Panel";
import type {
  AOIInfoResponse,
  DataAvailabilityResponse,
  GlobalDataStatusResponse,
  ModelScopesResponse,
  OutputMetadataResponse,
} from "../types";

interface GlobalStatusPanelProps {
  availability: DataAvailabilityResponse | null;
  scopes: ModelScopesResponse | null;
  outputMetadata: OutputMetadataResponse | null;
  aoi: AOIInfoResponse | null;
  loading: boolean;
  globalDataStatus: GlobalDataStatusResponse | null;
  globalDataLoading: boolean;
}

export function GlobalStatusPanel({
  availability,
  scopes,
  outputMetadata,
  aoi,
  loading,
  globalDataStatus,
  globalDataLoading,
}: GlobalStatusPanelProps) {
  const canPredict = outputMetadata?.inference?.can_predict ?? false;
  const scopeStatus =
    aoi?.model_scope?.status === "available"
      ? "available"
      : aoi?.model_scope?.status === "unavailable_for_aoi"
        ? "unavailable"
        : null;
  const gt = availability?.ground_truth;

  return (
    <>
      <Panel title="Globalization status">
        {loading && <p className="status-text">Loading scope status…</p>}
        {!loading && (
          <div className="kv-grid">
            <div>
              <span className="loc-label">AOI</span>
              <span className="loc-value">{aoi?.name ?? availability?.aoi?.name ?? "—"}</span>
            </div>
            <div>
              <span className="loc-label">Mode</span>
              <span className="loc-value">
                {aoi?.mode ?? availability?.aoi?.mode ?? "—"}
              </span>
            </div>
            <div>
              <span className="loc-label">Area</span>
              <span className="loc-value">
                {aoi?.area_km2 != null
                  ? `${aoi.area_km2.toLocaleString("en-IN")} km²`
                  : "—"}
              </span>
            </div>
            <div>
              <span className="loc-label">Metric CRS</span>
              <span className="loc-value">{aoi?.metric_crs ?? "—"}</span>
            </div>
            <div>
              <span className="loc-label">Model scope</span>
              <span className="loc-value">
                {aoi?.model_scope?.id ?? "—"}
                {scopeStatus === "available" && (
                  <span className="status-dot status-ok" title="Available" />
                )}
                {scopeStatus === "unavailable" && (
                  <span className="status-dot status-down" title="Unavailable" />
                )}
              </span>
            </div>
            <div>
              <span className="loc-label">Ground truth</span>
              <span className="loc-value">
                {gt
                  ? `${gt.n_stations_in_aoi} stations · ${gt.n_rows_in_aoi} rows`
                  : "—"}
              </span>
            </div>
            <div>
              <span className="loc-label">Ground truth coverage</span>
              <span className="loc-value">
                {gt && gt.countries.length > 0
                  ? gt.countries.join(", ")
                  : "—"}
              </span>
            </div>
            <div>
              <span className="loc-label">Prediction</span>
              <span className="loc-value">
                {canPredict ? "Available" : "Unavailable"}
              </span>
            </div>
          </div>
        )}
      </Panel>

      {!loading && canPredict && (
        <Panel title="Available outputs">
          <p className="body-text">
            Predicted PM2.5 at 500 m, PM2.5-derived AQI, and predicted
            high-pollution zones are available for this AOI on the map.
          </p>
        </Panel>
      )}

      {!loading && !canPredict && (
        <Panel title="Why is prediction unavailable?" className="panel-warning">
          <p className="body-text">
            No validated model trained on observations covering this AOI exists.
            The Delhi prototype model is never applied to other regions, and
            global predictions would be fabricated.
          </p>
          {aoi?.model_scope?.reason && (
            <p className="body-text muted">{aoi.model_scope.reason}</p>
          )}
          {availability && availability.notes.length > 0 && (
            <p className="body-text muted">{availability.notes[0]}</p>
          )}
          <p className="micro-note">
            Model scopes:{" "}
            {scopes
              ? Object.entries(scopes.scopes)
                  .map(([id, s]) => `${id}: ${s.status}`)
                  .join(" · ")
              : "—"}
          </p>
        </Panel>
      )}

      <Panel title="Global data acquisition">
        {globalDataLoading && <p className="status-text">Loading acquisition status…</p>}
        {!globalDataLoading && globalDataStatus === null && (
          <p className="body-text muted">Acquisition status unavailable.</p>
        )}
        {!globalDataLoading && globalDataStatus !== null && (
          <>
            <div className="kv-grid">
              <div>
                <span className="loc-label">Scope</span>
                <span className="loc-value">{globalDataStatus.scope}</span>
              </div>
              <div>
                <span className="loc-label">Overall</span>
                <span className="loc-value">
                  {globalDataStatus.overall}
                  {globalDataStatus.overall === "available" && (
                    <span className="status-dot status-ok" title="Available" />
                  )}
                  {globalDataStatus.overall === "unavailable" && (
                    <span className="status-dot status-down" title="Unavailable" />
                  )}
                </span>
              </div>
              <div>
                <span className="loc-label">Observations</span>
                <span className="loc-value">
                  {globalDataStatus.observations?.n_observations != null
                    ? `${globalDataStatus.observations.n_observations} rows`
                    : "—"}
                </span>
              </div>
            </div>
            {globalDataStatus.available_sources.length > 0 && (
              <p className="body-text muted">
                Acquired: {globalDataStatus.available_sources.join(", ")}
              </p>
            )}
            {globalDataStatus.unavailable_sources.length > 0 && (
              <p className="body-text muted">
                Unavailable: {globalDataStatus.unavailable_sources.join(", ")}
              </p>
            )}
            <p className="micro-note">
              Real datasets only. Missing credentials are reported, never
              substituted; no global prediction or AQI is produced from this layer.
            </p>
          </>
        )}
      </Panel>
    </>
  );
}
