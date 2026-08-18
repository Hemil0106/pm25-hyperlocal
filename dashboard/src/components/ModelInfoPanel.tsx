import { Panel, StatusPill } from "./Panel";
import type { FeatureImportanceResponse, UncertaintyResponse } from "../types";

interface ModelInfoPanelProps {
  metadata: Record<string, unknown> | null;
  featureImportance: FeatureImportanceResponse | null;
  uncertainty: UncertaintyResponse | null;
}

function metadataValue(
  metadata: Record<string, unknown> | null,
  segments: string[],
): string {
  if (!metadata) return "";
  let current: unknown = metadata;
  for (const segment of segments) {
    if (
      current &&
      typeof current === "object" &&
      segment in (current as Record<string, unknown>)
    ) {
      current = (current as Record<string, unknown>)[segment];
    } else {
      return "";
    }
  }
  if (current == null || current === "") return "";
  return String(current);
}

const FEATURE_LABELS: Record<string, string> = {
  temperature_c: "Temperature",
  relative_humidity_pct: "Relative Humidity",
  road_density: "Road Density",
  wind_u_mps: "Wind U Component",
  wind_v_mps: "Wind V Component",
  wind_direction_deg: "Wind Direction",
  elevation_m: "Elevation",
  wind_speed_mps: "Wind Speed",
  NDVI: "NDVI (Vegetation Index)",
  night_lights: "Night Lights",
  day_of_year: "Day of Year",
  sin_day_of_year: "Sine of Day of Year",
  cos_day_of_year: "Cosine of Day of Year",
  month: "Month",
};

function humanizeFeature(feature: string): string {
  if (feature in FEATURE_LABELS) return FEATURE_LABELS[feature];
  return feature.replace(/_/g, " ");
}

const BAR_GRADIENTS = [
  "linear-gradient(90deg, #3da5f4 0%, #7dd3fc 100%)",
  "linear-gradient(90deg, #34d399 0%, #6ee7b7 100%)",
  "linear-gradient(90deg, #f59e0b 0%, #fcd34d 100%)",
  "linear-gradient(90deg, #f87171 0%, #fca5a5 100%)",
  "linear-gradient(90deg, #a78bfa 0%, #c4b5fd 100%)",
  "linear-gradient(90deg, #f472b6 0%, #f9a8d4 100%)",
  "linear-gradient(90deg, #38bdf8 0%, #7dd3fc 100%)",
  "linear-gradient(90deg, #fb923c 0%, #fdba74 100%)",
];

export function ModelInfoPanel({
  metadata,
  featureImportance,
  uncertainty,
}: ModelInfoPanelProps) {
  const model = metadataValue(metadata, ["PM2.5", "model"]);
  const datasetMode = metadataValue(metadata, ["PM2.5", "dataset_mode"]);
  const aodUsed = metadataValue(metadata, ["PM2.5", "AOD_used"]);
  const trainingRows = metadataValue(metadata, ["Data limitations", "training_rows"]);
  const stationCount = metadataValue(metadata, ["Data limitations", "stations"]);
  const downscalingMethod = metadataValue(metadata, ["Downscaling", "method"]);
  const features = featureImportance?.features ?? [];
  const topFeatures = [...features]
    .sort(
      (a, b) =>
        (b.random_forest_importance ?? 0) - (a.random_forest_importance ?? 0),
    )
    .slice(0, 8);

  return (
    <>
      <Panel title="Model & data">
        <div className="kv-grid">
          <div>
            <span className="loc-label">Model</span>
            <span className="loc-value tech-value">{model || "—"}</span>
          </div>
          <div>
            <span className="loc-label">Training mode</span>
            <span className="loc-value">
              {datasetMode ? (
                <StatusPill
                  status={
                    datasetMode.charAt(0).toUpperCase() + datasetMode.slice(1)
                  }
                />
              ) : (
                "—"
              )}
            </span>
          </div>
          <div>
            <span className="loc-label">AOD</span>
            <span className="loc-value">
              <StatusPill
                status={aodUsed === "true" ? "Used" : "Unavailable"}
              />
            </span>
          </div>
          <div>
            <span className="loc-label">Training rows</span>
            <span className="loc-value tech-value">
              {trainingRows || "—"}
            </span>
          </div>
          <div>
            <span className="loc-label">Stations</span>
            <span className="loc-value tech-value">
              {stationCount || "—"}
            </span>
          </div>
          <div>
            <span className="loc-label">Downscaling</span>
            <span className="loc-value">
              {downscalingMethod || "Baseline prototype"}
            </span>
          </div>
          <div>
            <span className="loc-label">Uncertainty</span>
            <span className="loc-value">
              {uncertainty?.status ? (
                <StatusPill status={uncertainty.status} />
              ) : (
                "—"
              )}
            </span>
          </div>
        </div>
      </Panel>

      <Panel title="Model feature importance">
        <p className="micro-note">Not causal attribution.</p>
        {featureImportance === null && (
          <p className="status-text">Loading feature importance…</p>
        )}
        {topFeatures.length === 0 && featureImportance !== null && (
          <p className="status-text">No feature importance data available.</p>
        )}
        <div className="fi-bars" aria-label="Model feature importance">
          {topFeatures.map((item, idx) => {
            const value = item.random_forest_importance ?? 0;
            const max = Math.max(
              0.0001,
              ...topFeatures.map((f) => f.random_forest_importance ?? 0),
            );
            return (
              <div className="fi-row" key={item.feature}>
                <span className="fi-label" title={item.feature}>
                  {humanizeFeature(item.feature)}
                </span>
                <div className="fi-track">
                  <div
                    className="fi-bar"
                    style={{
                      width: `${(value / max) * 100}%`,
                      background: BAR_GRADIENTS[idx % BAR_GRADIENTS.length],
                      animationDelay: `${idx * 80}ms`,
                    }}
                  />
                </div>
                <span className="fi-value">{value.toFixed(3)}</span>
              </div>
            );
          })}
        </div>
      </Panel>

      <Panel title="Uncertainty">
        <p className="micro-note">Status: {uncertainty?.status ?? "…"}</p>
        {uncertainty?.status === "DEFERRED" && (
          <p className="body-text">
            Pixel-level uncertainty is currently deferred because the available
            training/validation dataset is insufficient for a defensible
            uncertainty estimate.
          </p>
        )}
        {uncertainty?.reason && (
          <p className="body-text muted">{uncertainty.reason}</p>
        )}
        {uncertainty?.status && uncertainty.status !== "DEFERRED" && (
          <p className="body-text">
            Method: {uncertainty.method ?? "not defined"}
          </p>
        )}
      </Panel>
    </>
  );
}
