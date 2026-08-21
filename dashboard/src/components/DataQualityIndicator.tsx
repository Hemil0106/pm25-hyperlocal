import { useState, useEffect } from "react";
import type { AvailabilityRegistryResponse, AvailabilitySourceEntry } from "../types";
import { getAvailabilityRegistry } from "../api";

const STATUS_COLORS: Record<string, string> = {
  AVAILABLE: "#34d399",
  PARTIAL: "#f59e0b",
  STALE: "#f59e0b",
  UNAVAILABLE: "#f87171",
  FAILED: "#f87171",
};

const CONFIDENCE_COLORS: Record<string, string> = {
  HIGH: "#34d399",
  MEDIUM: "#f59e0b",
  LOW: "#f59e0b",
  NONE: "#f87171",
};

function formatAge(ageS: number | null): string {
  if (ageS === null) return "never";
  const days = Math.floor(ageS / 86400);
  if (days === 0) return "< 1 day";
  if (days === 1) return "1 day";
  if (days < 30) return `${days} days`;
  const months = Math.floor(days / 30);
  return `${months} month${months > 1 ? "s" : ""}`;
}

function SourceRow({ source }: { source: AvailabilitySourceEntry }) {
  return (
    <div className="dq-source-row">
      <div className="dq-source-header">
        <span className="dq-source-name">{source.name}</span>
        <span
          className="dq-source-status"
          style={{ color: STATUS_COLORS[source.status] ?? "#888" }}
        >
          {source.status}
        </span>
      </div>
      <div className="dq-source-meta">
        <span>
          Age: {formatAge(source.freshness.age_s)}
        </span>
        <span>
          Confidence:{" "}
          <span style={{ color: CONFIDENCE_COLORS[source.confidence.level] ?? "#888" }}>
            {source.confidence.level}
          </span>
        </span>
        {source.artifact_checksums && (
          <span>Checksums: verified</span>
        )}
      </div>
      {source.reason && (
        <div className="dq-source-reason">{source.reason}</div>
      )}
    </div>
  );
}

export function DataQualityIndicator({ city }: { city?: string }) {
  const [registry, setRegistry] = useState<AvailabilityRegistryResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [expanded, setExpanded] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    getAvailabilityRegistry("global")
      .then((data) => {
        if (!cancelled) {
          setRegistry(data);
          setLoading(false);
        }
      })
      .catch((err) => {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Failed to load");
          setLoading(false);
        }
      });
    return () => { cancelled = true; };
  }, [city]);

  if (loading) {
    return (
      <section className="panel dq-panel">
        <div className="panel-header">
          <h2 className="panel-title">Data Quality</h2>
        </div>
        <div className="dq-loading">Loading data quality status…</div>
      </section>
    );
  }

  if (error || !registry) {
    return (
      <section className="panel dq-panel">
        <div className="panel-header">
          <h2 className="panel-title">Data Quality</h2>
        </div>
        <div className="dq-error">{error ?? "No data"}</div>
      </section>
    );
  }

  const sources = Object.values(registry.sources);
  const overallColor = STATUS_COLORS[registry.overall_status] ?? "#888";

  return (
    <section className="panel dq-panel">
      <div className="panel-header">
        <h2 className="panel-title">Data Quality</h2>
        <button
          className="dq-expand-btn"
          onClick={() => setExpanded(!expanded)}
          aria-label={expanded ? "Collapse" : "Expand"}
        >
          {expanded ? "−" : "+"}
        </button>
      </div>
      <div className="dq-overall">
        <span className="dq-overall-label">Overall:</span>
        <span className="dq-overall-status" style={{ color: overallColor }}>
          {registry.overall_status}
        </span>
        <span className="dq-overall-count">
          {registry.readiness_summary.available_source_count}/{registry.readiness_summary.total_source_count} sources
        </span>
      </div>
      {expanded && (
        <div className="dq-sources">
          {sources.map((src) => (
            <SourceRow key={src.id} source={src} />
          ))}
          <div className="dq-footer">
            Built: {new Date(registry.timestamp).toLocaleString()}
          </div>
        </div>
      )}
    </section>
  );
}
