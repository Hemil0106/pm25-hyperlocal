import type { ReactNode } from "react";

export function Panel({
  title,
  children,
  className = "",
  right,
}: {
  title?: string;
  children: ReactNode;
  className?: string;
  right?: ReactNode;
}) {
  return (
    <section className={`panel ${className}`}>
      {title && (
        <div className="panel-header">
          <h2 className="panel-title">{title}</h2>
          {right}
        </div>
      )}
      {children}
    </section>
  );
}

export function LoadingText({ label }: { label: string }) {
  return (
    <p className="status-text" role="status">
      {label}…
    </p>
  );
}

export function SkeletonLines({ lines = 3 }: { lines?: number }) {
  return (
    <div role="status" aria-label="Loading">
      {Array.from({ length: lines }).map((_, index) => (
        <div
          key={index}
          className={`skeleton sk-line ${index === 0 ? "wide" : index === lines - 1 ? "med" : "short"}`}
          style={{ marginBottom: index < lines - 1 ? 8 : 0 }}
        />
      ))}
    </div>
  );
}

export function ErrorBox({ children }: { children: ReactNode }) {
  return (
    <div className="error-box" role="alert">
      {children}
    </div>
  );
}

export function StatusPill({
  status,
  className,
}: {
  status: string;
  className?: string;
}) {
  const normalized = status.toLowerCase();
  const tone = normalized === "available" || normalized === "used"
    ? "pill-ok"
    : normalized === "fallback" || normalized === "partial" || normalized === "deferred"
      ? "pill-warn"
      : "pill-na";
  return <span className={`status-pill ${tone} ${className ?? ""}`}>{status}</span>;
}
