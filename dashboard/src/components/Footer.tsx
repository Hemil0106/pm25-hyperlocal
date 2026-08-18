interface FooterProps {
  regionName?: string;
  aodUsed?: boolean;
}

const REGION_LABELS: Record<string, string> = {
  delhi: "Delhi",
  pune: "Pune",
  mumbai: "Mumbai",
  india: "India",
  global: "Global",
};

export function Footer({ regionName, aodUsed }: FooterProps) {
  const city = REGION_LABELS[regionName ?? ""] ?? "Delhi";
  return (
    <footer className="footer">
      <div className="footer-inner">
        <div className="footer-left">
          <span className="footer-title">
            AI/ML-Based Downscaling of Satellite-Based Air Quality Maps for Hyperlocal PM2.5 Mapping
          </span>
          <span className="footer-sub">
            Smart India Hackathon 2025 · SIH Prototype · Model-derived estimates — not direct measurements.
          </span>
        </div>
        <div className="footer-right">
          <span className="footer-meta">
            <span className="footer-dot" />
            {city}
            {aodUsed && <span className="footer-aod-badge">AOD</span>}
          </span>
          <span className="footer-meta muted">v0.4.0</span>
        </div>
      </div>
    </footer>
  );
}
