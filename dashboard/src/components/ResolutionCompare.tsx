interface ResolutionCompareProps {
  resolution: "500m" | "1000m";
  onResolutionChange: (resolution: "500m" | "1000m") => void;
  pm25Loading: boolean;
}

export function ResolutionCompare({
  resolution,
  onResolutionChange,
  pm25Loading,
}: ResolutionCompareProps) {
  return (
    <div className="resolution-compare" aria-label="Resolution comparison">
      <h3 className="section-label">Resolution comparison</h3>
      <div className="segmented" role="radiogroup" aria-label="PM2.5 map resolution">
        <button
          type="button"
          role="radio"
          aria-checked={resolution === "500m"}
          className={resolution === "500m" ? "segment-active" : ""}
          onClick={() => onResolutionChange("500m")}
        >
          500 m
        </button>
        <button
          type="button"
          role="radio"
          aria-checked={resolution === "1000m"}
          className={resolution === "1000m" ? "segment-active" : ""}
          onClick={() => onResolutionChange("1000m")}
        >
          1 km
        </button>
      </div>
      {pm25Loading && <p className="micro-note">Loading {resolution} layer…</p>}
      <p className="micro-note">
        {resolution === "500m"
          ? "500 m spatial refinement active."
          : "1 km coarse PM2.5 active."}
      </p>
      <p className="micro-note">
        Higher resolution increases spatial detail; it does not imply higher
        accuracy.
      </p>
    </div>
  );
}
