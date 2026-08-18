import { useEffect, useRef, useState } from "react";

interface DateAnimatorProps {
  dates: string[];
  selectedDate: string;
  onDateChange: (date: string) => void;
}

export function DateAnimator({
  dates,
  selectedDate,
  onDateChange,
}: DateAnimatorProps) {
  const [playing, setPlaying] = useState(false);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    if (!playing || dates.length < 2) {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
      return;
    }
    intervalRef.current = setInterval(() => {
      const idx = dates.indexOf(selectedDate);
      const next = dates[(idx + 1) % dates.length];
      onDateChange(next);
    }, 1200);
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [playing, dates, selectedDate, onDateChange]);

  useEffect(() => {
    if (dates.length < 2) setPlaying(false);
  }, [dates]);

  if (dates.length < 2) return null;

  return (
    <button
      className={`date-animator-btn${playing ? " animating" : ""}`}
      onClick={() => setPlaying((p) => !p)}
      title={playing ? "Pause animation" : "Animate through dates"}
      aria-label={playing ? "Pause animation" : "Play animation"}
    >
      <svg
        width="14"
        height="14"
        viewBox="0 0 24 24"
        fill="currentColor"
        xmlns="http://www.w3.org/2000/svg"
      >
        {playing ? (
          <>
            <rect x="6" y="5" width="4" height="14" rx="1" />
            <rect x="14" y="5" width="4" height="14" rx="1" />
          </>
        ) : (
          <path d="M8 5v14l11-7z" />
        )}
      </svg>
    </button>
  );
}
