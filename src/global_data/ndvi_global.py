"""NDVI (MODIS Terra MOD13Q1) global adapter (Milestone 16).

NDVI uses 16-day composites. Spatial normalization later picks the nearest
VALID composite for each target date with ``no_future_match`` default (a
composite whose start date is after the target date is never used). The
composite date selection logic is pure and unit-testable.
"""

from __future__ import annotations

import logging
from datetime import date as _date, timedelta
from pathlib import Path
from typing import Optional

from .satellite import SatelliteSource

logger = logging.getLogger(__name__)


class NDVISource(SatelliteSource):
    source_id = "ndvi"
    product = "MODIS_Terra_MOD13Q1_v6.1"
    default_resolution = "250m"


def check_availability(config) -> dict:
    return NDVISource(config).check_availability()


def composite_dates(composite_period_days: int = 16, anchor: str = "2000-02-18") -> list[str]:
    """MODIS Terra MOD13Q1 composite start dates (every 16 days from anchor)."""
    start = _date.fromisoformat(anchor)
    period = int(composite_period_days)
    dates = []
    cursor = start
    while cursor.year <= 2050:
        dates.append(cursor.isoformat())
        cursor += timedelta(days=period)
    return dates


def nearest_valid_composite(target_date: str, valid_dates: list[str],
                            no_future_match: bool = True,
                            max_lookback_days: int = 32) -> Optional[str]:
    """Nearest valid NDVI composite for a target date.

    ``no_future_match=True`` (default): a composite starting after the target
    date is never selected (no-future rule). The nearest valid composite on or
    before the target date within ``max_lookback_days`` wins.
    """
    target = _date.fromisoformat(target_date)
    candidates = []
    for value in valid_dates:
        comp = _date.fromisoformat(value)
        offset = (target - comp).days
        if offset < 0 and no_future_match:
            continue
        if abs(offset) > max_lookback_days:
            continue
        candidates.append((comp, offset))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (abs(item[1]), item[0]))
    return candidates[0][0].isoformat()


def composite_metadata(target_date: str, selected: str,
                       no_future_match: bool = True) -> dict:
    """Record ndvi_date / target_date / temporal_offset_days for traceability."""
    target = _date.fromisoformat(target_date)
    selected_date = _date.fromisoformat(selected)
    return {
        "target_date": target_date,
        "ndvi_date": selected,
        "temporal_offset_days": (target - selected_date).days,
        "no_future_match": bool(no_future_match),
    }


def acquire(config, scope: str = "global", date: str = "2025-01-01") -> dict:
    source = NDVISource(config)
    cfg_temporal = source.source_cfg.get("temporal", {})
    period = int(cfg_temporal.get("composite_period_days", 16))
    no_future = bool(cfg_temporal.get("no_future_match", True))

    # Select the composite this acquisition targets.
    valid = composite_dates(period)
    selected = nearest_valid_composite(date, valid, no_future_match=no_future)

    def on_tile(tile: object, _date: str) -> Path:
        if selected is None:
            raise RuntimeError(f"No valid NDVI composite for date {_date}.")
        tile_id = tile.tile_id
        tile_bbox = tile.bbox
        raw_base = Path(config.get("global_data", {}).get("storage", {}).get(
            "raw_base", "data/raw/global"))
        out_dir = raw_base / "ndvi" / scope / str(tile_id) / selected
        out_dir.mkdir(parents=True, exist_ok=True)
        dest = out_dir / f"MOD13Q1_{selected}.tif"
        if dest.exists():
            return dest
        # Real download path (Earthdata); exercised when credentials are set.
        from .nasa_auth import (cmr_search_granules, find_download_url,
                                 download_with_earthdata)

        _MOD13Q1_CID = "C1748066515-LPCLOUD"
        granules = cmr_search_granules(
            _MOD13Q1_CID,
            bbox=tile_bbox,
            temporal_start=selected,
            temporal_end=selected,
        )
        url = find_download_url(granules)
        if not url:
            raise RuntimeError(
                f"No MOD13Q1 granule found for tile {tile_id} on {selected}")

        download_with_earthdata(url, str(dest))
        return dest

    report = source.attempt_acquire(scope, date, on_tile)
    report["composite_metadata"] = (
        composite_metadata(date, selected, no_future) if selected else None
    )
    return report
