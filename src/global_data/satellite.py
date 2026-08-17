"""Shared satellite source framework (Milestone 16).

All gridded satellite sources (AOD, NDVI, DEM, VIIRS, weather, OSM) share the
same rules:

  - tile-based processing (global rasters are NEVER loaded whole into RAM)
  - credentials from environment vars only; missing creds -> UNAVAILABLE
  - cache validation before re-download; checksums; failed-download manifest
  - graceful failure: report status, never fabricate, never substitute
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Callable, Optional

from .cache import GlobalCache, fetch_with_retry, record_failed_download
from .scope import validate_scope

logger = logging.getLogger(__name__)


class SourceUnavailable(Exception):
    """Raised when a source cannot be acquired (no creds / disabled / blocked)."""


class SatelliteSource:
    """Base class for gridded satellite sources."""

    source_id: str = ""
    product: str = ""
    default_resolution: str = ""

    def __init__(self, config):
        self.config = config
        self.global_data = config.get("global_data", {})
        self.source_cfg = self.global_data.get("sources", {}).get(self.source_id, {})
        self.fetch_cfg = self.global_data.get("fetch", {})
        self.scope = "global"
        cache_cfg = self.global_data.get("cache", {})
        self.cache = GlobalCache(
            cache_dir=cache_cfg.get("dir", "data/cache/global"),
            version=cache_cfg.get("version", "v1"),
            validate_before_redownload=cache_cfg.get("validate_before_redownload", True),
        )

    # -- credentials -----------------------------------------------------
    def credential_env_vars(self) -> list:
        return [
            self.source_cfg.get("credential_env_var"),
            self.source_cfg.get("credential_env_var_2"),
        ]

    def credentials_present(self) -> bool:
        env_vars = [v for v in self.credential_env_vars() if v]
        if not env_vars:
            return True
        return all(
            bool(os.environ.get(var)) for var in env_vars
        )

    def check_availability(self) -> dict:
        enabled = bool(self.source_cfg.get("enabled", False))
        creds = self.credentials_present()
        if not enabled:
            status = "disabled"
        elif not creds:
            status = "unavailable"
        else:
            status = "available"
        return {
            "source": self.source_id,
            "name": self.product,
            "enabled": enabled,
            "credentials_present": creds,
            "status": status,
            "reason": None if status == "available" else (
                "Credentials missing (env vars only). Set the required "
                "environment variables to enable acquisition."
                if status == "unavailable" else "Source disabled in config.yaml."
            ),
        }

    @staticmethod
    def probe_connectivity(url: str, timeout: float = 5.0) -> bool:
        """Quick reachability probe (used by credential-free sources)."""
        import requests

        try:
            response = requests.get(url, timeout=timeout,
                                    headers={"User-Agent": "pm25-hyperlocal-m16/0.1"})
            return response.status_code < 500
        except Exception:  # noqa: BLE001
            return False

    # -- helpers ---------------------------------------------------------
    def _download(self, url: str, dest: Path, tile: dict, date: str,
                  extra_headers: Optional[dict] = None) -> Path:
        import requests

        def _fetch(attempt: int):
            headers = {"User-Agent": "pm25-hyperlocal-m16/0.1"}
            headers.update(extra_headers or {})
            response = requests.get(url, headers=headers, timeout=60, stream=True)
            response.raise_for_status()
            with open(dest, "wb") as file:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        file.write(chunk)
            return dest

        failure_path = self._failed_manifest_path()
        return fetch_with_retry(
            _fetch,
            attempts=int(self.fetch_cfg.get("retries", 3)),
            backoff_base_s=float(self.fetch_cfg.get("backoff_base_s", 2.0)),
            backoff_max_s=float(self.fetch_cfg.get("backoff_max_s", 60.0)),
            on_failure=lambda record: record_failed_download(
                failure_path, self.source_id, self.product, date,
                tile.get("tile_id", "n/a"), record,
            ),
        )

    def _failed_manifest_path(self) -> Optional[Path]:
        processed_base = Path(self.global_data.get("storage", {}).get(
            "processed_base", "data/processed/global"))
        return processed_base / "availability" / f"global_failed_downloads_{self.scope}.json"

    def attempt_acquire(self, scope: str, date: str,
                        on_tile: Callable[[dict, str], Path]) -> dict:
        """Run tile-based acquisition; returns honest status dict."""
        self.scope = validate_scope(scope)
        availability = self.check_availability()
        report = {
            "source": self.source_id,
            "scope": scope,
            "status": availability["status"],
            "reason": availability["reason"],
            "tiles_completed": 0,
            "tiles_failed": 0,
            "artifacts": [],
        }
        if availability["status"] != "available":
            logger.warning("%s acquisition blocked: %s",
                           self.source_id, availability["reason"])
            return report

        from ..geospatial.tiles import generate_tiles

        from .scope import scope_bounds

        tiles = list(generate_tiles(scope_bounds(scope)))
        max_tiles = int(self.fetch_cfg.get("max_tiles", 0))
        if max_tiles > 0:
            tiles = tiles[:max_tiles]
            report["tiles_attempted"] = max_tiles
        for tile in tiles:
            try:
                path = on_tile(tile, date)
                report["artifacts"].append(str(path))
                report["tiles_completed"] += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning("Tile %s %s failed: %s", self.source_id,
                               tile.tile_id, exc)
                record_failed_download(
                    self._failed_manifest_path(), self.source_id,
                    self.product, date, tile.tile_id,
                    {"error": str(exc)},
                )
                report["tiles_failed"] += 1

        if report["tiles_completed"] == 0 and report["tiles_failed"] > 0:
            report["status"] = "failed"
        elif report["tiles_completed"] > 0:
            report["status"] = "available"
        else:
            report["status"] = "no_tiles_processed"
        return report
