"""Download caching, retry/backoff, and failed-download manifest (M16).

Caching policy (spec): before (re)downloading, validate the existing cache
entry; if valid, reuse it. Every cached file carries a SHA-256 checksum in its
provenance so corruption is detected (and triggers a redownload). Failed
downloads are recorded in ``global_failed_downloads.json`` -- never silently
ignored.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Callable, Optional

from ..geospatial.tiles import CACHE_MARKER, cache_is_valid, cache_key, cache_put
from .integrity import sha256_file, validate_checksum

logger = logging.getLogger(__name__)


class DownloadError(Exception):
    """Raised after retries are exhausted for a fetch."""


def _json_dump(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def fetch_with_retry(fetch: Callable[[int], object],
                     attempts: int = 3, backoff_base_s: float = 2.0,
                     backoff_max_s: float = 60.0,
                     on_failure: Optional[Callable[[dict], None]] = None) -> object:
    """Run ``fetch(attempt_index)`` with exponential backoff.

    ``attempts`` total tries; sleeps ``min(backoff_base * 2**n, backoff_max)``
    between tries. After the final failure, ``on_failure(failure_record)`` is
    called (used to append to the failed-downloads manifest) and DownloadError
    is raised.
    """
    last_error: Optional[Exception] = None
    for attempt in range(attempts):
        try:
            return fetch(attempt)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            is_last = attempt == attempts - 1
            delay = min(backoff_base_s * (2 ** attempt), backoff_max_s)
            logger.warning(
                "Download attempt %d/%d failed (%s); retrying in %.1fs",
                attempt + 1, attempts, exc, delay if not is_last else 0.0,
            )
            if is_last:
                break
            time.sleep(delay)

    record = {
        "error": str(last_error),
        "attempts": attempts,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if on_failure is not None:
        try:
            on_failure(record)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed-download callback error: %s", exc)
    raise DownloadError(str(last_error)) from last_error


def record_failed_download(manifest_path: Optional[Path],
                           source: str, product: str, date: str,
                           tile_id: str, failure: dict) -> None:
    """Append a failed download to the manifest (create if missing)."""
    if manifest_path is None:
        return
    path = Path(manifest_path)
    payload = {"source": source, "product": product, "date": date,
               "tile_id": tile_id, **failure}
    entries = []
    if path.exists():
        try:
            entries = json.loads(path.read_text(encoding="utf-8")).get("failures", [])
        except Exception:  # noqa: BLE001
            entries = []
    entries.append(payload)
    _json_dump(path, {"failures": entries})


class GlobalCache:
    """Cache wrapper with checksum validation for M16 downloads."""

    def __init__(self, cache_dir, version: str = "v1",
                 validate_before_redownload: bool = True):
        self.cache_dir = Path(cache_dir)
        self.version = version
        self.validate = bool(validate_before_redownload)

    def key(self, source: str, product: str, date: str, tile_id: str,
            resolution: str = "") -> Path:
        return cache_key(source, product, date, tile_id,
                         version=self.version, resolution=resolution,
                         cache_dir=self.cache_dir)

    def is_valid(self, key: Path) -> bool:
        if not cache_is_valid(key):
            return False
        meta = cache_get_meta(key)
        if self.validate and meta:
            data_file = meta.get("data_file")
            if data_file:
                return validate_checksum(key / data_file, meta.get("sha256"))
        return True

    def store(self, key: Path, data_file: str, metadata: Optional[dict] = None) -> Path:
        """Store a downloaded file under the cache key (with checksum)."""
        key.mkdir(parents=True, exist_ok=True)
        src = Path(data_file)
        if not src.exists():
            raise FileNotFoundError(f"Cached file source not found: {src}")
        dest = key / src.name
        import shutil

        shutil.copy2(src, dest)
        meta = dict(metadata or {})
        meta["data_file"] = src.name
        meta["sha256"] = sha256_file(dest)
        cache_put(key, meta)
        return dest


def cache_get_meta(key: Path) -> Optional[dict]:
    """Read provenance.json for a cache key (None if absent)."""
    meta_path = key / "provenance.json"
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
