"""NASA Earthdata authentication and data access helpers (Milestone 16).

Provides Earthdata bearer token generation from username/password, and a
CMR-based granule search helper that returns download URLs for LAADS/USGS data.
"""

from __future__ import annotations

import base64
import logging
import os
from typing import Optional

import requests

logger = logging.getLogger(__name__)

_EARTHDATA_TOKEN_CACHE: dict = {}


def get_earthdata_token() -> str:
    """Get Earthdata bearer token.

    First checks for LAADS_APP_TOKEN env var. Then lists existing tokens from
    the user's Earthdata account. Only creates a new token if none exist.
    Caches the token for the session.
    """
    if _EARTHDATA_TOKEN_CACHE.get("token"):
        return _EARTHDATA_TOKEN_CACHE["token"]

    # 1. Check env var first
    fallback = os.environ.get("LAADS_APP_TOKEN", "")
    if fallback:
        _EARTHDATA_TOKEN_CACHE["token"] = fallback
        return fallback

    username = os.environ.get("EARTHDATA_USERNAME", "")
    password = os.environ.get("EARTHDATA_PASSWORD", "")
    if not username or not password:
        return ""

    auth = (username, password)

    # 2. Try to use an existing token from the user's account
    try:
        resp = requests.get(
            "https://urs.earthdata.nasa.gov/api/users/tokens",
            auth=auth,
            timeout=30,
        )
        if resp.status_code == 200:
            tokens = resp.json()
            if tokens:
                token = tokens[0].get("access_token", "")
                if token:
                    _EARTHDATA_TOKEN_CACHE["token"] = token
                    logger.info("Earthdata bearer token acquired from existing tokens.")
                    return token
    except Exception as exc:
        logger.warning("Failed to list Earthdata tokens: %s", exc)

    # 3. Create new token (may fail with max_token_limit)
    try:
        resp = requests.post(
            "https://urs.earthdata.nasa.gov/api/users/token",
            auth=auth,
            timeout=30,
        )
        if resp.status_code == 200:
            token = resp.json().get("access_token", "")
            if token:
                _EARTHDATA_TOKEN_CACHE["token"] = token
                logger.info("Earthdata bearer token acquired successfully.")
                return token
        logger.warning("Earthdata token request returned status %d", resp.status_code)
    except Exception as exc:
        logger.warning("Failed to get Earthdata token: %s", exc)

    return ""


def earthdata_auth_headers() -> dict:
    """Return Authorization headers for NASA data downloads."""
    token = get_earthdata_token()
    if token:
        return {"Authorization": f"Bearer {token}"}
    return {}


def cmr_search_granules(
    collection_concept_id: str,
    bbox: Optional[dict] = None,
    temporal_start: Optional[str] = None,
    temporal_end: Optional[str] = None,
    page_size: int = 10,
) -> list[dict]:
    """Search NASA CMR for granules matching the given criteria.

    Returns a list of granule dicts with keys: concept_id, title,
    summary, time_start, links (list of dicts with href/rel).
    """
    url = "https://cmr.earthdata.nasa.gov/search/granules.json"
    params = {
        "collection_concept_id": collection_concept_id,
        "page_size": page_size,
    }
    if bbox:
        params["bounding_box"] = (
            f"{bbox['west']},{bbox['south']},{bbox['east']},{bbox['north']}"
        )
    if temporal_start and temporal_end:
        params["temporal"] = f"{temporal_start},{temporal_end}"
    elif temporal_start:
        params["temporal"] = f"{temporal_start},"

    headers = {"User-Agent": "pm25-hyperlocal-m16/0.1"}
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data.get("feed", {}).get("entry", [])
    except Exception as exc:
        logger.warning("CMR search failed: %s", exc)
        return []


def find_download_url(granules: list[dict], provider: str = "LAADS") -> Optional[str]:
    """Extract a download URL from CMR granule search results.

    Searches for URS, S3, or direct download links.
    """
    for granule in granules:
        links = granule.get("links", [])
        for link in links:
            href = link.get("href", "")
            rel = link.get("rel", "")
            # Prefer URS download links (direct file download).
            if rel == "" and "s3" not in href.lower() and href.startswith("http"):
                return href
        # Fallback: any http link.
        for link in links:
            href = link.get("href", "")
            if href.startswith("http") and "s3" not in href.lower():
                return href
    return None


def download_with_earthdata(url: str, dest, extra_headers: Optional[dict] = None,
                            timeout: float = 120) -> None:
    """Download a file from a NASA URL with Earthdata auth, handling redirects."""
    headers = {
        "User-Agent": "pm25-hyperlocal-m16/0.1",
        **(extra_headers or {}),
    }
    auth_headers = earthdata_auth_headers()
    headers.update(auth_headers)

    session = requests.Session()
    session.headers.update(headers)
    resp = session.get(url, timeout=timeout, stream=True, allow_redirects=True)
    resp.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
