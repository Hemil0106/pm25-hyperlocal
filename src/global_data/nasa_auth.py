"""NASA Earthdata authentication and data access helpers.

Thin wrapper around earthaccess for authentication and CMR search.
Delegates to earthaccess for robust auth (netrc, env vars, token management)
instead of hand-rolling bearer token logic.

Credentials: EARTHDATA_USERNAME + EARTHDATA_PASSWORD (env only, never logged).
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import requests

logger = logging.getLogger(__name__)


def earthdata_auth_headers() -> dict:
    """Return Authorization headers for NASA data downloads.

    Uses earthaccess session for proper redirect/auth handling.
    """
    try:
        import earthaccess

        username = os.environ.get("EARTHDATA_USERNAME", "")
        password = os.environ.get("EARTHDATA_PASSWORD", "")

        if username and password:
            auth = earthaccess.login(strategy="environment")
        else:
            auth = earthaccess.login()

        if auth.authenticated:
            session = earthaccess.get_requests_https_session()
            cookies = session.cookies.get_dict()
            if cookies:
                cookie_str = "; ".join(f"{k}={v}" for k, v in cookies.items())
                return {"Cookie": cookie_str}
    except ImportError:
        logger.warning("earthaccess not installed")
    except Exception as exc:
        logger.warning("earthaccess auth failed: %s", exc)

    return {}


def cmr_search_granules(
    collection_concept_id: str,
    bbox: Optional[dict] = None,
    temporal_start: Optional[str] = None,
    temporal_end: Optional[str] = None,
    page_size: int = 10,
) -> list:
    """Search NASA CMR for granules matching the given criteria.

    Returns a list of granule dicts. Delegates to earthaccess.search_data
    when available for more robust search.
    """
    try:
        import earthaccess

        kwargs = {
            "short_name": None,
            "concept_id": collection_concept_id,
            "page_size": page_size,
        }

        temporal = None
        if temporal_start and temporal_end:
            temporal = (temporal_start, temporal_end)

        spatial = None
        if bbox:
            spatial = {
                "bounding_box": (
                    bbox["west"], bbox["south"],
                    bbox["east"], bbox["north"],
                )
            }

        results = earthaccess.search_data(
            concept_id=collection_concept_id,
            temporal=temporal,
            bounding_box=spatial["bounding_box"] if spatial else None,
            count=page_size,
        )

        granules = []
        for r in results:
            granule = {
                "concept_id": collection_concept_id,
                "title": getattr(r, "title", ""),
                "links": [],
            }
            for href in getattr(r, "data_links", []):
                granule["links"].append({"href": href, "rel": ""})
            granules.append(granule)
        return granules

    except ImportError:
        pass
    except Exception as exc:
        logger.warning("earthaccess search failed, falling back to CMR API: %s", exc)

    return _cmr_search_fallback(
        collection_concept_id, bbox, temporal_start, temporal_end, page_size
    )


def _cmr_search_fallback(
    collection_concept_id: str,
    bbox: Optional[dict],
    temporal_start: Optional[str],
    temporal_end: Optional[str],
    page_size: int,
) -> list:
    """Fallback CMR search using raw requests."""
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

    try:
        resp = requests.get(
            url, params=params,
            headers={"User-Agent": "pm25-hyperlocal/0.1"},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("feed", {}).get("entry", [])
    except Exception as exc:
        logger.warning("CMR search failed: %s", exc)
        return []


def find_download_url(granules: list) -> Optional[str]:
    """Extract a download URL from CMR granule search results."""
    for granule in granules:
        links = granule.get("links", [])
        for link in links:
            href = link.get("href", "")
            rel = link.get("rel", "")
            if rel == "" and "s3" not in href.lower() and href.startswith("http"):
                return href
        for link in links:
            href = link.get("href", "")
            if href.startswith("http") and "s3" not in href.lower():
                return href
    return None


def download_with_earthdata(
    url: str, dest, extra_headers: Optional[dict] = None, timeout: float = 120
) -> None:
    """Download a file from NASA with earthaccess-managed auth.

    Uses earthaccess session for proper redirect/cookie handling.
    """
    try:
        import earthaccess

        session = earthaccess.get_requests_https_session()
        resp = session.get(url, timeout=timeout, stream=True, allow_redirects=True)
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        return
    except ImportError:
        pass
    except Exception as exc:
        logger.warning("earthaccess download failed, falling back to requests: %s", exc)

    headers = {
        "User-Agent": "pm25-hyperlocal/0.1",
        **(extra_headers or {}),
    }
    headers.update(earthdata_auth_headers())

    session = requests.Session()
    session.headers.update(headers)
    resp = session.get(url, timeout=timeout, stream=True, allow_redirects=True)
    resp.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
