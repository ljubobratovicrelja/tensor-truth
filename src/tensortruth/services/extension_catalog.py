"""Remote extension catalog fetcher with local caching.

Fetches catalog.json and individual YAML extension files from a remote
repository (GitHub raw URL by default), with 24-hour TTL caching.
"""

import json
import logging
import time
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)

CATALOG_CACHE_FILENAME = "catalog_cache.json"
CACHE_TTL_SECONDS = 86400  # 24 hours

DEFAULT_CATALOG_URL = (
    "https://raw.githubusercontent.com/"
    "ljubobratovicrelja/tensor-truth/main/extension_library"
)


class CatalogCache:
    """Local file cache for the extension catalog."""

    def __init__(self, cache_dir: Path):
        self._cache_path = cache_dir / CATALOG_CACHE_FILENAME

    def read(self) -> Optional[dict]:
        """Read cached catalog if present and not expired."""
        data = self._load()
        if data is None:
            return None
        fetched_at = data.get("fetched_at", 0)
        if time.time() - fetched_at > CACHE_TTL_SECONDS:
            return None
        return data.get("catalog")

    def read_any(self) -> Optional[dict]:
        """Read cached catalog even if expired (offline fallback)."""
        data = self._load()
        if data is None:
            return None
        return data.get("catalog")

    def write(self, catalog: dict) -> None:
        """Write catalog data with current timestamp."""
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"fetched_at": time.time(), "catalog": catalog}
        self._cache_path.write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )

    def _load(self) -> Optional[dict]:
        if not self._cache_path.is_file():
            return None
        try:
            return json.loads(self._cache_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to read catalog cache: %s", e)
            return None


def fetch_catalog(base_url: str) -> dict:
    """Fetch catalog.json from the remote URL.

    Args:
        base_url: Base URL of the extension library (no trailing slash).

    Returns:
        Parsed catalog dict.

    Raises:
        ConnectionError: On network failure.
    """
    url = f"{base_url.rstrip('/')}/catalog.json"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        raise ConnectionError(f"Failed to fetch catalog: {e}") from e


def fetch_extension_yaml(base_url: str, ext_type: str, filename: str) -> str:
    """Fetch a single extension YAML file from the remote URL.

    Args:
        base_url: Base URL of the extension library.
        ext_type: "command" or "agent".
        filename: YAML filename (e.g., "context7.yaml").

    Returns:
        Raw YAML text content.

    Raises:
        ConnectionError: On network failure.
    """
    subdir = "commands" if ext_type == "command" else "agents"
    url = f"{base_url.rstrip('/')}/{subdir}/{filename}"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        return resp.text
    except requests.RequestException as e:
        raise ConnectionError(
            f"Failed to fetch extension {subdir}/{filename}: {e}"
        ) from e
