"""Service for managing extensions: installed and library.

Scans installed extensions in ~/.tensortruth/{commands,agents}/ and fetches
available extensions from the remote catalog (GitHub by default).
"""

import logging
from pathlib import Path
from typing import Any, Optional

import yaml

from tensortruth.services.extension_catalog import (
    DEFAULT_CATALOG_URL,
    CatalogCache,
    fetch_catalog,
    fetch_extension_yaml,
)
from tensortruth.services.mcp_server_service import MCPServerService

logger = logging.getLogger(__name__)


def _parse_yaml_metadata(path: Path) -> Optional[dict[str, Any]]:
    """Parse a YAML extension file and extract metadata."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return None
        return {
            "name": raw.get("name", path.stem),
            "description": raw.get("description", ""),
            "requires_mcp": raw.get("requires_mcp"),
        }
    except Exception as e:
        logger.warning(f"Failed to parse {path.name}: {e}")
        return None


class ExtensionLibraryService:
    """Manages installed and library extensions."""

    def __init__(
        self,
        user_dir: Optional[Path] = None,
        catalog_url: Optional[str] = None,
        mcp_service: Optional[MCPServerService] = None,
    ):
        if user_dir is None:
            from tensortruth.app_utils.paths import get_user_data_dir

            user_dir = get_user_data_dir()
        self._user_dir = user_dir
        self._catalog_url = catalog_url or DEFAULT_CATALOG_URL
        self._mcp_service = mcp_service or MCPServerService()
        self._cache = CatalogCache(user_dir)

    def _get_configured_mcp_names(self) -> set[str]:
        """Get names of all configured and enabled MCP servers."""
        return self._mcp_service.get_configured_server_names()

    def _scan_dir(
        self, directory: Path, ext_type: str, configured_mcp: set[str]
    ) -> list[dict[str, Any]]:
        """Scan a directory for YAML extension files."""
        results: list[dict[str, Any]] = []
        if not directory.is_dir():
            return results

        for path in sorted(directory.iterdir()):
            if path.suffix not in (".yaml", ".yml"):
                continue
            meta = _parse_yaml_metadata(path)
            if meta is None:
                continue

            requires_mcp = meta.get("requires_mcp")
            mcp_available = True
            if requires_mcp:
                mcp_available = requires_mcp in configured_mcp

            results.append(
                {
                    "name": meta["name"],
                    "type": ext_type,
                    "description": meta["description"],
                    "filename": path.name,
                    "requires_mcp": requires_mcp,
                    "mcp_available": mcp_available,
                }
            )
        return results

    def list_installed(self) -> list[dict[str, Any]]:
        """List all installed user extensions."""
        configured_mcp = self._get_configured_mcp_names()
        extensions: list[dict[str, Any]] = []
        for ext_type, subdir in [("command", "commands"), ("agent", "agents")]:
            extensions.extend(
                self._scan_dir(self._user_dir / subdir, ext_type, configured_mcp)
            )
        return extensions

    def _fetch_catalog_data(self) -> Optional[dict]:
        """Fetch catalog with cache-first strategy and offline fallback."""
        # 1. Try valid cache
        cached = self._cache.read()
        if cached is not None:
            return cached

        # 2. Cache miss — fetch from remote
        try:
            catalog = fetch_catalog(self._catalog_url)
            self._cache.write(catalog)
            return catalog
        except ConnectionError:
            logger.warning("Failed to fetch remote catalog, trying expired cache")

        # 3. Offline fallback — use expired cache
        return self._cache.read_any()

    def list_library(self) -> list[dict[str, Any]]:
        """List all available extensions from the remote catalog."""
        catalog = self._fetch_catalog_data()
        if catalog is None:
            return []

        configured_mcp = self._get_configured_mcp_names()
        installed_filenames = self._get_installed_filenames()

        extensions: list[dict[str, Any]] = []
        for entry in catalog.get("extensions", []):
            ext_type = entry.get("type", "command")
            filename = entry.get("filename", "")
            requires_mcp = entry.get("requires_mcp")

            mcp_available = True
            if requires_mcp:
                mcp_available = requires_mcp in configured_mcp

            key = f"{ext_type}:{filename}"
            extensions.append(
                {
                    "name": entry.get("name", ""),
                    "type": ext_type,
                    "description": entry.get("description", ""),
                    "filename": filename,
                    "requires_mcp": requires_mcp,
                    "mcp_available": mcp_available,
                    "installed": key in installed_filenames,
                }
            )
        return extensions

    def _get_installed_filenames(self) -> set[str]:
        """Get set of 'type:filename' for installed extensions."""
        filenames: set[str] = set()
        for ext_type, subdir in [("command", "commands"), ("agent", "agents")]:
            d = self._user_dir / subdir
            if d.is_dir():
                for p in d.iterdir():
                    if p.suffix in (".yaml", ".yml"):
                        filenames.add(f"{ext_type}:{p.name}")
        return filenames

    def install(self, ext_type: str, filename: str) -> str:
        """Install an extension from the remote catalog to user dir.

        Args:
            ext_type: "command" or "agent"
            filename: YAML filename (e.g., "context7.yaml")

        Returns:
            The installed filename.

        Raises:
            ValueError: If filename is invalid.
            ConnectionError: If remote fetch fails.
        """
        if ext_type not in ("command", "agent"):
            raise ValueError(f"Invalid extension type: {ext_type}")

        # Security: sanitize filename
        safe_name = Path(filename).name
        if safe_name != filename or not safe_name.endswith((".yaml", ".yml")):
            raise ValueError(f"Invalid filename: {filename}")

        # Fetch YAML content from remote
        yaml_content = fetch_extension_yaml(self._catalog_url, ext_type, safe_name)

        subdir = "commands" if ext_type == "command" else "agents"
        dest_dir = self._user_dir / subdir
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / safe_name
        dest.write_text(yaml_content, encoding="utf-8")

        logger.info(f"Installed extension: {subdir}/{safe_name}")
        return safe_name

    def uninstall(self, ext_type: str, filename: str) -> None:
        """Uninstall a user extension.

        Args:
            ext_type: "command" or "agent"
            filename: YAML filename

        Raises:
            ValueError: If filename is invalid or not found.
        """
        if ext_type not in ("command", "agent"):
            raise ValueError(f"Invalid extension type: {ext_type}")

        # Security: sanitize filename
        safe_name = Path(filename).name
        if safe_name != filename or not safe_name.endswith((".yaml", ".yml")):
            raise ValueError(f"Invalid filename: {filename}")

        subdir = "commands" if ext_type == "command" else "agents"
        path = self._user_dir / subdir / safe_name
        if not path.is_file():
            raise ValueError(f"Extension not found: {subdir}/{safe_name}")

        path.unlink()
        logger.info(f"Uninstalled extension: {subdir}/{safe_name}")
