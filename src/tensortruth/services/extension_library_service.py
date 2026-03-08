"""Service for managing extensions: installed and library.

Scans installed extensions in ~/.tensortruth/{commands,agents}/ and available
extensions in the extension_library/ directory.
"""

import logging
from pathlib import Path
from typing import Any, Optional

import yaml

from tensortruth.services.mcp_server_service import MCPServerService

logger = logging.getLogger(__name__)


def _get_library_dir() -> Path:
    """Resolve the extension_library directory.

    Tries repo-root detection first (works in dev and installed from source),
    then falls back to looking relative to this file.
    """
    # Try repo root: go up from src/tensortruth/services/ to repo root
    repo_root = Path(__file__).resolve().parents[3]
    lib_dir = repo_root / "extension_library"
    if lib_dir.is_dir():
        return lib_dir

    # Fallback: look in the package data directory
    import importlib.resources

    try:
        ref = importlib.resources.files("tensortruth") / "extension_library"
        if hasattr(ref, "_path") and Path(ref._path).is_dir():
            return Path(ref._path)
    except (AttributeError, TypeError):
        pass

    return lib_dir  # Return the expected path even if not found


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
        library_dir: Optional[Path] = None,
        mcp_service: Optional[MCPServerService] = None,
    ):
        if user_dir is None:
            from tensortruth.app_utils.paths import get_user_data_dir

            user_dir = get_user_data_dir()
        self._user_dir = user_dir
        self._library_dir = library_dir or _get_library_dir()
        self._mcp_service = mcp_service or MCPServerService()

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

    def list_library(self) -> list[dict[str, Any]]:
        """List all available extensions from the library."""
        configured_mcp = self._get_configured_mcp_names()
        installed_filenames = self._get_installed_filenames()

        extensions: list[dict[str, Any]] = []
        for ext_type, subdir in [("command", "commands"), ("agent", "agents")]:
            lib_dir = self._library_dir / subdir
            if not lib_dir.is_dir():
                continue
            for path in sorted(lib_dir.iterdir()):
                if path.suffix not in (".yaml", ".yml"):
                    continue
                meta = _parse_yaml_metadata(path)
                if meta is None:
                    continue

                requires_mcp = meta.get("requires_mcp")
                mcp_available = True
                if requires_mcp:
                    mcp_available = requires_mcp in configured_mcp

                key = f"{ext_type}:{path.name}"
                extensions.append(
                    {
                        "name": meta["name"],
                        "type": ext_type,
                        "description": meta["description"],
                        "filename": path.name,
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
        """Install an extension from the library to user dir.

        Args:
            ext_type: "command" or "agent"
            filename: YAML filename (e.g., "context7.yaml")

        Returns:
            The installed filename.

        Raises:
            ValueError: If filename is invalid or not found in library.
        """
        if ext_type not in ("command", "agent"):
            raise ValueError(f"Invalid extension type: {ext_type}")

        # Security: sanitize filename
        safe_name = Path(filename).name
        if safe_name != filename or not safe_name.endswith((".yaml", ".yml")):
            raise ValueError(f"Invalid filename: {filename}")

        subdir = "commands" if ext_type == "command" else "agents"
        source = self._library_dir / subdir / safe_name
        if not source.is_file():
            raise ValueError(f"Extension not found in library: {subdir}/{safe_name}")

        dest_dir = self._user_dir / subdir
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / safe_name
        dest.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")

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
