"""Catalog of indexed knowledge modules with descriptions.

The orchestrator lists every indexed module in its system prompt so the agent
can decide which ones a question needs, then load them with ``load_modules``.
Listing 26 modules must not mean opening 26 Chroma databases on every turn, so
the catalog is read once (directly from each index's ``chroma.sqlite3``) and
cached until the indexes or ``sources.json`` change on disk.

Descriptions come from ``sources.json`` when the source group declares one
(papers and books) and are otherwise derived from the module's doc type,
library version and documentation root.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Group order in the prompt and in the modules API.
DOC_TYPE_SORT_ORDER: Dict[str, int] = {
    "book": 1,
    "paper": 2,
    "library_doc": 3,
}

# (name prefix in the index dir, section in sources.json)
_SOURCES_PREFIXES: Tuple[Tuple[str, str], ...] = (
    ("papers_", "papers"),
    ("book_", "books"),
    ("library_", "libraries"),
)

_METADATA_KEYS = (
    "display_name",
    "doc_type",
    "group_display_name",
    "book_display_name",
    "library_display_name",
)


@dataclass
class ModuleDescription:
    """Description of one indexed knowledge module.

    ``description`` and ``doc_count`` are optional so callers that only know a
    module's name/display name (and every older call site) keep working.
    """

    name: str
    display_name: str
    doc_type: str
    description: str = ""
    doc_count: int = 0


# Cache: (indexes dir, model_id, module count, newest mtime) -> catalog
_catalog_cache: Dict[Tuple[str, str, int, float], List[ModuleDescription]] = {}
_catalog_lock = threading.Lock()


def clear_module_catalog_cache() -> None:
    """Drop the catalog cache (tests, or after an index rebuild)."""
    with _catalog_lock:
        _catalog_cache.clear()


def _read_sources_config(sources_path: Path) -> Dict[str, Any]:
    """Load sources.json, returning {} when it is missing or malformed."""
    try:
        with open(sources_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception:
        logger.debug("Could not read sources config at %s", sources_path)
        return {}


def _sources_entry(name: str, sources: Dict[str, Any]) -> Dict[str, Any]:
    """Return the sources.json entry backing a module name, if any."""
    for prefix, section in _SOURCES_PREFIXES:
        if not name.startswith(prefix):
            continue
        group = sources.get(section)
        if not isinstance(group, dict):
            return {}
        entry = group.get(name[len(prefix) :])
        if isinstance(entry, dict):
            return entry
        return {}
    return {}


def _humanize(name: str) -> str:
    return name.replace("_", " ").replace("-", " ").strip().title()


def _derive_description(
    name: str,
    display_name: str,
    doc_type: str,
    entry: Dict[str, Any],
) -> str:
    """Build a one-line description when sources.json has none."""
    label = display_name or _humanize(name)
    if doc_type == "library_doc":
        version = entry.get("version")
        if version and str(version) not in label:
            label = f"{label} {version}"
        return f"{label} library documentation"
    if doc_type == "paper":
        return f"Research papers: {label}"
    if doc_type == "book":
        return f"Book: {label}"
    return label


def _read_index_metadata(index_path: Path) -> Tuple[str, str, int]:
    """Read display/document metadata for one index.

    Returns:
        (display_name, doc_type, doc_count). Raises on an unreadable index so
        the caller can decide how to degrade.
    """
    db_path = index_path / "chroma.sqlite3"
    uri = f"file:{db_path}?mode=ro"
    with sqlite3.connect(uri, uri=True, timeout=5) as conn:
        row = conn.execute(
            "SELECT key, string_value FROM embedding_metadata "
            "WHERE id = (SELECT id FROM embeddings LIMIT 1) "
            "AND key IN ({})".format(",".join("?" * len(_METADATA_KEYS))),
            _METADATA_KEYS,
        ).fetchall()
        metadata = {key: value for key, value in row}

        doc_count = conn.execute(
            "SELECT COUNT(DISTINCT string_value) FROM embedding_metadata "
            "WHERE key = 'doc_id'"
        ).fetchone()[0]
        if not doc_count:
            doc_count = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]

    doc_type = str(metadata.get("doc_type") or "unknown")
    display_name = (
        metadata.get("group_display_name")
        or metadata.get("book_display_name")
        or metadata.get("library_display_name")
        or metadata.get("display_name")
        or ""
    )
    return str(display_name), doc_type, int(doc_count or 0)


def _build_catalog(
    model_indexes_dir: Path,
    sources: Dict[str, Any],
) -> List[ModuleDescription]:
    """Build the catalog by walking the module directories once."""
    entries: List[ModuleDescription] = []
    if not model_indexes_dir.is_dir():
        return entries

    for path in sorted(model_indexes_dir.iterdir()):
        if not path.is_dir() or not (path / "chroma.sqlite3").exists():
            continue
        name = path.name
        try:
            display_name, doc_type, doc_count = _read_index_metadata(path)
        except Exception:
            logger.debug("Could not read index metadata for %s", name, exc_info=True)
            display_name, doc_type, doc_count = name, "unknown", 0

        source_entry = _sources_entry(name, sources)
        description = str(source_entry.get("description") or "").strip()
        if not description:
            description = _derive_description(
                name, display_name, doc_type, source_entry
            )

        entries.append(
            ModuleDescription(
                name=name,
                display_name=display_name or name,
                doc_type=doc_type,
                description=description,
                doc_count=doc_count,
            )
        )

    entries.sort(
        key=lambda m: (
            DOC_TYPE_SORT_ORDER.get(m.doc_type, 4),
            m.display_name.lower(),
        )
    )
    return entries


def _cache_signature(model_indexes_dir: Path, sources_path: Path) -> Tuple[float, int]:
    """Cheap fingerprint of the index dir + sources.json (stat only)."""
    newest = 0.0
    count = 0
    try:
        if sources_path.exists():
            newest = sources_path.stat().st_mtime
        if model_indexes_dir.is_dir():
            for path in model_indexes_dir.iterdir():
                db_path = path / "chroma.sqlite3"
                if db_path.exists():
                    count += 1
                    newest = max(newest, db_path.stat().st_mtime)
    except OSError:
        pass
    return newest, count


def load_module_catalog(
    config: Any,
    indexes_dir: Optional[Path] = None,
    sources_path: Optional[Path] = None,
) -> List[ModuleDescription]:
    """Return every indexed module for the configured embedding model.

    Cached on (model id, module count, newest index mtime) so repeat calls in
    the same turn — prompt build, cache identity, ``load_modules`` validation —
    are a dict lookup. Returns an empty list when the indexes cannot be read.

    Args:
        config: TensorTruthConfig instance (uses ``rag.default_embedding_model``).
        indexes_dir: Optional override for the indexes root (tests).
        sources_path: Optional override for sources.json (tests).
    """
    from tensortruth.app_utils.paths import get_indexes_dir, get_sources_config_path
    from tensortruth.indexing.metadata import sanitize_model_id

    try:
        model_id = sanitize_model_id(config.rag.default_embedding_model)
    except Exception:
        logger.debug("Could not resolve embedding model for module catalog")
        return []

    try:
        root = Path(indexes_dir) if indexes_dir is not None else Path(get_indexes_dir())
        model_indexes_dir = root / model_id
    except Exception:
        logger.debug("Could not resolve indexes directory for module catalog")
        return []
    if sources_path is not None:
        sources_file = Path(sources_path)
    else:
        try:
            sources_file = Path(get_sources_config_path())
        except Exception:
            sources_file = Path("/nonexistent/sources.json")
        if not sources_file.exists():
            # A fresh install has no user sources.json; the shipped catalog
            # descriptions are still the best source of one-liners.
            bundled = Path(__file__).resolve().parents[3] / "config" / "sources.json"
            if bundled.exists():
                sources_file = bundled

    newest, count = _cache_signature(model_indexes_dir, sources_file)
    # The directory itself is part of the key: two index roots can otherwise
    # share (model, count, mtime) and poison each other's cache.
    key = (str(model_indexes_dir), model_id, count, newest)

    with _catalog_lock:
        cached = _catalog_cache.get(key)
    if cached is not None:
        return cached

    try:
        sources = _read_sources_config(sources_file)
        entries = _build_catalog(model_indexes_dir, sources)
    except Exception:
        logger.warning("Failed to build module catalog", exc_info=True)
        entries = []

    with _catalog_lock:
        _catalog_cache.clear()
        _catalog_cache[key] = entries
    return entries


def describe_modules(
    modules: List[str],
    config: Any,
) -> List[ModuleDescription]:
    """Return catalog entries for ``modules``; unknown names keep bare names.

    Preserves the order of ``modules``.
    """
    if not modules:
        return []

    catalog = {entry.name: entry for entry in load_module_catalog(config)}
    descriptions: List[ModuleDescription] = []
    for name in modules:
        entry = catalog.get(name)
        if entry is not None:
            descriptions.append(entry)
        else:
            descriptions.append(
                ModuleDescription(name=name, display_name=name, doc_type="unknown")
            )
    return descriptions
