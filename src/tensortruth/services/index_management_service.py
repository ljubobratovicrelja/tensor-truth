"""Index management service for sources CRUD, fetching, and building.

Wraps existing CLI logic into a programmatic API for the web frontend.
"""

import functools
import json
import logging
import os
import shutil
import sqlite3
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from tensortruth.cli_paths import (
    get_base_indexes_dir,
    get_library_docs_dir,
    get_sources_config_path,
)
from tensortruth.indexing.metadata import (
    read_index_metadata,
    sanitize_model_id,
)
from tensortruth.utils.sources_config import load_user_sources, update_sources_config

logger = logging.getLogger(__name__)


class IndexManagementService:
    """Service for managing documentation sources, fetching docs, and building indexes."""

    def __init__(self) -> None:
        self._sources_config_path = get_sources_config_path()
        self._library_docs_dir = get_library_docs_dir()
        self._indexes_dir = get_base_indexes_dir()

    # ── Sources CRUD ──────────────────────────────────────────────────

    def get_sources(self) -> Dict[str, Any]:
        """Load sources.json, returning empty structure if not found."""
        try:
            return load_user_sources(self._sources_config_path)
        except IOError:
            return {"libraries": {}, "papers": {}, "books": {}}

    def add_source(self, source_type: str, name: str, config: Dict[str, Any]) -> None:
        """Add a new source entry. Raises ValueError if it already exists."""
        sources = self.get_sources()
        section = sources.get(source_type, {})
        if name in section:
            raise ValueError(
                f"Source '{name}' already exists in '{source_type}'. "
                f"Use update instead."
            )
        update_sources_config(self._sources_config_path, source_type, name, config)

    def update_source(
        self, source_type: str, name: str, config: Dict[str, Any]
    ) -> None:
        """Update an existing source entry. Raises ValueError if not found."""
        sources = self.get_sources()
        section = sources.get(source_type, {})
        if name not in section:
            raise ValueError(
                f"Source '{name}' not found in '{source_type}'. Use add instead."
            )
        update_sources_config(self._sources_config_path, source_type, name, config)

    def delete_source(self, source_type: str, name: str) -> None:
        """Delete a source entry from sources.json."""
        sources = self.get_sources()
        section = sources.get(source_type, {})
        if name not in section:
            raise ValueError(f"Source '{name}' not found in '{source_type}'.")
        del sources[source_type][name]
        os.makedirs(os.path.dirname(self._sources_config_path), exist_ok=True)
        with open(self._sources_config_path, "w", encoding="utf-8") as f:
            json.dump(sources, f, indent=2)
        logger.info(f"Deleted source {source_type}/{name}")

    # ── Modules overview ──────────────────────────────────────────────

    def get_modules_overview(
        self, embedding_model_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Build a merged overview from indexes + sources.json + library_docs.

        Primary source of truth is the indexes directory — modules with built
        indexes are always shown. Sources.json and library_docs entries that
        don't yet have an index are also included.

        Args:
            embedding_model_id: Sanitized embedding model ID to check indexes for.
                If None, uses the configured default model.
        """
        if embedding_model_id is None:
            try:
                from tensortruth.app_utils.config_loader import load_config

                config = load_config()
                embedding_model_id = sanitize_model_id(
                    config.rag.default_embedding_model
                )
            except Exception:
                embedding_model_id = "bge-m3"

        sources = self.get_sources()
        modules: List[Dict[str, Any]] = []
        seen: set[str] = set()

        # 1. Scan built indexes (primary source of truth)
        model_indexes_dir = Path(self._indexes_dir) / embedding_model_id
        if model_indexes_dir.is_dir():
            for idx_path in sorted(model_indexes_dir.iterdir()):
                if not idx_path.is_dir():
                    continue
                if not (idx_path / "chroma.sqlite3").is_file():
                    continue
                name = idx_path.name
                source_type, source_config = self._find_source_config(
                    name, sources, model_indexes_dir
                )
                overview = self._build_module_overview(
                    name, source_type, source_config, embedding_model_id
                )
                modules.append(overview)
                seen.add(name)

        # 2. Add sources.json entries that don't have a built index yet
        for source_type, entries in [
            ("libraries", sources.get("libraries", {})),
            ("papers", sources.get("papers", {})),
            ("books", sources.get("books", {})),
        ]:
            for name, config in entries.items():
                if name not in seen:
                    overview = self._build_module_overview(
                        name, source_type, config, embedding_model_id
                    )
                    modules.append(overview)
                    seen.add(name)

        # 3. Check library_docs/ for orphaned doc directories (skip empty ones)
        docs_dir = Path(self._library_docs_dir)
        if docs_dir.is_dir():
            for entry in sorted(docs_dir.iterdir()):
                if not entry.is_dir():
                    continue
                # Skip empty directories (leftovers from aborted fetches)
                if not any(entry.iterdir()):
                    continue
                dir_name = entry.name
                for prefix, stype in [
                    ("library_", "libraries"),
                    ("papers_", "papers"),
                    ("book_", "books"),
                ]:
                    if dir_name.startswith(prefix):
                        mod_name = dir_name[len(prefix) :]
                        if mod_name not in seen:
                            overview = self._build_module_overview(
                                mod_name, stype, {}, embedding_model_id
                            )
                            modules.append(overview)
                            seen.add(mod_name)
                        break

        return modules

    def _find_source_config(
        self,
        name: str,
        sources: Dict[str, Any],
        model_indexes_dir: Optional[Path] = None,
    ) -> tuple[str, Dict[str, Any]]:
        """Find a module's source type and config from sources.json.

        Returns (source_type, config) or infers type from index metadata.
        """
        for source_type in ("libraries", "papers", "books"):
            section = sources.get(source_type, {})
            if name in section:
                return source_type, section[name]

        # Not in sources.json — try to infer type from index metadata
        if model_indexes_dir is not None:
            try:
                from tensortruth.app_utils.helpers import get_module_display_name

                _, doc_type, _, _ = get_module_display_name(model_indexes_dir, name)
                # Map doc_type back to source_type
                doc_type_to_source = {
                    "book": "books",
                    "paper": "papers",
                    "library_doc": "libraries",
                }
                return doc_type_to_source.get(doc_type, "libraries"), {}
            except Exception:
                pass
        return "libraries", {}

    def _build_module_overview(
        self,
        name: str,
        source_type: str,
        config: Dict[str, Any],
        embedding_model_id: str,
    ) -> Dict[str, Any]:
        """Build a single module overview dict."""
        in_sources_config = bool(config)

        # Determine doc directory — check both with and without type prefix
        # Index names already include prefix (e.g., "library_pytorch_2.9"),
        # so try the name directly first, then with prefix prepended.
        doc_dir = self._find_doc_dir(name, source_type)

        has_fetched = (
            doc_dir.is_dir() and any(doc_dir.iterdir()) if doc_dir.is_dir() else False
        )
        doc_size_mb = None
        if has_fetched:
            total = sum(f.stat().st_size for f in doc_dir.rglob("*") if f.is_file())
            doc_size_mb = round(total / (1024 * 1024), 2)

        # Check index
        index_dir = Path(self._indexes_dir) / embedding_model_id / name
        has_index = (index_dir / "chroma.sqlite3").is_file()
        index_meta = read_index_metadata(index_dir) if has_index else None

        # Compute index size
        index_size_mb = None
        if has_index:
            total_bytes = sum(
                f.stat().st_size for f in index_dir.rglob("*") if f.is_file()
            )
            index_size_mb = round(total_bytes / (1024 * 1024), 1)

        # If no metadata file, compute basic stats from the index itself
        if has_index and not index_meta:
            index_meta = self._compute_index_stats(index_dir)

        # Build display name: prefer source config, then index metadata, then fallback
        display_name = config.get("display_name") or config.get("title")
        doc_type = ""
        if not display_name and has_index:
            try:
                from tensortruth.app_utils.helpers import get_module_display_name

                idx_display, idx_doc_type, _, _ = get_module_display_name(
                    Path(self._indexes_dir) / embedding_model_id, name
                )
                display_name = idx_display
                doc_type = idx_doc_type
            except Exception:
                pass
        if not display_name:
            display_name = name.replace("_", " ").title()

        return {
            "name": name,
            "source_type": source_type,
            "display_name": display_name,
            "doc_type": doc_type,
            "source_config": config,
            "in_sources_config": in_sources_config,
            "has_fetched_docs": has_fetched,
            "doc_dir_size_mb": doc_size_mb,
            "has_index": has_index,
            "index_size_mb": index_size_mb,
            "index_metadata": index_meta,
        }

    def _find_doc_dir(self, name: str, source_type: str) -> Path:
        """Find the docs directory for a module.

        Handles both bare names (from sources.json, e.g., "pytorch")
        and prefixed names (from index dirs, e.g., "library_pytorch_2.9").
        """
        docs_base = Path(self._library_docs_dir)
        doc_type_map = {
            "libraries": "library",
            "papers": "papers",
            "books": "book",
        }
        prefix = doc_type_map.get(source_type, source_type)

        # Try direct name first (already has prefix, e.g., "library_pytorch_2.9")
        direct = docs_base / name
        if direct.is_dir():
            return direct

        # Try with prefix prepended (bare name from sources.json, e.g., "pytorch")
        prefixed = docs_base / f"{prefix}_{name}"
        return prefixed

    @staticmethod
    def _compute_index_stats(index_dir: Path) -> Dict[str, Any]:
        """Compute basic stats from ChromaDB and docstore when metadata file is missing."""
        stats: Dict[str, Any] = {}
        chroma_path = index_dir / "chroma.sqlite3"
        if chroma_path.is_file():
            try:
                conn = sqlite3.connect(str(chroma_path))
                cur = conn.cursor()
                cur.execute("SELECT COUNT(*) FROM embeddings")
                stats["num_embeddings"] = cur.fetchone()[0]
                cur.execute("SELECT dimension FROM collections LIMIT 1")
                row = cur.fetchone()
                if row:
                    stats["embedding_dimension"] = row[0]
                conn.close()
            except Exception:
                pass

        docstore_path = index_dir / "docstore.json"
        if docstore_path.is_file():
            try:
                with open(docstore_path, "r", encoding="utf-8") as f:
                    docstore = json.load(f)
                if "docstore/data" in docstore:
                    stats["num_documents"] = len(docstore["docstore/data"])
                if "docstore/ref_doc_info" in docstore:
                    stats["num_source_docs"] = len(docstore["docstore/ref_doc_info"])
            except Exception:
                pass

        return stats if stats else None  # type: ignore[return-value]

    # ── Document browsing ─────────────────────────────────────────────

    def list_source_documents(
        self, embedding_model_id: str, module_name: str
    ) -> List[Dict[str, Any]]:
        """List source documents (ref_doc_info) for a module's index."""
        index_dir = Path(self._indexes_dir) / embedding_model_id / module_name
        docstore_path = index_dir / "docstore.json"
        if not docstore_path.is_file():
            raise ValueError(f"Docstore not found: {embedding_model_id}/{module_name}")

        docstore = _load_docstore(str(docstore_path))
        ref_doc_info = docstore.get("docstore/ref_doc_info", {})

        documents: List[Dict[str, Any]] = []
        for doc_id, info in ref_doc_info.items():
            meta = info.get("metadata", {})
            node_ids = info.get("node_ids", [])
            documents.append(
                {
                    "doc_id": doc_id,
                    "file_name": meta.get("file_name", ""),
                    "title": meta.get("display_name") or meta.get("title"),
                    "display_name": meta.get("display_name"),
                    "doc_type": meta.get("doc_type"),
                    "source_url": meta.get("source_url"),
                    "file_size": meta.get("file_size"),
                    "chunk_count": len(node_ids),
                }
            )

        documents.sort(key=lambda d: d.get("file_name", ""))
        return documents

    def get_document_content(
        self, embedding_model_id: str, module_name: str, doc_id: str
    ) -> Dict[str, Any]:
        """Reconstruct document text from top-level chunks with deduplication."""
        index_dir = Path(self._indexes_dir) / embedding_model_id / module_name
        docstore_path = index_dir / "docstore.json"
        if not docstore_path.is_file():
            raise ValueError(f"Docstore not found: {embedding_model_id}/{module_name}")

        docstore = _load_docstore(str(docstore_path))
        ref_doc_info = docstore.get("docstore/ref_doc_info", {})
        data = docstore.get("docstore/data", {})

        if doc_id not in ref_doc_info:
            raise ValueError(f"Document not found: {doc_id}")

        doc_info = ref_doc_info[doc_id]
        node_ids = doc_info.get("node_ids", [])
        metadata = doc_info.get("metadata", {})

        # Build set of all IDs referenced as children by other nodes
        child_ids: set[str] = set()
        for nid in node_ids:
            node = data.get(nid)
            if node is None:
                continue
            for ref in node.get("__data__", {}).get("relationships", {}).get("5", []):
                cid = ref.get("node_id")
                if cid:
                    child_ids.add(cid)

        # Top-level chunks: those NOT referenced as a child of any other node
        top_chunks: List[Dict[str, Any]] = []
        for nid in node_ids:
            if nid in child_ids:
                continue
            node = data.get(nid)
            if node is None:
                continue
            nd = node.get("__data__", {})
            top_chunks.append(
                {
                    "chunk_id": nid,
                    "start": nd.get("start_char_idx", 0),
                    "end": nd.get("end_char_idx", 0),
                    "text": nd.get("text", ""),
                }
            )

        # Sort by start position
        top_chunks.sort(key=lambda c: c["start"])

        # Reconstruct text with overlap deduplication, collecting all levels
        text_parts: List[str] = []
        all_chunks: List[Dict[str, Any]] = []
        current_pos = 0
        output_len = 0

        for chunk in top_chunks:
            chunk_text = chunk["text"]
            skip = 0
            if current_pos > 0 and chunk["start"] < current_pos:
                skip = current_pos - chunk["start"]
                if skip >= len(chunk_text):
                    continue
                chunk_text = chunk_text[skip:]

            recon_start = output_len
            text_parts.append(chunk_text)
            output_len += len(chunk_text)
            recon_end = output_len

            # Mapping: recon_pos = orig_pos + recon_offset
            effective_orig_start = chunk["start"] + skip
            recon_offset = recon_start - effective_orig_start

            all_chunks.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "start": recon_start,
                    "end": recon_end,
                    "level": 0,
                }
            )

            # Recursively collect child chunks at deeper levels.
            # Children store positions relative to their parent's full
            # text, so parent_doc_start is the parent's original start.
            _collect_children(
                data,
                chunk["chunk_id"],
                parent_doc_start=chunk["start"],
                recon_offset=recon_offset,
                effective_orig_start=effective_orig_start,
                level=1,
                out=all_chunks,
            )

            current_pos = max(current_pos, chunk["end"])

        return {
            "doc_id": doc_id,
            "metadata": metadata,
            "text": "".join(text_parts),
            "chunks": all_chunks,
        }

    # ── Fetch callable ────────────────────────────────────────────────

    def create_fetch_callable(
        self,
        source_type: str,
        name: str,
        *,
        workers: int = 20,
        converter: str = "marker",
        output_format: str = "markdown",
        enable_cleanup: bool = False,
        min_size: int = 0,
        pages_per_chunk: int = 15,
    ) -> Callable:
        """Create a callable suitable for TaskRunner.submit().

        Returns a function that accepts progress_callback(stage, current, total).
        """
        sources = self.get_sources()
        all_entries = {
            **sources.get("libraries", {}),
            **sources.get("papers", {}),
            **sources.get("books", {}),
        }
        if name not in all_entries:
            raise ValueError(f"Source '{name}' not found in sources.json")
        config = all_entries[name]
        library_docs_dir = self._library_docs_dir

        def _do_fetch(progress_callback: Callable) -> Dict[str, Any]:
            progress_callback("fetching", 0, 100)

            if source_type == "libraries":
                from tensortruth.scrapers.library import scrape_library

                result = scrape_library(
                    library_name=name,
                    config=config,
                    output_base_dir=library_docs_dir,
                    max_workers=workers,
                    output_format=output_format,
                    enable_cleanup=enable_cleanup,
                    min_size=min_size,
                )
            elif source_type == "papers":
                doc_type = config.get("type", "arxiv")
                if doc_type == "pdf_book":
                    from tensortruth.scrapers.book import fetch_book

                    result = fetch_book(
                        book_name=name,
                        book_config=config,
                        output_base_dir=library_docs_dir,
                        converter=converter,
                        pages_per_chunk=pages_per_chunk,
                    )
                else:
                    from tensortruth.scrapers.arxiv import fetch_paper_category

                    fetch_paper_category(
                        category_name=name,
                        category_config=config,
                        output_base_dir=library_docs_dir,
                        output_format=output_format,
                        converter=converter,
                    )
                    result = True
            elif source_type == "books":
                from tensortruth.scrapers.book import fetch_book

                result = fetch_book(
                    book_name=name,
                    book_config=config,
                    output_base_dir=library_docs_dir,
                    converter=converter,
                    pages_per_chunk=pages_per_chunk,
                )
            else:
                raise ValueError(f"Unknown source type: {source_type}")

            progress_callback("done", 100, 100)
            return {"success": bool(result), "source_type": source_type, "name": name}

        return _do_fetch

    # ── Build callable ────────────────────────────────────────────────

    def create_build_callable(
        self,
        module_name: str,
        *,
        chunking_strategy: str = "structure",
        chunk_sizes: Optional[List[int]] = None,
        chunk_overlap: int = 64,
        embedding_model: Optional[str] = None,
        semantic_buffer_size: int = 1,
        semantic_breakpoint_threshold: int = 95,
    ) -> Callable:
        """Create a build callable suitable for TaskRunner.submit()."""
        sources = self.get_sources()
        library_docs_dir = self._library_docs_dir
        indexes_dir = self._indexes_dir

        from tensortruth.utils.validation import validate_module_for_build

        validate_module_for_build(module_name, library_docs_dir, sources)

        def _do_build(progress_callback: Callable) -> Dict[str, Any]:
            from tensortruth.indexing.builder import build_module

            result = build_module(
                module_name=module_name,
                library_docs_dir=library_docs_dir,
                indexes_dir=indexes_dir,
                sources_config=sources,
                chunk_sizes=chunk_sizes,
                chunk_overlap=chunk_overlap,
                chunking_strategy=chunking_strategy,
                semantic_buffer_size=semantic_buffer_size,
                semantic_breakpoint_threshold=semantic_breakpoint_threshold,
                embedding_model=embedding_model,
                progress_callback=progress_callback,
            )
            return {"success": result, "module_name": module_name}

        return _do_build

    # ── Index operations ──────────────────────────────────────────────

    def get_index_metadata(
        self, embedding_model_id: str, module_name: str
    ) -> Optional[Dict[str, Any]]:
        """Read index metadata for a specific module + model combination."""
        index_dir = Path(self._indexes_dir) / embedding_model_id / module_name
        return read_index_metadata(index_dir)

    def delete_index(self, embedding_model_id: str, module_name: str) -> None:
        """Delete a built index from disk."""
        index_dir = Path(self._indexes_dir) / embedding_model_id / module_name
        if not index_dir.is_dir():
            raise ValueError(f"Index not found: {embedding_model_id}/{module_name}")
        shutil.rmtree(index_dir)
        logger.info(f"Deleted index {embedding_model_id}/{module_name}")

    # ── Validation ────────────────────────────────────────────────────

    def validate_sources(self) -> Dict[str, Any]:
        """Validate sources against filesystem. Returns issues list."""
        issues: List[str] = []
        sources = self.get_sources()

        for source_type in ("libraries", "papers", "books"):
            for name in sources.get(source_type, {}):
                doc_type_map = {
                    "libraries": "library",
                    "papers": "papers",
                    "books": "book",
                }
                prefix = doc_type_map.get(source_type, source_type)
                doc_dir = Path(self._library_docs_dir) / f"{prefix}_{name}"
                if not doc_dir.is_dir():
                    issues.append(
                        f"{source_type}/{name}: docs directory not found at {doc_dir}"
                    )
                elif not any(doc_dir.iterdir()):
                    issues.append(f"{source_type}/{name}: docs directory is empty")

        return {"valid": len(issues) == 0, "issues": issues}


def _collect_children(
    data: dict,
    parent_id: str,
    parent_doc_start: int,
    recon_offset: int,
    effective_orig_start: int,
    level: int,
    out: list,
    max_level: int = 10,
) -> None:
    """Recursively collect child chunks, mapping positions to reconstructed text.

    Child chunks store ``start_char_idx`` / ``end_char_idx`` relative to their
    parent's text, so we translate them to document-level coordinates using
    ``parent_doc_start`` before applying the reconstruction offset.
    """
    if level > max_level:
        return
    node = data.get(parent_id)
    if node is None:
        return
    children_refs = node.get("__data__", {}).get("relationships", {}).get("5", [])
    if not children_refs:
        return

    for ref in children_refs:
        child_id = ref.get("node_id")
        if not child_id:
            continue
        child_node = data.get(child_id)
        if child_node is None:
            continue
        cd = child_node["__data__"]
        local_start = cd.get("start_char_idx", 0)
        local_end = cd.get("end_char_idx", 0)

        # Convert parent-relative positions to document-absolute
        doc_start = parent_doc_start + local_start
        doc_end = parent_doc_start + local_end

        # Skip chunks entirely within the trimmed overlap region
        if doc_end <= effective_orig_start:
            continue

        mapped_start = max(doc_start, effective_orig_start) + recon_offset
        mapped_end = doc_end + recon_offset

        if mapped_start >= mapped_end:
            continue

        out.append(
            {
                "chunk_id": child_id,
                "start": mapped_start,
                "end": mapped_end,
                "level": level,
            }
        )

        _collect_children(
            data,
            child_id,
            doc_start,
            recon_offset,
            effective_orig_start,
            level + 1,
            out,
            max_level,
        )


@functools.lru_cache(maxsize=2)
def _load_docstore(path: str) -> dict:
    """Load and cache a docstore.json. LRU(2) since users browse one module at a time."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
