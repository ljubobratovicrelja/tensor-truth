"""Persistent JSON-backed metadata store for document scopes."""

import json
import logging
import tempfile
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class MetadataStore:
    """JSON-backed metadata store for a scope directory.

    Stores per-document metadata in ``{scope_dir}/metadata.json``.
    Supports atomic writes to avoid corruption.
    """

    def __init__(self, scope_dir: Path):
        self.scope_dir = Path(scope_dir)
        self._path = self.scope_dir / "metadata.json"
        self._data: Dict[str, Dict] = {}
        self._dirty = False

    def load(self) -> Dict[str, Dict]:
        """Load metadata from disk. Returns empty dict if file missing."""
        if self._path.exists():
            try:
                self._data = json.loads(self._path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Failed to load metadata from {self._path}: {e}")
                self._data = {}
        else:
            self._data = {}
        self._dirty = False
        return self._data

    def save(self) -> None:
        """Atomically write metadata to disk."""
        self.scope_dir.mkdir(parents=True, exist_ok=True)
        try:
            fd, tmp = tempfile.mkstemp(
                dir=str(self.scope_dir), suffix=".tmp", prefix="metadata_"
            )
            try:
                with open(fd, "w", encoding="utf-8") as f:
                    json.dump(self._data, f, indent=2, ensure_ascii=False)
                Path(tmp).replace(self._path)
            except BaseException:
                Path(tmp).unlink(missing_ok=True)
                raise
        except OSError as e:
            logger.error(f"Failed to save metadata to {self._path}: {e}")
            raise
        self._dirty = False

    def get(self, doc_id: str) -> Optional[Dict]:
        """Get metadata for a document."""
        return self._data.get(doc_id)

    def set(self, doc_id: str, metadata: Dict) -> None:
        """Set metadata for a document and mark dirty."""
        self._data[doc_id] = metadata
        self._dirty = True

    def delete(self, doc_id: str) -> None:
        """Remove metadata for a document."""
        if doc_id in self._data:
            del self._data[doc_id]
            self._dirty = True

    def update_from(self, cache: Dict[str, Dict]) -> None:
        """Merge a metadata cache dict into the store."""
        if cache:
            self._data.update(cache)
            self._dirty = True

    def persist_if_dirty(self) -> None:
        """Save only if there are unsaved changes."""
        if self._dirty:
            self.save()

    @property
    def is_dirty(self) -> bool:
        return self._dirty
