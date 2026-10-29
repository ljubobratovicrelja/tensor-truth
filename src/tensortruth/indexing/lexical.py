"""SQLite FTS5 lexical index that lives beside each vector index.

Dense retrieval reliably answers paraphrase questions but loses exact-term
queries (formulas, API names, identifiers) where a literal match is what the
user meant. A lexical index next to ``chroma.sqlite3`` gives that second
signal without a new dependency or another embedding pass, and without
invalidating indexes built before fusion existed: it can be backfilled from
``docstore.json`` (see ``scripts/build_lexical_index.py``).

Every failure mode degrades to "no lexical hits" instead of raising, because
retrieval must keep working when the auxiliary index is missing or stale.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)

LEXICAL_DB_FILENAME = "lexical.sqlite3"

# FTS5 treats these as syntax. They are stripped before building the MATCH
# expression so arbitrary user text can never produce a malformed query.
_FTS5_SPECIAL = re.compile(r'["*:^()\-]')
_WHITESPACE = re.compile(r"\s+")

# unicode61 has no stopword list, so queries made only of function words would
# match nearly every chunk and swamp the ranking. Dropping them also keeps a
# literal "and"/"or" from being read as output-affecting noise.
_STOPWORDS = frozenset("""
    a an and are as at be been being but by do does did for from had has have
    he her his i if in into is it its just me my no nor not of on or our out
    so than that the their them then there these they this those to too up us
    was we were what when where which who why will with would you your
    """.split())

_SCHEMA = (
    "CREATE VIRTUAL TABLE chunks USING fts5("
    "node_id UNINDEXED, doc_id UNINDEXED, text, "
    "tokenize='unicode61 remove_diacritics 2')"
)


def build_match_query(query: str) -> Optional[str]:
    """Turn free text into an FTS5 MATCH expression, or None when unusable.

    The whole phrase is OR'd with the disjunction of its tokens. A conjunction
    would require every word of a natural-language question to appear in one
    chunk, which almost never holds ("what does momentum change about the
    update direction" matched one chunk in 3689); bm25 still ranks chunks that
    share more of the query higher, so the disjunction is the useful default.
    """
    if not query:
        return None
    cleaned = _FTS5_SPECIAL.sub(" ", query).strip()
    tokens = [
        t for t in _WHITESPACE.split(cleaned) if t and t.lower() not in _STOPWORDS
    ]
    if not tokens:
        return None

    phrase = " ".join(tokens)
    if len(tokens) == 1:
        return f'"{phrase}"'
    disjunction = " OR ".join(f'"{t}"' for t in tokens)
    return f'"{phrase}" OR ({disjunction})'


class LexicalIndex:
    """FTS5 full-text index for one index directory."""

    def __init__(self, index_dir: str | Path) -> None:
        self.index_dir = Path(index_dir)
        self.db_path = self.index_dir / LEXICAL_DB_FILENAME

    # -- lifecycle -------------------------------------------------------

    def exists(self) -> bool:
        """True when a lexical database file is present in the index dir."""
        return self.db_path.is_file()

    def _connect(self) -> sqlite3.Connection:
        # A fresh connection per call keeps this safe from any thread the
        # parent multi-index retriever happens to use.
        return sqlite3.connect(str(self.db_path), timeout=5.0)

    def build(
        self,
        node_items: Iterable[tuple[str, str, str]],
        embedding_model: Optional[str] = None,
    ) -> int:
        """(Re)create the index from ``(node_id, text, doc_id)`` triples.

        Returns the number of chunks written. Dropping the old table makes a
        rebuild idempotent, which is what the backfill script relies on.
        """
        items = [
            (str(node_id), text or "", str(doc_id or ""))
            for node_id, text, doc_id in node_items
            if str(node_id)
        ]
        self.index_dir.mkdir(parents=True, exist_ok=True)

        conn = self._connect()
        try:
            with conn:
                conn.execute("DROP TABLE IF EXISTS chunks")
                conn.execute(_SCHEMA)
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS meta "
                    "(key TEXT PRIMARY KEY, value TEXT)"
                )
                conn.executemany(
                    "INSERT INTO chunks (node_id, text, doc_id) VALUES (?, ?, ?)",
                    items,
                )
                conn.execute(
                    "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                    ("built_at", datetime.now(timezone.utc).isoformat()),
                )
                conn.execute(
                    "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                    ("embedding_model", embedding_model or ""),
                )
        finally:
            conn.close()

        logger.info(f"Built lexical index at {self.db_path} with {len(items)} chunks")
        return len(items)

    @classmethod
    def build_from_store(
        cls,
        index_dir: str | Path,
        nodes: Iterable[Any],
        embedding_model: Optional[str] = None,
    ) -> "LexicalIndex":
        """Build from llama_index nodes, keeping only leaves.

        Parent and section nodes repeat their children's text, so indexing
        them would rank the same content twice and inflate parent hits.
        """
        from llama_index.core.node_parser import get_leaf_nodes

        node_list = list(nodes)
        leaves = get_leaf_nodes(node_list) if node_list else []
        index = cls(index_dir)
        items = (
            (node.node_id, _node_text(node), _node_doc_id(node)) for node in leaves
        )
        index.build(items, embedding_model=embedding_model)
        return index

    # -- reads -----------------------------------------------------------

    def count(self) -> int:
        """Number of indexed chunks; 0 when the database is absent or broken."""
        if not self.exists():
            return 0
        try:
            conn = self._connect()
            try:
                return int(conn.execute("SELECT count(*) FROM chunks").fetchone()[0])
            finally:
                conn.close()
        except sqlite3.Error as exc:
            logger.warning(f"Could not count lexical chunks at {self.db_path}: {exc}")
            return 0

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Return ``(node_id, score)`` pairs, best first, score in (0, 1]."""
        match_query = build_match_query(query)
        if match_query is None or not self.exists() or top_k <= 0:
            return []

        try:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT node_id, bm25(chunks) AS rank FROM chunks "
                    "WHERE chunks MATCH ? ORDER BY rank LIMIT ?",
                    (match_query, int(top_k)),
                ).fetchall()
            finally:
                conn.close()
        except sqlite3.Error as exc:
            logger.warning(f"Lexical search failed for {query!r}: {exc}")
            return []

        # bm25() is negative and lower is better in SQLite, so negate it into
        # a non-negative distance before mapping onto (0, 1].
        return [
            (str(node_id), 1.0 / (1.0 + max(0.0, -float(rank))))
            for node_id, rank in rows
        ]

    def get_chunk(self, node_id: str) -> Optional[tuple[str, str]]:
        """Return ``(text, doc_id)`` for a chunk, or None if unknown.

        Lets retrievers reconstruct a node when the docstore lookup misses.
        """
        if not self.exists():
            return None
        try:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT text, doc_id FROM chunks WHERE node_id = ? LIMIT 1",
                    (str(node_id),),
                ).fetchone()
            finally:
                conn.close()
        except sqlite3.Error as exc:
            logger.warning(f"Lexical lookup failed for {node_id}: {exc}")
            return None
        return (row[0], row[1]) if row else None


def _node_text(node: Any) -> str:
    getter = getattr(node, "get_content", None)
    if callable(getter):
        try:
            return getter(metadata_mode="none") or ""
        except TypeError:
            return getter() or ""
    return getattr(node, "text", "") or ""


def _node_doc_id(node: Any) -> str:
    ref_doc_id = getattr(node, "ref_doc_id", None)
    if ref_doc_id:
        return str(ref_doc_id)
    metadata = getattr(node, "metadata", None) or {}
    for key in ("doc_id", "file_name", "display_name"):
        if metadata.get(key):
            return str(metadata[key])
    return node.node_id
