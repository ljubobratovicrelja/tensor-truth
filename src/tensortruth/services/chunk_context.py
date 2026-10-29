"""Context expansion over the docstore hierarchy that indexing already built.

Retrieval returns short leaf chunks (~500 chars), and a leaf that starts
mid-sentence or stops mid-section is often not enough for the orchestrator to
answer from. Re-running retrieval with a rewording is expensive and unreliable,
while the docstore already stores the full PARENT/CHILD/PREVIOUS/NEXT
hierarchy plus every node's text.

This service therefore treats expansion as a pointer walk over the loaded
docstores: no re-embedding, no reindex, no JSON reload from disk. The
orchestrator calls it through the expand_section / read_neighbours /
document_outline tools using the short node-id prefixes printed by rag_query.

Access pattern matters here: on llama_index's KVDocumentStore the ``docs``
property deserialises every node on each access (~150 ms for a 6k-node book),
while ``get_node`` deserialises one node in microseconds and the key list is a
plain dict copy. So the service keeps only ids plus an id->store map, and
fetches node bodies one at a time.
"""

from __future__ import annotations

import bisect
import logging
from typing import Any, Dict, Iterator, List, Optional, Tuple

from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import BaseNode, NodeRelationship

logger = logging.getLogger(__name__)

# Node ids are 32+ hex chars (uuid4 hex). Shorter prefixes than this collide
# often enough across a large module set that resolving them would be a guess.
MIN_PREFIX_LEN = 6

# Guard against a corrupted docstore turning a walk into an infinite loop.
_MAX_WALK = 64

# Ancestors above this depth add no useful heading context and only make the
# heading path noisy.
_MAX_HEADING_DEPTH = 8

_LABEL_PREVIOUS = "previous"
_LABEL_NEXT = "next"

# Metadata keys that carry a human-facing document name, in preference order.
_DOCUMENT_NAME_KEYS = ("display_name", "title", "book_display_name", "file_name")

# File-header lines the book/paper loaders prepend; they are document-level
# metadata, not section titles.
_HEADER_PREFIXES = ("title:", "book:", "authors:", "source:", "url:")


def _docstore_of(obj: Any) -> Optional[Any]:
    """Return the docstore hanging off a retriever/index object, if any.

    Different llama_index retriever wrappers expose it under different names:
    AutoMergingRetriever keeps ``_storage_context``, a fused/lexical retriever
    may keep ``storage_context`` or ``docstore`` directly. Unknown shapes are
    skipped rather than diagnosed, because a retrieval stack can be built by
    extensions we do not control.
    """
    for attr in ("_storage_context", "storage_context"):
        ctx = getattr(obj, attr, None)
        store = getattr(ctx, "docstore", None)
        if store is not None and _looks_like_docstore(store):
            return store
    store = getattr(obj, "docstore", None)
    if store is not None and _looks_like_docstore(store):
        return store
    return None


def _looks_like_docstore(obj: Any) -> bool:
    return callable(getattr(obj, "get_node", None))


def _child_retrievers(obj: Any) -> List[Any]:
    """Return nested retrievers of a composite retriever, if any.

    MultiIndexRetriever and RRFusionRetriever both hold a plain list under
    ``retrievers``; checking the attribute instead of the class keeps this
    working for the fusion retriever that may not exist in every install.
    """
    children = getattr(obj, "retrievers", None)
    if isinstance(children, (list, tuple)):
        return list(children)
    return []


def _iter_docstores(
    obj: Any, seen: Optional[set] = None, depth: int = 0
) -> Iterator[Any]:
    """Yield every distinct docstore reachable from ``obj``.

    Recursion is bounded and cycle-safe: composite retrievers can be shared
    between sessions, and an unknown import shape must never raise here.
    """
    if obj is None or depth > 6:
        return
    if seen is None:
        seen = set()
    if id(obj) in seen:
        return
    seen.add(id(obj))

    store = _docstore_of(obj)
    if store is not None:
        yield store
    for child in _child_retrievers(obj):
        yield from _iter_docstores(child, seen=seen, depth=depth + 1)


def _node_ids(store: Any) -> List[str]:
    """Node ids of a docstore without deserialising any node.

    The KV backend exposes the raw JSON map, whose keys are exactly the node
    ids; the ``docs`` property would rebuild every node object just to read
    its keys. Non-KV docstores fall back to the plain mapping.
    """
    kvstore = getattr(store, "_kvstore", None)
    collection = getattr(store, "_node_collection", None)
    if kvstore is not None and collection is not None:
        try:
            return list(kvstore.get_all(collection=collection).keys())
        except Exception as exc:  # pragma: no cover - backend specific
            logger.debug("KV key listing failed, falling back to docs: %s", exc)
    try:
        return list((getattr(store, "docs", None) or {}).keys())
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Could not list docstore nodes: %s", exc)
        return []


def _content_of(node: Optional[BaseNode]) -> str:
    if node is None:
        return ""
    try:
        return node.get_content() or ""
    except Exception:  # pragma: no cover - defensive for exotic node types
        return getattr(node, "text", "") or ""


def _section_title_from_text(text: str, limit: int = 90) -> str:
    """Title from a markdown heading line, or "" when the text is prose.

    Legacy indexes contain section nodes whose text is a whole chapter body,
    so taking the first line unconditionally would print a paragraph of PDF
    text as the heading. Only real heading lines qualify.
    """
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if not line.startswith("#"):
            return ""
        title = line.lstrip("#").strip()
        if title.lower().startswith(_HEADER_PREFIXES):
            return ""
        return title[:limit]
    return ""


def _related_id(node: BaseNode, rel: NodeRelationship) -> Optional[str]:
    rels = getattr(node, "relationships", None) or {}
    info = rels.get(rel)
    if info is None:
        return None
    return getattr(info, "node_id", None)


def _child_ids(node: BaseNode) -> List[str]:
    rels = getattr(node, "relationships", None) or {}
    children = rels.get(NodeRelationship.CHILD) or []
    return [
        getattr(info, "node_id", "")
        for info in children
        if getattr(info, "node_id", None)
    ]


def _ref_doc_id(node: BaseNode) -> Optional[str]:
    """Best-effort document identity for cross-document boundary checks."""
    doc_id = getattr(node, "ref_doc_id", None)
    if doc_id:
        return str(doc_id)
    source = _related_id(node, NodeRelationship.SOURCE)
    if source:
        return source
    metadata = getattr(node, "metadata", None) or {}
    value = metadata.get("doc_id") or metadata.get("ref_doc_id")
    return str(value) if value else None


def format_char_size(num_chars: int) -> str:
    """Render a character count the way the rag_query header shows it."""
    if num_chars >= 1000:
        return f"~{num_chars / 1000:.1f}k chars"
    return f"~{num_chars} chars"


class ChunkContextService:
    """Resolve node ids against live docstores and walk the chunk hierarchy.

    Built from the retrievers that are actually loaded for a session, so it
    never opens an index directory itself and stays consistent with whatever
    the session is currently searching.
    """

    def __init__(self, retrievers: Optional[List[BaseRetriever]] = None) -> None:
        self._docstores: List[Any] = []
        self._id_to_store: Dict[str, Any] = {}
        self._sorted_ids: List[str] = []
        self._collect(retrievers or [])

    # --- construction -----------------------------------------------------

    def _collect(self, retrievers: List[Any]) -> None:
        seen_stores = set()
        for retriever in retrievers:
            for store in _iter_docstores(retriever):
                if id(store) in seen_stores:
                    continue
                seen_stores.add(id(store))
                self._docstores.append(store)
        if not self._docstores:
            logger.info(
                "ChunkContextService built with no reachable docstore; "
                "context tools will report chunk ids as unknown"
            )
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        # Listing ids costs ~0.1 ms per store, so this is cheap enough to redo
        # on every tool call (which rebuilds the service) and never goes stale.
        id_to_store: Dict[str, Any] = {}
        for store in self._docstores:
            for node_id in _node_ids(store):
                # The same node in two stores (shared index handles) is fine;
                # keep the first so id->store stays deterministic.
                id_to_store.setdefault(node_id, store)
        self._id_to_store = id_to_store
        self._sorted_ids = sorted(id_to_store)

    def refresh(self) -> None:
        """Re-read the docstore id maps (call after loading new modules)."""
        self._rebuild_index()

    @property
    def docstore_count(self) -> int:
        return len(self._docstores)

    @property
    def node_count(self) -> int:
        return len(self._id_to_store)

    # --- lookup primitives ------------------------------------------------

    def _get(self, node_id: str) -> Optional[BaseNode]:
        if not node_id:
            return None
        store = self._id_to_store.get(node_id)
        if store is None:
            return None
        try:
            node = store.get_node(node_id)
            if node is not None:
                return node
        except (KeyError, ValueError):
            return None
        except Exception as exc:  # pragma: no cover - backend specific
            logger.debug("get_node failed for %s: %s", node_id, exc)
            return None
        return (getattr(store, "docs", None) or {}).get(node_id)

    def _parent(self, node: BaseNode) -> Optional[BaseNode]:
        parent_id = _related_id(node, NodeRelationship.PARENT)
        return self._get(parent_id) if parent_id else None

    def resolve(self, node_ref: str) -> Optional[BaseNode]:
        """Return the node a reference points at, or None.

        Accepts a full node id or a unique prefix of at least six characters,
        which is what rag_query prints. An ambiguous prefix returns None on
        purpose: guessing would silently expand the wrong document.
        """
        ref = (node_ref or "").strip().strip("[]")
        if not ref:
            return None

        node = self._get(ref)
        if node is not None:
            return node

        if len(ref) < MIN_PREFIX_LEN:
            return None
        start = bisect.bisect_left(self._sorted_ids, ref)
        matches: List[str] = []
        for node_id in self._sorted_ids[start:]:
            if not node_id.startswith(ref):
                break
            matches.append(node_id)
            if len(matches) > 1:
                return None
        if len(matches) != 1:
            return None
        return self._get(matches[0])

    # --- relationship walks -----------------------------------------------

    def _reverse_next_map(self, store: Any) -> Dict[str, str]:
        """Map node id -> id of the node that lists it as PREVIOUS.

        Some indexes (built with the old hierarchical parser) carry PREVIOUS
        links but almost no NEXT links; without this the agent could read
        backwards but never forwards on those indexes. Built per docstore, on
        first miss, and kept for the life of the service (one tool call).
        """
        cache = getattr(self, "_rev_next_cache", None)
        if cache is None:
            cache = self._rev_next_cache = {}
        key = id(store)
        cached = cache.get(key)
        if cached is not None:
            return cached
        reverse: Dict[str, str] = {}
        for node_id, node_store in self._id_to_store.items():
            if node_store is not store:
                continue
            node = self._get(node_id)
            if node is None:
                continue
            prev_id = _related_id(node, NodeRelationship.PREVIOUS)
            if prev_id and prev_id not in reverse:
                reverse[prev_id] = node_id
        cache[key] = reverse
        return reverse

    def _neighbour_in_doc(
        self, node: BaseNode, rel: NodeRelationship, doc_id: Optional[str]
    ) -> Optional[BaseNode]:
        """Follow one PREVIOUS/NEXT hop, refusing to leave the document.

        Indexing links the last chunk of a document to nothing, but merged or
        re-indexed stores can leave dangling links; a neighbour from another
        document would show the agent unrelated text under a plausible header.
        """
        target_id = _related_id(node, rel)
        candidate = self._get(target_id) if target_id else None
        if candidate is None and rel is NodeRelationship.NEXT:
            store = self._id_to_store.get(node.node_id)
            fallback_id = (
                self._reverse_next_map(store).get(node.node_id)
                if store is not None
                else None
            )
            candidate = self._get(fallback_id) if fallback_id else None
        if candidate is None:
            return None
        if doc_id is not None and _ref_doc_id(candidate) not in (None, doc_id):
            return None
        return candidate

    def neighbours(
        self, node_ref: str, before: int = 2, after: int = 2
    ) -> List[Tuple[str, BaseNode]]:
        """Return adjacent chunks as (direction, node), in reading order.

        Direction is ``"previous"`` or ``"next"``; the caller uses it only for
        the markers it prints.
        """
        node = self.resolve(node_ref)
        if node is None:
            return []

        doc_id = _ref_doc_id(node)
        seen = {node.node_id}
        collected: List[Tuple[str, BaseNode]] = []

        cursor = node
        for _ in range(max(0, min(int(before), _MAX_WALK))):
            prev_node = self._neighbour_in_doc(
                cursor, NodeRelationship.PREVIOUS, doc_id
            )
            if prev_node is None or prev_node.node_id in seen:
                break
            seen.add(prev_node.node_id)
            collected.append((_LABEL_PREVIOUS, prev_node))
            cursor = prev_node
        collected.reverse()

        cursor = node
        for _ in range(max(0, min(int(after), _MAX_WALK))):
            next_node = self._neighbour_in_doc(cursor, NodeRelationship.NEXT, doc_id)
            if next_node is None or next_node.node_id in seen:
                break
            seen.add(next_node.node_id)
            collected.append((_LABEL_NEXT, next_node))
            cursor = next_node

        return collected

    def ancestors(self, node: BaseNode) -> List[BaseNode]:
        """Return the PARENT chain, immediate parent first."""
        chain: List[BaseNode] = []
        seen = {node.node_id}
        cursor = node
        for _ in range(_MAX_WALK):
            parent = self._parent(cursor)
            if parent is None or parent.node_id in seen:
                break
            seen.add(parent.node_id)
            chain.append(parent)
            cursor = parent
        return chain

    def section(
        self, node_ref: str, max_chars: int = 6000
    ) -> Tuple[Optional[BaseNode], List[BaseNode]]:
        """Return (largest enclosing ancestor that fits, ancestors visited).

        Walking stops at the first ancestor that would exceed ``max_chars``:
        parents only grow, so nothing above it can fit either. When the chunk
        itself is already over the cap there is nothing worth returning and
        the caller falls back to the chunk it already has.
        """
        node = self.resolve(node_ref)
        if node is None:
            return None, []
        if len(_content_of(node)) > max_chars:
            return None, []

        best = node
        visited: List[BaseNode] = []
        seen = {node.node_id}
        cursor = node
        for _ in range(_MAX_WALK):
            parent = self._parent(cursor)
            if parent is None or parent.node_id in seen:
                break
            seen.add(parent.node_id)
            visited.append(parent)
            if len(_content_of(parent)) > max_chars:
                break
            best = parent
            cursor = parent
        return best, visited

    # --- headings and outline ---------------------------------------------

    def heading_path(self, node: Optional[BaseNode]) -> str:
        """Human heading path for a node.

        The structure-aware parser writes ``heading_path`` into metadata; for
        indexes built by the older hierarchical parser the path is rebuilt
        from the ancestors' markdown headings, and finally from the document
        name.
        """
        if node is None:
            return ""
        metadata = getattr(node, "metadata", None) or {}
        explicit = metadata.get("heading_path")
        if explicit:
            return str(explicit)

        chain = self.ancestors(node)
        # With more than one ancestor the topmost is the document node: its
        # first line is the file header, not a section title, so the document
        # name is the better fallback there.
        if len(chain) > 1:
            chain = chain[:-1]

        # An ancestor with an explicit heading_path already names the whole
        # chain down to itself; only sections below it still need appending.
        parts: List[str] = []
        titles: List[str] = []
        for ancestor in reversed(chain[:_MAX_HEADING_DEPTH]):
            ancestor_meta = getattr(ancestor, "metadata", None) or {}
            explicit_ancestor = ancestor_meta.get("heading_path")
            if explicit_ancestor:
                parts = [p for p in str(explicit_ancestor).split(" > ") if p]
                titles = []
                continue
            title = str(ancestor_meta.get("section_title") or "") or (
                _section_title_from_text(_content_of(ancestor))
            )
            if title and title not in titles:
                titles.append(title)
        if parts or titles:
            return " > ".join(parts + titles)
        return self.document_name(node) or "Unknown document"

    def document_name(self, node: Optional[BaseNode]) -> str:
        """Document label for headers, independent of the heading chain."""
        if node is None:
            return ""
        metadata = getattr(node, "metadata", None) or {}
        for key in _DOCUMENT_NAME_KEYS:
            value = metadata.get(key)
            if value:
                return str(value)
        root = self._document_root(node)
        value = (getattr(root, "metadata", None) or {}).get("display_name")
        return str(value) if value else ""

    def _document_root(self, node: BaseNode) -> BaseNode:
        chain = self.ancestors(node)
        return chain[-1] if chain else node

    def _collect_sections(self, node: BaseNode, depth: int = 0) -> List[BaseNode]:
        """Return the lowest internal level under ``node`` (its sections).

        That level is a section in both layouts we have: the legacy
        HierarchicalNodeParser (document/section -> leaves) and the structure
        parser (document -> section -> leaves). Descending while the children
        are still internal keeps this correct for a deeper hierarchy too.
        """
        if depth > _MAX_HEADING_DEPTH:
            return []
        children = [self._get(child_id) for child_id in _child_ids(node)]
        children = [child for child in children if child is not None]
        if not children:
            return []
        if not any(_child_ids(child) for child in children):
            return [node]
        sections: List[BaseNode] = []
        for child in children:
            sections.extend(self._collect_sections(child, depth + 1))
        return sections

    def child_refs(self, node: BaseNode) -> List[str]:
        """8-char prefixes of a node's direct children, in order.

        The tools print these so the agent sees which chunks a section
        expansion actually covered before spending context on it.
        """
        return [child_id[:8] for child_id in _child_ids(node)]

    def outline_entries(
        self, node_ref: str, max_entries: int = 40
    ) -> List[Tuple[int, str, str, int]]:
        """List the document's sections as (index, title, node_id, chars).

        The node id is returned so document_outline can hand the agent a
        reference it can immediately pass to expand_section; outline() is the
        title-only view described by the service contract.
        """
        node = self.resolve(node_ref)
        if node is None:
            return []
        root = self._document_root(node)

        sections = self._collect_sections(root)
        if not sections and _child_ids(root):
            # The root has children that are all leaves: the root itself is
            # the only section level this document has.
            sections = [root]

        entries: List[Tuple[int, str, str, int]] = []
        for index, section in enumerate(sections[:max_entries], 1):
            text = _content_of(section)
            title = str(
                (getattr(section, "metadata", None) or {}).get("section_title") or ""
            ) or _section_title_from_text(text)
            if not title:
                # Legacy chapter files have no headings; their file name still
                # tells the agent which part of the book it is looking at.
                title = str(
                    (getattr(section, "metadata", None) or {}).get("file_name") or ""
                )
            entries.append(
                (index, title or f"Section {index}", section.node_id, len(text))
            )
        return entries

    def outline(self, node_ref: str, max_entries: int = 40) -> List[Tuple[int, str]]:
        """List the document's sections as (1-based index, title), in order."""
        return [
            (index, title)
            for index, title, _node_id, _size in self.outline_entries(
                node_ref, max_entries
            )
        ]


def build_for_session(
    rag_service: Any, owner: Optional[str] = None
) -> Optional[ChunkContextService]:
    """Build a context service from whatever retriever a session has loaded.

    Returns None instead of raising when the session has no retrieval set yet
    (the orchestrator calls this while tools are being constructed, possibly
    before load_modules has run).
    """
    if rag_service is None:
        return None
    sets = getattr(rag_service, "_sets", None) or {}
    from tensortruth.services.rag_service import DEFAULT_OWNER

    rag_set = sets.get(owner or DEFAULT_OWNER)
    retriever = getattr(rag_set, "retriever", None) if rag_set is not None else None
    if retriever is None:
        return None
    try:
        service = ChunkContextService([retriever])
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Could not build chunk context service: %s", exc)
        return None
    if service.docstore_count == 0:
        return None
    return service
