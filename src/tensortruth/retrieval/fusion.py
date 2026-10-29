"""Retrievers that combine dense and lexical evidence.

Dense-only retrieval answers paraphrases well but misses exact terms; lexical
search does the inverse. Reciprocal Rank Fusion (RRF) merges the two rankings
without calibrating their unrelated score scales, which is why it is used
here rather than a weighted sum of scores.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from tensortruth.indexing.lexical import LexicalIndex

logger = logging.getLogger(__name__)


class LexicalRetriever(BaseRetriever):
    """BaseRetriever over a :class:`LexicalIndex`.

    Stays leaf-level on purpose: fusing with an auto-merging dense retriever
    already reintroduces parents, and merging lexical hits as well would
    duplicate that text in the final context.
    """

    def __init__(
        self,
        lexical: LexicalIndex,
        top_k: int = 10,
        node_lookup: Optional[Callable[[str], Any]] = None,
    ) -> None:
        self.lexical = lexical
        self.top_k = top_k
        self.node_lookup = node_lookup
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        results = self.lexical.search(query_bundle.query_str, self.top_k)
        nodes: list[NodeWithScore] = []
        for node_id, score in results:
            node = self._lookup(node_id)
            if node is None:
                chunk = self.lexical.get_chunk(node_id)
                # Keep the indexed text on the synthetic node so callers can
                # still show the excerpt when the docstore does not know the id.
                text, doc_id = chunk if chunk else ("", "")
                node = TextNode(id_=node_id, text=text, metadata={"doc_id": doc_id})
            nodes.append(NodeWithScore(node=node, score=score))
        return nodes

    def _lookup(self, node_id: str) -> Any | None:
        if self.node_lookup is None:
            return None
        try:
            return self.node_lookup(node_id)
        except Exception as exc:
            # A missing id in the docstore is routine when an index was
            # re-chunked without rebuilding the lexical side.
            logger.debug(f"Docstore lookup failed for {node_id}: {exc}")
            return None


class RRFusionRetriever(BaseRetriever):
    """Reciprocal Rank Fusion over several retrievers, run sequentially.

    Children are already cheap (the parent MultiIndexRetriever parallelises
    across indexes), and sequential execution keeps one query from fanning out
    into nested threads.
    """

    def __init__(
        self,
        retrievers: list[BaseRetriever],
        rrf_k: int = 60,
        weights: Optional[list[float]] = None,
    ) -> None:
        self.retrievers = retrievers
        self.rrf_k = rrf_k
        if weights is None:
            weights = [1.0] * len(retrievers)
        if len(weights) != len(retrievers):
            raise ValueError(
                f"weights length {len(weights)} does not match "
                f"{len(retrievers)} retrievers"
            )
        self.weights = weights
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        merged: dict[str, tuple[float, NodeWithScore]] = {}

        for retriever, weight in zip(self.retrievers, self.weights):
            try:
                children = retriever.retrieve(query_bundle)
            except Exception as exc:
                # One dead backend must not take the whole query down.
                logger.warning(
                    f"RRF child {type(retriever).__name__} failed, "
                    f"continuing with the rest: {exc}"
                )
                continue

            for rank, item in enumerate(children):
                node_id = item.node.node_id
                contribution = weight / (self.rrf_k + rank + 1)
                existing = merged.get(node_id)
                if existing is None:
                    merged[node_id] = (contribution, item)
                else:
                    total, best = existing
                    if (item.score or 0.0) > (best.score or 0.0):
                        best = item
                    merged[node_id] = (total + contribution, best)

        fused = [
            NodeWithScore(node=item.node, score=score)
            for score, item in merged.values()
        ]
        fused.sort(key=lambda n: n.score or 0.0, reverse=True)
        return fused
