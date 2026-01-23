"""Retrieval quality metrics for RAG pipeline.

This module provides comprehensive metrics for evaluating retrieval quality,
including score distribution, diversity, coverage, and quality indicators.
"""

import math
import statistics
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class RetrievalMetrics:
    """Comprehensive retrieval quality metrics.

    Attributes:
        score_distribution: Statistical measures of relevance scores
        diversity: Source variety and distribution metrics
        coverage: Context size and chunk statistics
        quality: High-level quality indicators
    """

    # Score distribution
    score_mean: Optional[float] = None
    score_median: Optional[float] = None
    score_min: Optional[float] = None
    score_max: Optional[float] = None
    score_std: Optional[float] = None
    score_q1: Optional[float] = None
    score_q3: Optional[float] = None
    score_iqr: Optional[float] = None
    score_range: Optional[float] = None

    # Diversity
    unique_sources: int = 0
    source_types: int = 0
    source_entropy: Optional[float] = None

    # Coverage
    total_context_chars: int = 0
    avg_chunk_length: float = 0.0
    total_chunks: int = 0
    estimated_tokens: int = 0

    # Quality indicators
    high_confidence_ratio: float = 0.0
    low_confidence_ratio: float = 0.0

    # Configuration (for debugging/verification)
    configured_top_n: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize metrics for JSON response.

        Converts all numeric values to native Python types to ensure
        JSON serialization compatibility (handles numpy float32, etc.).

        Returns:
            Dictionary with nested structure for frontend consumption.
        """

        def to_native(value):
            """Convert value to native Python type for JSON serialization."""
            if value is None:
                return None
            # Handle numpy types (check for item() method which numpy scalars have)
            if hasattr(value, "item"):
                return value.item()
            # Convert to native Python types explicitly
            if isinstance(value, bool):
                return bool(value)
            if isinstance(value, int):
                return int(value)
            if isinstance(value, float):
                return float(value)
            if isinstance(value, str):
                return str(value)
            return value

        return {
            "score_distribution": {
                "mean": to_native(self.score_mean),
                "median": to_native(self.score_median),
                "min": to_native(self.score_min),
                "max": to_native(self.score_max),
                "std": to_native(self.score_std),
                "q1": to_native(self.score_q1),
                "q3": to_native(self.score_q3),
                "iqr": to_native(self.score_iqr),
                "range": to_native(self.score_range),
            },
            "diversity": {
                "unique_sources": int(self.unique_sources),
                "source_types": int(self.source_types),
                "source_entropy": to_native(self.source_entropy),
            },
            "coverage": {
                "total_context_chars": int(self.total_context_chars),
                "avg_chunk_length": to_native(self.avg_chunk_length),
                "total_chunks": int(self.total_chunks),
                "estimated_tokens": int(self.estimated_tokens),
            },
            "quality": {
                "high_confidence_ratio": to_native(self.high_confidence_ratio),
                "low_confidence_ratio": to_native(self.low_confidence_ratio),
            },
            "configuration": {
                "configured_top_n": to_native(self.configured_top_n),
            },
        }


def calculate_entropy(counts: List[int]) -> float:
    """Calculate Shannon entropy from count distribution.

    Args:
        counts: List of counts for each unique value.

    Returns:
        Shannon entropy in bits. Higher values indicate more diversity.
        Returns 0.0 for empty lists or single unique value.
    """
    if not counts or len(counts) == 1:
        return 0.0

    total = sum(counts)
    if total == 0:
        return 0.0

    entropy = 0.0
    for count in counts:
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)

    return entropy


def compute_retrieval_metrics(source_nodes: List[Any]) -> RetrievalMetrics:
    """Compute comprehensive retrieval quality metrics from source nodes.

    Args:
        source_nodes: List of NodeWithScore objects from retrieval.

    Returns:
        RetrievalMetrics object with all computed statistics.

    Note:
        - Handles empty node lists gracefully (returns zeros/None)
        - Extracts scores from node.score attribute
        - Extracts metadata from node.node.metadata dict
        - Uses postprocessor-reranked scores (not raw embeddings)
    """
    metrics = RetrievalMetrics()

    if not source_nodes:
        return metrics

    # Extract scores (filter out None values and convert to native Python float)
    scores = []
    for node in source_nodes:
        if hasattr(node, "score") and node.score is not None:
            score = node.score
            # Convert numpy types to native Python float
            if hasattr(score, "item"):
                score = float(score.item())
            else:
                score = float(score)
            scores.append(score)

    # === Score Distribution ===
    if scores:
        metrics.score_mean = statistics.mean(scores)
        metrics.score_median = statistics.median(scores)
        metrics.score_min = min(scores)
        metrics.score_max = max(scores)
        metrics.score_range = metrics.score_max - metrics.score_min

        # Standard deviation (requires at least 2 samples)
        if len(scores) >= 2:
            metrics.score_std = statistics.stdev(scores)

        # Quartiles (requires at least 2 samples)
        if len(scores) >= 2:
            sorted_scores = sorted(scores)
            metrics.score_q1 = statistics.median(
                sorted_scores[: len(sorted_scores) // 2]
            )
            metrics.score_q3 = statistics.median(
                sorted_scores[(len(sorted_scores) + 1) // 2 :]
            )
            metrics.score_iqr = metrics.score_q3 - metrics.score_q1

    # === Diversity Metrics ===
    source_files = []
    doc_types = []

    for node in source_nodes:
        # Extract metadata (handle different structures)
        metadata: Dict[str, Any] = {}
        if hasattr(node, "node") and hasattr(node.node, "metadata"):
            metadata = node.node.metadata or {}
        elif hasattr(node, "metadata"):
            metadata = node.metadata or {}

        # Collect source identifiers
        filename = (
            metadata.get("filename")
            or metadata.get("file_name")
            or metadata.get("source_url", "unknown")
        )
        source_files.append(filename)

        # Collect document types
        doc_type = metadata.get("doc_type", "unknown")
        doc_types.append(doc_type)

    # Unique sources and types
    unique_files = set(source_files)
    unique_types = set(doc_types)

    metrics.unique_sources = len(unique_files)
    metrics.source_types = len(unique_types)

    # Source entropy (distribution of chunks across sources)
    if unique_files:
        file_counts = list(Counter(source_files).values())
        metrics.source_entropy = calculate_entropy(file_counts)

    # === Coverage Metrics ===
    metrics.total_chunks = len(source_nodes)

    total_chars = 0
    for node in source_nodes:
        # Extract text content
        content = ""
        if hasattr(node, "node"):
            if hasattr(node.node, "get_content"):
                content = node.node.get_content()
            elif hasattr(node.node, "text"):
                content = node.node.text
        elif hasattr(node, "text"):
            content = node.text

        total_chars += len(content)

    metrics.total_context_chars = total_chars
    metrics.avg_chunk_length = total_chars / len(source_nodes) if source_nodes else 0.0
    metrics.estimated_tokens = total_chars // 4  # Rough estimate: 1 token ≈ 4 chars

    # === Quality Indicators ===
    if scores:
        high_count = sum(1 for s in scores if s >= 0.7)
        low_count = sum(1 for s in scores if s < 0.4)

        metrics.high_confidence_ratio = high_count / len(scores)
        metrics.low_confidence_ratio = low_count / len(scores)

    return metrics
