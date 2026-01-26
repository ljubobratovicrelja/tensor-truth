"""Unified source fetching pipeline for web research.

This module provides SourceFetchPipeline, which handles the complete workflow
for acquiring web sources: parallel fetching, content-based reranking, and
context window fitting. Used by both /web command and browse agent.
"""

import asyncio
import logging
from typing import Callable, Dict, List, Literal, Optional, Tuple

import aiohttp
from llama_index.core.postprocessor import SentenceTransformerRerank

from tensortruth.core.source_converter import SourceConverter
from tensortruth.core.source_metrics import compute_metrics
from tensortruth.core.sources import SourceNode
from tensortruth.core.unified_sources import SourceStatus, SourceType, UnifiedSource
from tensortruth.utils.web_search import (
    fetch_page_as_markdown,
    filter_by_threshold,
    fit_sources_to_context,
    get_reranker_for_web,
    rerank_fetched_pages,
)

logger = logging.getLogger(__name__)


class SourceFetchPipeline:
    """Unified pipeline for fetching, ranking, and fitting web sources.

    Handles the complete source acquisition workflow:
    1. Batch parallel fetching with adaptive retry (fetch until we have enough good pages)
    2. Content-based reranking (score based on actual fetched text)
    3. Threshold filtering (reject low-relevance pages)
    4. Context window fitting (fill context from top-scored pages)

    Used by both /web command and browse agent for consistency.

    Example:
        >>> pipeline = SourceFetchPipeline(
        ...     query="What are CNNs?",
        ...     max_pages=5,
        ...     context_window=8192,
        ...     reranker_model="BAAI/bge-reranker-v2-m3"
        ... )
        >>> fitted_pages, sources, allocations = await pipeline.execute(search_results)
    """

    def __init__(
        self,
        query: str,
        max_pages: int,
        context_window: int,
        reranker_model: Optional[str] = None,
        reranker_device: str = "cpu",
        rerank_content_threshold: float = 0.1,
        max_source_context_pct: float = 0.15,
        input_context_pct: float = 0.6,
        custom_instructions: Optional[str] = None,
        progress_callback: Optional[Callable] = None,
    ):
        """Initialize pipeline with configuration.

        Args:
            query: User's search query
            max_pages: Target number of successful pages to fetch
            context_window: Size of LLM context window for fitting
            reranker_model: Optional reranker model name (None = disabled)
            reranker_device: Device for reranker (cpu/cuda/mps)
            rerank_content_threshold: Min score for content reranking (0.0-1.0)
            max_source_context_pct: Max % of context per source
            input_context_pct: % of context for input (rest for output)
            custom_instructions: Optional custom instructions for reranking
            progress_callback: Optional callback(phase, message, details)
        """
        self.query = query
        self.max_pages = max_pages
        self.context_window = context_window
        self.reranker_model = reranker_model
        self.reranker_device = reranker_device
        self.rerank_content_threshold = rerank_content_threshold
        self.max_source_context_pct = max_source_context_pct
        self.input_context_pct = input_context_pct
        self.custom_instructions = custom_instructions
        self.progress_callback = progress_callback

        # Pipeline state
        self.sources: List[SourceNode] = []
        self.pages: List[Tuple[str, str, str]] = []  # (url, title, content)
        self.snippet_map: Dict[str, str] = {}
        self.reranker: Optional[SentenceTransformerRerank] = None

        # Unified source tracking (internal use)
        self._unified_sources: List[UnifiedSource] = []

    async def execute(
        self,
        search_results: List[Dict],
    ) -> Tuple[List[Tuple[str, str, str]], List[SourceNode], Dict[str, int]]:
        """Execute full pipeline: fetch → rerank → fit.

        Args:
            search_results: List of search result dicts with url, title, snippet

        Returns:
            Tuple of:
            - fitted_pages: List[(url, title, content)] ready for synthesis
            - sources: Rich SourceNode metadata with scores/status
            - allocations: Dict[url -> chars] context allocations
        """
        # Build snippet map for SourceNode metadata
        self.snippet_map = {r["url"]: r.get("snippet", "") for r in search_results}

        # Step 1: Fetch pages with adaptive retry
        await self._fetch_pages_adaptive(search_results)

        if not self.pages:
            logger.warning("No pages successfully fetched")
            return [], self.sources, {}

        # Step 2: Rerank by content
        await self._rerank_by_content()

        if not self.pages:
            logger.warning("All pages rejected after content reranking")
            return [], self.sources, {}

        # Step 3: Fit to context window
        fitted_pages, allocations = self._fit_to_context()

        # Update unified sources with content_chars from allocations
        for unified in self._unified_sources:
            if unified.url and unified.url in allocations:
                unified.content_chars = allocations[unified.url]

        # Compute and log metrics
        if self._unified_sources:
            metrics = compute_metrics(self._unified_sources)
            logger.info(
                f"Source metrics: {metrics.total_sources} total, "
                f"mean score: {metrics.score_mean:.2f}"
                if metrics.score_mean is not None
                else f"Source metrics: {metrics.total_sources} total, no scores"
            )

        logger.info(
            f"Pipeline complete: {len(fitted_pages)} pages fitted, "
            f"{len(self.sources)} total sources"
        )

        return fitted_pages, self.sources, allocations

    async def _fetch_pages_adaptive(self, search_results: List[Dict]):
        """Fetch pages with adaptive retry until max_pages good sources.

        Fetches in batches of (needed + 2) to handle failures gracefully.
        Continues fetching until we have max_pages successful fetches or
        run out of search results.

        Updates self.pages and self.sources.
        """
        current_idx = 0
        pages_fetched = 0
        pages_failed = 0

        logger.info(
            f"Starting adaptive fetch: target={self.max_pages} pages "
            f"from {len(search_results)} search results"
        )

        async with aiohttp.ClientSession() as session:
            while len(self.pages) < self.max_pages and current_idx < len(
                search_results
            ):
                # Calculate batch size: fetch a few extra to handle failures
                needed = self.max_pages - len(self.pages)
                batch_size = min(needed + 2, len(search_results) - current_idx)
                batch = search_results[current_idx : current_idx + batch_size]

                logger.debug(
                    f"Fetching batch: idx={current_idx}, size={batch_size}, "
                    f"pages_so_far={len(self.pages)}/{self.max_pages}"
                )

                # Send progress update
                if self.progress_callback:
                    self.progress_callback(
                        "fetching",
                        f"Fetching batch {current_idx // batch_size + 1}...",
                        {
                            "pages_target": self.max_pages,
                            "pages_fetched": pages_fetched,
                            "pages_failed": pages_failed,
                        },
                    )

                # Fetch batch in parallel
                tasks = [fetch_page_as_markdown(r["url"], session) for r in batch]
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)

                # Process results
                for search_result, result in zip(batch, batch_results):
                    url = search_result["url"]
                    title = search_result["title"]

                    if isinstance(result, BaseException):
                        # Fetch failed with exception
                        pages_failed += 1
                        error_str = str(result)
                        self.sources.append(
                            SourceNode(
                                url=url,
                                title=title,
                                status="failed",
                                error=error_str,
                                snippet=self.snippet_map.get(url),
                            )
                        )
                        # Create unified source
                        self._unified_sources.append(
                            UnifiedSource(
                                id=SourceConverter._generate_id(url),
                                url=url,
                                title=title,
                                source_type=SourceType.WEB,
                                status=SourceStatus.FAILED,
                                error=error_str,
                                snippet=self.snippet_map.get(url),
                            )
                        )
                        logger.debug(f"Fetch failed: {title} - {result}")

                        if self.progress_callback:
                            self.progress_callback(
                                "fetching",
                                f"Failed: {title}",
                                {
                                    "url": url,
                                    "status": "failed",
                                    "error": error_str,
                                },
                            )

                    else:
                        # Unpack result tuple
                        markdown, status, error_msg = result

                        if status == "success" and markdown:
                            # Success - add to pages
                            self.pages.append((url, title, markdown))
                            pages_fetched += 1
                            self.sources.append(
                                SourceNode(
                                    url=url,
                                    title=title,
                                    status="success",
                                    error=None,
                                    snippet=self.snippet_map.get(url),
                                    content=markdown,
                                    content_chars=len(markdown),
                                )
                            )
                            # Create unified source
                            self._unified_sources.append(
                                UnifiedSource(
                                    id=SourceConverter._generate_id(url),
                                    url=url,
                                    title=title,
                                    source_type=SourceType.WEB,
                                    status=SourceStatus.SUCCESS,
                                    content=markdown,
                                    snippet=self.snippet_map.get(url),
                                    content_chars=len(markdown),
                                )
                            )
                            logger.debug(
                                f"Fetch success: {title} ({len(markdown)} chars)"
                            )

                            if self.progress_callback:
                                self.progress_callback(
                                    "fetching",
                                    f"Fetched: {title}",
                                    {
                                        "url": url,
                                        "status": "success",
                                        "content_chars": len(markdown),
                                    },
                                )

                        else:
                            # Failed or skipped (too short, parse error, etc)
                            source_status: Literal["success", "failed", "skipped"] = (
                                "skipped"
                                if status in ("too_short", "parse_error")
                                else "failed"
                            )
                            unified_status = (
                                SourceStatus.SKIPPED
                                if source_status == "skipped"
                                else SourceStatus.FAILED
                            )
                            pages_failed += 1
                            self.sources.append(
                                SourceNode(
                                    url=url,
                                    title=title,
                                    status=source_status,
                                    error=error_msg,
                                    snippet=self.snippet_map.get(url),
                                )
                            )
                            # Create unified source
                            self._unified_sources.append(
                                UnifiedSource(
                                    id=SourceConverter._generate_id(url),
                                    url=url,
                                    title=title,
                                    source_type=SourceType.WEB,
                                    status=unified_status,
                                    error=error_msg,
                                    snippet=self.snippet_map.get(url),
                                )
                            )
                            logger.debug(
                                f"Fetch {source_status}: {title} - {error_msg}"
                            )

                            if self.progress_callback:
                                self.progress_callback(
                                    "fetching",
                                    f"{source_status.title()}: {title}",
                                    {
                                        "url": url,
                                        "status": source_status,
                                        "error": error_msg,
                                    },
                                )

                    # Stop if we have enough pages
                    if len(self.pages) >= self.max_pages:
                        break

                current_idx += batch_size

        logger.info(
            f"Fetch complete: {len(self.pages)} pages fetched, "
            f"{pages_failed} failed/skipped"
        )

    async def _rerank_by_content(self):
        """Rerank fetched pages by actual content.

        Uses reranker model to score pages based on full content.
        Updates SourceNode.relevance_score for all pages.
        Marks low-scoring pages as skipped.

        Updates self.pages (filters out low-scoring) and self.sources (adds scores).
        """
        if not self.reranker_model or not self.pages:
            logger.info("Content reranking disabled or no pages to rerank")
            return

        logger.info(
            f"Content reranking: {len(self.pages)} pages with {self.reranker_model}"
        )

        # Check if model needs loading (first use detection)
        from tensortruth.services.model_manager import ModelManager

        manager = ModelManager.get_instance()
        model_needs_loading = not manager.is_reranker_loaded(self.reranker_model)

        if model_needs_loading and self.progress_callback:
            self.progress_callback(
                "loading_model",
                "Loading reranker model...",
                {"model": self.reranker_model},
            )

        if self.progress_callback:
            self.progress_callback(
                "ranking_content",
                "Ranking content by relevance...",
                {"page_count": len(self.pages)},
            )

        # Get reranker
        self.reranker = get_reranker_for_web(
            self.reranker_model,
            self.reranker_device,
            top_n=len(self.pages),
        )

        # Rerank pages by content
        ranked_pages = rerank_fetched_pages(
            self.query, self.custom_instructions, self.pages, self.reranker
        )

        # Apply threshold filtering
        passing_pages, rejected_pages = filter_by_threshold(
            ranked_pages, self.rerank_content_threshold
        )

        logger.info(
            f"Content reranking: {len(passing_pages)} passed, "
            f"{len(rejected_pages)} rejected (threshold={self.rerank_content_threshold})"
        )

        # Update sources with relevance scores for ALL pages
        for (url, title, content), score in ranked_pages:
            # Find and update the matching legacy source
            for source in self.sources:
                if source.url == url and source.status == "success":
                    source.relevance_score = score
                    logger.debug(f"Score: {title} -> {score:.2f}")
                    break
            # Also update unified source
            for unified in self._unified_sources:
                if unified.url == url and unified.status == SourceStatus.SUCCESS:
                    unified.score = score
                    break

        # Mark rejected sources as skipped/filtered with reason
        rejected_urls = {url for (url, title, content), score in rejected_pages}
        for source in self.sources:
            if source.url in rejected_urls and source.status == "success":
                score_pct = int((source.relevance_score or 0) * 100)
                threshold_pct = int(self.rerank_content_threshold * 100)
                source.status = "skipped"
                source.error = f"Low relevance ({score_pct}% < {threshold_pct}%)"
                logger.debug(f"Rejected: {source.title} (score={score_pct}%)")
        # Also mark unified sources as filtered
        for unified in self._unified_sources:
            if unified.url in rejected_urls and unified.status == SourceStatus.SUCCESS:
                score_pct = int((unified.score or 0) * 100)
                threshold_pct = int(self.rerank_content_threshold * 100)
                unified.status = SourceStatus.FILTERED
                unified.error = f"Low relevance ({score_pct}% < {threshold_pct}%)"

        # Update pages to only include passing pages
        if passing_pages:
            self.pages = [
                (url, title, content) for (url, title, content), _ in passing_pages
            ]

            if self.progress_callback:
                self.progress_callback(
                    "ranking_content",
                    f"{len(passing_pages)} pages passed threshold",
                    {
                        "passed": len(passing_pages),
                        "rejected": len(rejected_pages),
                    },
                )
        else:
            self.pages = []
            if self.progress_callback:
                self.progress_callback(
                    "ranking_content",
                    f"All {len(rejected_pages)} pages below relevance threshold",
                    {"rejected": len(rejected_pages)},
                )

    def _fit_to_context(
        self,
    ) -> Tuple[List[Tuple[str, str, str]], Dict[str, int]]:
        """Fit top-scored sources to context window.

        Sorts pages by relevance score (desc) and fills context from top
        until window is full. Trimmed pages are included in sources but
        marked appropriately.

        Returns:
            Tuple of:
            - fitted_pages: List[(url, title, content)] that fit in context
            - allocations: Dict[url -> chars] showing how much of each page was used
        """
        if not self.pages:
            return [], {}

        logger.info(
            f"Context fitting: {len(self.pages)} pages into {self.context_window} tokens"
        )

        # Build source_scores dict for fit_sources_to_context
        source_scores: Dict[str, float] = {}
        for source in self.sources:
            if source.url and source.relevance_score is not None:
                source_scores[source.url] = source.relevance_score

        # Fit sources to context window
        fitted_pages, allocations = fit_sources_to_context(
            pages=self.pages,
            source_scores=source_scores,
            context_window=self.context_window,
            input_context_pct=self.input_context_pct,
            max_source_context_pct=self.max_source_context_pct,
        )

        # Update sources with content_chars based on allocations
        for source in self.sources:
            if source.url in allocations:
                source.content_chars = allocations[source.url]

        logger.info(
            f"Context fitting complete: {len(fitted_pages)}/{len(self.pages)} pages fit"
        )

        if self.progress_callback:
            self.progress_callback(
                "fitting",
                f"Fitted {len(fitted_pages)} pages to context",
                {
                    "fitted": len(fitted_pages),
                    "total": len(self.pages),
                },
            )

        return fitted_pages, allocations

    def get_unified_sources(self) -> List[UnifiedSource]:
        """Get sources as UnifiedSource objects.

        Returns a copy of the internal unified sources list for external use.
        Useful for metrics computation and API conversion.
        """
        return self._unified_sources.copy()
