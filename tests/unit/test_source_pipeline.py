"""Unit tests for tensortruth.core.source_pipeline module."""

from unittest.mock import MagicMock, patch

import pytest

from tensortruth.core.source_pipeline import SourceFetchPipeline


@pytest.fixture
def sample_search_results():
    """Sample search results for pipeline testing."""
    return [
        {
            "url": "https://example.com/page1",
            "title": "Page One",
            "snippet": "Snippet for page one",
        },
        {
            "url": "https://example.com/page2",
            "title": "Page Two",
            "snippet": "Snippet for page two",
        },
        {
            "url": "https://example.com/page3",
            "title": "Page Three",
            "snippet": "Snippet for page three",
        },
    ]


@pytest.fixture
def pipeline_config():
    """Default pipeline configuration."""
    return {
        "query": "test query",
        "max_pages": 3,
        "context_window": 8192,
        "reranker_model": None,  # Disabled by default for unit tests
        "reranker_device": "cpu",
        "rerank_content_threshold": 0.1,
        "max_source_context_pct": 0.15,
        "input_context_pct": 0.6,
        "custom_instructions": None,
        "progress_callback": None,
    }


@pytest.mark.unit
class TestSourceFetchPipelineInit:
    """Tests for SourceFetchPipeline initialization."""

    def test_initialization_with_defaults(self):
        """Test pipeline initializes with required params and defaults."""
        pipeline = SourceFetchPipeline(
            query="test query",
            max_pages=5,
            context_window=8192,
        )

        assert pipeline.query == "test query"
        assert pipeline.max_pages == 5
        assert pipeline.context_window == 8192
        assert pipeline.reranker_model is None
        assert pipeline.reranker_device == "cpu"
        assert pipeline.rerank_content_threshold == 0.1
        assert pipeline.sources == []
        assert pipeline.pages == []

    def test_initialization_with_custom_config(self):
        """Test pipeline initializes with all custom parameters."""
        callback = MagicMock()
        pipeline = SourceFetchPipeline(
            query="custom query",
            max_pages=10,
            context_window=16384,
            reranker_model="BAAI/bge-reranker-v2-m3",
            reranker_device="cuda",
            rerank_content_threshold=0.2,
            max_source_context_pct=0.1,
            input_context_pct=0.5,
            custom_instructions="focus on code examples",
            progress_callback=callback,
        )

        assert pipeline.query == "custom query"
        assert pipeline.max_pages == 10
        assert pipeline.context_window == 16384
        assert pipeline.reranker_model == "BAAI/bge-reranker-v2-m3"
        assert pipeline.reranker_device == "cuda"
        assert pipeline.rerank_content_threshold == 0.2
        assert pipeline.custom_instructions == "focus on code examples"
        assert pipeline.progress_callback is callback


@pytest.mark.unit
@pytest.mark.asyncio
class TestSourceFetchPipelineExecute:
    """Tests for SourceFetchPipeline.execute() method."""

    async def test_empty_search_results(self, pipeline_config):
        """Test pipeline handles empty search results."""
        pipeline = SourceFetchPipeline(**pipeline_config)

        with patch("aiohttp.ClientSession"):
            fitted_pages, sources, allocations = await pipeline.execute([])

        assert fitted_pages == []
        assert sources == []
        assert allocations == {}

    async def test_execute_fetch_success(self, pipeline_config, sample_search_results):
        """Test successful fetch phase populates pages and sources."""
        pipeline = SourceFetchPipeline(**pipeline_config)

        with patch(
            "tensortruth.core.source_pipeline.fetch_page_as_markdown"
        ) as mock_fetch:
            # Mock successful fetches
            mock_fetch.side_effect = [
                ("# Page 1 content", "success", None),
                ("# Page 2 content", "success", None),
                ("# Page 3 content", "success", None),
            ]

            fitted_pages, sources, allocations = await pipeline.execute(
                sample_search_results
            )

        # Should have fetched pages
        assert len(fitted_pages) == 3
        assert len(sources) == 3

        # Check sources have correct status
        for source in sources:
            assert source.status == "success"
            assert source.content is not None

    async def test_execute_with_fetch_failures(
        self, pipeline_config, sample_search_results
    ):
        """Test pipeline handles mix of success and failures."""
        pipeline = SourceFetchPipeline(**pipeline_config)

        with patch(
            "tensortruth.core.source_pipeline.fetch_page_as_markdown"
        ) as mock_fetch:
            # Mix of success and failures
            mock_fetch.side_effect = [
                ("# Page 1 content", "success", None),
                (None, "http_error", "404 Not Found"),
                ("# Page 3 content", "success", None),
            ]

            fitted_pages, sources, allocations = await pipeline.execute(
                sample_search_results
            )

        # Only successful pages
        assert len(fitted_pages) == 2
        # All sources tracked
        assert len(sources) == 3

        # Verify statuses
        statuses = [s.status for s in sources]
        assert statuses.count("success") == 2
        assert statuses.count("failed") == 1

    async def test_execute_with_all_failures(
        self, pipeline_config, sample_search_results
    ):
        """Test pipeline handles all fetch failures gracefully."""
        pipeline = SourceFetchPipeline(**pipeline_config)

        with patch(
            "tensortruth.core.source_pipeline.fetch_page_as_markdown"
        ) as mock_fetch:
            # All failures
            mock_fetch.side_effect = [
                (None, "timeout", "Connection timed out"),
                (None, "http_error", "500 Server Error"),
                (None, "parse_error", "Invalid HTML"),
            ]

            fitted_pages, sources, allocations = await pipeline.execute(
                sample_search_results
            )

        assert fitted_pages == []
        assert len(sources) == 3
        # All should be failed/skipped
        for source in sources:
            assert source.status in ("failed", "skipped")

    async def test_fetch_exceptions_handled(
        self, pipeline_config, sample_search_results
    ):
        """Test pipeline handles exceptions during fetch."""
        pipeline = SourceFetchPipeline(**pipeline_config)

        with patch(
            "tensortruth.core.source_pipeline.fetch_page_as_markdown"
        ) as mock_fetch:
            # Exception thrown
            mock_fetch.side_effect = Exception("Network error")

            fitted_pages, sources, allocations = await pipeline.execute(
                sample_search_results
            )

        # Should handle gracefully
        assert fitted_pages == []
        assert len(sources) == 3
        for source in sources:
            assert source.status == "failed"
            assert "Network error" in source.error


@pytest.mark.unit
@pytest.mark.asyncio
class TestSourceFetchPipelineReranking:
    """Tests for SourceFetchPipeline content reranking."""

    async def test_reranking_disabled_when_no_model(
        self, pipeline_config, sample_search_results
    ):
        """Test reranking is skipped when no model specified."""
        pipeline_config["reranker_model"] = None
        pipeline = SourceFetchPipeline(**pipeline_config)

        with (
            patch(
                "tensortruth.core.source_pipeline.fetch_page_as_markdown"
            ) as mock_fetch,
            patch(
                "tensortruth.core.source_pipeline.rerank_fetched_pages"
            ) as mock_rerank,
        ):
            mock_fetch.return_value = ("# Content", "success", None)

            await pipeline.execute(sample_search_results)

            # Reranking should not be called
            mock_rerank.assert_not_called()

    async def test_reranking_updates_scores(
        self, pipeline_config, sample_search_results
    ):
        """Test reranking updates source relevance_score."""
        pipeline_config["reranker_model"] = "BAAI/bge-reranker-v2-m3"
        pipeline = SourceFetchPipeline(**pipeline_config)

        with (
            patch(
                "tensortruth.core.source_pipeline.fetch_page_as_markdown"
            ) as mock_fetch,
            patch(
                "tensortruth.core.source_pipeline.get_reranker_for_web"
            ) as mock_get_reranker,
            patch(
                "tensortruth.core.source_pipeline.rerank_fetched_pages"
            ) as mock_rerank,
            patch(
                "tensortruth.core.source_pipeline.filter_by_threshold"
            ) as mock_filter,
            patch("tensortruth.services.model_manager.ModelManager") as mock_manager,
        ):
            # Setup mocks
            mock_fetch.side_effect = [
                ("# Content 1", "success", None),
                ("# Content 2", "success", None),
            ]

            mock_manager_instance = MagicMock()
            mock_manager_instance.is_reranker_loaded.return_value = True
            mock_manager.get_instance.return_value = mock_manager_instance

            mock_get_reranker.return_value = MagicMock()
            mock_rerank.return_value = [
                (
                    (
                        "https://example.com/page1",
                        "Page One",
                        "# Content 1",
                    ),
                    0.9,
                ),
                (
                    (
                        "https://example.com/page2",
                        "Page Two",
                        "# Content 2",
                    ),
                    0.7,
                ),
            ]
            mock_filter.return_value = (mock_rerank.return_value, [])

            await pipeline.execute(sample_search_results[:2])

            # Reranking should be called
            mock_rerank.assert_called_once()

    async def test_reranking_threshold_filtering(
        self, pipeline_config, sample_search_results
    ):
        """Test pages below threshold are marked as skipped."""
        pipeline_config["reranker_model"] = "BAAI/bge-reranker-v2-m3"
        pipeline_config["rerank_content_threshold"] = 0.5
        pipeline = SourceFetchPipeline(**pipeline_config)

        with (
            patch(
                "tensortruth.core.source_pipeline.fetch_page_as_markdown"
            ) as mock_fetch,
            patch("tensortruth.core.source_pipeline.get_reranker_for_web"),
            patch(
                "tensortruth.core.source_pipeline.rerank_fetched_pages"
            ) as mock_rerank,
            patch(
                "tensortruth.core.source_pipeline.filter_by_threshold"
            ) as mock_filter,
            patch("tensortruth.services.model_manager.ModelManager") as mock_manager,
        ):
            mock_fetch.side_effect = [
                ("# High score content", "success", None),
                ("# Low score content", "success", None),
            ]

            mock_manager_instance = MagicMock()
            mock_manager_instance.is_reranker_loaded.return_value = True
            mock_manager.get_instance.return_value = mock_manager_instance

            # One passes, one rejected
            passing = [
                (
                    ("https://example.com/page1", "Page One", "# High score content"),
                    0.8,
                )
            ]
            rejected = [
                (
                    ("https://example.com/page2", "Page Two", "# Low score content"),
                    0.3,
                )
            ]
            mock_rerank.return_value = passing + rejected
            mock_filter.return_value = (passing, rejected)

            fitted_pages, sources, allocations = await pipeline.execute(
                sample_search_results[:2]
            )

            # Only passing page should be in fitted_pages
            assert len(fitted_pages) == 1


@pytest.mark.unit
@pytest.mark.asyncio
class TestSourceFetchPipelineContextFitting:
    """Tests for SourceFetchPipeline context window fitting."""

    async def test_fit_to_context_called(self, pipeline_config, sample_search_results):
        """Test context fitting is performed on successful pages."""
        pipeline = SourceFetchPipeline(**pipeline_config)

        with (
            patch(
                "tensortruth.core.source_pipeline.fetch_page_as_markdown"
            ) as mock_fetch,
            patch(
                "tensortruth.core.source_pipeline.fit_sources_to_context"
            ) as mock_fit,
        ):
            mock_fetch.return_value = ("# Content", "success", None)
            mock_fit.return_value = (
                [("https://example.com/page1", "Page One", "# Content")],
                {"https://example.com/page1": 100},
            )

            await pipeline.execute(sample_search_results[:1])

            mock_fit.assert_called_once()

    async def test_allocations_update_content_chars(
        self, pipeline_config, sample_search_results
    ):
        """Test that allocations update source.content_chars."""
        pipeline = SourceFetchPipeline(**pipeline_config)

        with (
            patch(
                "tensortruth.core.source_pipeline.fetch_page_as_markdown"
            ) as mock_fetch,
            patch(
                "tensortruth.core.source_pipeline.fit_sources_to_context"
            ) as mock_fit,
        ):
            mock_fetch.return_value = ("# Long content here", "success", None)
            mock_fit.return_value = (
                [
                    (
                        "https://example.com/page1",
                        "Page One",
                        "# Truncated",
                    )
                ],
                {"https://example.com/page1": 500},  # 500 chars allocated
            )

            fitted_pages, sources, allocations = await pipeline.execute(
                sample_search_results[:1]
            )

            # Source should have updated content_chars
            assert sources[0].content_chars == 500
            assert allocations == {"https://example.com/page1": 500}


@pytest.mark.unit
@pytest.mark.asyncio
class TestSourceFetchPipelineProgress:
    """Tests for SourceFetchPipeline progress callbacks."""

    async def test_progress_callback_called(
        self, pipeline_config, sample_search_results
    ):
        """Test progress callback is called during execution."""
        callback = MagicMock()
        pipeline_config["progress_callback"] = callback
        pipeline = SourceFetchPipeline(**pipeline_config)

        with patch(
            "tensortruth.core.source_pipeline.fetch_page_as_markdown"
        ) as mock_fetch:
            mock_fetch.return_value = ("# Content", "success", None)

            await pipeline.execute(sample_search_results[:1])

        # Callback should have been called at least once
        assert callback.called

    async def test_progress_callback_phases(
        self, pipeline_config, sample_search_results
    ):
        """Test progress callback is called with correct phases."""
        phases_seen = []

        def track_phases(phase, message, details):
            phases_seen.append(phase)

        pipeline_config["progress_callback"] = track_phases
        pipeline = SourceFetchPipeline(**pipeline_config)

        with patch(
            "tensortruth.core.source_pipeline.fetch_page_as_markdown"
        ) as mock_fetch:
            mock_fetch.return_value = ("# Content", "success", None)

            await pipeline.execute(sample_search_results[:1])

        # Should see fetching phase at minimum
        assert "fetching" in phases_seen


@pytest.mark.unit
@pytest.mark.asyncio
class TestSourceFetchPipelineState:
    """Tests for SourceFetchPipeline state management."""

    async def test_state_reset_between_executions(
        self, pipeline_config, sample_search_results
    ):
        """Test pipeline state is properly managed across executions."""
        pipeline = SourceFetchPipeline(**pipeline_config)

        with patch(
            "tensortruth.core.source_pipeline.fetch_page_as_markdown"
        ) as mock_fetch:
            mock_fetch.return_value = ("# Content", "success", None)

            # First execution
            await pipeline.execute(sample_search_results[:1])
            first_sources = pipeline.sources.copy()

            # Second execution with different data
            mock_fetch.return_value = ("# Different content", "success", None)
            await pipeline.execute(sample_search_results[1:2])
            second_sources = pipeline.sources

        # Each execution should have its own sources
        # (state accumulated but distinguishable by URL)
        assert len(first_sources) == 1
        assert first_sources[0].url == "https://example.com/page1"
        # Second execution adds to state
        assert len(second_sources) == 2

    async def test_snippet_map_populated(self, pipeline_config, sample_search_results):
        """Test snippet_map is populated from search results."""
        pipeline = SourceFetchPipeline(**pipeline_config)

        with patch(
            "tensortruth.core.source_pipeline.fetch_page_as_markdown"
        ) as mock_fetch:
            mock_fetch.return_value = ("# Content", "success", None)

            await pipeline.execute(sample_search_results)

        # Snippet map should contain all URLs
        assert "https://example.com/page1" in pipeline.snippet_map
        assert (
            pipeline.snippet_map["https://example.com/page1"] == "Snippet for page one"
        )
