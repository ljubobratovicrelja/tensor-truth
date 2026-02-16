"""
Unit tests for tensortruth.rag_engine module.
"""

from unittest.mock import MagicMock, patch

import pytest
from llama_index.core import QueryBundle

from tensortruth.rag_engine import MultiIndexRetriever

# ============================================================================
# Tests for MultiIndexRetriever
# ============================================================================


@pytest.mark.unit
class TestMultiIndexRetriever:
    """Tests for MultiIndexRetriever class."""

    def test_initialization(self):
        """Test retriever initialization."""
        mock_retriever1 = MagicMock()
        mock_retriever2 = MagicMock()

        multi_retriever = MultiIndexRetriever([mock_retriever1, mock_retriever2])

        assert len(multi_retriever.retrievers) == 2
        assert multi_retriever.retrievers[0] == mock_retriever1
        assert multi_retriever.retrievers[1] == mock_retriever2

    def test_retrieve_combines_results(self):
        """Test that retrieve combines results from multiple retrievers."""
        # Create mock retrievers with mock nodes
        mock_node1 = MagicMock()
        mock_node1.text = "Node from retriever 1"
        mock_node1.score = 0.9

        mock_node2 = MagicMock()
        mock_node2.text = "Node from retriever 2"
        mock_node2.score = 0.8

        mock_retriever1 = MagicMock()
        mock_retriever1.retrieve.return_value = [mock_node1]

        mock_retriever2 = MagicMock()
        mock_retriever2.retrieve.return_value = [mock_node2]

        # Create multi-retriever
        multi_retriever = MultiIndexRetriever([mock_retriever1, mock_retriever2])

        # Test retrieval
        query_bundle = QueryBundle(query_str="test query")
        results = multi_retriever._retrieve(query_bundle)

        assert len(results) == 2
        assert mock_node1 in results
        assert mock_node2 in results

    def test_retrieve_with_empty_retriever(self):
        """Test retrieval when one retriever returns empty."""
        mock_node = MagicMock()
        mock_node.text = "Node from retriever 1"

        mock_retriever1 = MagicMock()
        mock_retriever1.retrieve.return_value = [mock_node]

        mock_retriever2 = MagicMock()
        mock_retriever2.retrieve.return_value = []

        multi_retriever = MultiIndexRetriever([mock_retriever1, mock_retriever2])

        query_bundle = QueryBundle(query_str="test query")
        results = multi_retriever._retrieve(query_bundle)

        assert len(results) == 1
        assert results[0] == mock_node

    def test_retrieve_with_single_retriever(self):
        """Test retrieval with single retriever."""
        mock_node = MagicMock()
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [mock_node]

        multi_retriever = MultiIndexRetriever([mock_retriever])

        query_bundle = QueryBundle(query_str="test query")
        results = multi_retriever._retrieve(query_bundle)

        assert len(results) == 1
        assert results[0] == mock_node

    def test_retrieve_preserves_order(self):
        """Test that retrieval preserves order from retrievers when balancing is disabled."""
        nodes = [MagicMock() for _ in range(5)]

        retriever1 = MagicMock()
        retriever1.retrieve.return_value = nodes[:3]

        retriever2 = MagicMock()
        retriever2.retrieve.return_value = nodes[3:]

        # Disable balancing to test order preservation
        multi_retriever = MultiIndexRetriever(
            [retriever1, retriever2], balance_strategy="none"
        )

        query_bundle = QueryBundle(query_str="test query")
        results = multi_retriever._retrieve(query_bundle)

        # Should get nodes from retriever1 first, then retriever2
        assert len(results) == 5
        assert results[:3] == nodes[:3]
        assert results[3:] == nodes[3:]

    def test_clear_cache_with_caching_enabled(self):
        """Test that clear_cache clears the LRU cache when caching is enabled."""
        mock_node = MagicMock()
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [mock_node]

        # Create retriever with caching enabled (default)
        multi_retriever = MultiIndexRetriever([mock_retriever], enable_cache=True)

        # Perform a retrieval to populate the cache
        query_bundle = QueryBundle(query_str="test query")
        multi_retriever._retrieve(query_bundle)

        # Verify cache has entries
        cache_info = multi_retriever._retrieve_cached.cache_info()
        assert cache_info.currsize > 0

        # Clear the cache
        multi_retriever.clear_cache()

        # Verify cache is empty
        cache_info = multi_retriever._retrieve_cached.cache_info()
        assert cache_info.currsize == 0

    def test_clear_cache_with_caching_disabled(self):
        """Test that clear_cache is safe to call when caching is disabled."""
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = []

        # Create retriever with caching disabled
        multi_retriever = MultiIndexRetriever([mock_retriever], enable_cache=False)

        # Should not raise when cache is disabled
        multi_retriever.clear_cache()

    def test_clear_cache_idempotent(self):
        """Test that multiple clear_cache calls don't raise."""
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = []

        multi_retriever = MultiIndexRetriever([mock_retriever], enable_cache=True)

        # Multiple calls should be safe
        multi_retriever.clear_cache()
        multi_retriever.clear_cache()
        multi_retriever.clear_cache()

    def test_multi_index_balancing(self):
        """Test per-index balancing distributes sources fairly."""
        # Create mock nodes with different scores
        # Retriever 1 has higher embedding scores
        nodes1 = [
            MagicMock(score=0.9, metadata={}),
            MagicMock(score=0.8, metadata={}),
            MagicMock(score=0.7, metadata={}),
        ]

        # Retriever 2 has lower embedding scores
        nodes2 = [
            MagicMock(score=0.6, metadata={}),
            MagicMock(score=0.5, metadata={}),
            MagicMock(score=0.4, metadata={}),
        ]

        retriever1 = MagicMock()
        retriever1.retrieve.return_value = nodes1

        retriever2 = MagicMock()
        retriever2.retrieve.return_value = nodes2

        # Create balanced retriever
        multi_retriever = MultiIndexRetriever(
            [retriever1, retriever2], balance_strategy="top_k_per_index"
        )

        # Execute retrieval
        query_bundle = QueryBundle(query_str="test")
        results = multi_retriever._retrieve(query_bundle)

        # Verify nodes are tagged with source index
        for node in results:
            assert "_source_index" in node.metadata

        # Count nodes from each index
        index0_count = sum(1 for n in results if n.metadata.get("_source_index") == 0)
        index1_count = sum(1 for n in results if n.metadata.get("_source_index") == 1)

        # Should be balanced (3 from each)
        assert index0_count == 3
        assert index1_count == 3

    def test_multi_index_balancing_disabled(self):
        """Test that balancing can be disabled."""
        nodes1 = [MagicMock(score=0.9, metadata={})]
        nodes2 = [MagicMock(score=0.6, metadata={})]

        retriever1 = MagicMock()
        retriever1.retrieve.return_value = nodes1

        retriever2 = MagicMock()
        retriever2.retrieve.return_value = nodes2

        # Create retriever with balancing disabled
        multi_retriever = MultiIndexRetriever(
            [retriever1, retriever2], balance_strategy="none"
        )

        query_bundle = QueryBundle(query_str="test")
        results = multi_retriever._retrieve(query_bundle)

        # Should still get all nodes, just not balanced
        assert len(results) == 2

    def test_multi_index_balancing_single_index(self):
        """Test that balancing doesn't break with single index."""
        nodes = [MagicMock(score=0.9, metadata={})]

        retriever = MagicMock()
        retriever.retrieve.return_value = nodes

        # Create retriever with single index (balancing should be skipped)
        multi_retriever = MultiIndexRetriever(
            [retriever], balance_strategy="top_k_per_index"
        )

        query_bundle = QueryBundle(query_str="test")
        results = multi_retriever._retrieve(query_bundle)

        # Should work normally
        assert len(results) == 1


# ============================================================================
# Tests for get_embed_model (mocked)
# ============================================================================


@pytest.mark.unit
class TestGetEmbedModel:
    """Tests for get_embed_model function via ModelManager."""

    def setup_method(self):
        """Reset ModelManager singleton before each test."""
        from tensortruth.services.model_manager import ModelManager

        ModelManager.reset_instance()

    @patch("tensortruth.services.model_manager.HuggingFaceEmbedding")
    def test_get_embed_model_cuda(self, mock_embedding_class):
        """Test embedding model initialization with CUDA."""
        from tensortruth.rag_engine import get_embed_model

        mock_model = MagicMock()
        mock_embedding_class.return_value = mock_model

        result = get_embed_model(device="cuda")

        assert result == mock_model
        mock_embedding_class.assert_called_once()

        # Check that device was passed correctly
        call_kwargs = mock_embedding_class.call_args.kwargs
        assert call_kwargs["device"] == "cuda"
        assert "BAAI/bge-m3" in call_kwargs["model_name"]

    @patch("tensortruth.services.model_manager.HuggingFaceEmbedding")
    def test_get_embed_model_cpu(self, mock_embedding_class):
        """Test embedding model initialization with CPU."""
        from tensortruth.rag_engine import get_embed_model

        mock_model = MagicMock()
        mock_embedding_class.return_value = mock_model

        result = get_embed_model(device="cpu")
        assert result == mock_model

        call_kwargs = mock_embedding_class.call_args.kwargs
        assert call_kwargs["device"] == "cpu"

    @patch("tensortruth.services.model_manager.HuggingFaceEmbedding")
    def test_get_embed_model_mps(self, mock_embedding_class):
        """Test embedding model initialization with MPS (skipped if not available)."""
        import torch

        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            pytest.skip("MPS not available on this system")

        from tensortruth.rag_engine import get_embed_model

        mock_model = MagicMock()
        mock_embedding_class.return_value = mock_model

        result = get_embed_model(device="mps")
        assert result == mock_model

        call_kwargs = mock_embedding_class.call_args.kwargs
        assert call_kwargs["device"] == "mps"


# ============================================================================
# Tests for get_llm (mocked)
# ============================================================================


@pytest.mark.unit
class TestGetLLM:
    """Tests for get_llm function."""

    @patch("tensortruth.rag_engine.Ollama")
    def test_get_llm_defaults(self, mock_ollama_class):
        """Test LLM initialization with default parameters."""
        from tensortruth.rag_engine import get_llm

        mock_llm = MagicMock()
        mock_ollama_class.return_value = mock_llm

        params = {}
        result = get_llm(params)

        assert result == mock_llm
        mock_ollama_class.assert_called_once()

    @patch("tensortruth.rag_engine.Ollama")
    def test_get_llm_custom_params(self, mock_ollama_class):
        """Test LLM initialization with custom parameters."""
        from tensortruth.rag_engine import get_llm

        mock_llm = MagicMock()
        mock_ollama_class.return_value = mock_llm

        params = {
            "model": "llama2:7b",
            "temperature": 0.5,
            "context_window": 8192,
            "system_prompt": "You are a helpful assistant.",
            "llm_device": "gpu",
        }

        result = get_llm(params)
        assert result == mock_llm

        call_kwargs = mock_ollama_class.call_args[1]
        assert call_kwargs["model"] == "llama2:7b"
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["context_window"] == 8192

    @patch("tensortruth.rag_engine.Ollama")
    def test_get_llm_cpu_mode(self, mock_ollama_class):
        """Test LLM initialization with CPU mode."""
        from tensortruth.rag_engine import get_llm

        mock_llm = MagicMock()
        mock_ollama_class.return_value = mock_llm

        params = {"llm_device": "cpu"}
        result = get_llm(params)
        assert result == mock_llm

        # Should set num_gpu to 0 for CPU mode
        call_kwargs = mock_ollama_class.call_args[1]
        assert call_kwargs["additional_kwargs"]["num_gpu"] == 0


# ============================================================================
# Tests for get_reranker (mocked)
# ============================================================================


@pytest.mark.unit
class TestGetReranker:
    """Tests for get_reranker function via ModelManager."""

    def setup_method(self):
        """Reset ModelManager singleton before each test."""
        from tensortruth.services.model_manager import ModelManager

        ModelManager.reset_instance()

    @patch("tensortruth.services.model_manager.SentenceTransformerRerank")
    def test_get_reranker_defaults(self, mock_reranker_class):
        """Test reranker initialization with defaults."""
        from tensortruth.rag_engine import get_reranker

        mock_reranker = MagicMock()
        mock_reranker_class.return_value = mock_reranker

        params = {}
        result = get_reranker(params, device="cuda")

        assert result == mock_reranker
        call_kwargs = mock_reranker_class.call_args.kwargs
        assert call_kwargs["device"] == "cuda"
        assert call_kwargs["top_n"] == 3  # default

    @patch("tensortruth.services.model_manager.SentenceTransformerRerank")
    def test_get_reranker_custom_model(self, mock_reranker_class):
        """Test reranker with custom model."""
        from tensortruth.rag_engine import get_reranker

        mock_reranker = MagicMock()
        mock_reranker_class.return_value = mock_reranker

        params = {"reranker_model": "BAAI/bge-reranker-base", "reranker_top_n": 5}
        result = get_reranker(params, device="cpu")
        assert result == mock_reranker

        call_kwargs = mock_reranker_class.call_args.kwargs
        assert call_kwargs["model"] == "BAAI/bge-reranker-base"
        assert call_kwargs["top_n"] == 5
        assert call_kwargs["device"] == "cpu"


# ============================================================================
# Tests for Metadata Filtering (Phase 3)
# ============================================================================


@pytest.mark.unit
class TestMetadataFiltering:
    """Tests for metadata filtering functionality."""

    def test_build_metadata_filters_function_exists(self):
        """Test that _build_metadata_filters function exists."""
        from tensortruth.rag_engine import _build_metadata_filters

        assert callable(_build_metadata_filters)

    def test_build_metadata_filters_simple_equality(self):
        """Test building filters with simple equality checks."""
        from tensortruth.rag_engine import _build_metadata_filters

        filter_spec = {"doc_type": "library"}
        result = _build_metadata_filters(filter_spec)

        assert result is not None
        assert len(result.filters) == 1
        assert result.filters[0].key == "doc_type"
        assert result.filters[0].value == "library"

    def test_build_metadata_filters_multiple_conditions(self):
        """Test building filters with multiple conditions."""
        from tensortruth.rag_engine import _build_metadata_filters

        filter_spec = {"doc_type": "library", "source": "pytorch"}
        result = _build_metadata_filters(filter_spec)

        assert result is not None
        assert len(result.filters) == 2

    def test_build_metadata_filters_with_operators(self):
        """Test building filters with explicit operators."""
        from tensortruth.rag_engine import _build_metadata_filters

        # Filter spec with operator syntax
        filter_spec = {"version": {"$gte": "2.0"}}
        result = _build_metadata_filters(filter_spec)

        assert result is not None
        assert len(result.filters) == 1
        # The filter should use GTE operator
        assert result.filters[0].key == "version"

    def test_build_metadata_filters_returns_none_for_empty(self):
        """Test that empty filter spec returns None."""
        from tensortruth.rag_engine import _build_metadata_filters

        result = _build_metadata_filters({})
        assert result is None

        result = _build_metadata_filters(None)
        assert result is None

    def test_build_metadata_filters_list_values(self):
        """Test building filters with list values (IN operator)."""
        from tensortruth.rag_engine import _build_metadata_filters

        filter_spec = {"doc_type": ["library", "book"]}
        result = _build_metadata_filters(filter_spec)

        assert result is not None
        # Should use IN operator for lists
        assert len(result.filters) == 1
