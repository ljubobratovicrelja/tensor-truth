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
        """Test that retrieval preserves order from retrievers."""
        nodes = [MagicMock() for _ in range(5)]

        retriever1 = MagicMock()
        retriever1.retrieve.return_value = nodes[:3]

        retriever2 = MagicMock()
        retriever2.retrieve.return_value = nodes[3:]

        multi_retriever = MultiIndexRetriever([retriever1, retriever2])

        query_bundle = QueryBundle(query_str="test query")
        results = multi_retriever._retrieve(query_bundle)

        # Should get nodes from retriever1 first, then retriever2
        assert len(results) == 5
        assert results[:3] == nodes[:3]
        assert results[3:] == nodes[3:]


# ============================================================================
# Tests for get_embed_model (mocked)
# ============================================================================


@pytest.mark.unit
class TestGetEmbedModel:
    """Tests for get_embed_model function."""

    @patch("tensortruth.rag_engine.HuggingFaceEmbedding")
    def test_get_embed_model_cuda(self, mock_embedding_class):
        """Test embedding model initialization with CUDA."""
        from tensortruth.rag_engine import get_embed_model

        mock_model = MagicMock()
        mock_embedding_class.return_value = mock_model

        result = get_embed_model(device="cuda")

        assert result == mock_model
        mock_embedding_class.assert_called_once()

        # Check that device was passed correctly
        call_kwargs = mock_embedding_class.call_args[1]
        assert call_kwargs["device"] == "cuda"
        assert "BAAI/bge-m3" in call_kwargs["model_name"]

    @patch("tensortruth.rag_engine.HuggingFaceEmbedding")
    def test_get_embed_model_cpu(self, mock_embedding_class):
        """Test embedding model initialization with CPU."""
        from tensortruth.rag_engine import get_embed_model

        mock_model = MagicMock()
        mock_embedding_class.return_value = mock_model

        result = get_embed_model(device="cpu")

        call_kwargs = mock_embedding_class.call_args[1]
        assert call_kwargs["device"] == "cpu"

    @patch("tensortruth.rag_engine.HuggingFaceEmbedding")
    def test_get_embed_model_mps(self, mock_embedding_class):
        """Test embedding model initialization with MPS."""
        from tensortruth.rag_engine import get_embed_model

        mock_model = MagicMock()
        mock_embedding_class.return_value = mock_model

        result = get_embed_model(device="mps")

        call_kwargs = mock_embedding_class.call_args[1]
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

        # Should set num_gpu to 0 for CPU mode
        call_kwargs = mock_ollama_class.call_args[1]
        assert call_kwargs["additional_kwargs"]["options"]["num_gpu"] == 0


# ============================================================================
# Tests for get_reranker (mocked)
# ============================================================================


@pytest.mark.unit
class TestGetReranker:
    """Tests for get_reranker function."""

    @patch("tensortruth.rag_engine.SentenceTransformerRerank")
    def test_get_reranker_defaults(self, mock_reranker_class):
        """Test reranker initialization with defaults."""
        from tensortruth.rag_engine import get_reranker

        mock_reranker = MagicMock()
        mock_reranker_class.return_value = mock_reranker

        params = {}
        result = get_reranker(params, device="cuda")

        assert result == mock_reranker
        call_kwargs = mock_reranker_class.call_args[1]
        assert call_kwargs["device"] == "cuda"
        assert call_kwargs["top_n"] == 3  # default

    @patch("tensortruth.rag_engine.SentenceTransformerRerank")
    def test_get_reranker_custom_model(self, mock_reranker_class):
        """Test reranker with custom model."""
        from tensortruth.rag_engine import get_reranker

        mock_reranker = MagicMock()
        mock_reranker_class.return_value = mock_reranker

        params = {"reranker_model": "BAAI/bge-reranker-base", "reranker_top_n": 5}
        result = get_reranker(params, device="cpu")

        call_kwargs = mock_reranker_class.call_args[1]
        assert call_kwargs["model"] == "BAAI/bge-reranker-base"
        assert call_kwargs["top_n"] == 5
        assert call_kwargs["device"] == "cpu"
