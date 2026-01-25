"""Unit tests for ModelManager service."""

import pytest

from tensortruth.services.model_manager import ModelManager


@pytest.mark.unit
def test_get_embedder_returns_non_none():
    """Verify get_embedder always returns valid instance."""
    manager = ModelManager.get_instance()
    manager.set_default_device("cpu")

    embedder = manager.get_embedder()

    assert embedder is not None
    assert hasattr(embedder, "get_text_embedding")


@pytest.mark.unit
def test_get_embedder_raises_on_invalid_model():
    """Verify clear error when model loading fails."""
    manager = ModelManager.get_instance()

    with pytest.raises(RuntimeError, match="Failed to load embedding model"):
        manager.get_embedder(model_name="invalid/nonexistent-model-12345")


@pytest.mark.unit
def test_get_reranker_returns_non_none():
    """Verify get_reranker always returns valid instance."""
    manager = ModelManager.get_instance()
    manager.set_default_device("cpu")

    reranker = manager.get_reranker()

    assert reranker is not None
    assert hasattr(reranker, "postprocess_nodes")
