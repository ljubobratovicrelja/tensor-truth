"""Unit tests for tensortruth.api.routes.chat module."""

from unittest.mock import MagicMock

import pytest

from tensortruth.api.routes.chat import _extract_sources
from tensortruth.api.schemas.chat import SourceNode


@pytest.mark.unit
class TestExtractSources:
    """Tests for _extract_sources helper function."""

    def test_extract_from_node_with_score(self):
        """Test extraction from LlamaIndex NodeWithScore objects."""
        # Mock NodeWithScore with inner node
        inner_node = MagicMock()
        inner_node.get_content.return_value = (
            "This is the merged content from retriever"
        )

        node_with_score = MagicMock()
        node_with_score.node = inner_node
        node_with_score.score = 0.95
        node_with_score.metadata = {"source": "test.pdf", "page": 5}

        result = _extract_sources([node_with_score])

        assert len(result) == 1
        assert isinstance(result[0], SourceNode)
        assert result[0].text == "This is the merged content from retriever"
        assert result[0].score == 0.95
        assert result[0].metadata["source"] == "test.pdf"
        assert result[0].metadata["page"] == 5

    def test_empty_source_nodes_list(self):
        """Test that empty list returns empty list."""
        result = _extract_sources([])

        assert result == []

    def test_metadata_extraction_all_fields(self):
        """Test metadata extraction with various field types."""
        inner_node = MagicMock()
        inner_node.get_content.return_value = "Content"

        node = MagicMock()
        node.node = inner_node
        node.score = 0.8
        node.metadata = {
            "doc_type": "paper",
            "source": "arxiv",
            "title": "Neural Networks Paper",
            "file_name": "paper.pdf",
            "page_label": "3",
            "authors": ["Author A", "Author B"],
        }

        result = _extract_sources([node])

        assert result[0].metadata["doc_type"] == "paper"
        assert result[0].metadata["source"] == "arxiv"
        assert result[0].metadata["title"] == "Neural Networks Paper"
        assert result[0].metadata["authors"] == ["Author A", "Author B"]

    def test_score_field_mapping(self):
        """Test score is correctly mapped from node."""
        inner_node = MagicMock()
        inner_node.get_content.return_value = "Text"

        node = MagicMock()
        node.node = inner_node
        node.score = 0.7654
        node.metadata = {}

        result = _extract_sources([node])

        assert result[0].score == 0.7654

    def test_missing_score_attribute(self):
        """Test handling of nodes without score attribute.

        When a node has no score, effective_score returns a status-based default:
        - SUCCESS (default status): 1.0
        - Other statuses: 0.0
        """
        inner_node = MagicMock()
        inner_node.get_content.return_value = "Text"

        node = MagicMock(spec=["node", "metadata"])  # No 'score' in spec
        node.node = inner_node
        node.metadata = {}
        # hasattr(node, 'score') will return False

        result = _extract_sources([node])

        # No explicit score + SUCCESS status = effective_score of 1.0
        assert result[0].score == 1.0

    def test_missing_metadata_attribute(self):
        """Test handling of nodes without metadata attribute.

        With SourceConverter, unified source fields are added to metadata
        even when the original node has no metadata.
        """
        inner_node = MagicMock()
        inner_node.get_content.return_value = "Text"

        node = MagicMock(spec=["node", "score"])  # No 'metadata' in spec
        node.node = inner_node
        node.score = 0.5

        result = _extract_sources([node])

        # SourceConverter adds unified source fields to metadata
        assert "doc_type" in result[0].metadata
        assert result[0].metadata["fetch_status"] == "success"

    def test_fallback_to_text_attribute(self):
        """Test fallback to .text when get_content() not available."""
        # Node without get_content method
        inner_node = MagicMock(spec=["text"])
        inner_node.text = "Text from .text attribute"

        node = MagicMock()
        node.node = inner_node
        node.text = "Fallback text"
        node.score = 0.6
        node.metadata = {}

        result = _extract_sources([node])

        # Should use node.text as fallback
        assert result[0].text == "Fallback text"

    def test_fallback_to_str_conversion(self):
        """Test fallback to str(node) when no text methods available."""
        # Node without get_content or text
        inner_node = MagicMock(spec=[])

        node = MagicMock(spec=["node", "score", "metadata"])
        node.node = inner_node
        node.score = 0.4
        node.metadata = {}

        result = _extract_sources([node])

        # Should convert to string
        assert isinstance(result[0].text, str)

    def test_raw_node_without_inner_node(self):
        """Test handling of raw nodes without .node wrapper."""
        # Direct node without NodeWithScore wrapper
        node = MagicMock()
        # No .node attribute - simulate raw node
        del node.node
        node.get_content = MagicMock(return_value="Direct content")
        node.score = 0.9
        node.metadata = {"source": "direct"}

        result = _extract_sources([node])

        assert len(result) == 1
        assert result[0].text == "Direct content"
        assert result[0].score == 0.9

    def test_multiple_source_nodes(self):
        """Test extraction from multiple source nodes."""
        nodes = []
        for i in range(3):
            inner_node = MagicMock()
            inner_node.get_content.return_value = f"Content {i}"

            node = MagicMock()
            node.node = inner_node
            node.score = 0.9 - i * 0.1
            node.metadata = {"index": i}
            nodes.append(node)

        result = _extract_sources(nodes)

        assert len(result) == 3
        assert result[0].text == "Content 0"
        assert result[0].score == 0.9
        assert result[1].text == "Content 1"
        assert result[1].score == 0.8
        assert result[2].text == "Content 2"
        assert result[2].score == 0.7

    def test_exception_handling_skips_bad_node(self):
        """Test that exceptions are caught and bad nodes skipped."""
        # Good node
        good_inner = MagicMock()
        good_inner.get_content.return_value = "Good content"
        good_node = MagicMock()
        good_node.node = good_inner
        good_node.score = 0.8
        good_node.metadata = {}

        # Bad node that raises exception
        bad_inner = MagicMock()
        bad_inner.get_content.side_effect = Exception("Extraction error")
        bad_node = MagicMock()
        bad_node.node = bad_inner
        bad_node.score = 0.5
        bad_node.metadata = {}

        result = _extract_sources([good_node, bad_node])

        # Only good node should be extracted
        assert len(result) == 1
        assert result[0].text == "Good content"

    def test_none_metadata_handled_gracefully(self):
        """Test that None metadata is handled gracefully by SourceConverter.

        With SourceConverter, None metadata is treated as empty dict
        and unified source fields are added.
        """
        inner_node = MagicMock()
        inner_node.get_content.return_value = "Content"

        node = MagicMock()
        node.node = inner_node
        node.score = 0.7
        node.metadata = None  # Explicitly None - now handled gracefully

        result = _extract_sources([node])

        # Node should be extracted with unified metadata fields
        assert len(result) == 1
        assert result[0].text == "Content"
        assert result[0].score == 0.7
        assert "doc_type" in result[0].metadata
        assert result[0].metadata["fetch_status"] == "success"

    def test_get_content_preferred_over_text(self):
        """Test that get_content() is preferred when both available."""
        inner_node = MagicMock()
        inner_node.get_content.return_value = "Merged parent content"
        inner_node.text = "Leaf node text"

        node = MagicMock()
        node.node = inner_node
        node.score = 0.8
        node.metadata = {}

        result = _extract_sources([node])

        # get_content() should be preferred (returns merged content)
        assert result[0].text == "Merged parent content"
