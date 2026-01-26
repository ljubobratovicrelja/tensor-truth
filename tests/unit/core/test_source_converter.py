"""Unit tests for source converter."""

from dataclasses import dataclass
from typing import Any, Dict, Optional

import pytest

from tensortruth.core.source_converter import SourceConverter
from tensortruth.core.unified_sources import SourceStatus, SourceType, UnifiedSource


@dataclass
class MockWebSourceNode:
    """Mock of core/sources.py SourceNode for testing."""

    url: str
    title: str
    status: str  # "success", "failed", "skipped"
    error: Optional[str] = None
    snippet: Optional[str] = None
    content: Optional[str] = None
    content_chars: int = 0
    relevance_score: Optional[float] = None


class MockRAGNode:
    """Mock of llama_index NodeWithScore for testing."""

    def __init__(
        self,
        text: str,
        score: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        node_id: Optional[str] = None,
    ):
        self.text = text
        self.score = score
        self.metadata = metadata or {}
        self.id_ = node_id

    def get_content(self) -> str:
        return self.text


class MockNodeWithScore:
    """Mock of llama_index NodeWithScore wrapper."""

    def __init__(self, inner_node: MockRAGNode, score: float):
        self.node = inner_node
        self.score = score
        self.metadata = inner_node.metadata
        self.id_ = inner_node.id_


@pytest.mark.unit
class TestFromWebSource:
    """Tests for from_web_source converter."""

    def test_basic_conversion(self):
        """Test basic conversion with all fields populated."""
        web_source = MockWebSourceNode(
            url="https://example.com/page",
            title="Example Page",
            status="success",
            error=None,
            snippet="A preview snippet",
            content="Full page content here",
            content_chars=100,
            relevance_score=0.85,
        )

        result = SourceConverter.from_web_source(web_source)

        assert result.url == "https://example.com/page"
        assert result.title == "Example Page"
        assert result.status == SourceStatus.SUCCESS
        assert result.error is None
        assert result.snippet == "A preview snippet"
        assert result.content == "Full page content here"
        assert result.content_chars == 100
        assert result.score == 0.85
        assert result.source_type == SourceType.WEB

    def test_failed_status_mapping(self):
        """Test that failed status is mapped correctly."""
        web_source = MockWebSourceNode(
            url="https://example.com",
            title="Failed Page",
            status="failed",
            error="Connection timeout",
        )

        result = SourceConverter.from_web_source(web_source)

        assert result.status == SourceStatus.FAILED
        assert result.error == "Connection timeout"

    def test_skipped_status_mapping(self):
        """Test that skipped status is mapped correctly."""
        web_source = MockWebSourceNode(
            url="https://example.com",
            title="Skipped Page",
            status="skipped",
        )

        result = SourceConverter.from_web_source(web_source)

        assert result.status == SourceStatus.SKIPPED

    def test_missing_optional_fields(self):
        """Test conversion with missing optional fields."""
        web_source = MockWebSourceNode(
            url="https://example.com",
            title="Minimal Page",
            status="success",
            snippet=None,
            content=None,
            relevance_score=None,
        )

        result = SourceConverter.from_web_source(web_source)

        assert result.snippet is None
        assert result.content is None
        assert result.score is None

    def test_id_generated_from_url(self):
        """Test that stable ID is generated from URL."""
        web_source = MockWebSourceNode(
            url="https://example.com/specific-page",
            title="Test",
            status="success",
        )

        result = SourceConverter.from_web_source(web_source)

        assert result.id is not None
        assert len(result.id) == 16  # SHA256 hash truncated to 16 chars

    def test_same_url_produces_same_id(self):
        """Test that same URL produces same ID."""
        web_source1 = MockWebSourceNode(
            url="https://example.com/page",
            title="Page 1",
            status="success",
        )
        web_source2 = MockWebSourceNode(
            url="https://example.com/page",
            title="Page 2",
            status="failed",
        )

        result1 = SourceConverter.from_web_source(web_source1)
        result2 = SourceConverter.from_web_source(web_source2)

        assert result1.id == result2.id


@pytest.mark.unit
class TestFromRAGNode:
    """Tests for from_rag_node converter."""

    def test_basic_conversion(self):
        """Test basic RAG node conversion."""
        node = MockRAGNode(
            text="Document content from RAG",
            score=0.75,
            metadata={
                "source_url": "https://docs.example.com/api",
                "display_name": "API Documentation",
                "doc_type": "sphinx",
            },
            node_id="node-123",
        )

        result = SourceConverter.from_rag_node(node)

        assert result.content == "Document content from RAG"
        assert result.score == 0.75
        assert result.url == "https://docs.example.com/api"
        assert result.title == "API Documentation"
        assert result.source_type == SourceType.LIBRARY_DOC

    def test_node_with_score_wrapper(self):
        """Test conversion of NodeWithScore wrapper."""
        inner = MockRAGNode(
            text="Inner node content",
            metadata={"title": "Test Doc"},
            node_id="inner-456",
        )
        wrapper = MockNodeWithScore(inner, score=0.88)

        result = SourceConverter.from_rag_node(wrapper)

        assert result.content == "Inner node content"
        assert result.score == 0.88
        assert result.title == "Test Doc"

    def test_paper_doc_type_mapping(self):
        """Test that arxiv/paper doc types map to PAPER."""
        node = MockRAGNode(
            text="Paper abstract",
            metadata={"doc_type": "arxiv"},
        )

        result = SourceConverter.from_rag_node(node)

        assert result.source_type == SourceType.PAPER

    def test_book_doc_type_mapping(self):
        """Test that book doc types map to BOOK."""
        node = MockRAGNode(
            text="Book chapter",
            metadata={"doc_type": "pdf_book"},
        )

        result = SourceConverter.from_rag_node(node)

        assert result.source_type == SourceType.BOOK

    def test_uploaded_pdf_mapping(self):
        """Test that pdf doc type maps to UPLOADED_PDF."""
        node = MockRAGNode(
            text="PDF content",
            metadata={"doc_type": "pdf"},
        )

        result = SourceConverter.from_rag_node(node)

        assert result.source_type == SourceType.UPLOADED_PDF

    def test_missing_metadata(self):
        """Test conversion with minimal metadata."""
        node = MockRAGNode(text="Just text")

        result = SourceConverter.from_rag_node(node)

        assert result.content == "Just text"
        assert result.title == "Untitled"
        assert result.url is None
        assert result.source_type == SourceType.WEB  # default

    def test_snippet_generated_from_content(self):
        """Test that snippet is generated from long content."""
        long_text = "x" * 1000
        node = MockRAGNode(text=long_text)

        result = SourceConverter.from_rag_node(node)

        assert result.snippet == long_text[:500]

    def test_content_chars_computed(self):
        """Test that content_chars is computed."""
        node = MockRAGNode(text="Hello world")

        result = SourceConverter.from_rag_node(node)

        assert result.content_chars == 11


@pytest.mark.unit
class TestToAPISchema:
    """Tests for to_api_schema converter."""

    def test_basic_conversion(self):
        """Test conversion to API schema dict."""
        source = UnifiedSource(
            id="api-test",
            url="https://example.com",
            title="Test Source",
            content="Full content",
            snippet="Snippet",
            score=0.8,
            status=SourceStatus.SUCCESS,
            source_type=SourceType.WEB,
            content_chars=50,
            metadata={"extra": "data"},
        )

        result = SourceConverter.to_api_schema(source)

        assert result["text"] == "Full content"
        assert result["score"] == 0.8
        assert result["metadata"]["source_url"] == "https://example.com"
        assert result["metadata"]["display_name"] == "Test Source"
        assert result["metadata"]["doc_type"] == "web"
        assert result["metadata"]["fetch_status"] == "success"
        assert result["metadata"]["content_chars"] == 50
        assert result["metadata"]["extra"] == "data"

    def test_text_fallback_to_snippet(self):
        """Test that text falls back to snippet when no content."""
        source = UnifiedSource(
            id="fallback",
            title="Test",
            source_type=SourceType.WEB,
            content=None,
            snippet="Just snippet",
        )

        result = SourceConverter.to_api_schema(source)

        assert result["text"] == "Just snippet"

    def test_text_empty_when_no_content(self):
        """Test that text is empty when no content or snippet."""
        source = UnifiedSource(
            id="empty",
            title="Test",
            source_type=SourceType.WEB,
        )

        result = SourceConverter.to_api_schema(source)

        assert result["text"] == ""

    def test_error_included_in_metadata(self):
        """Test that error is included in metadata when present."""
        source = UnifiedSource(
            id="error-test",
            title="Failed",
            source_type=SourceType.WEB,
            status=SourceStatus.FAILED,
            error="Fetch failed",
        )

        result = SourceConverter.to_api_schema(source)

        assert result["metadata"]["fetch_error"] == "Fetch failed"


@pytest.mark.unit
class TestToWebSearchSchema:
    """Tests for to_web_search_schema converter."""

    def test_basic_conversion(self):
        """Test conversion to WebSearchSource schema."""
        source = UnifiedSource(
            id="web-schema",
            url="https://example.com",
            title="Web Page",
            snippet="Page snippet",
            status=SourceStatus.SUCCESS,
            error=None,
            source_type=SourceType.WEB,
        )

        result = SourceConverter.to_web_search_schema(source)

        assert result["url"] == "https://example.com"
        assert result["title"] == "Web Page"
        assert result["status"] == "success"
        assert result["snippet"] == "Page snippet"
        assert result["error"] is None

    def test_filtered_status_becomes_skipped(self):
        """Test that FILTERED status is mapped to 'skipped' for API."""
        source = UnifiedSource(
            id="filtered",
            title="Low Score",
            source_type=SourceType.WEB,
            status=SourceStatus.FILTERED,
        )

        result = SourceConverter.to_web_search_schema(source)

        assert result["status"] == "skipped"

    def test_empty_url_becomes_empty_string(self):
        """Test that None URL becomes empty string."""
        source = UnifiedSource(
            id="no-url",
            title="No URL",
            source_type=SourceType.WEB,
            url=None,
        )

        result = SourceConverter.to_web_search_schema(source)

        assert result["url"] == ""


@pytest.mark.unit
class TestBatchOperations:
    """Tests for batch conversion operations."""

    def test_batch_from_rag_nodes(self):
        """Test batch conversion of RAG nodes."""
        nodes = [
            MockRAGNode(text="First", metadata={"title": "Doc 1"}),
            MockRAGNode(text="Second", metadata={"title": "Doc 2"}),
            MockRAGNode(text="Third", metadata={"title": "Doc 3"}),
        ]

        results = SourceConverter.batch_from_rag_nodes(nodes)

        assert len(results) == 3
        assert results[0].content == "First"
        assert results[1].content == "Second"
        assert results[2].content == "Third"

    def test_batch_to_api_schema(self):
        """Test batch conversion to API schema."""
        sources = [
            UnifiedSource(
                id="1", title="A", source_type=SourceType.WEB, content="Content A"
            ),
            UnifiedSource(
                id="2", title="B", source_type=SourceType.PAPER, content="Content B"
            ),
        ]

        results = SourceConverter.batch_to_api_schema(sources)

        assert len(results) == 2
        assert results[0]["text"] == "Content A"
        assert results[1]["text"] == "Content B"


@pytest.mark.unit
class TestRoundtripConversion:
    """Tests for roundtrip conversion consistency."""

    def test_web_source_roundtrip(self):
        """Test that web source -> unified -> api preserves key data."""
        web_source = MockWebSourceNode(
            url="https://example.com/test",
            title="Test Page",
            status="success",
            content="Page content",
            content_chars=12,
            relevance_score=0.9,
        )

        unified = SourceConverter.from_web_source(web_source)
        api_dict = SourceConverter.to_api_schema(unified)

        assert api_dict["text"] == "Page content"
        assert api_dict["score"] == 0.9
        assert api_dict["metadata"]["source_url"] == "https://example.com/test"
        assert api_dict["metadata"]["display_name"] == "Test Page"
        assert api_dict["metadata"]["fetch_status"] == "success"

    def test_rag_node_roundtrip(self):
        """Test that rag node -> unified -> api preserves key data."""
        rag_node = MockRAGNode(
            text="RAG document text",
            score=0.77,
            metadata={
                "source_url": "https://docs.example.com",
                "display_name": "Documentation",
                "doc_type": "sphinx",
            },
        )

        unified = SourceConverter.from_rag_node(rag_node)
        api_dict = SourceConverter.to_api_schema(unified)

        assert api_dict["text"] == "RAG document text"
        assert api_dict["score"] == 0.77
        assert api_dict["metadata"]["source_url"] == "https://docs.example.com"
        assert api_dict["metadata"]["display_name"] == "Documentation"
        assert api_dict["metadata"]["doc_type"] == "library_doc"
