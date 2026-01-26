"""Unit tests for unified source model."""

import pytest

from tensortruth.core.unified_sources import (
    SourceStatus,
    SourceType,
    UnifiedSource,
)


@pytest.mark.unit
class TestSourceStatus:
    """Tests for SourceStatus enum."""

    def test_status_values(self):
        """Test that SourceStatus has correct string values."""
        assert SourceStatus.SUCCESS.value == "success"
        assert SourceStatus.FAILED.value == "failed"
        assert SourceStatus.SKIPPED.value == "skipped"
        assert SourceStatus.FILTERED.value == "filtered"

    def test_status_from_string(self):
        """Test creating SourceStatus from string."""
        assert SourceStatus("success") == SourceStatus.SUCCESS
        assert SourceStatus("failed") == SourceStatus.FAILED
        assert SourceStatus("skipped") == SourceStatus.SKIPPED
        assert SourceStatus("filtered") == SourceStatus.FILTERED


@pytest.mark.unit
class TestSourceType:
    """Tests for SourceType enum."""

    def test_type_values(self):
        """Test that SourceType has correct string values."""
        assert SourceType.WEB.value == "web"
        assert SourceType.PAPER.value == "paper"
        assert SourceType.LIBRARY_DOC.value == "library_doc"
        assert SourceType.UPLOADED_PDF.value == "uploaded_pdf"
        assert SourceType.BOOK.value == "book"

    def test_type_from_string(self):
        """Test creating SourceType from string."""
        assert SourceType("web") == SourceType.WEB
        assert SourceType("paper") == SourceType.PAPER
        assert SourceType("library_doc") == SourceType.LIBRARY_DOC


@pytest.mark.unit
class TestUnifiedSource:
    """Tests for UnifiedSource dataclass."""

    def test_creation_with_all_fields(self):
        """Test creating UnifiedSource with all fields populated."""
        source = UnifiedSource(
            id="test-123",
            url="https://example.com/doc",
            title="Test Document",
            content="Full content here",
            snippet="Preview snippet",
            score=0.85,
            status=SourceStatus.SUCCESS,
            error=None,
            source_type=SourceType.WEB,
            metadata={"custom": "value"},
            content_chars=100,
        )

        assert source.id == "test-123"
        assert source.url == "https://example.com/doc"
        assert source.title == "Test Document"
        assert source.content == "Full content here"
        assert source.snippet == "Preview snippet"
        assert source.score == 0.85
        assert source.status == SourceStatus.SUCCESS
        assert source.error is None
        assert source.source_type == SourceType.WEB
        assert source.metadata == {"custom": "value"}
        assert source.content_chars == 100

    def test_creation_with_defaults(self):
        """Test creating UnifiedSource with default values."""
        source = UnifiedSource(
            id="test-456",
            title="Minimal Source",
            source_type=SourceType.PAPER,
        )

        assert source.id == "test-456"
        assert source.url is None
        assert source.title == "Minimal Source"
        assert source.content is None
        assert source.snippet is None
        assert source.score is None
        assert source.status == SourceStatus.SUCCESS
        assert source.error is None
        assert source.source_type == SourceType.PAPER
        assert source.metadata == {}
        assert source.content_chars == 0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        source = UnifiedSource(
            id="dict-test",
            url="https://example.com",
            title="Dict Test",
            content="Content",
            snippet="Snippet",
            score=0.9,
            status=SourceStatus.FILTERED,
            error=None,
            source_type=SourceType.LIBRARY_DOC,
            metadata={"key": "val"},
            content_chars=50,
        )

        result = source.to_dict()

        assert result["id"] == "dict-test"
        assert result["url"] == "https://example.com"
        assert result["title"] == "Dict Test"
        assert result["content"] == "Content"
        assert result["snippet"] == "Snippet"
        assert result["score"] == 0.9
        assert result["status"] == "filtered"
        assert result["error"] is None
        assert result["source_type"] == "library_doc"
        assert result["metadata"] == {"key": "val"}
        assert result["content_chars"] == 50

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "id": "from-dict",
            "url": "https://test.com",
            "title": "From Dict",
            "content": "Some content",
            "snippet": "Some snippet",
            "score": 0.75,
            "status": "success",
            "error": None,
            "source_type": "paper",
            "metadata": {"foo": "bar"},
            "content_chars": 25,
        }

        source = UnifiedSource.from_dict(data)

        assert source.id == "from-dict"
        assert source.url == "https://test.com"
        assert source.title == "From Dict"
        assert source.content == "Some content"
        assert source.score == 0.75
        assert source.status == SourceStatus.SUCCESS
        assert source.source_type == SourceType.PAPER
        assert source.metadata == {"foo": "bar"}

    def test_from_dict_with_defaults(self):
        """Test creation from dict with missing optional fields."""
        data = {
            "id": "minimal",
            "title": "Minimal",
        }

        source = UnifiedSource.from_dict(data)

        assert source.id == "minimal"
        assert source.title == "Minimal"
        assert source.url is None
        assert source.content is None
        assert source.status == SourceStatus.SUCCESS
        assert source.source_type == SourceType.WEB

    def test_roundtrip_conversion(self):
        """Test that to_dict -> from_dict preserves data."""
        original = UnifiedSource(
            id="roundtrip",
            url="https://roundtrip.test",
            title="Roundtrip Test",
            content="Round trip content",
            snippet="RT snippet",
            score=0.55,
            status=SourceStatus.SKIPPED,
            error="Some error",
            source_type=SourceType.BOOK,
            metadata={"round": "trip"},
            content_chars=200,
        )

        data = original.to_dict()
        restored = UnifiedSource.from_dict(data)

        assert restored.id == original.id
        assert restored.url == original.url
        assert restored.title == original.title
        assert restored.content == original.content
        assert restored.snippet == original.snippet
        assert restored.score == original.score
        assert restored.status == original.status
        assert restored.error == original.error
        assert restored.source_type == original.source_type
        assert restored.metadata == original.metadata
        assert restored.content_chars == original.content_chars

    def test_get_display_text_with_content(self):
        """Test get_display_text returns content when available."""
        source = UnifiedSource(
            id="display-1",
            title="Test",
            source_type=SourceType.WEB,
            content="Full content",
            snippet="Just snippet",
        )

        assert source.get_display_text() == "Full content"

    def test_get_display_text_with_snippet_only(self):
        """Test get_display_text falls back to snippet."""
        source = UnifiedSource(
            id="display-2",
            title="Test",
            source_type=SourceType.WEB,
            content=None,
            snippet="Just snippet",
        )

        assert source.get_display_text() == "Just snippet"

    def test_get_display_text_empty(self):
        """Test get_display_text returns empty string when no content."""
        source = UnifiedSource(
            id="display-3",
            title="Test",
            source_type=SourceType.WEB,
        )

        assert source.get_display_text() == ""

    def test_is_successful(self):
        """Test is_successful method."""
        success = UnifiedSource(
            id="s1", title="T", source_type=SourceType.WEB, status=SourceStatus.SUCCESS
        )
        failed = UnifiedSource(
            id="s2", title="T", source_type=SourceType.WEB, status=SourceStatus.FAILED
        )
        filtered = UnifiedSource(
            id="s3", title="T", source_type=SourceType.WEB, status=SourceStatus.FILTERED
        )

        assert success.is_successful() is True
        assert failed.is_successful() is False
        assert filtered.is_successful() is False

    def test_is_usable(self):
        """Test is_usable method."""
        success = UnifiedSource(
            id="u1", title="T", source_type=SourceType.WEB, status=SourceStatus.SUCCESS
        )
        filtered = UnifiedSource(
            id="u2", title="T", source_type=SourceType.WEB, status=SourceStatus.FILTERED
        )
        failed = UnifiedSource(
            id="u3", title="T", source_type=SourceType.WEB, status=SourceStatus.FAILED
        )
        skipped = UnifiedSource(
            id="u4", title="T", source_type=SourceType.WEB, status=SourceStatus.SKIPPED
        )

        assert success.is_usable() is True
        assert filtered.is_usable() is True
        assert failed.is_usable() is False
        assert skipped.is_usable() is False

    def test_metadata_default_is_empty_dict(self):
        """Test that metadata defaults to empty dict, not shared instance."""
        source1 = UnifiedSource(id="m1", title="T", source_type=SourceType.WEB)
        source2 = UnifiedSource(id="m2", title="T", source_type=SourceType.WEB)

        source1.metadata["key"] = "value"

        assert source2.metadata == {}
        assert source1.metadata == {"key": "value"}
