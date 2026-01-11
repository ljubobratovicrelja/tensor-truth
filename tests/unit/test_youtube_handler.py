"""Tests for YouTube handler."""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest

from tensortruth.utils.youtube_handler import YouTubeHandler


class TestYouTubeHandler:
    """Test suite for YouTubeHandler."""

    @pytest.fixture
    def handler(self):
        """Create a YouTubeHandler instance."""
        return YouTubeHandler()

    def test_name(self, handler):
        """Test handler name property."""
        assert handler.name == "YouTube"

    @pytest.mark.parametrize(
        "url,expected",
        [
            ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", True),
            ("https://youtu.be/dQw4w9WgXcQ", True),
            ("https://www.youtube.com/embed/dQw4w9WgXcQ", True),
            ("https://m.youtube.com/watch?v=dQw4w9WgXcQ", True),
            ("https://www.youtube.com/v/dQw4w9WgXcQ", True),
            ("https://youtu.be/dQw4w9WgXcQ?t=42", True),
            # Should NOT match these:
            ("https://www.youtube.com/", False),  # No video ID
            ("https://www.youtube.com/channel/UCxxxxx", False),  # Channel page
            ("https://www.youtube.com/playlist?list=PLxxxxx", False),  # Playlist
            ("https://example.com/watch?v=dQw4w9WgXcQ", False),  # Not YouTube
            ("https://github.com/repo", False),
            ("not-a-url", False),
        ],
    )
    def test_matches(self, handler, url, expected):
        """Test URL matching for YouTube videos."""
        assert handler.matches(url) == expected

    @pytest.mark.parametrize(
        "url,expected_id",
        [
            ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("https://youtu.be/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("https://www.youtube.com/embed/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("https://m.youtube.com/watch?v=ABC123def45", "ABC123def45"),
            ("https://www.youtube.com/v/test_video_id", "test_video_id"),
            (
                "https://youtu.be/dQw4w9WgXcQ?t=42",
                "dQw4w9WgXcQ",
            ),  # With timestamp
        ],
    )
    def test_extract_video_id(self, handler, url, expected_id):
        """Test video ID extraction from URLs."""
        assert handler._extract_video_id(url) == expected_id

    def test_extract_video_id_invalid(self, handler):
        """Test video ID extraction with invalid URLs."""
        assert handler._extract_video_id("https://www.youtube.com/") is None
        assert handler._extract_video_id("https://example.com") is None

    def test_extract_metadata_from_html(self, handler):
        """Test metadata extraction from YouTube HTML."""
        html = """
        <html>
        <meta property="og:title" content="Test Video Title">
        <meta property="og:description" content="This is a test video description">
        <script>{"channelName":"Test Channel","uploadDate":"2024-01-15T10:30:00Z"}</script>
        </html>
        """

        metadata = handler._extract_metadata_from_html(html)

        assert metadata["title"] == "Test Video Title"
        assert metadata["channel"] == "Test Channel"
        assert metadata["upload_date"] == "2024-01-15"
        assert metadata["description"] == "This is a test video description"

    @pytest.mark.asyncio
    @patch("tensortruth.utils.youtube_handler.YouTubeTranscriptApi")
    async def test_fetch_success_with_transcript(self, mock_transcript_api, handler):
        """Test successful YouTube video fetch with transcript."""
        # Mock HTTP response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_html = """
        <html>
        <meta property="og:title" content="Introduction to Machine Learning">
        <meta property="og:description" content="Learn ML basics">
        <script>{"channelName":"Tech Channel","uploadDate":"2024-01-01T00:00:00Z"}</script>
        </html>
        """
        mock_response.text = AsyncMock(return_value=mock_html)

        mock_cm = Mock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_cm.__aexit__ = AsyncMock(return_value=None)

        mock_session = Mock()
        mock_session.get.return_value = mock_cm

        # Mock transcript
        mock_transcript = Mock()
        mock_transcript.fetch.return_value = [
            {"text": "Hello and welcome", "start": 0.0},
            {"text": "to this tutorial", "start": 2.5},
            {"text": "on machine learning.", "start": 5.0},
        ]

        mock_transcript_list = Mock()
        mock_transcript_list.find_manually_created_transcript.return_value = (
            mock_transcript
        )

        mock_transcript_api.list_transcripts.return_value = mock_transcript_list

        url = "https://www.youtube.com/watch?v=test123"
        markdown, status, error = await handler.fetch(url, mock_session, timeout=10)

        assert status == "success"
        assert error is None
        assert markdown is not None
        assert "# Introduction to Machine Learning" in markdown
        assert "**Channel**: Tech Channel" in markdown
        assert "**Upload Date**: 2024-01-01" in markdown
        assert "Learn ML basics" in markdown
        assert "## Transcript" in markdown
        assert "Hello and welcome to this tutorial on machine learning." in markdown
        assert "https://www.youtube.com/watch?v=test123" in markdown

    @pytest.mark.asyncio
    @patch("tensortruth.utils.youtube_handler.YouTubeTranscriptApi")
    async def test_fetch_success_no_transcript(self, mock_transcript_api, handler):
        """Test successful fetch but no transcript available."""
        # Mock HTTP response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_html = """
        <html>
        <meta property="og:title" content="No Transcript Video">
        <script>{"channelName":"Test Channel"}</script>
        </html>
        """
        mock_response.text = AsyncMock(return_value=mock_html)

        mock_cm = Mock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_cm.__aexit__ = AsyncMock(return_value=None)

        mock_session = Mock()
        mock_session.get.return_value = mock_cm

        # Mock transcript API to raise NoTranscriptFound
        from youtube_transcript_api._errors import NoTranscriptFound

        mock_transcript_api.list_transcripts.side_effect = NoTranscriptFound(
            "test123", [], None
        )

        url = "https://www.youtube.com/watch?v=test123"
        markdown, status, error = await handler.fetch(url, mock_session, timeout=10)

        assert status == "success"
        assert error is None
        assert "# No Transcript Video" in markdown
        assert "*Note: Transcript not available for this video.*" in markdown

    @pytest.mark.asyncio
    @patch("tensortruth.utils.youtube_handler.YouTubeTranscriptApi")
    async def test_fetch_auto_generated_transcript(self, mock_transcript_api, handler):
        """Test falling back to auto-generated transcript."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_html = """
        <html>
        <meta property="og:title" content="Auto Transcript Video">
        </html>
        """
        mock_response.text = AsyncMock(return_value=mock_html)

        mock_cm = Mock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_cm.__aexit__ = AsyncMock(return_value=None)

        mock_session = Mock()
        mock_session.get.return_value = mock_cm

        # Mock transcript - manual fails, auto succeeds
        from youtube_transcript_api._errors import NoTranscriptFound

        mock_auto_transcript = Mock()
        mock_auto_transcript.fetch.return_value = [
            {"text": "Auto generated text", "start": 0.0}
        ]

        mock_transcript_list = Mock()
        mock_transcript_list.find_manually_created_transcript.side_effect = (
            NoTranscriptFound("test123", [], None)
        )
        mock_transcript_list.find_generated_transcript.return_value = (
            mock_auto_transcript
        )

        mock_transcript_api.list_transcripts.return_value = mock_transcript_list

        url = "https://www.youtube.com/watch?v=test123"
        markdown, status, error = await handler.fetch(url, mock_session, timeout=10)

        assert status == "success"
        assert "Auto generated text" in markdown

    @pytest.mark.asyncio
    async def test_fetch_invalid_url(self, handler):
        """Test handling of invalid URLs (ID extraction fails)."""
        mock_session = AsyncMock()
        url = "https://www.youtube.com/channel/UCxxxxx"

        markdown, status, error = await handler.fetch(url, mock_session, timeout=10)

        assert status == "parse_error"
        assert markdown is None
        assert "Could not extract video ID" in error

    @pytest.mark.asyncio
    async def test_fetch_http_error(self, handler):
        """Test handling of HTTP errors."""
        mock_response = AsyncMock()
        mock_response.status = 404

        mock_cm = Mock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_cm.__aexit__ = AsyncMock(return_value=None)

        mock_session = Mock()
        mock_session.get.return_value = mock_cm

        url = "https://www.youtube.com/watch?v=invalid123"
        markdown, status, error = await handler.fetch(url, mock_session, timeout=10)

        assert status == "http_error"
        assert markdown is None
        assert "404" in error

    @pytest.mark.asyncio
    @patch("tensortruth.utils.youtube_handler.YouTubeTranscriptApi")
    async def test_fetch_transcript_truncation(self, mock_transcript_api, handler):
        """Test that very long transcripts are truncated."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_html = """
        <html>
        <meta property="og:title" content="Long Video">
        </html>
        """
        mock_response.text = AsyncMock(return_value=mock_html)

        mock_cm = Mock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_cm.__aexit__ = AsyncMock(return_value=None)

        mock_session = Mock()
        mock_session.get.return_value = mock_cm

        # Create a very long transcript
        long_transcript = [
            {"text": "word " * 100, "start": i * 10.0} for i in range(100)
        ]

        mock_transcript = Mock()
        mock_transcript.fetch.return_value = long_transcript

        mock_transcript_list = Mock()
        mock_transcript_list.find_manually_created_transcript.return_value = (
            mock_transcript
        )

        mock_transcript_api.list_transcripts.return_value = mock_transcript_list

        url = "https://www.youtube.com/watch?v=test123"
        markdown, status, error = await handler.fetch(url, mock_session, timeout=10)

        assert status == "success"
        assert "[Transcript truncated for length...]" in markdown

    @pytest.mark.asyncio
    async def test_fetch_timeout(self, handler):
        """Test handling of timeouts."""
        mock_session = Mock()

        # Mock a slow response that times out
        async def slow_get(*args, **kwargs):
            await asyncio.sleep(20)

        mock_cm = Mock()
        mock_cm.__aenter__ = AsyncMock(side_effect=asyncio.TimeoutError())
        mock_cm.__aexit__ = AsyncMock(return_value=None)

        mock_session.get.return_value = mock_cm

        url = "https://www.youtube.com/watch?v=test123"
        markdown, status, error = await handler.fetch(url, mock_session, timeout=1)

        assert status == "timeout"
        assert markdown is None
        assert error == "YouTube fetch timeout"
