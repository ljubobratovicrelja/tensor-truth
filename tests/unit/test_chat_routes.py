"""Unit tests for tensortruth.api.routes.chat module.

Note: Tests for ChatService.execute() and ChatService.extract_sources()
are in tests/unit/test_chat_service.py.
"""

from unittest.mock import MagicMock

import pytest

from tensortruth.api.routes.chat import ChatContext


@pytest.mark.unit
class TestChatContext:
    """Tests for ChatContext dataclass."""

    def test_construction_no_modules_no_pdfs(self):
        """ChatContext construction with modules=[] and no PDF index."""
        context = ChatContext(
            session_id="test-session",
            prompt="Hello",
            modules=[],
            params={},
            session_messages=[],
            session_index_path=None,
        )
        assert context.modules == []
        assert context.session_index_path is None

    def test_construction_with_modules(self):
        """ChatContext construction with modules=['pytorch']."""
        context = ChatContext(
            session_id="test-session",
            prompt="Hello",
            modules=["pytorch"],
            params={},
            session_messages=[],
            session_index_path=None,
        )
        assert context.modules == ["pytorch"]

    def test_construction_with_pdf_index(self):
        """ChatContext construction with PDF index (even without modules)."""
        context = ChatContext(
            session_id="test-session",
            prompt="Hello",
            modules=[],
            params={},
            session_messages=[],
            session_index_path="/path/to/index",
        )
        assert context.session_index_path == "/path/to/index"

    def test_construction_with_both(self):
        """ChatContext construction with both modules and PDF index."""
        context = ChatContext(
            session_id="test-session",
            prompt="Hello",
            modules=["pytorch"],
            params={},
            session_messages=[],
            session_index_path="/path/to/index",
        )
        assert context.modules == ["pytorch"]
        assert context.session_index_path == "/path/to/index"

    def test_from_session_creates_context(self):
        """Test from_session factory method creates valid context."""
        session = {
            "modules": ["pytorch"],
            "params": {"temperature": 0.7},
            "messages": [{"role": "user", "content": "previous"}],
        }

        # Mock PDF service that returns an index path
        mock_pdf_service = MagicMock()
        mock_pdf_service.get_index_path.return_value = "/path/to/index"

        context = ChatContext.from_session(
            session_id="test-123",
            prompt="current question",
            session=session,
            pdf_service=mock_pdf_service,
        )

        assert context.session_id == "test-123"
        assert context.prompt == "current question"
        assert context.modules == ["pytorch"]
        assert context.params == {"temperature": 0.7}
        assert context.session_messages == [{"role": "user", "content": "previous"}]
        assert context.session_index_path == "/path/to/index"

    def test_from_session_handles_no_index(self):
        """Test from_session handles None index path."""
        session = {
            "modules": [],
            "params": {},
            "messages": [],
        }

        mock_pdf_service = MagicMock()
        mock_pdf_service.get_index_path.return_value = None

        context = ChatContext.from_session(
            session_id="test-123",
            prompt="question",
            session=session,
            pdf_service=mock_pdf_service,
        )

        assert context.session_index_path is None
        assert context.modules == []

    def test_from_session_handles_missing_modules(self):
        """Test from_session handles session with no modules key."""
        session = {
            "params": {},
            "messages": [],
            # modules key is missing
        }

        mock_pdf_service = MagicMock()
        mock_pdf_service.get_index_path.return_value = None

        context = ChatContext.from_session(
            session_id="test-123",
            prompt="question",
            session=session,
            pdf_service=mock_pdf_service,
        )

        assert context.modules == []
