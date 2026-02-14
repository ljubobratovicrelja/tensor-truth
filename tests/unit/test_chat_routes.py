"""Unit tests for tensortruth.api.routes.chat module.

Note: Tests for ChatService.execute() and ChatService.extract_sources()
are in tests/unit/test_chat_service.py.
"""

from unittest.mock import MagicMock, patch

import pytest

from tensortruth.api.routes.chat import ChatContext


@pytest.mark.unit
class TestChatContext:
    """Tests for ChatContext dataclass."""

    def test_construction_no_modules_no_pdfs(self):
        """ChatContext construction with modules=[] and no additional indexes."""
        context = ChatContext(
            session_id="test-session",
            prompt="Hello",
            modules=[],
            params={},
            session_messages=[],
        )
        assert context.modules == []
        assert context.additional_index_paths == []

    def test_construction_with_modules(self):
        """ChatContext construction with modules=['pytorch']."""
        context = ChatContext(
            session_id="test-session",
            prompt="Hello",
            modules=["pytorch"],
            params={},
            session_messages=[],
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
            additional_index_paths=["/path/to/index"],
        )
        assert context.additional_index_paths == ["/path/to/index"]

    def test_construction_with_both(self):
        """ChatContext construction with both modules and additional indexes."""
        context = ChatContext(
            session_id="test-session",
            prompt="Hello",
            modules=["pytorch"],
            params={},
            session_messages=[],
            additional_index_paths=["/path/to/index"],
        )
        assert context.modules == ["pytorch"]
        assert context.additional_index_paths == ["/path/to/index"]

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
        assert context.additional_index_paths == ["/path/to/index"]

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

        assert context.additional_index_paths == []
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

    @patch("tensortruth.api.routes.chat.get_project_index_dir")
    def test_from_session_with_project_merges_modules(self, mock_get_index_dir):
        """Test from_session merges project and session catalog modules."""
        mock_project_index = MagicMock()
        mock_project_index.__truediv__ = MagicMock(
            return_value=MagicMock(exists=MagicMock(return_value=False))
        )
        mock_get_index_dir.return_value = mock_project_index

        session = {
            "modules": ["session_module"],
            "params": {},
            "messages": [],
            "project_id": "proj-1",
        }

        mock_pdf_service = MagicMock()
        mock_pdf_service.get_index_path.return_value = None

        mock_project_service = MagicMock()
        mock_project_data = MagicMock()
        mock_project_service.load.return_value = mock_project_data
        mock_project_service.get_project.return_value = {
            "catalog_modules": {
                "project_module": {"status": "indexed"},
                "pending_module": {"status": "pending"},
            },
        }

        context = ChatContext.from_session(
            session_id="test-123",
            prompt="question",
            session=session,
            pdf_service=mock_pdf_service,
            project_service=mock_project_service,
        )

        # project_module (indexed) comes first, then session_module
        # pending_module is excluded (not indexed)
        assert context.modules == ["project_module", "session_module"]

    @patch("tensortruth.api.routes.chat.get_project_index_dir")
    def test_from_session_with_project_adds_project_index_path(
        self, mock_get_index_dir
    ):
        """Test from_session adds project index path when chroma.sqlite3 exists."""
        mock_chroma_path = MagicMock()
        mock_chroma_path.exists.return_value = True

        mock_project_index = MagicMock()
        mock_project_index.__truediv__ = MagicMock(return_value=mock_chroma_path)
        mock_project_index.__str__ = MagicMock(return_value="/projects/proj-1/index")
        mock_get_index_dir.return_value = mock_project_index

        session = {
            "modules": [],
            "params": {},
            "messages": [],
            "project_id": "proj-1",
        }

        mock_pdf_service = MagicMock()
        mock_pdf_service.get_index_path.return_value = "/session/pdf/index"

        mock_project_service = MagicMock()
        mock_project_data = MagicMock()
        mock_project_service.load.return_value = mock_project_data
        mock_project_service.get_project.return_value = {
            "catalog_modules": {},
        }

        context = ChatContext.from_session(
            session_id="test-123",
            prompt="question",
            session=session,
            pdf_service=mock_pdf_service,
            project_service=mock_project_service,
        )

        # Project index first, then session PDF index
        assert context.additional_index_paths == [
            "/projects/proj-1/index",
            "/session/pdf/index",
        ]

    @patch("tensortruth.api.routes.chat.get_project_index_dir")
    def test_from_session_deduplicates_modules(self, mock_get_index_dir):
        """Test from_session deduplicates modules across project and session."""
        mock_project_index = MagicMock()
        mock_project_index.__truediv__ = MagicMock(
            return_value=MagicMock(exists=MagicMock(return_value=False))
        )
        mock_get_index_dir.return_value = mock_project_index

        session = {
            "modules": ["shared_module", "session_only"],
            "params": {},
            "messages": [],
            "project_id": "proj-1",
        }

        mock_pdf_service = MagicMock()
        mock_pdf_service.get_index_path.return_value = None

        mock_project_service = MagicMock()
        mock_project_data = MagicMock()
        mock_project_service.load.return_value = mock_project_data
        mock_project_service.get_project.return_value = {
            "catalog_modules": {
                "shared_module": {"status": "indexed"},
                "project_only": {"status": "indexed"},
            },
        }

        context = ChatContext.from_session(
            session_id="test-123",
            prompt="question",
            session=session,
            pdf_service=mock_pdf_service,
            project_service=mock_project_service,
        )

        # shared_module appears once (from project), project_only, then session_only
        assert context.modules == ["shared_module", "project_only", "session_only"]

    def test_from_session_project_not_found(self):
        """Test from_session gracefully handles deleted/missing project."""
        session = {
            "modules": ["session_module"],
            "params": {},
            "messages": [],
            "project_id": "deleted-project",
        }

        mock_pdf_service = MagicMock()
        mock_pdf_service.get_index_path.return_value = None

        mock_project_service = MagicMock()
        mock_project_data = MagicMock()
        mock_project_service.load.return_value = mock_project_data
        mock_project_service.get_project.return_value = None  # Project not found

        context = ChatContext.from_session(
            session_id="test-123",
            prompt="question",
            session=session,
            pdf_service=mock_pdf_service,
            project_service=mock_project_service,
        )

        # Falls back to session-only modules
        assert context.modules == ["session_module"]
        assert context.additional_index_paths == []
