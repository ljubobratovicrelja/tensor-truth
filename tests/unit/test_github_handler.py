"""Tests for GitHub handler."""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from tensortruth.utils.github_handler import GitHubHandler


class TestGitHubHandler:
    """Test suite for GitHubHandler."""

    @pytest.fixture
    def handler(self):
        """Create a GitHubHandler instance."""
        return GitHubHandler()

    def test_name(self, handler):
        """Test handler name property."""
        assert handler.name == "GitHub"

    @pytest.mark.parametrize(
        "url,expected",
        [
            ("https://github.com/openai/whisper", True),
            ("https://github.com/microsoft/TypeScript", True),
            ("https://github.com/pytorch/pytorch", True),
            ("https://github.com/user/repo-name", True),
            ("https://github.com/user/repo.git", True),
            ("https://github.com/user/repo/tree/main/src", True),  # File in repo
            ("https://github.com/user/repo/blob/main/README.md", True),  # Specific file
            # Should NOT match these:
            ("https://github.com/user/repo/issues", False),  # Issues page
            ("https://github.com/user/repo/pulls", False),  # PRs page
            ("https://github.com/user/repo/pull/123", False),  # Specific PR
            ("https://github.com/user/repo/actions", False),  # Actions
            ("https://github.com/user/repo/wiki", False),  # Wiki
            ("https://github.com/user/repo/settings", False),  # Settings
            ("https://github.com/user", False),  # User profile, not repo
            ("https://github.com", False),  # Homepage
            ("https://example.com/repo", False),  # Not GitHub
            ("not-a-url", False),
        ],
    )
    def test_matches(self, handler, url, expected):
        """Test URL matching for GitHub repositories."""
        assert handler.matches(url) == expected

    @pytest.mark.parametrize(
        "url,expected",
        [
            ("https://github.com/openai/whisper", ("openai", "whisper")),
            ("https://github.com/microsoft/TypeScript", ("microsoft", "TypeScript")),
            ("https://github.com/user/repo.git", ("user", "repo")),  # .git removed
            ("https://github.com/user/my-repo", ("user", "my-repo")),
            ("https://github.com/user/repo/tree/main", ("user", "repo")),
        ],
    )
    def test_extract_repo_info(self, handler, url, expected):
        """Test repository info extraction from URLs."""
        assert handler._extract_repo_info(url) == expected

    def test_extract_repo_info_invalid(self, handler):
        """Test repository info extraction with invalid URLs."""
        assert handler._extract_repo_info("https://github.com") is None
        assert handler._extract_repo_info("https://github.com/user") is None

    @pytest.mark.asyncio
    async def test_fetch_success_main_branch(self, handler):
        """Test successful README fetch from main branch."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(
            return_value="# Test Repo\n\nThis is a test README."
        )

        # Mock the context manager properly - use Mock not AsyncMock
        mock_cm = Mock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_cm.__aexit__ = AsyncMock(return_value=None)

        mock_session = Mock()
        mock_session.get.return_value = mock_cm

        url = "https://github.com/user/test-repo"
        markdown, status, error = await handler.fetch(url, mock_session, timeout=10)

        assert status == "success"
        assert error is None
        assert markdown is not None
        assert "# user/test-repo" in markdown
        assert "# Test Repo" in markdown
        assert "This is a test README." in markdown
        assert "https://github.com/user/test-repo" in markdown

    @pytest.mark.asyncio
    async def test_fetch_success_master_branch(self, handler):
        """Test successful README fetch from master branch (fallback)."""
        # Create a mock that returns 404 for main branch, 200 for master
        call_count = [0]

        def mock_get(*args, **kwargs):
            call_count[0] += 1
            mock_response = AsyncMock()
            if call_count[0] <= 5:  # First 5 calls (main branch attempts) fail
                mock_response.status = 404
            else:  # 6th call (master branch) succeeds
                mock_response.status = 200
                mock_response.text = AsyncMock(
                    return_value=(
                        "# README content\n\nThis is a longer README "
                        "to pass the minimum length check. "
                    )
                    * 5
                )

            mock_cm = Mock()
            mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
            mock_cm.__aexit__ = AsyncMock(return_value=None)
            return mock_cm

        mock_session = Mock()
        mock_session.get = mock_get

        url = "https://github.com/user/old-repo"
        markdown, status, error = await handler.fetch(url, mock_session, timeout=10)

        assert status == "success"
        assert error is None
        assert "# README content" in markdown

    @pytest.mark.asyncio
    async def test_fetch_readme_not_found(self, handler):
        """Test handling when README doesn't exist."""
        mock_response = AsyncMock()
        mock_response.status = 404

        mock_cm = Mock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_cm.__aexit__ = AsyncMock(return_value=None)

        mock_session = Mock()
        mock_session.get.return_value = mock_cm

        url = "https://github.com/user/no-readme-repo"
        markdown, status, error = await handler.fetch(url, mock_session, timeout=10)

        assert status == "http_error"
        assert markdown is None
        assert "README not found" in error

    @pytest.mark.asyncio
    async def test_fetch_invalid_url(self, handler):
        """Test handling of invalid GitHub URLs."""
        mock_session = AsyncMock()
        url = "https://github.com/invalid"

        markdown, status, error = await handler.fetch(url, mock_session, timeout=10)

        assert status == "parse_error"
        assert markdown is None
        assert "Could not extract repository info" in error

    @pytest.mark.asyncio
    async def test_fetch_timeout(self, handler):
        """Test handling of timeout errors."""
        # Note: Timeout errors are caught inside _fetch_readme_from_branch,
        # so they result in "README not found" rather than a timeout status
        mock_session = AsyncMock()
        mock_session.get.side_effect = asyncio.TimeoutError()

        url = "https://github.com/user/repo"
        markdown, status, error = await handler.fetch(url, mock_session, timeout=10)

        # Timeout is caught internally, so we get http_error
        assert status == "http_error"
        assert markdown is None
        assert "README not found" in error

    @pytest.mark.asyncio
    async def test_fetch_tries_multiple_readme_variants(self, handler):
        """Test that handler tries multiple README filename variants."""
        call_count = [0]

        def mock_get(*args, **kwargs):
            call_count[0] += 1
            mock_response = AsyncMock()
            if call_count[0] == 1:  # First call (README.md) fails
                mock_response.status = 404
            else:  # Second call (readme.md) succeeds
                mock_response.status = 200
                mock_response.text = AsyncMock(
                    return_value=(
                        "# Content\n\nThis is longer content to "
                        "pass the minimum length check. "
                    )
                    * 5
                )

            mock_cm = Mock()
            mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
            mock_cm.__aexit__ = AsyncMock(return_value=None)
            return mock_cm

        mock_session = Mock()
        mock_session.get = mock_get

        url = "https://github.com/user/repo"
        markdown, status, error = await handler.fetch(url, mock_session, timeout=10)

        assert status == "success"
        assert "# Content" in markdown

    @pytest.mark.asyncio
    async def test_fetch_content_too_short(self, handler):
        """Test handling of README content that's too short."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value="short")

        mock_cm = Mock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_cm.__aexit__ = AsyncMock(return_value=None)

        mock_session = Mock()
        mock_session.get.return_value = mock_cm

        url = "https://github.com/user/minimal-repo"
        markdown, status, error = await handler.fetch(url, mock_session, timeout=10)

        assert status == "too_short"
        assert markdown is None
        assert "too short" in error

    @pytest.mark.asyncio
    async def test_fetch_preserves_markdown_formatting(self, handler):
        """Test that markdown formatting is preserved."""
        readme_content = """# Project Title

## Installation

```bash
pip install package
```

## Usage

- Step 1
- Step 2

[Link](https://example.com)
"""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value=readme_content)

        mock_cm = Mock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_cm.__aexit__ = AsyncMock(return_value=None)

        mock_session = Mock()
        mock_session.get.return_value = mock_cm

        url = "https://github.com/user/repo"
        markdown, status, error = await handler.fetch(url, mock_session, timeout=10)

        assert status == "success"
        assert "# Project Title" in markdown
        assert "## Installation" in markdown
        assert "```bash" in markdown
        assert "[Link](https://example.com)" in markdown


def _response(status=200, text=None, json_data=None):
    """Build a mocked aiohttp response usable as an async context manager."""
    mock_response = AsyncMock()
    mock_response.status = status
    if text is not None:
        mock_response.text = AsyncMock(return_value=text)
    if json_data is not None:
        mock_response.json = AsyncMock(return_value=json_data)

    mock_cm = Mock()
    mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
    mock_cm.__aexit__ = AsyncMock(return_value=None)
    return mock_cm


def _recording_session(responses=None, default_status=404):
    """
    Build a mock session that records every requested URL.

    Args:
        responses: mapping of URL -> async context manager (Mock)
        default_status: status returned for URLs not present in ``responses``

    Returns:
        (session, requested_urls)
    """
    requested = []
    responses = responses or {}

    def mock_get(url, *args, **kwargs):
        requested.append(url)
        if url in responses:
            return responses[url]
        return _response(status=default_status)

    mock_session = Mock()
    mock_session.get = mock_get
    return mock_session, requested


class TestGitHubHandlerPathDispatch:
    """Tests for blob/tree URL dispatch (URL path must be honoured)."""

    @pytest.fixture
    def handler(self):
        return GitHubHandler()

    @pytest.mark.asyncio
    async def test_blob_requests_exact_raw_url_and_returns_file(self, handler):
        """A blob URL must fetch that exact file, not the README."""
        blob_url = "https://github.com/user/repo/blob/main/src/x.py"
        expected_raw = "https://raw.githubusercontent.com/user/repo/main/src/x.py"
        session, requested = _recording_session(
            {expected_raw: _response(status=200, text="print('hello')\n")}
        )

        markdown, status, error = await handler.fetch(blob_url, session, timeout=10)

        assert status == "success", error
        assert error is None
        assert requested == [expected_raw]
        assert "user/repo" in markdown
        assert "main" in markdown
        assert "src/x.py" in markdown
        # File body returned verbatim
        assert "print('hello')" in markdown
        assert blob_url in markdown
        # The README was never requested
        assert not any("README" in u or "readme" in u for u in requested)

    @pytest.mark.asyncio
    async def test_blob_short_file_not_rejected_as_too_short(self, handler):
        """Single files are exempt from the README minimum-length check."""
        blob_url = "https://github.com/user/repo/blob/main/VERSION"
        expected_raw = "https://raw.githubusercontent.com/user/repo/main/VERSION"
        session, _ = _recording_session(
            {expected_raw: _response(status=200, text="1.2.3\n")}
        )

        markdown, status, error = await handler.fetch(blob_url, session, timeout=10)

        assert status == "success", error
        assert markdown is not None
        assert "1.2.3" in markdown

    @pytest.mark.asyncio
    async def test_blob_empty_body_rejected(self, handler):
        """An empty file body is rejected even though short files are allowed."""
        blob_url = "https://github.com/user/repo/blob/main/empty.txt"
        expected_raw = "https://raw.githubusercontent.com/user/repo/main/empty.txt"
        session, _ = _recording_session(
            {expected_raw: _response(status=200, text="   \n")}
        )

        markdown, status, error = await handler.fetch(blob_url, session, timeout=10)

        assert status == "http_error"
        assert markdown is None
        assert "empty.txt" in error
        assert "Empty" in error

    @pytest.mark.asyncio
    async def test_blob_404_reports_path_and_ref(self, handler):
        """A missing blob reports the path and ref, and never falls back."""
        blob_url = "https://github.com/user/repo/blob/dev/docs/missing.md"
        expected_raw = "https://raw.githubusercontent.com/user/repo/dev/docs/missing.md"
        session, requested = _recording_session({expected_raw: _response(status=404)})

        markdown, status, error = await handler.fetch(blob_url, session, timeout=10)

        assert status == "http_error"
        assert markdown is None
        assert "docs/missing.md" in error
        assert "dev" in error
        assert requested == [expected_raw]

    @pytest.mark.asyncio
    async def test_blob_nested_path_requested_verbatim(self, handler):
        """Nested paths are joined with slashes, not truncated."""
        blob_url = "https://github.com/user/repo/blob/v1.0/a/b/c/d.json"
        expected_raw = "https://raw.githubusercontent.com/user/repo/v1.0/a/b/c/d.json"
        session, requested = _recording_session(
            {expected_raw: _response(status=200, text='{"a": 1}')}
        )

        markdown, status, error = await handler.fetch(blob_url, session, timeout=10)

        assert status == "success", error
        assert requested == [expected_raw]
        assert '{"a": 1}' in markdown
        assert "a/b/c/d.json" in markdown

    @pytest.mark.asyncio
    async def test_tree_requests_contents_api_and_renders_entries(self, handler):
        """A tree URL must use the contents API and render a markdown listing."""
        tree_url = "https://github.com/user/repo/tree/main/src"
        expected_api = "https://api.github.com/repos/user/repo/contents/src?ref=main"
        entries = [
            {"name": "module.py", "type": "file", "size": 1234},
            {"name": "nested", "type": "dir", "size": 0},
            {"name": "LICENSE", "type": "file", "size": 40},
        ]
        session, requested = _recording_session(
            {expected_api: _response(status=200, json_data=entries)}
        )

        markdown, status, error = await handler.fetch(tree_url, session, timeout=10)

        assert status == "success", error
        assert requested == [expected_api]
        assert "- `module.py` — file, 1234 bytes" in markdown
        assert "- `nested` — directory" in markdown
        assert "- `LICENSE` — file, 40 bytes" in markdown
        assert "src" in markdown
        assert "main" in markdown
        assert not any("README" in u or "readme" in u for u in requested)

    @pytest.mark.asyncio
    async def test_tree_root_uses_contents_endpoint_without_path(self, handler):
        """A tree URL with no path lists the repository root."""
        tree_url = "https://github.com/user/repo/tree/main"
        expected_api = "https://api.github.com/repos/user/repo/contents?ref=main"
        session, requested = _recording_session(
            {expected_api: _response(status=200, json_data=[])}
        )

        markdown, status, error = await handler.fetch(tree_url, session, timeout=10)

        assert status == "success", error
        assert requested == [expected_api]
        assert "empty directory" in markdown

    @pytest.mark.asyncio
    async def test_tree_403_rate_limit_reports_error(self, handler):
        """A 403 from the contents API is reported, not silently README'd."""
        tree_url = "https://github.com/user/repo/tree/main/docs"
        expected_api = "https://api.github.com/repos/user/repo/contents/docs?ref=main"
        session, requested = _recording_session({expected_api: _response(status=403)})

        markdown, status, error = await handler.fetch(tree_url, session, timeout=10)

        assert status == "http_error"
        assert markdown is None
        assert "403" in error
        assert "docs" in error
        assert "main" in error
        assert requested == [expected_api]

    @pytest.mark.asyncio
    async def test_tree_non_200_reports_error(self, handler):
        """A non-200, non-403 contents API response is reported clearly."""
        tree_url = "https://github.com/user/repo/tree/main/docs"
        expected_api = "https://api.github.com/repos/user/repo/contents/docs?ref=main"
        session, _ = _recording_session({expected_api: _response(status=500)})

        markdown, status, error = await handler.fetch(tree_url, session, timeout=10)

        assert status == "http_error"
        assert markdown is None
        assert "500" in error
        assert "docs" in error

    @pytest.mark.asyncio
    async def test_bare_repo_still_fetches_readme(self, handler):
        """Bare repo URLs keep the original README behaviour."""
        url = "https://github.com/user/plain-repo"
        expected_raw = (
            "https://raw.githubusercontent.com/user/plain-repo/main/README.md"
        )
        session, requested = _recording_session(
            {
                expected_raw: _response(
                    status=200,
                    text=(
                        "# Plain Repo\n\nA long enough README body to pass the "
                        "minimum length check applied to repository READMEs.\n"
                    ),
                )
            }
        )

        markdown, status, error = await handler.fetch(url, session, timeout=10)

        assert status == "success", error
        assert requested == [expected_raw]
        assert markdown.startswith("# user/plain-repo")
        assert "# Plain Repo" in markdown

    @pytest.mark.asyncio
    async def test_classify_url(self, handler):
        """URL classification extracts kind, ref and path."""
        assert handler._classify_url("https://github.com/o/r") == ("repo", None, None)
        assert handler._classify_url("https://github.com/o/r.git") == (
            "repo",
            None,
            None,
        )
        assert handler._classify_url("https://github.com/o/r/blob/main/a/b.py") == (
            "blob",
            "main",
            "a/b.py",
        )
        assert handler._classify_url("https://github.com/o/r/tree/main/src") == (
            "tree",
            "main",
            "src",
        )
        assert handler._classify_url("https://github.com/o/r/tree/main") == (
            "tree",
            "main",
            "",
        )
        # Blob without a file path is treated as a repo URL (README fallback)
        assert handler._classify_url("https://github.com/o/r/blob/main") == (
            "repo",
            None,
            None,
        )
