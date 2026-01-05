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
