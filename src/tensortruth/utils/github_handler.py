"""GitHub-specific content handler for web search.

Fetches README.md directly from GitHub repositories using raw.githubusercontent.com
instead of scraping the main page (which loads README dynamically via JavaScript).
"""

import asyncio
import logging
from typing import Optional, Tuple
from urllib.parse import urlparse

import aiohttp

from .domain_handlers import ContentHandler, register_handler

logger = logging.getLogger(__name__)


class GitHubHandler(ContentHandler):
    """Handler for GitHub repository URLs."""

    @property
    def name(self) -> str:
        return "GitHub"

    def matches(self, url: str) -> bool:
        """Check if URL is a GitHub repository page."""
        try:
            parsed = urlparse(url)
            # Match github.com domain
            if "github.com" not in parsed.netloc:
                return False

            # Match repository URLs: github.com/owner/repo (not issues, pulls, etc.)
            path_parts = [p for p in parsed.path.split("/") if p]

            # Should have at least owner/repo
            if len(path_parts) < 2:
                return False

            # Exclude non-repository paths
            excluded_paths = [
                "issues",
                "pull",
                "pulls",
                "actions",
                "wiki",
                "projects",
                "security",
                "settings",
                "graphs",
                "pulse",
                "community",
                "compare",
                "blame",
                "commits",
                "releases",
                "tags",
            ]

            # If there are more than 2 path parts, check if it's an excluded section
            if len(path_parts) > 2 and path_parts[2] in excluded_paths:
                return False

            return True
        except Exception:
            return False

    def _extract_repo_info(self, url: str) -> Optional[Tuple[str, str]]:
        """
        Extract owner and repo name from GitHub URL.

        Args:
            url: GitHub URL (e.g., https://github.com/owner/repo)

        Returns:
            Tuple of (owner, repo) or None if extraction fails
        """
        try:
            parsed = urlparse(url)
            path_parts = [p for p in parsed.path.split("/") if p]

            if len(path_parts) >= 2:
                owner = path_parts[0]
                repo = path_parts[1]
                # Remove .git suffix if present
                if repo.endswith(".git"):
                    repo = repo[:-4]
                return owner, repo
        except Exception as e:
            logger.warning(f"Failed to extract repo info from {url}: {e}")
        return None

    async def _fetch_readme_from_branch(
        self,
        owner: str,
        repo: str,
        branch: str,
        session: aiohttp.ClientSession,
        timeout: int,
    ) -> Optional[str]:
        """
        Try to fetch README.md from a specific branch.

        Args:
            owner: Repository owner
            repo: Repository name
            branch: Branch name (e.g., 'main', 'master')
            session: aiohttp ClientSession
            timeout: Timeout in seconds

        Returns:
            README content or None if not found
        """
        # Try common README filenames
        readme_variants = [
            "README.md",
            "readme.md",
            "Readme.md",
            "README.MD",
            "README",
        ]

        for readme_name in readme_variants:
            try:
                raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{readme_name}"
                logger.debug(f"Trying {raw_url}")

                timeout_obj = aiohttp.ClientTimeout(total=timeout)
                async with session.get(raw_url, timeout=timeout_obj) as response:
                    if response.status == 200:
                        content = await response.text()
                        logger.info(f"✅ Found README at {raw_url}")
                        return content
            except Exception as e:
                logger.debug(f"Failed to fetch {readme_name} from {branch}: {e}")
                continue

        return None

    async def fetch(
        self, url: str, session: aiohttp.ClientSession, timeout: int = 10
    ) -> Tuple[Optional[str], str, Optional[str]]:
        """
        Fetch GitHub repository README using raw.githubusercontent.com.

        Args:
            url: GitHub repository URL
            session: aiohttp ClientSession
            timeout: Timeout in seconds

        Returns:
            Tuple of (markdown_content, status, error_message)
        """
        logger.info(f"Fetching GitHub repository: {url}")

        # Extract owner and repo
        repo_info = self._extract_repo_info(url)
        if not repo_info:
            return None, "parse_error", "Could not extract repository info from URL"

        owner, repo = repo_info

        try:
            # Try main branch first (most common now), then master
            branches = ["main", "master"]
            readme_content = None

            for branch in branches:
                readme_content = await self._fetch_readme_from_branch(
                    owner, repo, branch, session, timeout
                )
                if readme_content:
                    break

            if not readme_content:
                error_msg = "README not found in main or master branch"
                logger.warning(f"{error_msg} for {owner}/{repo}")
                return None, "http_error", error_msg

            # Build markdown with metadata
            markdown_lines = []

            # Add title
            markdown_lines.append(f"# {owner}/{repo}")
            markdown_lines.append("")

            # Add source metadata
            markdown_lines.append(f"<!-- Source: {url} -->")
            markdown_lines.append("")

            # Add README content
            markdown_lines.append(readme_content)

            markdown = "\n".join(markdown_lines)

            # Quality check
            if len(markdown.strip()) < 100:
                return None, "too_short", "README content too short"

            logger.info(
                f"✅ Fetched GitHub README for {owner}/{repo} ({len(markdown)} chars)"
            )
            return markdown, "success", None

        except asyncio.TimeoutError:
            error_msg = "Timeout fetching README"
            logger.warning(f"{error_msg} for {owner}/{repo}")
            return None, "timeout", error_msg

        except Exception as e:
            error_msg = f"Error fetching README: {str(e)}"
            logger.error(f"Error fetching GitHub repo {owner}/{repo}: {e}")
            return None, "parse_error", error_msg


# Register the GitHub handler
register_handler(GitHubHandler())
