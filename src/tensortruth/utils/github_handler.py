"""GitHub-specific content handler for web search.

GitHub repository pages load their README dynamically via JavaScript, so
scraping the HTML page yields nothing useful. This handler talks to the
underlying content services instead:

* Bare repository URLs (``github.com/owner/repo``) fetch ``README.md`` (and
  common variants) from ``main`` then ``master`` via
  ``raw.githubusercontent.com``.
* Blob URLs (``github.com/owner/repo/blob/<ref>/<path>``) fetch exactly that
  file from ``raw.githubusercontent.com`` and return it verbatim.
* Tree URLs (``github.com/owner/repo/tree/<ref>[/<path>]``) render a directory
  listing from the GitHub REST contents API.

Path classification caveat: a git ref may itself contain slashes
(``feature/foo``), which makes ``<ref>`` and the first ``<path>`` segment
indistinguishable from the URL alone. This handler always treats the single
segment immediately after ``blob``/``tree`` as the ref; branch names with
slashes are therefore misparsed (their first segment is taken as the ref and
the remainder as the path).
"""

import asyncio
import logging
from typing import List, Optional, Tuple
from urllib.parse import urlparse

import aiohttp

from .domain_handlers import ContentHandler, register_handler

logger = logging.getLogger(__name__)

#: URL kinds this handler can distinguish.
KIND_REPO = "repo"
KIND_BLOB = "blob"
KIND_TREE = "tree"


class GitHubHandler(ContentHandler):
    """Handler for GitHub repository, file (blob) and directory (tree) URLs."""

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

    def _classify_url(self, url: str) -> Tuple[str, Optional[str], Optional[str]]:
        """
        Classify a GitHub URL into repo / blob / tree and extract ref + path.

        The segment right after ``blob`` or ``tree`` is always taken as the ref;
        a ref containing a slash cannot be distinguished from a path, see the
        module docstring.

        Args:
            url: GitHub URL

        Returns:
            Tuple of (kind, ref, path). ``ref``/``path`` are None for kind
            ``repo``. ``path`` may be an empty string for a tree URL pointing at
            the repository root.
        """
        try:
            parsed = urlparse(url)
            parts = [p for p in parsed.path.split("/") if p]
        except Exception:
            return KIND_REPO, None, None

        if len(parts) < 4 or parts[2] not in (KIND_BLOB, KIND_TREE):
            return KIND_REPO, None, None

        kind = parts[2]
        ref = parts[3]
        rest: List[str] = parts[4:]

        # A blob URL without a file path is not a file reference; fall back to
        # the repository README rather than requesting a directory as a file.
        if kind == KIND_BLOB and not rest:
            return KIND_REPO, None, None

        path = "/".join(rest)
        return kind, ref, path

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

    async def _fetch_readme(
        self,
        owner: str,
        repo: str,
        url: str,
        session: aiohttp.ClientSession,
        timeout: int,
    ) -> Tuple[Optional[str], str, Optional[str]]:
        """Fetch the repository README (bare repo URL behaviour)."""
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

    async def _fetch_blob(
        self,
        owner: str,
        repo: str,
        ref: str,
        path: str,
        url: str,
        session: aiohttp.ClientSession,
        timeout: int,
    ) -> Tuple[Optional[str], str, Optional[str]]:
        """Fetch a single file referenced by a blob URL, verbatim."""
        raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{ref}/{path}"
        logger.debug(f"Fetching blob {raw_url}")

        try:
            timeout_obj = aiohttp.ClientTimeout(total=timeout)
            async with session.get(raw_url, timeout=timeout_obj) as response:
                status = response.status
                if status == 404:
                    error_msg = (
                        f"File '{path}' not found at ref '{ref}' in {owner}/{repo}"
                    )
                    logger.warning(error_msg)
                    return None, "http_error", error_msg
                if status != 200:
                    error_msg = (
                        f"HTTP {status} fetching '{path}' at ref '{ref}' "
                        f"in {owner}/{repo}"
                    )
                    logger.warning(error_msg)
                    return None, "http_error", error_msg

                content = await response.text()

            if not content or not content.strip():
                error_msg = f"Empty file body for '{path}' at ref '{ref}'"
                logger.warning(error_msg)
                return None, "http_error", error_msg

            markdown_lines = [
                f"# {owner}/{repo}: {path}",
                "",
                f"<!-- Source: {url} -->",
                f"<!-- Ref: {ref} -->",
                "",
                content,
            ]
            markdown = "\n".join(markdown_lines)

            # Note: no minimum-length check here — a single small file is a
            # legitimate result.
            logger.info(f"✅ Fetched GitHub blob {owner}/{repo}@{ref}/{path}")
            return markdown, "success", None

        except asyncio.TimeoutError:
            error_msg = f"Timeout fetching '{path}' at ref '{ref}'"
            logger.warning(error_msg)
            return None, "timeout", error_msg

        except Exception as e:
            error_msg = f"Error fetching '{path}' at ref '{ref}': {str(e)}"
            logger.error(error_msg)
            return None, "parse_error", error_msg

    async def _fetch_tree(
        self,
        owner: str,
        repo: str,
        ref: str,
        path: str,
        url: str,
        session: aiohttp.ClientSession,
        timeout: int,
    ) -> Tuple[Optional[str], str, Optional[str]]:
        """Render a directory listing via the GitHub REST contents API."""
        api_path = f"contents/{path}" if path else "contents"
        api_url = f"https://api.github.com/repos/{owner}/{repo}/{api_path}?ref={ref}"
        logger.debug(f"Fetching tree listing {api_url}")

        display_path = path or "/"

        try:
            timeout_obj = aiohttp.ClientTimeout(total=timeout)
            async with session.get(api_url, timeout=timeout_obj) as response:
                status = response.status
                if status == 403:
                    error_msg = (
                        f"GitHub contents API rate limit (403) for path "
                        f"'{display_path}' at ref '{ref}'"
                    )
                    logger.warning(error_msg)
                    return None, "http_error", error_msg
                if status != 200:
                    error_msg = (
                        f"GitHub contents API returned {status} for path "
                        f"'{display_path}' at ref '{ref}'"
                    )
                    logger.warning(error_msg)
                    return None, "http_error", error_msg

                entries = await response.json()

            if not isinstance(entries, list):
                error_msg = (
                    f"Unexpected contents API payload for path '{display_path}' "
                    f"at ref '{ref}'"
                )
                logger.warning(error_msg)
                return None, "parse_error", error_msg

            markdown_lines = [
                f"# {owner}/{repo} — {display_path} ({ref})",
                "",
                f"<!-- Source: {url} -->",
                f"<!-- Ref: {ref} -->",
                "",
            ]

            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                entry_name = entry.get("name", "<unknown>")
                entry_type = entry.get("type", "file")
                if entry_type == "dir":
                    markdown_lines.append(f"- `{entry_name}` — directory")
                else:
                    size = entry.get("size")
                    size_text = (
                        f"{size} bytes" if isinstance(size, int) else "size unknown"
                    )
                    markdown_lines.append(
                        f"- `{entry_name}` — {entry_type}, {size_text}"
                    )

            if not entries:
                markdown_lines.append("_(empty directory)_")

            markdown = "\n".join(markdown_lines)

            logger.info(
                f"✅ Fetched GitHub tree listing for {owner}/{repo}@{ref}/{display_path}"
            )
            return markdown, "success", None

        except asyncio.TimeoutError:
            error_msg = f"Timeout fetching directory '{display_path}' at ref '{ref}'"
            logger.warning(error_msg)
            return None, "timeout", error_msg

        except Exception as e:
            error_msg = (
                f"Error fetching directory '{display_path}' at ref '{ref}': {str(e)}"
            )
            logger.error(error_msg)
            return None, "parse_error", error_msg

    async def fetch(
        self, url: str, session: aiohttp.ClientSession, timeout: int = 10
    ) -> Tuple[Optional[str], str, Optional[str]]:
        """
        Fetch GitHub content, dispatching on the URL path.

        Bare repository URLs return the README, blob URLs return one file
        verbatim, and tree URLs return a directory listing.

        Args:
            url: GitHub URL
            session: aiohttp ClientSession
            timeout: Timeout in seconds

        Returns:
            Tuple of (markdown_content, status, error_message)
        """
        logger.info(f"Fetching GitHub content: {url}")

        # Extract owner and repo
        repo_info = self._extract_repo_info(url)
        if not repo_info:
            return None, "parse_error", "Could not extract repository info from URL"

        owner, repo = repo_info
        kind, ref, path = self._classify_url(url)

        if kind == KIND_BLOB:
            assert ref is not None and path is not None
            return await self._fetch_blob(owner, repo, ref, path, url, session, timeout)

        if kind == KIND_TREE:
            assert ref is not None and path is not None
            return await self._fetch_tree(owner, repo, ref, path, url, session, timeout)

        # Bare repo URL (or anything else we do not classify): README, as before.
        return await self._fetch_readme(owner, repo, url, session, timeout)


# Register the GitHub handler
register_handler(GitHubHandler())
