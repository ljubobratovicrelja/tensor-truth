"""Domain-specific content handlers for web search.

This module provides a pluggable architecture for handling different types of web content.
Special domains (Wikipedia, GitHub, arXiv, etc.) can have custom handlers that extract
content more effectively than generic HTML scraping.
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import aiohttp

logger = logging.getLogger(__name__)


class ContentHandler(ABC):
    """Base class for domain-specific content handlers."""

    @abstractmethod
    def matches(self, url: str) -> bool:
        """
        Check if this handler can process the given URL.

        Args:
            url: URL to check

        Returns:
            True if this handler should process the URL
        """
        pass

    @abstractmethod
    async def fetch(
        self, url: str, session: aiohttp.ClientSession, timeout: int = 10
    ) -> Tuple[Optional[str], str, Optional[str]]:
        """
        Fetch and convert content from URL to markdown.

        Args:
            url: URL to fetch
            session: aiohttp ClientSession for making requests
            timeout: Timeout in seconds

        Returns:
            Tuple of (markdown_content, status, error_message)
            - markdown_content: Markdown string or None on failure
            - status: "success", "http_error", "timeout", "parse_error", "too_short", etc.
            - error_message: Human-readable error description or None
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this handler (e.g., 'Wikipedia')."""
        pass


class ContentHandlerRegistry:
    """Registry for domain-specific content handlers."""

    def __init__(self):
        self._handlers: List[ContentHandler] = []

    def register(self, handler: ContentHandler) -> None:
        """
        Register a new content handler.

        Handlers are checked in registration order, so register more
        specific handlers before generic ones.

        Args:
            handler: ContentHandler instance to register
        """
        self._handlers.append(handler)
        logger.debug(f"Registered content handler: {handler.name}")

    def get_handler(self, url: str) -> Optional[ContentHandler]:
        """
        Find the first handler that can process the given URL.

        Args:
            url: URL to check

        Returns:
            ContentHandler instance or None if no handler matches
        """
        for handler in self._handlers:
            if handler.matches(url):
                logger.debug(f"Using {handler.name} handler for {url}")
                return handler
        return None

    def list_handlers(self) -> List[str]:
        """Get list of registered handler names."""
        return [h.name for h in self._handlers]


# Global registry instance
_registry = ContentHandlerRegistry()


def register_handler(handler: ContentHandler) -> None:
    """
    Register a content handler with the global registry.

    Args:
        handler: ContentHandler instance
    """
    _registry.register(handler)


def get_handler_for_url(url: str) -> Optional[ContentHandler]:
    """
    Get appropriate handler for a URL from the global registry.

    Args:
        url: URL to check

    Returns:
        ContentHandler instance or None if no special handler available
    """
    return _registry.get_handler(url)


def list_registered_handlers() -> List[str]:
    """Get list of all registered handler names."""
    return _registry.list_handlers()
