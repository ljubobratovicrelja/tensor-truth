"""Shared prompt utilities."""

from datetime import datetime, timezone


def current_date_context() -> str:
    """Return a short current-date string for system prompts."""
    now = datetime.now(timezone.utc)
    return f"Current date: {now.strftime('%A, %B %d, %Y')} (UTC)"
