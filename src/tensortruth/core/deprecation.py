"""Deprecation utilities for marking code as deprecated.

This module provides a simple, type-safe decorator for marking functions,
methods, and classes as deprecated. It makes deprecated code easy to find
with grep and provides runtime warnings.

Usage:
    @deprecated("Use new_function() instead")
    def old_function():
        pass

    # Can also specify version when it will be removed
    @deprecated("Use new_function() instead", removal_version="2.0.0")
    def old_function():
        pass
"""

import functools
import warnings
from typing import Callable, Optional, TypeVar

F = TypeVar("F", bound=Callable)


def deprecated(
    reason: str,
    removal_version: Optional[str] = None,
) -> Callable[[F], F]:
    """Mark a function, method, or class as deprecated.

    This decorator:
    - Issues a DeprecationWarning when the decorated item is called
    - Preserves the original function signature for type checking
    - Makes deprecated code easy to find with grep

    Args:
        reason: Explanation of why it's deprecated and what to use instead.
        removal_version: Optional version when this will be removed.

    Returns:
        A decorator that marks the function as deprecated.

    Example:
        @deprecated("Use SourceConverter.to_api_schema() instead")
        def web_source_to_source_node(source):
            ...
    """

    def decorator(func: F) -> F:
        message = f"{func.__name__} is deprecated. {reason}"
        if removal_version:
            message += f" Will be removed in version {removal_version}."

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(
                message,
                category=DeprecationWarning,
                stacklevel=2,
            )
            return func(*args, **kwargs)

        # Mark with attribute for easy detection
        wrapper.__deprecated__ = True  # type: ignore[attr-defined]
        wrapper.__deprecation_reason__ = reason  # type: ignore[attr-defined]

        return wrapper  # type: ignore[return-value]

    return decorator
