"""Browse agent implementation using router pattern.

BrowseAgent is the default web research agent in TensorTruth.
Uses router pattern with overflow protection and reranking.
"""

from .agent import BrowseAgent
from .factory import create_browse_agent

__all__ = ["BrowseAgent", "create_browse_agent"]
