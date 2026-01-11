"""Conftest for directory naming tests - handles pymupdf.layout import issue."""

import sys
from unittest.mock import MagicMock

# Mock pymupdf.layout before any imports
layout_mock = MagicMock()
layout_mock.activate = MagicMock()
sys.modules["pymupdf.layout"] = layout_mock
