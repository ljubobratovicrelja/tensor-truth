"""Tests for extension_catalog module."""

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tensortruth.services.extension_catalog import (
    CACHE_TTL_SECONDS,
    CatalogCache,
    fetch_catalog,
    fetch_extension_yaml,
)

SAMPLE_CATALOG = {
    "version": 1,
    "generated_at": "2026-03-08T12:00:00Z",
    "extensions": [
        {
            "name": "test_cmd",
            "type": "command",
            "filename": "test_cmd.yaml",
            "description": "A test command",
        }
    ],
}


class TestCatalogCache:
    def test_read_returns_none_when_no_cache(self, tmp_path):
        cache = CatalogCache(tmp_path)
        assert cache.read() is None

    def test_write_then_read(self, tmp_path):
        cache = CatalogCache(tmp_path)
        cache.write(SAMPLE_CATALOG)
        result = cache.read()
        assert result == SAMPLE_CATALOG

    def test_read_returns_none_when_expired(self, tmp_path):
        cache = CatalogCache(tmp_path)
        # Write with a timestamp far in the past
        cache_path = tmp_path / "catalog_cache.json"
        payload = {
            "fetched_at": time.time() - CACHE_TTL_SECONDS - 100,
            "catalog": SAMPLE_CATALOG,
        }
        cache_path.write_text(json.dumps(payload), encoding="utf-8")
        assert cache.read() is None

    def test_read_any_returns_expired_data(self, tmp_path):
        cache = CatalogCache(tmp_path)
        cache_path = tmp_path / "catalog_cache.json"
        payload = {
            "fetched_at": time.time() - CACHE_TTL_SECONDS - 100,
            "catalog": SAMPLE_CATALOG,
        }
        cache_path.write_text(json.dumps(payload), encoding="utf-8")
        result = cache.read_any()
        assert result == SAMPLE_CATALOG

    def test_read_any_returns_none_when_no_cache(self, tmp_path):
        cache = CatalogCache(tmp_path)
        assert cache.read_any() is None

    def test_read_handles_corrupt_cache(self, tmp_path):
        cache = CatalogCache(tmp_path)
        cache_path = tmp_path / "catalog_cache.json"
        cache_path.write_text("not json", encoding="utf-8")
        assert cache.read() is None
        assert cache.read_any() is None


class TestFetchCatalog:
    @patch("tensortruth.services.extension_catalog.requests.get")
    def test_success(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = SAMPLE_CATALOG
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = fetch_catalog("https://example.com/ext")
        assert result == SAMPLE_CATALOG
        mock_get.assert_called_once_with(
            "https://example.com/ext/catalog.json", timeout=15
        )

    @patch("tensortruth.services.extension_catalog.requests.get")
    def test_failure_raises_connection_error(self, mock_get):
        import requests

        mock_get.side_effect = requests.ConnectionError("no network")
        with pytest.raises(ConnectionError):
            fetch_catalog("https://example.com/ext")


class TestFetchExtensionYaml:
    @patch("tensortruth.services.extension_catalog.requests.get")
    def test_success(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.text = "name: test\ndescription: A test"
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = fetch_extension_yaml("https://example.com/ext", "command", "test.yaml")
        assert "name: test" in result
        mock_get.assert_called_once_with(
            "https://example.com/ext/commands/test.yaml", timeout=15
        )

    @patch("tensortruth.services.extension_catalog.requests.get")
    def test_agent_type_uses_agents_subdir(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.text = "name: agent1"
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        fetch_extension_yaml("https://example.com/ext", "agent", "agent1.yaml")
        mock_get.assert_called_once_with(
            "https://example.com/ext/agents/agent1.yaml", timeout=15
        )

    @patch("tensortruth.services.extension_catalog.requests.get")
    def test_failure_raises_connection_error(self, mock_get):
        import requests

        mock_get.side_effect = requests.ConnectionError("no network")
        with pytest.raises(ConnectionError):
            fetch_extension_yaml("https://example.com/ext", "command", "test.yaml")
