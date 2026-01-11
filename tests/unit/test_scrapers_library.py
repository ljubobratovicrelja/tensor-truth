"""
Unit tests for scrapers.library module.

Tests the scrape_library function that handles scraping of library documentation
from Sphinx and Doxygen sources.
"""

from unittest.mock import MagicMock, patch

import pytest

from tensortruth.scrapers.library import DEFAULT_MAX_WORKERS, scrape_library


@pytest.mark.unit
class TestScrapeLibrary:
    """Tests for scrape_library function."""

    @pytest.fixture
    def sphinx_config(self):
        """Mock configuration for Sphinx documentation."""
        return {
            "type": "sphinx",
            "version": "2.9",
            "doc_root": "https://pytorch.org/docs/stable/",
            "inventory_url": "https://pytorch.org/docs/stable/objects.inv",
            "selector": "div[role='main']",
        }

    @pytest.fixture
    def doxygen_config(self):
        """Mock configuration for Doxygen documentation."""
        return {
            "type": "doxygen",
            "version": "1.0",
            "doc_root": "https://example.com/docs/",
            "urls": ["https://example.com/docs/index.html"],
        }

    def test_scrape_library_sphinx_success(self, tmp_path, sphinx_config):
        """Test successful scraping of Sphinx documentation."""
        output_base_dir = tmp_path / "library_docs"
        output_base_dir.mkdir()

        mock_urls = [
            "https://pytorch.org/docs/stable/torch.html",
            "https://pytorch.org/docs/stable/nn.html",
        ]

        with (
            patch(
                "tensortruth.scrapers.library.fetch_inventory", return_value=mock_urls
            ),
            patch("tensortruth.scrapers.library.process_url", return_value=True),
        ):
            result = scrape_library(
                library_name="pytorch_2.9",
                config=sphinx_config,
                output_base_dir=str(output_base_dir),
            )

        assert result is True
        # Verify directory was created with correct prefix
        expected_dir = output_base_dir / "library_pytorch_2.9"
        assert expected_dir.exists()

    def test_scrape_library_doxygen_success(self, tmp_path, doxygen_config):
        """Test successful scraping of Doxygen documentation."""
        output_base_dir = tmp_path / "library_docs"
        output_base_dir.mkdir()

        mock_urls = [
            "https://example.com/docs/annotated.html",
            "https://example.com/docs/classes.html",
        ]

        with (
            patch(
                "tensortruth.scrapers.library.fetch_doxygen_urls",
                return_value=mock_urls,
            ),
            patch("tensortruth.scrapers.library.process_url", return_value=True),
        ):
            result = scrape_library(
                library_name="mylib_1.0",
                config=doxygen_config,
                output_base_dir=str(output_base_dir),
            )

        assert result is True
        expected_dir = output_base_dir / "library_mylib_1.0"
        assert expected_dir.exists()

    def test_scrape_library_no_urls_found(self, tmp_path, sphinx_config):
        """Test handling when no URLs are found."""
        output_base_dir = tmp_path / "library_docs"
        output_base_dir.mkdir()

        with patch("tensortruth.scrapers.library.fetch_inventory", return_value=[]):
            result = scrape_library(
                library_name="empty_lib",
                config=sphinx_config,
                output_base_dir=str(output_base_dir),
            )

        assert result is False

    def test_scrape_library_unknown_doc_type(self, tmp_path):
        """Test handling of unknown documentation type."""
        output_base_dir = tmp_path / "library_docs"
        output_base_dir.mkdir()

        unknown_config = {
            "type": "unknown_type",
            "version": "1.0",
        }

        result = scrape_library(
            library_name="test_lib",
            config=unknown_config,
            output_base_dir=str(output_base_dir),
        )

        assert result is False

    def test_scrape_library_with_custom_workers(self, tmp_path, sphinx_config):
        """Test scraping with custom number of workers."""
        output_base_dir = tmp_path / "library_docs"
        output_base_dir.mkdir()

        mock_urls = ["https://example.com/page1.html"]

        with (
            patch(
                "tensortruth.scrapers.library.fetch_inventory", return_value=mock_urls
            ),
            patch("tensortruth.scrapers.library.process_url", return_value=True),
            patch("tensortruth.scrapers.library.ThreadPoolExecutor") as mock_executor,
        ):
            mock_executor.return_value.__enter__.return_value.map.return_value = [True]

            scrape_library(
                library_name="test_lib",
                config=sphinx_config,
                output_base_dir=str(output_base_dir),
                max_workers=10,
            )

            # Verify executor was called with custom worker count
            mock_executor.assert_called_once_with(max_workers=10)

    def test_scrape_library_default_workers(self, tmp_path, sphinx_config):
        """Test scraping uses default workers when not specified."""
        output_base_dir = tmp_path / "library_docs"
        output_base_dir.mkdir()

        mock_urls = ["https://example.com/page1.html"]

        with (
            patch(
                "tensortruth.scrapers.library.fetch_inventory", return_value=mock_urls
            ),
            patch("tensortruth.scrapers.library.process_url", return_value=True),
            patch("tensortruth.scrapers.library.ThreadPoolExecutor") as mock_executor,
        ):
            mock_executor.return_value.__enter__.return_value.map.return_value = [True]

            scrape_library(
                library_name="test_lib",
                config=sphinx_config,
                output_base_dir=str(output_base_dir),
            )

            # Verify default worker count
            mock_executor.assert_called_once_with(max_workers=DEFAULT_MAX_WORKERS)

    def test_scrape_library_with_output_format(self, tmp_path, sphinx_config):
        """Test scraping with different output formats."""
        output_base_dir = tmp_path / "library_docs"
        output_base_dir.mkdir()

        mock_urls = ["https://example.com/page1.html"]

        with (
            patch(
                "tensortruth.scrapers.library.fetch_inventory", return_value=mock_urls
            ),
            patch(
                "tensortruth.scrapers.library.process_url", return_value=True
            ) as mock_process,
        ):
            scrape_library(
                library_name="test_lib",
                config=sphinx_config,
                output_base_dir=str(output_base_dir),
                output_format="html",
            )

            # Verify process_url was called with html format
            call_args = mock_process.call_args_list[0][0]
            assert call_args[3] == "html"  # output_format parameter

    def test_scrape_library_with_cleanup(self, tmp_path, doxygen_config):
        """Test scraping with HTML cleanup enabled."""
        output_base_dir = tmp_path / "library_docs"
        output_base_dir.mkdir()

        mock_urls = ["https://example.com/page1.html"]

        with (
            patch(
                "tensortruth.scrapers.library.fetch_doxygen_urls",
                return_value=mock_urls,
            ),
            patch(
                "tensortruth.scrapers.library.process_url", return_value=True
            ) as mock_process,
        ):
            scrape_library(
                library_name="test_lib",
                config=doxygen_config,
                output_base_dir=str(output_base_dir),
                enable_cleanup=True,
            )

            # Verify process_url was called with cleanup enabled
            call_args = mock_process.call_args_list[0][0]
            assert call_args[4] is True  # enable_cleanup parameter

    def test_scrape_library_with_min_size(self, tmp_path, sphinx_config):
        """Test scraping with minimum file size filter."""
        output_base_dir = tmp_path / "library_docs"
        output_base_dir.mkdir()

        mock_urls = ["https://example.com/page1.html", "https://example.com/page2.html"]

        with (
            patch(
                "tensortruth.scrapers.library.fetch_inventory", return_value=mock_urls
            ),
            patch(
                "tensortruth.scrapers.library.process_url",
                side_effect=[True, "skipped"],
            ) as mock_process,
        ):
            result = scrape_library(
                library_name="test_lib",
                config=sphinx_config,
                output_base_dir=str(output_base_dir),
                min_size=1000,
            )

            # Verify min_size was passed to process_url
            call_args = mock_process.call_args_list[0][0]
            assert call_args[5] == 1000  # min_size parameter
            assert result is True

    def test_scrape_library_statistics(self, tmp_path, sphinx_config):
        """Test that scraping correctly calculates statistics."""
        output_base_dir = tmp_path / "library_docs"
        output_base_dir.mkdir()

        mock_urls = [f"https://example.com/page{i}.html" for i in range(10)]

        # Simulate: 7 successful, 2 skipped, 1 failed
        mock_results = [True] * 7 + ["skipped"] * 2 + [False]

        with (
            patch(
                "tensortruth.scrapers.library.fetch_inventory", return_value=mock_urls
            ),
            patch("tensortruth.scrapers.library.process_url", side_effect=mock_results),
        ):
            result = scrape_library(
                library_name="test_lib",
                config=sphinx_config,
                output_base_dir=str(output_base_dir),
            )

            assert result is True

    def test_scrape_library_progress_callback(self, tmp_path, sphinx_config):
        """Test that progress callback is called with correct statistics."""
        output_base_dir = tmp_path / "library_docs"
        output_base_dir.mkdir()

        mock_urls = [f"https://example.com/page{i}.html" for i in range(5)]
        mock_results = [True, True, "skipped", True, False]

        callback_mock = MagicMock()

        with (
            patch(
                "tensortruth.scrapers.library.fetch_inventory", return_value=mock_urls
            ),
            patch("tensortruth.scrapers.library.process_url", side_effect=mock_results),
        ):
            result = scrape_library(
                library_name="test_lib",
                config=sphinx_config,
                output_base_dir=str(output_base_dir),
                progress_callback=callback_mock,
            )

            assert result is True
            # Verify callback was called with (successful=3, skipped=1, failed=1)
            callback_mock.assert_called_once_with(3, 1, 1)

    def test_scrape_library_no_callback_when_none(self, tmp_path, sphinx_config):
        """Test that no error occurs when progress_callback is None."""
        output_base_dir = tmp_path / "library_docs"
        output_base_dir.mkdir()

        mock_urls = ["https://example.com/page1.html"]

        with (
            patch(
                "tensortruth.scrapers.library.fetch_inventory", return_value=mock_urls
            ),
            patch("tensortruth.scrapers.library.process_url", return_value=True),
        ):
            result = scrape_library(
                library_name="test_lib",
                config=sphinx_config,
                output_base_dir=str(output_base_dir),
                progress_callback=None,
            )

            assert result is True

    def test_scrape_library_creates_output_directory(self, tmp_path, sphinx_config):
        """Test that output directory is created if it doesn't exist."""
        output_base_dir = tmp_path / "library_docs"
        # Don't create the directory

        mock_urls = ["https://example.com/page1.html"]

        with (
            patch(
                "tensortruth.scrapers.library.fetch_inventory", return_value=mock_urls
            ),
            patch("tensortruth.scrapers.library.process_url", return_value=True),
        ):
            result = scrape_library(
                library_name="test_lib",
                config=sphinx_config,
                output_base_dir=str(output_base_dir),
            )

            assert result is True
            expected_dir = output_base_dir / "library_test_lib"
            assert expected_dir.exists()

    def test_scrape_library_defaults_to_sphinx(self, tmp_path):
        """Test that missing 'type' field defaults to sphinx."""
        output_base_dir = tmp_path / "library_docs"
        output_base_dir.mkdir()

        # Config without 'type' field
        config = {
            "version": "1.0",
            "doc_root": "https://example.com/docs/",
        }

        mock_urls = ["https://example.com/page1.html"]

        with (
            patch(
                "tensortruth.scrapers.library.fetch_inventory", return_value=mock_urls
            ) as mock_sphinx,
            patch("tensortruth.scrapers.library.process_url", return_value=True),
        ):
            result = scrape_library(
                library_name="test_lib",
                config=config,
                output_base_dir=str(output_base_dir),
            )

            # Verify sphinx fetcher was called (not doxygen)
            mock_sphinx.assert_called_once()
            assert result is True
