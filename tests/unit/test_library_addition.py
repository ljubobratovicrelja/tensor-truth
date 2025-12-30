"""
Tests for interactive library addition feature.

Tests cover auto-detection of documentation types, URL validation,
and configuration management for adding libraries to sources.json.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from tensortruth.utils.detection import (
    detect_css_selector,
    detect_doc_type,
    detect_objects_inv,
)
from tensortruth.utils.interactive import add_library_interactive


@pytest.mark.unit
class TestDetectDocType:
    """Tests for detect_doc_type function (to be implemented)."""

    def test_detect_sphinx_from_objects_inv(self):
        """Test detection of Sphinx docs by objects.inv presence."""

        # Mock HEAD request that finds objects.inv
        with patch("requests.head") as mock_head:
            mock_head.return_value.status_code = 200
            mock_head.return_value.url = "https://example.com/objects.inv"

            # Should detect as Sphinx
            result = detect_doc_type("https://example.com/docs/")
            assert result == "sphinx"

    def test_detect_doxygen_from_index_pages(self):
        """Test detection of Doxygen docs by index page patterns."""

        # Mock HEAD request fails for objects.inv
        with patch("requests.head") as mock_head:
            mock_head.return_value.status_code = 404

            # Mock GET request finding annotated.html
            with patch("requests.get") as mock_get:
                mock_get.return_value.status_code = 200
                mock_get.return_value.text = '<a href="annotated.html">Classes</a>'

                # Should detect as Doxygen
                result = detect_doc_type("https://example.com/docs/")
                assert result == "doxygen"

    def test_unknown_doc_type_returns_none(self):
        """Test that unrecognizable doc types return None."""

        # Mock HEAD request fails for objects.inv
        with patch("requests.head") as mock_head:
            mock_head.return_value.status_code = 404

            # Mock GET request returns generic HTML
            with patch("requests.get") as mock_get:
                mock_get.return_value.status_code = 200
                mock_get.return_value.text = "<html><body>Generic docs</body></html>"

                # Should return None
                result = detect_doc_type("https://example.com/docs/")
                assert result is None


@pytest.mark.unit
class TestDetectObjectsInv:
    """Tests for detect_objects_inv function (to be implemented)."""

    def test_find_objects_inv_in_root(self):
        """Test finding objects.inv in doc root."""

        with patch("requests.head") as mock_head:
            mock_head.return_value.status_code = 200

            # Should find objects.inv at root
            result = detect_objects_inv("https://example.com/docs/")
            assert result == "https://example.com/docs/objects.inv"

    def test_find_objects_inv_in_subdirectory(self):
        """Test finding objects.inv in common subdirectories."""

        with patch("requests.head") as mock_head:
            # Root fails, but finds in _static/
            def side_effect(url, **kwargs):
                response = MagicMock()
                if "_static/objects.inv" in url:
                    response.status_code = 200
                else:
                    response.status_code = 404
                return response

            mock_head.side_effect = side_effect

            # Should find in _static/ subdirectory
            result = detect_objects_inv("https://example.com/docs/")
            assert result == "https://example.com/docs/_static/objects.inv"

    def test_objects_inv_not_found_returns_none(self):
        """Test that missing objects.inv returns None."""

        with patch("requests.head") as mock_head:
            mock_head.return_value.status_code = 404

            result = detect_objects_inv("https://example.com/docs/")
            assert result is None


@pytest.mark.unit
class TestDetectCssSelector:
    """Tests for detect_css_selector function (to be implemented)."""

    def test_detect_main_role_selector(self):
        """Test detection of div[role='main'] selector."""

        html = """
        <html>
            <div role="main">
                <h1>Content</h1>
            </div>
        </html>
        """

        with patch("requests.get") as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.text = html

            result = detect_css_selector("https://example.com/docs/")
            assert result == "div[role='main']"

    def test_detect_article_selector(self):
        """Test detection of article[role='main'] selector."""

        html = """
        <html>
            <article role="main">
                <h1>Content</h1>
            </article>
        </html>
        """

        with patch("requests.get") as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.text = html

            result = detect_css_selector("https://example.com/docs/")
            assert result == "article[role='main']"

    def test_fallback_to_main_tag(self):
        """Test fallback to <main> tag."""

        html = """
        <html>
            <main>
                <h1>Content</h1>
            </main>
        </html>
        """

        with patch("requests.get") as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.text = html

            result = detect_css_selector("https://example.com/docs/")
            assert result == "main"

    def test_no_selector_found_returns_none(self):
        """Test that undetectable selector returns None."""

        html = "<html><body>Content</body></html>"

        with patch("requests.get") as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.text = html

            result = detect_css_selector("https://example.com/docs/")
            assert result is None


@pytest.mark.integration
class TestAddLibraryInteractive:
    """Tests for add_library_interactive function (to be implemented)."""

    @pytest.fixture
    def sources_config(self, tmp_path):
        """Create temporary sources config file."""
        config_file = tmp_path / "sources.json"
        initial = {"libraries": {}, "papers": {}, "books": {}}
        config_file.write_text(json.dumps(initial, indent=2))
        return str(config_file)

    def test_add_sphinx_library_with_auto_detection(self, tmp_path, sources_config):
        """Test adding Sphinx library with full auto-detection."""

        args = MagicMock()
        args.url = "https://example.com/docs/"  # Provide URL directly

        # Mock auto-detection and URL validation
        with patch("tensortruth.utils.validation.validate_url") as mock_validate:
            with patch(
                "tensortruth.utils.interactive.detect_doc_type"
            ) as mock_detect_type:
                with patch(
                    "tensortruth.utils.interactive.detect_objects_inv"
                ) as mock_detect_inv:
                    with patch(
                        "tensortruth.utils.interactive.detect_css_selector"
                    ) as mock_detect_css:
                        mock_validate.return_value = True  # URL is valid
                        mock_detect_type.return_value = "sphinx"
                        mock_detect_inv.return_value = "https://example.com/objects.inv"
                        mock_detect_css.return_value = "div[role='main']"

                        # Mock user inputs: accept inv, accept selector,
                        # name, version, confirm, don't fetch
                        with patch(
                            "builtins.input",
                            side_effect=[
                                "y",  # Accept detected inventory URL
                                "y",  # Accept detected CSS selector
                                "test_lib",  # Name
                                "1.0",  # Version
                                "y",  # Confirm
                                "n",  # Don't fetch now
                            ],
                        ):
                            result = add_library_interactive(
                                sources_config, str(tmp_path), args
                            )

                            assert result == 0

                            # Verify library was added
                            config = json.loads(open(sources_config).read())
                            assert "test_lib" in config["libraries"]
                            lib = config["libraries"]["test_lib"]
                            assert lib["type"] == "sphinx"
                            assert (
                                lib["inventory_url"]
                                == "https://example.com/objects.inv"
                            )
                            assert lib["selector"] == "div[role='main']"
                            # Schema validation: libraries don't have display_name
                            assert "display_name" not in lib

    def test_add_doxygen_library(self, tmp_path, sources_config):
        """Test adding Doxygen library."""

        args = MagicMock()
        args.url = "https://example.com/doxygen/"

        with patch("tensortruth.utils.interactive.detect_doc_type") as mock_detect_type:
            with patch(
                "tensortruth.utils.interactive.detect_css_selector"
            ) as mock_detect_css:
                mock_detect_type.return_value = "doxygen"
                mock_detect_css.return_value = "div.contents"

                # Mock inputs for Doxygen-specific config
                with patch(
                    "builtins.input",
                    side_effect=[
                        "y",  # Accept detected CSS selector
                        "doxygen_lib",  # Name
                        "1.0",  # Version
                        "y",  # Confirm
                        "n",  # Don't fetch now
                    ],
                ):
                    result = add_library_interactive(
                        sources_config, str(tmp_path), args
                    )

                    assert result == 0

                    config = json.loads(open(sources_config).read())
                    assert config["libraries"]["doxygen_lib"]["type"] == "doxygen"
                    assert (
                        config["libraries"]["doxygen_lib"]["selector"] == "div.contents"
                    )

    def test_manual_override_auto_detection(self, tmp_path, sources_config):
        """Test that user can override auto-detected values."""

        args = MagicMock()
        args.url = "https://example.com/docs/"

        with patch("tensortruth.utils.interactive.detect_doc_type") as mock_detect_type:
            with patch(
                "tensortruth.utils.interactive.detect_objects_inv"
            ) as mock_detect_inv:
                with patch(
                    "tensortruth.utils.interactive.detect_css_selector"
                ) as mock_detect_css:
                    mock_detect_type.return_value = "sphinx"
                    mock_detect_inv.return_value = "https://example.com/objects.inv"
                    mock_detect_css.return_value = "div[role='main']"

                    # User chooses to override selector
                    with patch(
                        "builtins.input",
                        side_effect=[
                            "y",  # Accept detected inventory URL
                            "n",  # Don't use auto-detected selector
                            "article.content",  # Custom selector
                            "test_lib",  # Name
                            "1.0",  # Version
                            "y",  # Confirm
                            "n",  # Don't fetch now
                        ],
                    ):
                        result = add_library_interactive(
                            sources_config, str(tmp_path), args
                        )

                        assert result == 0

                        config = json.loads(open(sources_config).read())
                        assert (
                            config["libraries"]["test_lib"]["selector"]
                            == "article.content"
                        )

    def test_invalid_url_rejected(self, tmp_path, sources_config):
        """Test that invalid URLs are rejected."""

        args = MagicMock()
        args.url = None

        with patch("tensortruth.utils.validation.validate_url") as mock_validate:
            mock_validate.return_value = False

            # User enters invalid URL, then cancels
            with patch(
                "builtins.input",
                side_effect=[
                    "not-a-valid-url",  # Invalid URL
                    "",  # Cancel (empty input)
                ],
            ):
                with pytest.raises(SystemExit) as exc_info:
                    add_library_interactive(sources_config, str(tmp_path), args)

                # prompt_for_url raises SystemExit(1) on cancel
                assert exc_info.value.code == 1

    def test_duplicate_library_name_rejected(self, tmp_path, sources_config):
        """Test that duplicate library names are rejected."""

        # Add existing library
        config = json.loads(open(sources_config).read())
        config["libraries"]["existing_lib"] = {"type": "sphinx"}
        open(sources_config, "w").write(json.dumps(config, indent=2))

        args = MagicMock()
        args.url = "https://example.com/docs/"

        with patch("tensortruth.utils.interactive.detect_doc_type") as mock_detect_type:
            with patch(
                "tensortruth.utils.interactive.detect_objects_inv"
            ) as mock_detect_inv:
                with patch(
                    "tensortruth.utils.interactive.detect_css_selector"
                ) as mock_detect_css:
                    mock_detect_type.return_value = "sphinx"
                    mock_detect_inv.return_value = "https://example.com/objects.inv"
                    mock_detect_css.return_value = "div[role='main']"

                    # User tries to use duplicate name, then rejects overwrite
                    with patch(
                        "builtins.input",
                        side_effect=[
                            "y",  # Accept detected inventory URL
                            "y",  # Accept detected selector
                            "existing_lib",  # Duplicate name
                            "",  # Version (will default)
                            "n",  # Don't overwrite
                        ],
                    ):
                        result = add_library_interactive(
                            sources_config, str(tmp_path), args
                        )

                        # Should return error code 1
                        assert result == 1

                        # Config should not be modified
                        config = json.loads(open(sources_config).read())
                        assert "type" in config["libraries"]["existing_lib"]
                        assert config["libraries"]["existing_lib"]["type"] == "sphinx"
