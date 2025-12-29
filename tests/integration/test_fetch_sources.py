"""
Integration tests for fetch_sources.py CLI functionality.

Tests the full CLI workflows including argument parsing, interactive flows,
and the complete add_paper_interactive function.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from tensortruth.fetch_sources import add_paper_interactive, interactive_add
from tensortruth.fetch_sources import main as fetch_main


@pytest.mark.integration
class TestAddPaperInteractive:
    """Tests for add_paper_interactive function."""

    @pytest.fixture
    def mock_arxiv_paper(self):
        """Mock ArXiv paper result."""
        paper = MagicMock()
        paper.title = "Attention Is All You Need"

        # Create proper author mocks
        author1 = MagicMock()
        author1.name = "Vaswani"
        author2 = MagicMock()
        author2.name = "Shazeer"
        paper.authors = [author1, author2]

        paper.published = MagicMock()
        paper.published.year = 2017
        return paper

    @pytest.fixture
    def sources_config(self, tmp_path):
        """Create temporary sources config file."""
        config_file = tmp_path / "sources.json"
        initial = {
            "libraries": {},
            "papers": {
                "existing_category": {
                    "type": "arxiv",
                    "display_name": "Existing Category",
                    "description": "Test category",
                    "items": {},
                }
            },
            "books": {},
        }
        config_file.write_text(json.dumps(initial, indent=2))
        return str(config_file)

    def test_add_to_existing_category(self, tmp_path, sources_config, mock_arxiv_paper):
        """Test adding papers to an existing category."""

        # Mock args
        args = MagicMock()
        args.category = "existing_category"
        args.arxiv_ids = ["1706.03762"]

        # Mock ArXiv search
        with patch("arxiv.Search") as mock_search_cls:
            mock_search = MagicMock()
            mock_search.results.return_value = iter([mock_arxiv_paper])
            mock_search_cls.return_value = mock_search

            # Mock user input (confirm adding)
            with patch("builtins.input", side_effect=["y", "n"]):
                result = add_paper_interactive(sources_config, str(tmp_path), args)

        assert result == 0

        # Verify config was updated
        config = json.loads(open(sources_config).read())
        assert "1706.03762" in config["papers"]["existing_category"]["items"]
        paper = config["papers"]["existing_category"]["items"]["1706.03762"]
        assert paper["title"] == "Attention Is All You Need"
        assert "Vaswani" in paper["authors"]

    def test_create_new_category(self, tmp_path, sources_config, mock_arxiv_paper):
        """Test creating a new category and adding papers."""

        args = MagicMock()
        args.category = "new_category"
        args.arxiv_ids = ["1706.03762"]

        with patch("arxiv.Search") as mock_search_cls:
            mock_search = MagicMock()
            mock_search.results.return_value = iter([mock_arxiv_paper])
            mock_search_cls.return_value = mock_search

            # Mock user inputs: display name, description, confirm, no fetch
            with patch(
                "builtins.input",
                side_effect=[
                    "New Category",
                    "Test description",
                    "y",
                    "n",
                ],
            ):
                result = add_paper_interactive(sources_config, str(tmp_path), args)

        assert result == 0

        # Verify new category created
        config = json.loads(open(sources_config).read())
        assert "new_category" in config["papers"]
        assert config["papers"]["new_category"]["display_name"] == "New Category"
        assert "1706.03762" in config["papers"]["new_category"]["items"]

    def test_interactive_category_prompt(
        self, tmp_path, sources_config, mock_arxiv_paper
    ):
        """Test that category is prompted when not provided."""

        args = MagicMock()
        args.category = None
        args.arxiv_ids = ["1706.03762"]

        with patch("arxiv.Search") as mock_search_cls:
            mock_search = MagicMock()
            mock_search.results.return_value = iter([mock_arxiv_paper])
            mock_search_cls.return_value = mock_search

            # Mock inputs: category name, confirm, no fetch
            with patch(
                "builtins.input",
                side_effect=[
                    "existing_category",
                    "y",
                    "n",
                ],
            ):
                result = add_paper_interactive(sources_config, str(tmp_path), args)

        assert result == 0

    def test_interactive_arxiv_ids_prompt(
        self, tmp_path, sources_config, mock_arxiv_paper
    ):
        """Test that ArXiv IDs are prompted when not provided."""

        args = MagicMock()
        args.category = "existing_category"
        args.arxiv_ids = None

        with patch("arxiv.Search") as mock_search_cls:
            mock_search = MagicMock()
            mock_search.results.return_value = iter([mock_arxiv_paper])
            mock_search_cls.return_value = mock_search

            # Mock inputs: arxiv IDs, confirm, no fetch
            with patch(
                "builtins.input",
                side_effect=[
                    "1706.03762",
                    "y",
                    "n",
                ],
            ):
                result = add_paper_interactive(sources_config, str(tmp_path), args)

        assert result == 0

    def test_skip_duplicate_papers(self, tmp_path, sources_config):
        """Test that duplicate papers are skipped."""

        # Add paper to config first
        config = json.loads(open(sources_config).read())
        config["papers"]["existing_category"]["items"]["1706.03762"] = {
            "title": "Existing Paper",
            "arxiv_id": "1706.03762",
        }
        open(sources_config, "w").write(json.dumps(config, indent=2))

        args = MagicMock()
        args.category = "existing_category"
        args.arxiv_ids = ["1706.03762"]

        # Should detect duplicate and skip
        with patch("builtins.input", return_value="n"):
            result = add_paper_interactive(sources_config, str(tmp_path), args)

        # Should complete but not add duplicate
        assert result == 0

    def test_arxiv_fetch_error_manual_entry(self, tmp_path, sources_config):
        """Test manual entry when ArXiv API fails."""

        args = MagicMock()
        args.category = "existing_category"
        args.arxiv_ids = ["9999.99999"]  # Invalid ID

        with patch("arxiv.Search") as mock_search_cls:
            mock_search_cls.side_effect = Exception("API Error")

            # Mock inputs: yes to manual, title, authors, year, confirm, no fetch
            with patch(
                "builtins.input",
                side_effect=[
                    "y",  # Add manually?
                    "Manual Paper",
                    "Test Author",
                    "2023",
                    "y",  # Confirm
                    "n",  # Fetch
                ],
            ):
                result = add_paper_interactive(sources_config, str(tmp_path), args)

        assert result == 0

        # Verify manual entry
        config = json.loads(open(sources_config).read())
        assert "9999.99999" in config["papers"]["existing_category"]["items"]
        paper = config["papers"]["existing_category"]["items"]["9999.99999"]
        assert paper["title"] == "Manual Paper"

    def test_user_cancels_at_confirmation(
        self, tmp_path, sources_config, mock_arxiv_paper
    ):
        """Test that user can cancel at confirmation step."""

        args = MagicMock()
        args.category = "existing_category"
        args.arxiv_ids = ["1706.03762"]

        with patch("arxiv.Search") as mock_search_cls:
            mock_search = MagicMock()
            mock_search.results.return_value = iter([mock_arxiv_paper])
            mock_search_cls.return_value = mock_search

            # Mock input: cancel at confirmation
            with patch("builtins.input", return_value="n"):
                result = add_paper_interactive(sources_config, str(tmp_path), args)

        # Should return error code
        assert result == 1

        # Config should not be modified
        config = json.loads(open(sources_config).read())
        assert "1706.03762" not in config["papers"]["existing_category"]["items"]

    def test_invalid_arxiv_ids_rejected(self, tmp_path, sources_config):
        """Test that invalid ArXiv IDs are rejected."""

        args = MagicMock()
        args.category = "existing_category"
        args.arxiv_ids = ["invalid-id", "also-invalid"]

        result = add_paper_interactive(sources_config, str(tmp_path), args)

        # Should fail due to no valid IDs
        assert result == 1

    def test_reject_book_category(self, tmp_path):
        """Test that book categories are rejected for paper addition."""

        # Create config with book category
        config_file = tmp_path / "sources.json"
        initial = {
            "libraries": {},
            "papers": {
                "book_category": {
                    "type": "pdf_book",
                    "title": "Some Book",
                }
            },
            "books": {},
        }
        config_file.write_text(json.dumps(initial, indent=2))

        args = MagicMock()
        args.category = "book_category"
        args.arxiv_ids = ["1706.03762"]

        result = add_paper_interactive(str(config_file), str(tmp_path), args)

        # Should reject book category
        assert result == 1


@pytest.mark.integration
class TestInteractiveAdd:
    """Tests for interactive_add main entry point."""

    @pytest.fixture
    def sources_config(self, tmp_path):
        """Create temporary sources config file."""
        config_file = tmp_path / "sources.json"
        initial = {"libraries": {}, "papers": {}, "books": {}}
        config_file.write_text(json.dumps(initial, indent=2))
        return str(config_file)

    def test_type_selection_paper(self, tmp_path, sources_config):
        """Test selecting paper type interactively."""

        args = MagicMock()
        args.type = None

        with patch(
            "builtins.input",
            side_effect=[
                "3",  # Choose paper
                "test_cat",  # Category
                "1706.03762",  # ArXiv ID
                "n",  # Cancel at confirmation
            ],
        ):
            with patch("tensortruth.fetch_sources.add_paper_interactive") as mock_add:
                mock_add.return_value = 0
                interactive_add(sources_config, str(tmp_path), args)

        # Should have called paper addition
        mock_add.assert_called_once()

    def test_type_from_cli_skip_prompt(self, tmp_path, sources_config):
        """Test that --type flag skips type selection prompt."""

        args = MagicMock()
        args.type = "paper"

        with patch("tensortruth.fetch_sources.add_paper_interactive") as mock_add:
            mock_add.return_value = 0
            interactive_add(sources_config, str(tmp_path), args)

        # Should skip to paper addition without prompting
        mock_add.assert_called_once()

    def test_library_addition_called(self, tmp_path, sources_config):
        """Test that library addition is called."""

        args = MagicMock()
        args.type = "library"
        args.url = None  # Will trigger URL prompt in the function

        with patch("tensortruth.fetch_sources.add_library_interactive") as mock_add:
            mock_add.return_value = 0
            result = interactive_add(sources_config, str(tmp_path), args)

        # Should call library addition
        mock_add.assert_called_once()
        assert result == 0

    def test_book_addition_called(self, tmp_path, sources_config):
        """Test that book addition is called."""

        args = MagicMock()
        args.type = "book"
        args.url = None  # Will trigger URL prompt in the function

        with patch("tensortruth.fetch_sources.add_book_interactive") as mock_add:
            mock_add.return_value = 0
            result = interactive_add(sources_config, str(tmp_path), args)

        # Should call book addition
        mock_add.assert_called_once()
        assert result == 0

    def test_invalid_type_rejected(self, tmp_path, sources_config):
        """Test that invalid --type is rejected."""

        args = MagicMock()
        args.type = "invalid"

        result = interactive_add(sources_config, str(tmp_path), args)

        # Should return error
        assert result == 1

    def test_plural_type_normalized(self, tmp_path, sources_config):
        """Test that plural types (papers/books) are normalized."""

        args = MagicMock()
        args.type = "papers"  # Plural

        with patch("tensortruth.fetch_sources.add_paper_interactive") as mock_add:
            mock_add.return_value = 0
            interactive_add(sources_config, str(tmp_path), args)

        # Should normalize to "paper" and call addition
        mock_add.assert_called_once()


@pytest.mark.integration
class TestCLIArgumentParsing:
    """Tests for CLI argument combinations."""

    @patch("tensortruth.fetch_sources.load_user_sources")
    @patch("tensortruth.fetch_sources.interactive_add")
    def test_add_flag_triggers_interactive(self, mock_interactive, mock_load):
        """Test that --add flag triggers interactive mode."""
        mock_interactive.return_value = 0
        mock_load.return_value = {"libraries": {}, "papers": {}, "books": {}}

        with patch("sys.argv", ["tensor-truth-docs", "--add"]):
            fetch_main()

        mock_interactive.assert_called_once()

    @patch("tensortruth.fetch_sources.load_user_sources")
    @patch("tensortruth.fetch_sources.interactive_add")
    def test_add_with_type_flag(self, mock_interactive, mock_load):
        """Test --add with --type flag."""
        mock_interactive.return_value = 0
        mock_load.return_value = {"libraries": {}, "papers": {}, "books": {}}

        with patch("sys.argv", ["tensor-truth-docs", "--add", "--type", "paper"]):
            fetch_main()

        mock_interactive.assert_called_once()
        # Verify args.type is set
        call_args = mock_interactive.call_args
        assert call_args[0][2].type == "paper"

    @patch("tensortruth.fetch_sources.load_user_sources")
    @patch("tensortruth.fetch_sources.interactive_add")
    def test_add_with_arxiv_ids(self, mock_interactive, mock_load):
        """Test --add with --arxiv-ids for non-interactive flow."""
        mock_interactive.return_value = 0
        mock_load.return_value = {"libraries": {}, "papers": {}, "books": {}}

        with patch(
            "sys.argv",
            [
                "tensor-truth-docs",
                "--add",
                "--type",
                "paper",
                "--category",
                "test",
                "--arxiv-ids",
                "1706.03762",
            ],
        ):
            fetch_main()

        mock_interactive.assert_called_once()

    def test_list_flag(self, capsys):
        """Test --list flag displays sources."""

        with patch("sys.argv", ["tensor-truth-docs", "--list"]):
            with patch("tensortruth.fetch_sources.load_user_sources") as mock_load:
                mock_load.return_value = {
                    "libraries": {"pytorch": {"version": "2.0"}},
                    "papers": {},
                    "books": {},
                }

                fetch_main()

        # Should display sources
        captured = capsys.readouterr()
        assert "pytorch" in captured.out

    def test_traditional_paper_fetch(self):
        """Test traditional --type papers --category workflow."""

        with patch(
            "sys.argv",
            [
                "tensor-truth-docs",
                "--type",
                "papers",
                "--category",
                "test_cat",
                "--arxiv-ids",
                "1706.03762",
            ],
        ):
            with patch("tensortruth.fetch_sources.load_user_sources") as mock_load:
                with patch("arxiv.Search"):
                    with patch("tensortruth.fetch_sources.fetch_arxiv_paper"):
                        with patch("tensortruth.fetch_sources.update_sources_config"):
                            mock_load.return_value = {
                                "libraries": {},
                                "papers": {
                                    "test_cat": {
                                        "type": "arxiv",
                                        "items": {},
                                    }
                                },
                                "books": {},
                            }

                            # Should not raise
                            try:
                                result = fetch_main()
                            except SystemExit as e:
                                result = e.code

        # Should succeed (0 or None)
        assert result in [0, None]
