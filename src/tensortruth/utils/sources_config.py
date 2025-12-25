"""Sources configuration management utilities."""

import json
import logging
import os

logger = logging.getLogger(__name__)


def load_user_sources(config_path):
    """
    Load user's sources.json (auto-managed registry).

    Args:
        config_path: Path to user's sources.json file

    Returns:
        Dictionary with 'libraries' and 'papers' sections
    """
    if not os.path.exists(config_path):
        raise IOError(f"User config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def update_sources_config(config_path, source_type, name, entry):
    """
    Add or update a source entry in user's sources.json.

    Creates the config file if it doesn't exist. Auto-writes after
    successful source fetching.

    Args:
        config_path: Path to sources.json
        source_type: 'libraries' or 'papers'
        name: Source name (e.g., 'pytorch')
        entry: Source configuration dictionary

    Examples:
        >>> update_sources_config(
        ...     "~/.tensortruth/sources.json",
        ...     "libraries",
        ...     "pytorch",
        ...     {"type": "sphinx", "version": "2.9", ...}
        ... )
    """
    # Load existing or create new
    config = load_user_sources(config_path)

    # Ensure structure
    if source_type not in config:
        config[source_type] = {}

    # Add/update entry
    config[source_type][name] = entry

    # Write back
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    logger.info(f"Updated {config_path}: added {source_type}/{name}")


def load_config(config_path):
    """Load unified sources configuration.

    Args:
        config_path: Path to JSON configuration file

    Returns:
        Dictionary with 'libraries' and 'papers' sections
    """
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        return {"libraries": {}, "papers": {}}

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        # Validate structure
        if "libraries" not in config:
            config["libraries"] = {}
        if "papers" not in config:
            config["papers"] = {}

        return config
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return {"libraries": {}, "papers": {}}


def list_sources(config):
    """List all available sources (libraries, books, and papers)."""
    print("\n=== Available Libraries ===")
    if config.get("libraries"):
        for name, lib_config in sorted(config["libraries"].items()):
            doc_type = lib_config.get("type", "unknown")
            version = lib_config.get("version", "?")
            print(f"  • {name:<20} ({doc_type}, v{version})")
    else:
        print("  (none)")

    # Separate books and paper categories
    books = {}
    paper_categories = {}

    if config.get("papers"):
        for name, cat_config in config["papers"].items():
            cat_type = cat_config.get("type", "unknown")
            if cat_type == "pdf_book":
                books[name] = cat_config
            else:
                paper_categories[name] = cat_config

    print("\n=== Available Books ===")
    if books:
        for name, book_config in sorted(books.items()):
            title = book_config.get("title", name)
            authors = book_config.get("authors", [])
            desc = book_config.get("description", "")

            # Format: "Title by Author1, Author2 (item_name)"
            if authors:
                author_str = ", ".join(authors)
                print(f"  • {title} by {author_str} ({name})")
            else:
                print(f"  • {title} ({name})")

            if desc:
                print(f"    {desc}")
    else:
        print("  (none)")

    print("\n=== Available Paper Categories ===")
    if paper_categories:
        for name, cat_config in sorted(paper_categories.items()):
            cat_type = cat_config.get("type", "unknown")
            display_name = cat_config.get("display_name", name)
            desc = cat_config.get("description", "")
            item_count = len(cat_config.get("items", []))

            # Format: "Display Name (item_name, N items)"
            print(f"  • {display_name} ({name}, {item_count} items)")
            if desc:
                print(f"    {desc}")
    else:
        print("  (none)")
