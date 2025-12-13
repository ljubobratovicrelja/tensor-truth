#!/usr/bin/env python3
"""
Command-line interface for Tensor-Truth.
Unified CLI for managing documentation, papers, databases, and the web interface.
"""

import sys
from pathlib import Path


def main():
    """Main entry point - launches the Streamlit web application."""
    # Get the project root directory (where app.py lives)
    package_dir = Path(__file__).parent.resolve()
    app_path = package_dir / "app.py"

    if not app_path.exists():
        print(f"Error: Could not find app.py at {app_path}", file=sys.stderr)
        sys.exit(1)

    # Import streamlit.web.cli as st_cli to avoid loading the entire streamlit module
    try:
        from streamlit.web import cli as st_cli
    except ImportError:
        print(
            "Error: Streamlit is not installed. Install with: pip install streamlit",
            file=sys.stderr,
        )
        sys.exit(1)

    # Run the Streamlit app
    sys.argv = ["streamlit", "run", str(app_path)] + sys.argv[1:]
    sys.exit(st_cli.main())


def fetch_paper():
    """Entry point for paper fetching tool."""
    from tensortruth.fetch_paper import main as fetch_main

    sys.exit(fetch_main())


def scrape_docs():
    """Entry point for documentation scraping tool."""
    from tensortruth.scrape_docs import main as scrape_main

    sys.exit(scrape_main())


def build_db():
    """Entry point for database building tool."""
    from tensortruth.build_db import main as build_main

    sys.exit(build_main())


if __name__ == "__main__":
    main()
