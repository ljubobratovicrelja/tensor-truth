#!/usr/bin/env python3
"""Command-line interface for Tensor-Truth.

Unified CLI for managing documentation, papers, databases, and the web interface.
"""

import os
import sys
from pathlib import Path


def _missing_imports_message(tool, package, error):
    """Print standardized message for missing dependencies.

    Args:
        tool: Name of the CLI tool
        package: Package name to install
        error: The import error that occurred
    """
    print(
        (
            f"Missing dependencies for {tool} CLI tool. Error: {error}.\n"
            f"Install with: pip install tensor-truth[{package}]\n"
        ),
        file=sys.stderr,
    )


def main():
    """Main entry point - launches the Streamlit web application."""
    # Get the project root directory (where app.py lives)
    package_dir = Path(__file__).parent.resolve()
    app_path = package_dir / "app.py"

    # Check and pull required Ollama models before starting the app
    try:
        from tensortruth.core.ollama import ensure_required_models_available

        print("🔍 Checking for required Ollama models...")
        pulled_models = ensure_required_models_available()
        if pulled_models:
            print(
                f"✅ Successfully pulled {len(pulled_models)} models: {', '.join(pulled_models)}"
            )
        else:
            print("✅ All required models are already available")
    except ImportError:
        print("⚠️  Could not import Ollama utilities - model auto-pulling disabled")
    except Exception as e:
        print(f"⚠️  Error checking/pulling models: {e}")
        print("💡 You can manually pull required models with:")
        from tensortruth.core.constants import (
            DEFAULT_AGENT_REASONING_MODEL,
            DEFAULT_FALLBACK_MODEL,
            DEFAULT_RAG_MODEL,
        )

        print(f"   ollama pull {DEFAULT_RAG_MODEL}")
        print(f"   ollama pull {DEFAULT_FALLBACK_MODEL}")
        print(f"   ollama pull {DEFAULT_AGENT_REASONING_MODEL}")

    if not app_path.exists():
        print(f"Error: Could not find app.py at {app_path}", file=sys.stderr)
        sys.exit(1)

    # Extract custom flags before passing to Streamlit
    filtered_args = []
    debug_context = False

    for arg in sys.argv[1:]:
        if arg == "--debug-context":
            debug_context = True
        else:
            filtered_args.append(arg)

    # Set environment variable for app to read
    if debug_context:
        os.environ["TENSOR_TRUTH_DEBUG_CONTEXT"] = "1"

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
    sys.argv = ["streamlit", "run", str(app_path)] + filtered_args
    sys.exit(st_cli.main())


def scrape_docs():
    """Entry point for unified source fetching tool."""
    try:
        from tensortruth.fetch_sources import main as fetch_main
    except ImportError as e:
        _missing_imports_message("tensor-truth-docs", "docs", e)
        sys.exit(1)

    sys.exit(fetch_main())


def build_db():
    """Entry point for database building tool."""
    from tensortruth.build_db import main as build_main

    sys.exit(build_main())


if __name__ == "__main__":
    main()
