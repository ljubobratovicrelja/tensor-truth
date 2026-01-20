#!/usr/bin/env python3
"""Command-line interface for Tensor-Truth.

Unified CLI for managing documentation, papers, databases, and the web interface.
"""

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


def run_ui():
    """Entry point for React frontend dev server."""
    import argparse
    import os
    import subprocess
    import shutil

    parser = argparse.ArgumentParser(
        description="Start the TensorTruth React frontend dev server"
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="Host to bind to (use 0.0.0.0 for network access). Default: localhost",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5173,
        help="Port to run on. Default: 5173",
    )
    args = parser.parse_args()

    # Find the frontend directory relative to the package
    package_dir = Path(__file__).parent.resolve()
    # Go up to project root: src/tensortruth -> src -> project_root
    project_root = package_dir.parent.parent
    frontend_dir = project_root / "frontend"

    if not frontend_dir.exists():
        print(f"Error: Frontend directory not found at {frontend_dir}", file=sys.stderr)
        print("Make sure you're running from a development installation.", file=sys.stderr)
        sys.exit(1)

    # Check if npm is available
    npm_path = shutil.which("npm")
    if not npm_path:
        print("Error: npm is not installed or not in PATH.", file=sys.stderr)
        print("Install Node.js from https://nodejs.org/", file=sys.stderr)
        sys.exit(1)

    # Check if node_modules exists
    node_modules = frontend_dir / "node_modules"
    if not node_modules.exists():
        print("📦 Installing frontend dependencies...")
        result = subprocess.run(
            ["npm", "install"],
            cwd=frontend_dir,
            env={**os.environ},
        )
        if result.returncode != 0:
            print("Error: Failed to install dependencies.", file=sys.stderr)
            sys.exit(1)

    # Build the URL for display
    display_host = args.host if args.host != "0.0.0.0" else "localhost"
    url = f"http://{display_host}:{args.port}"

    print("🚀 Starting frontend dev server...")
    print(f"   Directory: {frontend_dir}")
    print(f"   URL: {url}")
    if args.host == "0.0.0.0":
        print("   Network: Available on all interfaces")
    print()

    # Run npm run dev with host and port arguments
    try:
        result = subprocess.run(
            ["npm", "run", "dev", "--", "--host", args.host, "--port", str(args.port)],
            cwd=frontend_dir,
            env={**os.environ},
        )
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        print("\n👋 Frontend server stopped.")
        sys.exit(0)


if __name__ == "__main__":
    main()
