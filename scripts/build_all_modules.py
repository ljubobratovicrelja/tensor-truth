#!/usr/bin/env python3
"""Build all modules with appropriate chunk sizes."""

import json
import subprocess
import sys
from pathlib import Path


def load_modules_from_config():
    """Load and categorize modules from config/sources.json.

    Returns:
        dict: Dictionary with keys 'books', 'papers', 'api_docs' containing module lists
    """
    # Get the path to config/sources.json relative to this script
    script_dir = Path(__file__).parent
    config_path = script_dir.parent / "config" / "sources.json"

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            sources = json.load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {config_path}: {e}", file=sys.stderr)
        sys.exit(1)

    # Extract books - those with category in deep_learning or machine_learning
    books = []
    papers_modules = []
    api_docs = []

    papers_section = sources.get("papers", {})

    for key, value in papers_section.items():
        if isinstance(value, dict):
            # Check if it's a paper collection (has 'items' or is arxiv type)
            if value.get("type") == "arxiv" or "items" in value:
                papers_modules.append(key)
            # Check if it's a book (has category field and type pdf_book)
            elif value.get("type") == "pdf_book":
                books.append(key)

    # Extract API documentation from libraries section
    libraries = sources.get("libraries", {})
    for lib_name, lib_config in libraries.items():
        version = lib_config.get("version", "")
        # Create module name: libraryname_version
        module_name = f"{lib_name}_{version}"
        api_docs.append(module_name)

    return {"books": books, "papers": papers_modules, "api_docs": api_docs}


def run_build(modules, chunk_sizes, description):
    """Run tensor-truth-build command with specified modules and chunk sizes.

    Args:
        modules: List of module names to build
        chunk_sizes: List of chunk sizes (e.g., [3072, 768, 384])
        description: Description of what's being built (e.g., "books", "papers")
    """
    print(f"\nBuilding {description} with chunk sizes {chunk_sizes}...")

    cmd = (
        ["tensor-truth-build", "--modules"]
        + modules
        + ["--chunk-sizes"]
        + [str(s) for s in chunk_sizes]
    )

    try:
        subprocess.run(cmd, check=True, capture_output=False)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error building modules: {e}", file=sys.stderr)
        return False
    except FileNotFoundError:
        print(
            "Error: tensor-truth-build command not found. Make sure it's installed and in PATH.",
            file=sys.stderr,
        )
        return False


def main():
    """Main function to build all modules."""

    # Load modules from config/sources.json
    modules = load_modules_from_config()

    books = modules["books"]
    papers = modules["papers"]
    api_docs = modules["api_docs"]

    print(f"Loaded {len(books)} book modules")
    print(f"Loaded {len(papers)} paper modules")
    print(f"Loaded {len(api_docs)} API documentation modules")

    success = True

    # Build books with chunk sizes [3072, 768, 384]
    if books and not run_build(books, [3072, 768, 384], "books"):
        success = False

    # Build papers with chunk sizes [2048, 512, 256]
    if papers and not run_build(papers, [2048, 512, 256], "papers"):
        success = False

    # Build API docs with chunk sizes [2048, 512, 256]
    if api_docs and not run_build(api_docs, [2048, 512, 256], "api_docs"):
        success = False

    if success:
        print("\n✓ Build complete!", file=sys.stdout)
        return 0
    else:
        print("\n✗ Build completed with errors", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
