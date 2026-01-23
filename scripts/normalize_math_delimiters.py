#!/usr/bin/env python3
"""Convert LaTeX math delimiters in markdown files for LLM compatibility.

Converts:
- $$...$$ -> \\[...\\] (display math)
- $...$ -> \\(...\\) (inline math)

Usage:
    # Single file mode
    python scripts/normalize_math_delimiters.py --file /path/to/file.md

    # Batch mode (all .md files in library_docs)
    python scripts/normalize_math_delimiters.py

    # Dry run (show what would change without modifying files)
    python scripts/normalize_math_delimiters.py --dry-run
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tensortruth.utils.pdf import normalize_math_delimiters  # noqa: E402

DEFAULT_LIBRARY_PATH = Path.home() / ".tensortruth" / "library_docs"


def process_file(file_path: Path, dry_run: bool = False) -> tuple[bool, str]:
    """Process a single markdown file.

    Args:
        file_path: Path to the markdown file
        dry_run: If True, don't write changes

    Returns:
        Tuple of (was_modified, sample_diff)
    """
    original = file_path.read_text(encoding="utf-8")
    converted = normalize_math_delimiters(original)

    if original == converted:
        return False, ""

    # Generate a sample diff (first change)
    sample_diff = ""
    orig_lines = original.split("\n")
    conv_lines = converted.split("\n")

    for i, (orig, conv) in enumerate(zip(orig_lines, conv_lines)):
        if orig != conv:
            sample_diff = f"  Line {i + 1}:\n    - {orig[:100]}\n    + {conv[:100]}"
            break

    if not dry_run:
        file_path.write_text(converted, encoding="utf-8")

    return True, sample_diff


def main():
    parser = argparse.ArgumentParser(
        description="Convert LaTeX math delimiters in markdown files"
    )
    parser.add_argument(
        "--file",
        type=Path,
        help="Process a single file instead of the entire library",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would change without modifying files",
    )
    parser.add_argument(
        "--library-path",
        type=Path,
        default=DEFAULT_LIBRARY_PATH,
        help=f"Path to library_docs directory (default: {DEFAULT_LIBRARY_PATH})",
    )
    args = parser.parse_args()

    if args.file:
        # Single file mode
        if not args.file.exists():
            print(f"Error: File not found: {args.file}")
            sys.exit(1)

        modified, sample_diff = process_file(args.file, dry_run=args.dry_run)

        if modified:
            action = "Would modify" if args.dry_run else "Modified"
            print(f"{action}: {args.file}")
            if sample_diff:
                print(sample_diff)
        else:
            print(f"No changes needed: {args.file}")

    else:
        # Batch mode
        if not args.library_path.exists():
            print(f"Error: Library path not found: {args.library_path}")
            sys.exit(1)

        md_files = list(args.library_path.rglob("*.md"))
        print(f"Found {len(md_files)} markdown files in {args.library_path}")

        modified_count = 0
        for file_path in md_files:
            modified, sample_diff = process_file(file_path, dry_run=args.dry_run)

            if modified:
                modified_count += 1
                action = "Would modify" if args.dry_run else "Modified"
                print(f"{action}: {file_path.relative_to(args.library_path)}")

        action = "Would modify" if args.dry_run else "Modified"
        print(f"\n{action} {modified_count} of {len(md_files)} files")


if __name__ == "__main__":
    main()
