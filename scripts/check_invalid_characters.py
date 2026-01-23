#!/usr/bin/env python3
"""Check for invalid Unicode characters in markdown files.

Detects:
- U+FFFD: Replacement character (diamond with question mark)
- U+0000: Null character
- U+FFFE, U+FFFF: Unicode noncharacters

Usage:
    # Check single file
    python scripts/check_invalid_characters.py --file /path/to/file.md

    # Check all files in library_docs (report only)
    python scripts/check_invalid_characters.py

    # Check and remove invalid characters
    python scripts/check_invalid_characters.py --fix

    # Show detailed line-by-line info
    python scripts/check_invalid_characters.py --verbose
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tensortruth.utils.pdf import (  # noqa: E402
    detect_invalid_characters,
    remove_invalid_characters,
)

DEFAULT_LIBRARY_PATH = Path.home() / ".tensortruth" / "library_docs"


def process_file(
    file_path: Path, fix: bool = False, verbose: bool = False
) -> tuple[bool, dict]:
    """Process a single markdown file.

    Args:
        file_path: Path to the markdown file
        fix: If True, remove invalid characters
        verbose: If True, include line-by-line details

    Returns:
        Tuple of (has_invalid, result_dict)
    """
    content = file_path.read_text(encoding="utf-8", errors="replace")
    result = detect_invalid_characters(content, include_line_info=verbose)

    if result["has_invalid"] and fix:
        cleaned = remove_invalid_characters(content)
        file_path.write_text(cleaned, encoding="utf-8")
        result["fixed"] = True

    return result["has_invalid"], result


def main():
    parser = argparse.ArgumentParser(
        description="Check for invalid Unicode characters in markdown files"
    )
    parser.add_argument(
        "--file",
        type=Path,
        help="Check a single file instead of the entire library",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Remove invalid characters from files",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show line-by-line details for affected files",
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

        has_invalid, result = process_file(args.file, fix=args.fix, verbose=True)

        if has_invalid:
            print(f"Invalid characters found in: {args.file}")
            for desc, count in result["counts"].items():
                print(f"  {desc}: {count} occurrence(s)")

            if result.get("occurrences"):
                print("\nOccurrences:")
                for line_num, context in result["occurrences"]:
                    print(f"  Line {line_num}: {context}")

            if result.get("fixed"):
                print("\n✓ Invalid characters removed")
        else:
            print(f"No invalid characters found: {args.file}")

    else:
        # Batch mode
        if not args.library_path.exists():
            print(f"Error: Library path not found: {args.library_path}")
            sys.exit(1)

        md_files = list(args.library_path.rglob("*.md"))
        print(f"Scanning {len(md_files)} markdown files in {args.library_path}...")

        affected_files = []
        total_counts: dict[str, int] = {}

        for file_path in md_files:
            has_invalid, result = process_file(
                file_path, fix=args.fix, verbose=args.verbose
            )

            if has_invalid:
                rel_path = file_path.relative_to(args.library_path)
                affected_files.append((rel_path, result))

                for desc, count in result["counts"].items():
                    total_counts[desc] = total_counts.get(desc, 0) + count

        # Report
        if affected_files:
            print(f"\nFound invalid characters in {len(affected_files)} files:\n")

            for rel_path, result in affected_files:
                counts_str = ", ".join(
                    f"{desc}: {count}" for desc, count in result["counts"].items()
                )
                status = " [FIXED]" if result.get("fixed") else ""
                print(f"  {rel_path}{status}")
                print(f"    {counts_str}")

                if args.verbose and result.get("occurrences"):
                    for line_num, context in result["occurrences"][:5]:
                        print(f"      Line {line_num}: {context[:80]}...")

            print("\nTotal counts:")
            for desc, count in total_counts.items():
                print(f"  {desc}: {count}")

            if args.fix:
                print(f"\n✓ Fixed {len(affected_files)} files")
            else:
                print("\nRun with --fix to remove invalid characters")
        else:
            print("\n✓ No invalid characters found in any files")


if __name__ == "__main__":
    main()
