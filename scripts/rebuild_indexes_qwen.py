#!/usr/bin/env python3
"""
Rebuild all indexes with Qwen3-Embedding-0.6B and optimized chunking strategies.

This script rebuilds all indexes (libraries, papers, books) with:
- Qwen/Qwen3-Embedding-0.6B embedding model
- Optimized chunk sizes per content type
- Semantic/hierarchical chunking strategies
- Progress tracking and ETA estimation

Designed for overnight runs with detailed logging.
"""

import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path


@dataclass
class BuildConfig:
    """Configuration for a build task."""

    name: str
    modules: list[str]
    chunk_sizes: list[int]
    chunk_overlap: int
    chunking_strategy: str
    semantic_buffer_size: int | None = None
    semantic_breakpoint_threshold: int | None = None


def load_sources_config() -> dict:
    """Load sources from ~/.tensortruth/sources.json."""
    config_path = Path.home() / ".tensortruth" / "sources.json"

    if not config_path.exists():
        print(f"Error: sources.json not found at {config_path}", file=sys.stderr)
        sys.exit(1)

    with open(config_path, encoding="utf-8") as f:
        return json.load(f)


def get_module_lists(sources: dict) -> dict[str, list[str]]:
    """Extract module lists from sources config."""
    libraries = list(sources.get("libraries", {}).keys())
    papers = list(sources.get("papers", {}).keys())
    books = list(sources.get("books", {}).keys())

    return {
        "libraries": libraries,
        "papers": papers,
        "books": books,
    }


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        mins = seconds / 60
        return f"{mins:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def format_eta(seconds: float) -> str:
    """Format ETA as timestamp."""
    eta = datetime.now() + timedelta(seconds=seconds)
    return eta.strftime("%H:%M:%S")


def print_banner(text: str, char: str = "=") -> None:
    """Print a banner with the given text."""
    width = 70
    print()
    print(char * width)
    print(f" {text}")
    print(char * width)


def print_progress(
    current: int,
    total: int,
    module_name: str,
    elapsed: float,
    category_elapsed: float,
) -> None:
    """Print progress update with ETA."""
    pct = (current / total) * 100
    avg_time = category_elapsed / current if current > 0 else 0
    remaining = (total - current) * avg_time

    print(f"\n[{current}/{total}] ({pct:.0f}%) Building: {module_name}")
    print(f"    Elapsed: {format_duration(elapsed)} | ", end="")
    print(f"Avg per module: {format_duration(avg_time)} | ", end="")
    print(f"ETA: {format_eta(remaining)} ({format_duration(remaining)} remaining)")


def run_build(config: BuildConfig, embedding_model: str) -> tuple[bool, float]:
    """Run tensor-truth-build with the given configuration.

    Returns:
        Tuple of (success, duration_seconds)
    """
    cmd = [
        "tensor-truth-build",
        "--modules",
        *config.modules,
        "--embedding-model",
        embedding_model,
        "--chunking-strategy",
        config.chunking_strategy,
        "--chunk-sizes",
        *[str(s) for s in config.chunk_sizes],
        "--chunk-overlap",
        str(config.chunk_overlap),
    ]

    # Add semantic parameters if using semantic chunking
    if config.chunking_strategy in ("semantic", "semantic_hierarchical"):
        if config.semantic_buffer_size is not None:
            cmd.extend(["--semantic-buffer-size", str(config.semantic_buffer_size)])
        if config.semantic_breakpoint_threshold is not None:
            cmd.extend(
                [
                    "--semantic-breakpoint-threshold",
                    str(config.semantic_breakpoint_threshold),
                ]
            )

    print(f"    Command: {' '.join(cmd)}")

    start_time = time.time()
    try:
        subprocess.run(cmd, check=True, capture_output=False)
        duration = time.time() - start_time
        return True, duration
    except subprocess.CalledProcessError as e:
        duration = time.time() - start_time
        print(f"    ERROR: Build failed with code {e.returncode}", file=sys.stderr)
        return False, duration
    except FileNotFoundError:
        print(
            "    ERROR: tensor-truth-build not found. Is the venv activated?",
            file=sys.stderr,
        )
        return False, 0


def build_category(
    category_name: str,
    modules: list[str],
    chunk_sizes: list[int],
    chunk_overlap: int,
    chunking_strategy: str,
    embedding_model: str,
    semantic_buffer_size: int | None = None,
    semantic_breakpoint_threshold: int | None = None,
    total_start_time: float = 0,
) -> tuple[int, int, float]:
    """Build all modules in a category one by one with progress tracking.

    Returns:
        Tuple of (successful_count, failed_count, total_duration)
    """
    print_banner(f"{category_name.upper()} ({len(modules)} modules)")
    print(f"Strategy: {chunking_strategy}")
    print(f"Chunk sizes: {chunk_sizes}")
    print(f"Chunk overlap: {chunk_overlap}")
    if semantic_buffer_size:
        print(f"Semantic buffer size: {semantic_buffer_size}")
    if semantic_breakpoint_threshold:
        print(f"Semantic breakpoint threshold: {semantic_breakpoint_threshold}")

    successful = 0
    failed = 0
    category_start = time.time()

    for i, module in enumerate(modules, 1):
        elapsed_total = time.time() - total_start_time
        elapsed_category = time.time() - category_start

        print_progress(i, len(modules), module, elapsed_total, elapsed_category)

        config = BuildConfig(
            name=module,
            modules=[module],
            chunk_sizes=chunk_sizes,
            chunk_overlap=chunk_overlap,
            chunking_strategy=chunking_strategy,
            semantic_buffer_size=semantic_buffer_size,
            semantic_breakpoint_threshold=semantic_breakpoint_threshold,
        )

        success, duration = run_build(config, embedding_model)

        if success:
            successful += 1
            print(f"    ✓ Completed in {format_duration(duration)}")
        else:
            failed += 1
            print(f"    ✗ Failed after {format_duration(duration)}")

    category_duration = time.time() - category_start
    print(f"\n{category_name} complete: {successful} succeeded, {failed} failed")
    print(f"Category duration: {format_duration(category_duration)}")

    return successful, failed, category_duration


def main():
    """Main entry point."""
    embedding_model = "Qwen/Qwen3-Embedding-0.6B"

    print_banner("TENSOR-TRUTH INDEX REBUILD", char="#")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Embedding model: {embedding_model}")

    # Load sources
    sources = load_sources_config()
    module_lists = get_module_lists(sources)

    print("\nModules to build:")
    print(f"  Libraries: {len(module_lists['libraries'])}")
    print(f"  Papers:    {len(module_lists['papers'])}")
    print(f"  Books:     {len(module_lists['books'])}")

    total_modules = sum(len(m) for m in module_lists.values())
    print(f"  TOTAL:     {total_modules}")

    # Rough time estimates based on typical build times
    # Libraries: ~2-5 min each (hierarchical is fast)
    # Papers: ~3-8 min each (semantic_hierarchical)
    # Books: ~5-15 min each (semantic, larger chunks)
    est_libraries = len(module_lists["libraries"]) * 3.5  # 3.5 min avg
    est_papers = len(module_lists["papers"]) * 5.5  # 5.5 min avg
    est_books = len(module_lists["books"]) * 10  # 10 min avg
    est_total = (est_libraries + est_papers + est_books) * 60  # in seconds

    print(f"\nEstimated total time: {format_duration(est_total)}")
    print(f"Estimated completion: {format_eta(est_total)}")

    total_start = time.time()
    results = {"successful": 0, "failed": 0}

    # === BUILD LIBRARIES ===
    if module_lists["libraries"]:
        s, f, _ = build_category(
            category_name="Libraries",
            modules=module_lists["libraries"],
            chunk_sizes=[2048, 512, 256],
            chunk_overlap=32,
            chunking_strategy="hierarchical",
            embedding_model=embedding_model,
            total_start_time=total_start,
        )
        results["successful"] += s
        results["failed"] += f

    # === BUILD PAPERS ===
    if module_lists["papers"]:
        s, f, _ = build_category(
            category_name="Papers",
            modules=module_lists["papers"],
            chunk_sizes=[2048, 512],
            chunk_overlap=64,
            chunking_strategy="semantic_hierarchical",
            embedding_model=embedding_model,
            semantic_buffer_size=2,
            semantic_breakpoint_threshold=92,
            total_start_time=total_start,
        )
        results["successful"] += s
        results["failed"] += f

    # === BUILD BOOKS ===
    if module_lists["books"]:
        s, f, _ = build_category(
            category_name="Books",
            modules=module_lists["books"],
            chunk_sizes=[4096, 1024, 512],
            chunk_overlap=96,
            chunking_strategy="semantic",
            embedding_model=embedding_model,
            semantic_buffer_size=3,
            semantic_breakpoint_threshold=95,
            total_start_time=total_start,
        )
        results["successful"] += s
        results["failed"] += f

    # === FINAL SUMMARY ===
    total_duration = time.time() - total_start

    print_banner("BUILD COMPLETE", char="#")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total duration: {format_duration(total_duration)}")
    print("\nResults:")
    print(f"  Successful: {results['successful']}")
    print(f"  Failed:     {results['failed']}")
    print(f"  Total:      {results['successful'] + results['failed']}")

    if results["failed"] > 0:
        print("\n⚠ Some builds failed. Check the output above for details.")
        return 1
    else:
        print("\n✓ All builds completed successfully!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
