#!/usr/bin/env python
"""Backfill FTS5 lexical indexes for vector indexes built before fusion.

Reindexing every module just to gain lexical retrieval would cost hours of
embedding on the live library, and the text is already sitting in each
``docstore.json``. This walks the docstore leaves and writes ``lexical.sqlite3``
next to ``chroma.sqlite3``, which is what ``rag_engine._open_index`` looks for.

Usage:
    python scripts/build_lexical_index.py --all
    python scripts/build_lexical_index.py --module book_neural_networks_nielsen
    python scripts/build_lexical_index.py --all --indexes-dir /tmp/copy
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from llama_index.core.node_parser import get_leaf_nodes
from llama_index.core.storage.docstore import SimpleDocumentStore

from tensortruth.app_utils.paths import get_indexes_dir_for_model
from tensortruth.indexing.lexical import LexicalIndex

DOCSTORE_FILENAME = "docstore.json"


def load_docstore_nodes(index_dir: Path) -> list:
    """Read every node out of a persisted docstore."""
    docstore = SimpleDocumentStore.from_persist_path(str(index_dir / DOCSTORE_FILENAME))
    return list(docstore.docs.values())


def backfill(module_dir: Path, embedding_model: str) -> int:
    """Build the lexical index for one module directory; returns the chunk count."""
    nodes = load_docstore_nodes(module_dir)
    # get_leaf_nodes rather than every node: parents repeat child text, and
    # duplicate text would double-count the same passage in bm25.
    leaves = get_leaf_nodes(nodes)
    index = LexicalIndex(module_dir)
    index.build_from_store(module_dir, leaves, embedding_model=embedding_model)
    return index.count()


def resolve_module_dirs(model_dir: Path, modules: list[str] | None) -> list[Path]:
    if modules:
        return [model_dir / name for name in modules]
    return sorted(
        p
        for p in model_dir.iterdir()
        if p.is_dir() and (p / DOCSTORE_FILENAME).is_file()
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build FTS5 lexical indexes from persisted docstores."
    )
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument(
        "--module",
        action="append",
        metavar="NAME",
        help="Module/index directory name; repeatable.",
    )
    scope.add_argument(
        "--all",
        action="store_true",
        help="Backfill every index directory that has a docstore.json.",
    )
    parser.add_argument(
        "--indexes-dir",
        default=None,
        help="Base indexes directory (default: the app's indexes dir).",
    )
    parser.add_argument(
        "--embedding-model",
        default="bge-m3",
        help="Embedding model whose index subdirectory to use (default: bge-m3).",
    )
    args = parser.parse_args(argv)

    # One line per module is the contract; library INFO logs would bury it.
    logging.getLogger("tensortruth").setLevel(logging.WARNING)

    model_dir = Path(get_indexes_dir_for_model(args.embedding_model, args.indexes_dir))
    if not model_dir.is_dir():
        print(f"error: no such indexes directory: {model_dir}", file=sys.stderr)
        return 1

    module_dirs = resolve_module_dirs(model_dir, args.module)
    if not module_dirs:
        print(f"error: nothing to backfill under {model_dir}", file=sys.stderr)
        return 1

    failures = 0
    for module_dir in module_dirs:
        if not (module_dir / DOCSTORE_FILENAME).is_file():
            print(f"{module_dir.name}: skipped (no {DOCSTORE_FILENAME})")
            failures += 1
            continue
        try:
            count = backfill(module_dir, args.embedding_model)
        except Exception as exc:
            print(f"{module_dir.name}: failed ({exc})", file=sys.stderr)
            failures += 1
            continue
        print(f"{module_dir.name}: {count} chunks")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
