import argparse
import logging
import os
import shutil
import sys

import chromadb
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.vector_stores.chroma import ChromaVectorStore

from tensortruth.rag_engine import get_embed_model

SOURCE_DIR = "./library_docs"
BASE_INDEX_DIR = "./indexes"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BUILDER")


def build_module(module_name, chunk_sizes=[2048, 512, 128]):

    source_dir = os.path.join(SOURCE_DIR, module_name)
    persist_dir = os.path.join(BASE_INDEX_DIR, module_name)

    print(f"\n--- BUILDING MODULE: {module_name} ---")
    print(f"Source: {source_dir}")
    print(f"Target: {persist_dir}")

    # 1. Clean Slate for THIS module only
    if os.path.exists(persist_dir):
        print(f"Removing old index at {persist_dir}...")
        shutil.rmtree(persist_dir)

    # 2. Load Documents
    if not os.path.exists(source_dir):
        print(f"❌ Source directory missing: {source_dir}")
        return

    documents = SimpleDirectoryReader(
        source_dir, recursive=True, required_exts=[".md", ".html"]
    ).load_data()

    print(f"Loaded {len(documents)} documents.")

    # 3. Parse
    node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=chunk_sizes)
    nodes = node_parser.get_nodes_from_documents(documents)
    leaf_nodes = get_leaf_nodes(nodes)
    print(f"Parsed {len(nodes)} nodes ({len(leaf_nodes)} leaves).")

    # 4. Create Isolated DB
    # We use a unique collection name, though it's less critical since folders are separate
    db = chromadb.PersistentClient(path=persist_dir)
    collection = db.get_or_create_collection("data")
    vector_store = ChromaVectorStore(chroma_collection=collection)

    # 5. Index & Persist
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    storage_context.docstore.add_documents(nodes)

    print("Embedding (GPU)...")
    VectorStoreIndex(
        leaf_nodes,
        storage_context=storage_context,
        embed_model=get_embed_model(),
        show_progress=True,
    )

    storage_context.persist(persist_dir=persist_dir)
    print(f"✅ Module '{module_name}' built successfully!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--modules",
        nargs="+",
        help="Module names to build (subfolders in library_docs)",
    )
    parser.add_argument(
        "--all", action="store_true", help="Build all modules found in library_docs"
    )
    parser.add_argument(
        "--chunk-sizes",
        nargs="+",
        type=int,
        default=[2048, 512, 128],
        help="Chunk sizes for hierarchical parsing",
    )

    args = parser.parse_args()

    if args.all:
        # Check if modules were also specified
        if args.modules:
            print("❌ Cannot use --all and --modules together.")
            return 1

        args.modules = [
            name
            for name in os.listdir(SOURCE_DIR)
            if os.path.isdir(os.path.join(SOURCE_DIR, name))
        ]

    print()
    print(f"\nModules to build: {args.modules}")
    print()

    for module in args.modules:

        print()
        print("=" * 60)
        print(f" Building Module: {module} ")
        print("=" * 60)
        print()

        build_module(module, args.chunk_sizes)

        print()
        print("=" * 60)
        print(f"\n✅ Completed Module: {module} ")
        print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
