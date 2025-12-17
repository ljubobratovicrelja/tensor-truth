import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path

import chromadb
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.vector_stores.chroma import ChromaVectorStore

from tensortruth.app_utils.config_schema import TensorTruthConfig
from tensortruth.core.ollama import get_ollama_url
from tensortruth.rag_engine import get_base_index_dir, get_embed_model
from tensortruth.utils.metadata import extract_document_metadata

# Source directory is in the current working directory (where docs are placed)
SOURCE_DIR = "./library_docs"
# Indexes are built directly into the user data directory
BASE_INDEX_DIR = get_base_index_dir()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BUILDER")


def build_module(module_name, chunk_sizes=[2048, 512, 128], extract_metadata=True):

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

    # 2a. Extract Metadata (NEW)
    if extract_metadata:
        print("Extracting document metadata...")

        # Load sources.json config
        sources_config = None
        sources_path = Path("config/sources.json")
        if sources_path.exists():
            with open(sources_path, "r", encoding="utf-8") as f:
                sources_config = json.load(f)

        # Get Ollama URL for LLM fallback
        ollama_url = get_ollama_url()

        # Extract metadata for each document
        for i, doc in enumerate(documents):
            file_path = Path(doc.metadata.get("file_path", ""))

            try:
                metadata = extract_document_metadata(
                    doc=doc,
                    file_path=file_path,
                    module_name=module_name,
                    sources_config=sources_config,
                    ollama_url=ollama_url,
                    use_llm_fallback=True,
                )

                # Inject enriched metadata into document
                doc.metadata.update(metadata)

                if (i + 1) % 10 == 0 or (i + 1) == len(documents):
                    print(f"  Processed {i + 1}/{len(documents)} documents...")

            except Exception as e:
                logger.warning(f"Failed to extract metadata for {file_path.name}: {e}")
                # Continue with default metadata

        print(f"✓ Metadata extraction complete for {len(documents)} documents")

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

    device = TensorTruthConfig._detect_default_device()
    print(f"Embedding on {device.upper()}...")
    VectorStoreIndex(
        leaf_nodes,
        storage_context=storage_context,
        embed_model=get_embed_model(device=device),
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
    parser.add_argument(
        "--no-extract-metadata",
        action="store_true",
        help="Skip metadata extraction (faster but less informative citations)",
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

        build_module(
            module, args.chunk_sizes, extract_metadata=not args.no_extract_metadata
        )

        print()
        print("=" * 60)
        print(f"\n✅ Completed Module: {module} ")
        print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
