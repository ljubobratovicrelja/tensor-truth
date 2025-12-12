import os
import shutil
import logging
import argparse
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext
)
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

from rag_engine import get_embed_model

# --- CONFIGURATION: MODULE MAP ---
# Map a "Key" to a "Source Directory"
# You can add as many as you want here.
INDEX_REGISTRY = {
    "pytorch": "./library_docs/pytorch_2.9",
    "papers":  "./library_docs/papers",
    # "numpy": "./library_docs/numpy_1.26",
}

BASE_INDEX_DIR = "./indexes"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BUILDER")

def build_module(module_name):
    if module_name not in INDEX_REGISTRY:
        print(f"❌ Error: Module '{module_name}' not found in registry.")
        print(f"Available modules: {list(INDEX_REGISTRY.keys())}")
        return

    source_dir = INDEX_REGISTRY[module_name]
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
        source_dir,
        recursive=True,
        required_exts=[".md"]
    ).load_data()

    print(f"Loaded {len(documents)} documents.")

    # 3. Parse
    node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=[2048, 512, 128])
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
        show_progress=True
    )
    
    storage_context.persist(persist_dir=persist_dir)
    print(f"✅ Module '{module_name}' built successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("module", help="Name of the module to build (e.g., 'pytorch')")
    args = parser.parse_args()
    
    build_module(args.module)