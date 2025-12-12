import os
import shutil
import logging
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext
)
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

# Import config from our shared library
from rag_engine import (
    DB_PATH, 
    COLLECTION_NAME, 
    get_embed_model,
    load_inference_index
)

DOCS_DIR = "./library_docs/pytorch_2.5" # Point this to your specific version folder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BUILDER")

def build():
    print(f"--- STARTING BUILD: {DOCS_DIR} ---")
    
    # 1. Clean Slate (Optional: removes old DB to ensure freshness)
    if os.path.exists(DB_PATH):
        print("Removing old database...")
        shutil.rmtree(DB_PATH)
    
    # 2. Load Documents
    print("Loading documents (this may take a moment)...")
    documents = SimpleDirectoryReader(DOCS_DIR, recursive=True).load_data()
    print(f"Loaded {len(documents)} source files.")

    # 3. Advanced Parsing (Hierarchical)
    node_parser = HierarchicalNodeParser.from_defaults(
        chunk_sizes=[2048, 512, 128]
    )
    nodes = node_parser.get_nodes_from_documents(documents)
    leaf_nodes = get_leaf_nodes(nodes)
    print(f"Parsed {len(nodes)} total nodes ({len(leaf_nodes)} leaf nodes to embed).")

    # 4. Setup Database
    db = chromadb.PersistentClient(path=DB_PATH)
    chroma_collection = db.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    # 5. Create Storage Context
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    storage_context.docstore.add_documents(nodes) # Store ALL nodes (parents + children)

    # 6. Indexing (The VRAM Heavy Part)
    print("--- EMBEDDING & INDEXING (GPU) ---")
    VectorStoreIndex(
        leaf_nodes,
        storage_context=storage_context,
        embed_model=get_embed_model(), # Use our CUDA config
        show_progress=True
    )
    
    # 7. Persist to Disk
    print("--- SAVING TO DISK ---")
    storage_context.persist(persist_dir=DB_PATH)
    print("✅ Build Complete. You can now run 'streamlit run app.py'")

if __name__ == "__main__":
    # Stop Ollama to free up VRAM for building? 
    # Optional, but recommended if you hit OOM.
    build()