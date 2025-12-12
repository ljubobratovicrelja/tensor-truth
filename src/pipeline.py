import os
import logging
import sys

# LlamaIndex Core
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings,
    load_index_from_storage,
    Document
)
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SentenceTransformerRerank

# Integrations
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

# Logging setup
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
DOCS_DIR = "./library_docs/pytorch_2.5"  # Put your .md/.txt files here
DB_PATH = "./chroma_db"      # Persistence path
COLLECTION_NAME = "deepseek_library_docs"

# 1. SETUP LLM & EMBEDDINGS
# Using bge-m3 for robust dense retrieval on code/technical terms
embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-m3",
    device="cuda", # Change to "cpu" if Ollama hogs all VRAM
    embed_batch_size=16
)

# DeepSeek-r1 via Ollama
llm = Ollama(
    model="deepseek-r1:32b", 
    request_timeout=300.0,
    temperature=0.1 # Low temp for factual API answers
)

# Apply Global Settings
Settings.llm = llm
Settings.embedding_model = embed_model

def get_vector_store():
    """Initialize persistent ChromaDB."""
    db = chromadb.PersistentClient(path=DB_PATH)
    chroma_collection = db.get_or_create_collection(COLLECTION_NAME)
    return ChromaVectorStore(chroma_collection=chroma_collection)

def build_or_load_index():
    """
    Builds the index using Hierarchical Node Parsing or loads it if it exists.
    Crucial: Must persist the DocStore to support AutoMergingRetriever.
    """
    vector_store = get_vector_store()
    
    # Check if we already have a storage context on disk
    if os.path.exists(DB_PATH) and os.listdir(DB_PATH):
        try:
            logger.info("Loading existing index from storage...")
            # We load the docstore/index_store from disk, but attach the Chroma vector store
            storage_context = StorageContext.from_defaults(
                persist_dir=DB_PATH, 
                vector_store=vector_store,
            )
            index = load_index_from_storage(
                storage_context,
                embed_model=embed_model
            )
            return index
        except Exception as e:
            logger.warning(f"Could not load index (might be partial or empty): {e}")

    logger.info("Creating new index from documents...")
    
    # 1. Load Documents
    if not os.path.exists(DOCS_DIR):
        os.makedirs(DOCS_DIR)
        logger.error(f"Please place your documentation files in {DOCS_DIR}")
        return None

    documents = SimpleDirectoryReader(DOCS_DIR, recursive=True).load_data()
    logger.info(f"Loaded {len(documents)} documents.")

    # 2. Hierarchical Node Parsing
    # Parent chunk: 1024 (Good for class context)
    # Child chunk: 256 (Good for specific function lookup)
    node_parser = HierarchicalNodeParser.from_defaults(
        chunk_sizes=[2048, 512, 128] 
    )
    
    nodes = node_parser.get_nodes_from_documents(documents)
    leaf_nodes = get_leaf_nodes(nodes) # We only index the leaves (smallest chunks)
    logger.info(f"Parsed into {len(nodes)} hierarchical nodes ({len(leaf_nodes)} leaf nodes).")

    # 3. Create Storage Context
    # We must add 'nodes' to the docstore explicitly so the parent can be retrieved later
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    storage_context.docstore.add_documents(nodes)

    # 4. Build Index (Indexing only leaf nodes!)
    index = VectorStoreIndex(
        leaf_nodes,
        storage_context=storage_context,
        embed_model=embed_model,
    )

    # 5. Persist the DocStore (Chroma saves itself, but we need the docstore mappings)
    storage_context.persist(persist_dir=DB_PATH)
    
    return index

def get_query_engine(index):
    """
    Constructs an advanced Query Engine with:
    1. AutoMergingRetriever (Context expansion)
    2. BGE-Reranker (Precision filtering)
    """
    
    # 1. Base Retriever
    base_retriever = index.as_retriever(similarity_top_k=10)
    
    # 2. Auto Merging Retriever 
    # If >50% of a parent's children are retrieved, swap them for the parent.
    retriever = AutoMergingRetriever(
        base_retriever, 
        index.storage_context, 
        verbose=True
    )

    # 3. Reranker
    # Using a cross-encoder to re-score the top 10 results -> top 3
    # BAAI/bge-reranker-v2-m3 is excellent for code
    reranker = SentenceTransformerRerank(
        model="BAAI/bge-reranker-v2-m3", 
        top_n=3,
        device="cuda" # or cpu
    )

    # 4. Construct Engine
    query_engine = RetrieverQueryEngine.from_args(
        retriever=retriever,
        node_postprocessors=[reranker],
        llm=llm
    )
    
    return query_engine

def main():
    # 1. Initialize Index
    index = build_or_load_index()
    if not index:
        print("Index initialization failed. Exiting.")
        return

    # 2. Setup Engine
    query_engine = get_query_engine(index)

    # 3. Interactive Loop
    print("\n--- RAG Pipeline Ready (Type 'exit' to quit) ---")
    while True:
        question = input("\nQuery: ")
        if question.lower() in ["exit", "quit"]:
            break
            
        try:
            print("Retrieving and Reasoning...")
            response = query_engine.query(question)
            
            # Print Response
            print("\n--- DeepSeek Response ---")
            print(response)
            
            # Debug: Show which nodes were actually used
            print("\n--- Source Nodes (Verification) ---")
            for node in response.source_nodes:
                print(f"Score: {node.score:.3f} | File: {node.metadata.get('file_name')}")
                # print(f"Content Preview: {node.node.get_content()[:200]}...")
                
        except Exception as e:
            logger.error(f"Error during query: {e}")

if __name__ == "__main__":
    main()