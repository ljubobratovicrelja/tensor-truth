import os
import logging
from llama_index.core import (
    StorageContext,
    load_index_from_storage,
    Settings
)
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

# --- GLOBAL CONFIG ---
DB_PATH = "./chroma_db"
COLLECTION_NAME = "deepseek_library_docs"

# 1. GPU CONFIGURATION
# We define these once here so both Build and App use the exact same settings
def get_llm():
    return Ollama(
        model="deepseek-r1:14b", 
        request_timeout=300.0,
        temperature=0.1,
        context_window=4096,  # Limits VRAM usage significantly
        additional_kwargs={"num_ctx": 4096} # Explicit Ollama arg just to be safe
    )

def get_embed_model():
    return HuggingFaceEmbedding(
        model_name="BAAI/bge-m3",
        device="cuda",
        model_kwargs={"trust_remote_code": True}, 
        embed_batch_size=16
    )

def get_reranker():
    return SentenceTransformerRerank(
        model="BAAI/bge-reranker-v2-m3", 
        top_n=3,
        device="cuda",
    )

# Apply Global Settings
Settings.llm = get_llm()
Settings.embedding_model = get_embed_model()

def load_inference_index():
    """
    STRICTLY LOADS the existing index. 
    Fails loudly if the database is missing.
    """
    if not os.path.exists(DB_PATH) or not os.path.exists(os.path.join(DB_PATH, "docstore.json")):
        raise FileNotFoundError(f"Database not found at {DB_PATH}. Please run src/build_db.py first.")

    print("--- MOUNTING VECTOR STORE ---")
    db = chromadb.PersistentClient(path=DB_PATH)
    chroma_collection = db.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    print("--- LOADING INDEX METADATA ---")
    # We explicitly pass embed_model here to prevent the 'OpenAI' crash
    storage_context = StorageContext.from_defaults(
        persist_dir=DB_PATH, 
        vector_store=vector_store
    )
    
    index = load_index_from_storage(
        storage_context, 
        embed_model=get_embed_model() # <--- THE CRITICAL FIX
    )
    return index

def get_query_engine(index):
    print("--- ASSEMBLING RAG PIPELINE ---")
    base_retriever = index.as_retriever(similarity_top_k=10)
    retriever = AutoMergingRetriever(base_retriever, index.storage_context, verbose=True)
    
    query_engine = RetrieverQueryEngine.from_args(
        retriever=retriever,
        node_postprocessors=[get_reranker()],
        llm=get_llm()
    )
    return query_engine