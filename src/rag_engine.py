import os
from llama_index.core import (
    StorageContext,
    load_index_from_storage,
    Settings,
    QueryBundle
)
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.retrievers import AutoMergingRetriever, BaseRetriever
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

# --- GLOBAL CONFIG ---
BASE_INDEX_DIR = "./indexes"

def get_embed_model():
    return HuggingFaceEmbedding(
        model_name="BAAI/bge-m3",
        device="cuda",
        model_kwargs={"trust_remote_code": True},
        embed_batch_size=16
    )

# CHANGED: Now accepts dynamic params
def get_llm(params):
    return Ollama(
        model="deepseek-r1:14b", 
        request_timeout=300.0,
        temperature=params.get("temperature", 0.1),
        context_window=params.get("context_window", 4096),
        additional_kwargs={"num_ctx": params.get("context_window", 4096)}
    )

def get_reranker():
    return SentenceTransformerRerank(
        model="BAAI/bge-reranker-v2-m3", 
        top_n=3,
        device="cuda"
    )

Settings.embedding_model = get_embed_model()
# Note: Settings.llm will be overridden per-session now

# --- CUSTOM COMPOSITE RETRIEVER ---
class MultiIndexRetriever(BaseRetriever):
    def __init__(self, retrievers):
        self.retrievers = retrievers
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle):
        combined_nodes = []
        for r in self.retrievers:
            nodes = r.retrieve(query_bundle)
            combined_nodes.extend(nodes)
        return combined_nodes

# CHANGED: Accepts engine_params
def load_engine_for_modules(selected_modules, engine_params=None):
    if not selected_modules:
        raise ValueError("No modules selected!")
    
    # Default params if none provided
    if engine_params is None:
        engine_params = {"temperature": 0.1, "context_window": 4096}

    active_retrievers = []
    print(f"--- MOUNTING MODULES: {selected_modules} WITH PARAMS: {engine_params} ---")
    
    for module in selected_modules:
        path = os.path.join(BASE_INDEX_DIR, module)
        if not os.path.exists(path):
            continue
            
        db = chromadb.PersistentClient(path=path)
        collection = db.get_or_create_collection("data")
        vector_store = ChromaVectorStore(chroma_collection=collection)
        
        storage_context = StorageContext.from_defaults(persist_dir=path, vector_store=vector_store)
        index = load_index_from_storage(storage_context, embed_model=get_embed_model())
        
        base = index.as_retriever(similarity_top_k=10)
        am_retriever = AutoMergingRetriever(base, index.storage_context, verbose=False)
        active_retrievers.append(am_retriever)

    if not active_retrievers:
        raise FileNotFoundError("No valid indices loaded.")

    composite_retriever = MultiIndexRetriever(active_retrievers)
    
    memory = ChatMemoryBuffer.from_defaults(token_limit=3000)

    # Initialize LLM with custom params
    llm = get_llm(engine_params)

    chat_engine = CondensePlusContextChatEngine.from_defaults(
        retriever=composite_retriever,
        node_postprocessors=[get_reranker()],
        llm=llm,
        memory=memory,
        condense_prompt=(
            "Given the following conversation between a user and an AI assistant and a follow up question from user, "
            "rephrase the follow up question to be a standalone question.\n\n"
            "Chat History:\n{chat_history}\n\n"
            "Follow Up Input: {question}\n\n"
            "Standalone question:"
        ),
        verbose=True
    )
    
    return chat_engine