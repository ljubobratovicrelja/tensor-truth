import os
from llama_index.core import (
    StorageContext,
    load_index_from_storage,
    Settings,
    QueryBundle,
)

from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.retrievers import AutoMergingRetriever, BaseRetriever
from llama_index.core.postprocessor import SentenceTransformerRerank, SimilarityPostprocessor
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

# --- GLOBAL CONFIG ---
BASE_INDEX_DIR = "./indexes"

def get_embed_model(device="cuda"):
    print(f"Loading Embedder on: {device.upper()}")
    return HuggingFaceEmbedding(
        model_name="BAAI/bge-m3",
        device=device,
        model_kwargs={"trust_remote_code": True},
        embed_batch_size=16
    )

def get_llm(params):
    model_name = params.get("model", "deepseek-r1:14b")
    user_system_prompt = params.get("system_prompt", "").strip()
    device_mode = params.get("llm_device", "gpu") # 'gpu' or 'cpu'
    
    # Ollama specific options
    ollama_options = {"num_predict": -1} # Prevent truncation
    
    # Force CPU if requested
    if device_mode == "cpu":
        print(f"Loading LLM {model_name} on: CPU (Forced)")
        ollama_options["num_gpu"] = 0
    
    return Ollama(
        model=model_name, 
        request_timeout=300.0,
        temperature=params.get("temperature", 0.3),
        context_window=params.get("context_window", 4096),
        additional_kwargs={
            "num_ctx": params.get("context_window", 4096),
            "options": ollama_options
        },
        system_prompt=user_system_prompt
    )

def get_reranker(params, device="cuda"):
    # Default to the high-precision BGE-M3 v2 if not specified
    model = params.get("reranker_model", "BAAI/bge-reranker-v2-m3")
    top_n = params.get("reranker_top_n", 3)
    
    print(f"Loading Reranker on: {device.upper()}")
    return SentenceTransformerRerank(
        model=model, 
        top_n=top_n,
        device=device
    )

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

def load_engine_for_modules(selected_modules, engine_params=None):
    if not selected_modules:
        raise ValueError("No modules selected!")
    
    if engine_params is None: engine_params = {}

    similarity_cutoff = engine_params.get("confidence_cutoff", 0.0)
    
    # Determine devices
    rag_device = engine_params.get("rag_device", "cuda")
    
    # Set Global Settings for this session (Embedder)
    embed_model = get_embed_model(rag_device)
    Settings.embedding_model = embed_model

    active_retrievers = []
    print(f"--- MOUNTING: {selected_modules} | MODEL: {engine_params.get('model')} | RAG DEVICE: {rag_device} ---")
    
    for module in selected_modules:
        path = os.path.join(BASE_INDEX_DIR, module)
        if not os.path.exists(path): continue
            
        db = chromadb.PersistentClient(path=path)
        collection = db.get_or_create_collection("data")
        vector_store = ChromaVectorStore(chroma_collection=collection)
        
        storage_context = StorageContext.from_defaults(persist_dir=path, vector_store=vector_store)
        
        # Explicitly pass the embed_model to ensure consistency
        index = load_index_from_storage(storage_context, embed_model=embed_model)
        
        base = index.as_retriever(similarity_top_k=10)
        am_retriever = AutoMergingRetriever(base, index.storage_context, verbose=False)
        active_retrievers.append(am_retriever)

    if not active_retrievers:
        raise FileNotFoundError("No valid indices loaded.")

    composite_retriever = MultiIndexRetriever(active_retrievers)
    
    memory = ChatMemoryBuffer.from_defaults(token_limit=3000)
    llm = get_llm(engine_params)

    # Pass device to reranker
    node_postprocessors = [get_reranker(engine_params, device=rag_device)]

    if similarity_cutoff > 0:
        node_postprocessors.append(SimilarityPostprocessor(similarity_cutoff=similarity_cutoff))

    chat_engine = CondensePlusContextChatEngine.from_defaults(
        retriever=composite_retriever,
        node_postprocessors=node_postprocessors,
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