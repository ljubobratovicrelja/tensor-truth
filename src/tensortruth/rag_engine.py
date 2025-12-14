import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

import chromadb
from llama_index.core import (
    QueryBundle,
    Settings,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.postprocessor import (
    SentenceTransformerRerank,
    SimilarityPostprocessor,
)
from llama_index.core.retrievers import AutoMergingRetriever, BaseRetriever
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore

# --- GLOBAL CONFIG ---
BASE_INDEX_DIR = "./indexes"

# --- CUSTOM PROMPTS ---
# This fixes "Context Blindness" by forcing the model to check History if RAG fails.
CUSTOM_CONTEXT_PROMPT_TEMPLATE = (
    "The following is a friendly conversation between a user and an AI assistant.\n"
    "The assistant is a coding expert and helpful assistant.\n\n"
    "Here are the relevant documents from the knowledge base:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n\n"
    "Chat History:\n"
    "{chat_history}\n\n"
    "Instructions:\n"
    "1. FIRST check if the question refers to something specific from the "
    "Chat History (e.g., code we discussed, an example I showed).\n"
    "2. If YES: prioritize the Chat History and only use documents if they "
    "add relevant detail to that specific context.\n"
    "3. If NO: use the documents above if they contain the answer.\n"
    "4. If documents are empty, generic, or don't match the specific context, "
    "ignore them and rely on Chat History.\n"
    "5. Never say 'I could not find relevant context' if the answer is in "
    "the Chat History.\n\n"
    "User: {query_str}\n"
    "Assistant:"
)

CUSTOM_CONDENSE_PROMPT_TEMPLATE = (
    "Given the following conversation between a user and an AI assistant "
    "and a follow up question from user, "
    "rephrase the follow up question to be a standalone question.\n\n"
    "IMPORTANT RULES:\n"
    "1. If the question uses pronouns (it, this, that) or refers to "
    "something previously discussed, "
    "include enough context to make it clear what is being referenced.\n"
    "2. If the question is asking to elaborate on a specific example or "
    "code shown earlier, "
    "mention that specific context (e.g., 'in the example we discussed' or "
    "'in the code I just showed').\n"
    "3. Preserve conversational references - don't make the question "
    "overly general.\n"
    "4. When the question clearly follows from the conversation, maintain "
    "that connection.\n\n"
    "Chat History:\n{chat_history}\n\n"
    "Follow Up Input: {question}\n\n"
    "Standalone question:"
)


def get_embed_model(device="cuda"):
    print(f"Loading Embedder on: {device.upper()}")
    return HuggingFaceEmbedding(
        model_name="BAAI/bge-m3",
        device=device,
        model_kwargs={"trust_remote_code": True},
        embed_batch_size=16,
    )


def get_llm(params):
    model_name = params.get("model", "deepseek-r1:14b")
    user_system_prompt = params.get("system_prompt", "").strip()
    device_mode = params.get("llm_device", "gpu")  # 'gpu' or 'cpu'

    # Ollama specific options
    ollama_options = {"num_predict": -1}  # Prevent truncation

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
            "options": ollama_options,
        },
        system_prompt=user_system_prompt,
    )


def get_reranker(params, device="cuda"):
    # Default to the high-precision BGE-M3 v2 if not specified
    model = params.get("reranker_model", "BAAI/bge-reranker-v2-m3")
    top_n = params.get("reranker_top_n", 3)

    print(f"Loading Reranker on: {device.upper()}")
    return SentenceTransformerRerank(model=model, top_n=top_n, device=device)


class MultiIndexRetriever(BaseRetriever):
    def __init__(self, retrievers, max_workers=None, enable_cache=True, cache_size=128):
        self.retrievers = retrievers
        self.max_workers = max_workers or min(len(retrievers), 8)
        self.enable_cache = enable_cache
        super().__init__()

        # Create LRU cache for retrieve operations if enabled
        if self.enable_cache:
            self._retrieve_cached = lru_cache(maxsize=cache_size)(self._retrieve_impl)
        else:
            self._retrieve_cached = self._retrieve_impl

    def _retrieve_impl(self, query_text: str):
        """Actual retrieval implementation that can be cached."""
        # Recreate QueryBundle from cached query text
        query_bundle = QueryBundle(query_str=query_text)
        combined_nodes = []

        # Parallelize retrieval across all indices
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_retriever = {
                executor.submit(r.retrieve, query_bundle): r for r in self.retrievers
            }

            for future in as_completed(future_to_retriever):
                try:
                    nodes = future.result()
                    combined_nodes.extend(nodes)
                except Exception as e:
                    # Log error but continue with other retrievers
                    print(f"Retriever failed: {e}")

        return combined_nodes

    def _retrieve(self, query_bundle: QueryBundle):
        """Public retrieve method that leverages caching."""
        return self._retrieve_cached(query_bundle.query_str)


def load_engine_for_modules(selected_modules, engine_params=None):
    if not selected_modules:
        raise ValueError("No modules selected!")

    if engine_params is None:
        engine_params = {}

    similarity_cutoff = engine_params.get("confidence_cutoff", 0.0)

    # Determine devices
    rag_device = engine_params.get("rag_device", "cuda")

    # Calculate adaptive similarity_top_k based on reranker_top_n
    # Retrieve 2-3x more candidates than final target to ensure quality
    reranker_top_n = engine_params.get("reranker_top_n", 3)
    similarity_top_k = max(5, reranker_top_n * 2)

    # Set Global Settings for this session (Embedder)
    embed_model = get_embed_model(rag_device)
    Settings.embedding_model = embed_model

    active_retrievers = []
    print(
        f"--- MOUNTING: {selected_modules} | MODEL: {engine_params.get('model')} | "
        f"RAG DEVICE: {rag_device} | RETRIEVAL: {similarity_top_k} per index → "
        f"RERANK: top {reranker_top_n} ---"
    )

    for module in selected_modules:
        path = os.path.join(BASE_INDEX_DIR, module)
        if not os.path.exists(path):
            continue

        db = chromadb.PersistentClient(path=path)
        collection = db.get_or_create_collection("data")
        vector_store = ChromaVectorStore(chroma_collection=collection)

        storage_context = StorageContext.from_defaults(
            persist_dir=path, vector_store=vector_store
        )

        # Explicitly pass the embed_model to ensure consistency
        index = load_index_from_storage(storage_context, embed_model=embed_model)

        base = index.as_retriever(similarity_top_k=similarity_top_k)
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
        node_postprocessors.append(
            SimilarityPostprocessor(similarity_cutoff=similarity_cutoff)
        )

    chat_engine = CondensePlusContextChatEngine.from_defaults(
        retriever=composite_retriever,
        node_postprocessors=node_postprocessors,
        llm=llm,
        memory=memory,
        context_prompt=CUSTOM_CONTEXT_PROMPT_TEMPLATE,
        condense_prompt=CUSTOM_CONDENSE_PROMPT_TEMPLATE,
        verbose=False,
    )

    return chat_engine
