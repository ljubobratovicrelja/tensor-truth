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
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.retrievers import AutoMergingRetriever, BaseRetriever
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore

# --- GLOBAL CONFIG ---
_BASE_INDEX_DIR_CACHE = None


def get_base_index_dir():
    """
    Get the base index directory, preferring user data dir if available.

    This uses lazy loading to avoid circular import issues.
    """
    global _BASE_INDEX_DIR_CACHE
    if _BASE_INDEX_DIR_CACHE is None:
        try:
            from tensortruth.app_utils.paths import get_indexes_dir

            _BASE_INDEX_DIR_CACHE = get_indexes_dir()
        except (ImportError, AttributeError):
            # Fallback for standalone usage or during circular imports
            _BASE_INDEX_DIR_CACHE = "./indexes"
    return _BASE_INDEX_DIR_CACHE


# For backwards compatibility, provide BASE_INDEX_DIR as a constant
# Note: This will be "./indexes" at import time, but
# get_base_index_dir() will return the correct path.
BASE_INDEX_DIR = "./indexes"


# --- CUSTOM PROMPTS ---
CUSTOM_CONTEXT_PROMPT_TEMPLATE = (
    "Role: Technical Research & Development Assistant.\n"
    "Objective: Provide direct, factual answers based strictly on the provided context "
    "and chat history. Eliminate conversational filler.\n\n"
    "--- CONTEXT START ---\n"
    "{context_str}\n"
    "--- CONTEXT END ---\n\n"
    "--- HISTORY START ---\n"
    "{chat_history}\n"
    "--- HISTORY END ---\n\n"
    "OPERATIONAL RULES:\n"
    "1. MODE SELECTION:\n"
    "   - IF CODING: Output strictly the code or diffs. Do not re-print unchanged code. "
    "Use standard technical terminology. No 'happy to help' intros.\n"
    "   - IF RESEARCH: Synthesize facts from the Context. Cite specific sources if available. "
    "Resolve conflicts between sources by noting the discrepancy.\n"
    "2. HISTORY INTEGRATION: Do not repeat information already established in the History. "
    "Reference it directly (e.g., 'As shown in the previous ResNet block...').\n"
    "3. PRECISION: If the Context is insufficient, state exactly what is missing. "
    "Do not halluciation or fill gaps with generic fluff.\n"
    "4. FORMATTING: Use Markdown headers for structure. Use LaTeX for math.\n\n"
    "User Query: {query_str}\n"
    "Response:"
)

# Prompt used when confidence is low but sources are still provided
CUSTOM_CONTEXT_PROMPT_LOW_CONFIDENCE = (
    "Role: Technical Research & Development Assistant.\n"
    "Status: LOW CONFIDENCE MATCH - DATA INTEGRITY WARNING.\n\n"
    "--- RETRIEVED CONTEXT (LOW RELEVANCE) ---\n"
    "{context_str}\n"
    "--- END CONTEXT ---\n\n"
    "--- HISTORY ---\n"
    "{chat_history}\n"
    "--- END HISTORY ---\n\n"
    "OPERATIONAL CONSTRAINTS:\n"
    "1. INTEGRITY CHECK: The retrieved context has low similarity scores. "
    "It may be irrelevant.\n"
    "2. MANDATORY PREFACE: You must start the response with: "
    "'[NOTICE: Low confidence in retrieved sources. Response may rely on general knowledge.]'\n"
    "3. PRIORITIZATION: If the Chat History contains the answer, ignore the "
    "retrieved context entirely.\n"
    "4. NO HALLUCINATION: If neither History nor Context supports a factual answer, "
    "state 'Insufficient data available' and stop.\n\n"
    "User Query: {query_str}\n"
    "Response:"
)

# Prompt used when confidence cutoff filters all sources - includes warning acknowledgment
CUSTOM_CONTEXT_PROMPT_NO_SOURCES = (
    "Role: Technical Research & Development Assistant.\n"
    "Status: NO RETRIEVED DOCUMENTS.\n\n"
    "--- HISTORY ---\n"
    "{chat_history}\n"
    "--- END HISTORY ---\n\n"
    "INSTRUCTIONS:\n"
    "1. SYSTEM ALERT: The knowledge base returned zero matches. "
    "You are now operating on GENERAL MODEL KNOWLEDGE only.\n"
    "2. MANDATORY FORMATTING: Start your response with one of the following labels:\n"
    "   - 'NO INDEXED DATA FOUND. General knowledge fallback:'\n"
    "   - 'OUT OF SCOPE. Using general training data:'\n"
    "3. SCOPE: If the query is strictly about the internal database (e.g., 'What is in file X?'), "
    "state 'No data found' and terminate.\n"
    "4. CONTINUITY: If the answer is in the Chat History, output it without the "
    "no-data warning.\n\n"
    "User Query: {query_str}\n"
    "Response:"
)

# Context string injected when confidence cutoff filters all nodes
NO_CONTEXT_FALLBACK_CONTEXT = (
    "[SYSTEM FLAG: NULL_CONTEXT. No documents met the confidence threshold. "
    "Proceed with caution using internal knowledge only.]"
)

CUSTOM_CONDENSE_PROMPT_TEMPLATE = (
    "Role: Technical Query Engineer.\n"
    "Task: Convert the user's follow-up input into a precise, standalone technical directive "
    "or search query based on the chat history.\n\n"
    "Chat History:\n{chat_history}\n\n"
    "User Input: {question}\n\n"
    "TRANSFORMATION RULES:\n"
    "1. PRESERVE ENTITIES: Keep all variable names, file paths, error codes, "
    "and library names exactly as they appear.\n"
    "2. RESOLVE REFERENCES: Replace 'it', 'this', 'that code' with the specific "
    "object/concept from history "
    "(e.g., replace 'fix it' with 'Debug the BasicBlock class implementation').\n"
    "3. MAINTAIN IMPERATIVE: If the user gives a command (e.g., 'refactor'), "
    "keep the output as a command, "
    "do not turn it into a question (e.g., 'How do I refactor?').\n"
    "4. NO FLUFF: Output ONLY the standalone query. Do not add 'The user wants to know...' "
    "or polite padding.\n\n"
    "Standalone Query:"
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


def load_engine_for_modules(
    selected_modules, engine_params=None, preserved_chat_history=None
):
    if not selected_modules:
        raise ValueError("No modules selected!")

    if engine_params is None:
        engine_params = {}

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
        path = os.path.join(get_base_index_dir(), module)
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

    # Restore chat history from previous engine if provided
    if preserved_chat_history:
        for msg in preserved_chat_history:
            # Restore each message to the memory buffer
            memory.put(msg)

    llm = get_llm(engine_params)

    # Pass device to reranker
    # Note: similarity_cutoff is no longer used as a hard filter here
    # Instead, it's used in the app layer to show soft warnings when confidence is low
    node_postprocessors = [get_reranker(engine_params, device=rag_device)]

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
