import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from typing import Any, Dict, List, Optional

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

from tensortruth.core.ollama import check_thinking_support, get_ollama_url

# --- GLOBAL CONFIG ---
_BASE_INDEX_DIR_CACHE = None


def get_base_index_dir() -> str:
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

# Common code execution instructions (appended to all prompts)
_CODE_EXECUTION_RULES = (
    "CODE EXECUTION:\n"
    "   - Python code blocks will be AUTOMATICALLY EXECUTED in an isolated container.\n"
    '   - Include `if __name__ == "__main__":` blocks for executable examples.\n'
    "   - Make code print informative output - the output will be shown directly to the user.\n"
    "   - DO NOT predict or describe the output - the actual execution result will be displayed.\n"
    "   - Write self-contained, runnable code that demonstrates the concept.\n"
    "   - For matplotlib plots: Save figures with plt.savefig('plot.png') - images will be displayed automatically.\n"
    "   - You can generate multiple outputs: plot1.png, plot2.png, data.csv, etc.\n"
)

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
    "4. FORMATTING: Use Markdown headers for structure. Use LaTeX for math.\n"
    f"5. {_CODE_EXECUTION_RULES}\n"
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
    "state 'Insufficient data available' and stop.\n"
    f"5. {_CODE_EXECUTION_RULES}\n"
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
    "no-data warning.\n"
    f"5. {_CODE_EXECUTION_RULES}\n"
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


def get_embed_model(device: str = "cuda") -> HuggingFaceEmbedding:
    """Load HuggingFace embedding model.

    Args:
        device: Device to load model on ('cuda' or 'cpu')

    Returns:
        HuggingFaceEmbedding instance
    """
    print(f"Loading Embedder on: {device.upper()}")
    batch_size = 128 if device == "cuda" else 16
    return HuggingFaceEmbedding(
        model_name="BAAI/bge-m3",
        device=device,
        model_kwargs={"trust_remote_code": True},
        embed_batch_size=batch_size,
    )


def get_llm(params: Dict[str, Any]) -> Ollama:
    """Initialize Ollama LLM with configuration parameters.

    Args:
        params: Dictionary with model configuration

    Returns:
        Ollama LLM instance
    """
    model_name = params.get("model", "deepseek-r1:14b")
    user_system_prompt = params.get("system_prompt", "").strip()
    device_mode = params.get("llm_device", "gpu")  # 'gpu' or 'cpu'

    # Ollama specific options
    ollama_options = {}

    # Force CPU if requested
    if device_mode == "cpu":
        print(f"Loading LLM {model_name} on: CPU (Forced)")
        ollama_options["num_gpu"] = 0

    # Check if model supports thinking by querying Ollama API
    thinking_enabled = check_thinking_support(model_name)

    # For thinking models, limit total tokens to prevent runaway reasoning
    # For non-thinking models, use unlimited (-1) to prevent truncation
    if thinking_enabled:
        # Limit thinking models to ~4K tokens total (thinking + response)
        # This prevents endless loops while allowing reasonable reasoning
        ollama_options["num_predict"] = params.get("max_tokens", 4096)
    else:
        # Non-thinking models get unlimited to prevent truncation
        ollama_options["num_predict"] = -1

    return Ollama(
        model=model_name,
        base_url=get_ollama_url(),
        request_timeout=300.0,
        temperature=params.get("temperature", 0.3),
        context_window=params.get("context_window", 16384),
        thinking=thinking_enabled,
        additional_kwargs={
            "num_ctx": params.get("context_window", 16384),
            "options": ollama_options,
        },
        system_prompt=user_system_prompt,
    )


def get_reranker(
    params: Dict[str, Any], device: str = "cuda"
) -> SentenceTransformerRerank:
    """Initialize cross-encoder reranker model.

    Args:
        params: Dictionary with reranker configuration
        device: Device to load model on ('cuda' or 'cpu')

    Returns:
        SentenceTransformerRerank instance
    """
    # Default to the high-precision BGE-M3 v2 if not specified
    model = params.get("reranker_model", "BAAI/bge-reranker-v2-m3")
    top_n = params.get("reranker_top_n", 3)

    print(f"Loading Reranker on: {device.upper()}")
    return SentenceTransformerRerank(model=model, top_n=top_n, device=device)


class MultiIndexRetriever(BaseRetriever):
    """Retriever that queries multiple vector indexes in parallel.

    Combines results from multiple index retrievers using concurrent execution.
    """

    def __init__(
        self,
        retrievers: List[BaseRetriever],
        max_workers: Optional[int] = None,
        enable_cache: bool = True,
        cache_size: int = 128,
    ) -> None:
        """Initialize multi-index retriever.

        Args:
            retrievers: List of retriever instances
            max_workers: Maximum parallel workers (default: min(len(retrievers), 8))
            enable_cache: Whether to cache retrieval results
            cache_size: LRU cache size
        """
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
        """Actual retrieval implementation that can be cached.

        Args:
            query_text: Query string

        Returns:
            List of retrieved nodes from all indexes
        """
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
        """Public retrieve method that leverages caching.

        Args:
            query_bundle: Query bundle with query string and embeddings

        Returns:
            List of retrieved nodes
        """
        return self._retrieve_cached(query_bundle.query_str)


def load_engine_for_modules(
    selected_modules: List[str],
    engine_params: Optional[Dict[str, Any]] = None,
    preserved_chat_history: Optional[List] = None,
    session_index_path: Optional[str] = None,
) -> CondensePlusContextChatEngine:
    """Load RAG chat engine with selected module indexes.

    Args:
        selected_modules: List of module names to load
        engine_params: Engine configuration parameters
        preserved_chat_history: Chat history to restore
        session_index_path: Optional session-specific index path

    Returns:
        Configured CondensePlusContextChatEngine instance

    Raises:
        ValueError: If no modules or session index selected
        FileNotFoundError: If no valid indices loaded
    """
    if not selected_modules and not session_index_path:
        raise ValueError("No modules or session index selected!")

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

    # Load session-specific PDF index if provided
    if session_index_path and os.path.exists(session_index_path):
        print(f"--- LOADING SESSION INDEX: {session_index_path} ---")
        try:
            db = chromadb.PersistentClient(path=session_index_path)
            collection = db.get_or_create_collection("data")
            vector_store = ChromaVectorStore(chroma_collection=collection)

            storage_context = StorageContext.from_defaults(
                persist_dir=session_index_path, vector_store=vector_store
            )

            index = load_index_from_storage(storage_context, embed_model=embed_model)

            base = index.as_retriever(similarity_top_k=similarity_top_k)
            am_retriever = AutoMergingRetriever(
                base, index.storage_context, verbose=False
            )
            active_retrievers.append(am_retriever)
            print("Session index loaded successfully")
        except Exception as e:
            print(f"Failed to load session index: {e}")

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

    # Build node postprocessors chain
    # Order: Reranker first, then hard cutoff filter on reranked scores
    node_postprocessors = []

    # Add reranker first
    node_postprocessors.append(get_reranker(engine_params, device=rag_device))

    # Add hard cutoff filter AFTER reranking (filters on final cross-encoder scores)
    confidence_cutoff_hard = engine_params.get("confidence_cutoff_hard", 0.0)
    if confidence_cutoff_hard > 0.0:
        print(
            f"--- HARD CUTOFF: Filtering reranked nodes below "
            f"{confidence_cutoff_hard} ---"
        )
        node_postprocessors.append(
            SimilarityPostprocessor(similarity_cutoff=confidence_cutoff_hard)
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
