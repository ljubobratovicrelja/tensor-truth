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
from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores import (
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore

# Config import for model defaults
from tensortruth.app_utils.config import load_config
from tensortruth.core.ollama import check_thinking_support, get_ollama_url

# --- GLOBAL CONFIG ---
_BASE_INDEX_DIR_CACHE: str | None = None


def get_base_index_dir() -> str:
    """
    Get the base index directory, preferring user data dir if available.

    This uses lazy loading to avoid circular import issues.
    """
    global _BASE_INDEX_DIR_CACHE
    if _BASE_INDEX_DIR_CACHE is None:
        try:
            from tensortruth.app_utils.paths import get_indexes_dir

            _BASE_INDEX_DIR_CACHE = str(get_indexes_dir())
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

# Prompt used for LLM-only mode (no RAG modules attached)
LLM_ONLY_SYSTEM_PROMPT = (
    "You are a helpful AI assistant within TensorTruth, a RAG (Retrieval-Augmented Generation) "
    "application for document intelligence.\n\n"
    "CURRENT STATUS: No knowledge base is attached to this session.\n\n"
    "GUIDELINES:\n"
    "1. Answer questions helpfully using your general knowledge.\n"
    "2. For factual or domain-specific questions where verified sources would help, "
    "briefly mention that the user can:\n"
    "   - Select a knowledge module when starting a new chat (using the Knowledge dropdown)\n"
    "   - Upload PDFs to this session using the PDF button in the chat header\n"
    "3. Keep disclaimers minimal and natural - a short note at the end is fine.\n"
    "4. For coding, creative writing, or general conversation - no disclaimer needed.\n"
    "5. Use Markdown formatting. Be concise and direct.\n"
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


def get_embed_model(
    device: str = "cpu", model_name: str | None = None
) -> HuggingFaceEmbedding:
    """Get HuggingFace embedding model via ModelManager singleton.

    This function provides backward compatibility while using the new
    ModelManager singleton for efficient model lifecycle management.

    Args:
        device: Device to load model on ('cuda', 'cpu', or 'mps')
        model_name: HuggingFace model path (default: from config or BAAI/bge-m3)

    Returns:
        HuggingFaceEmbedding instance (managed by ModelManager)
    """
    from tensortruth.services.model_manager import ModelManager

    manager = ModelManager.get_instance()
    manager.set_default_device(device)
    return manager.get_embedder(model_name=model_name, device=device)


def get_llm(params: Dict[str, Any]) -> Ollama:
    """Initialize Ollama LLM with configuration parameters.

    Args:
        params: Dictionary with model configuration

    Returns:
        Ollama LLM instance
    """
    # Try to get model from params, then from config, then use default
    model_name = params.get("model")
    if model_name is None:
        try:
            config = load_config()
            model_name = config.models.default_rag_model
        except Exception:
            from tensortruth.core.constants import DEFAULT_RAG_MODEL

            model_name = DEFAULT_RAG_MODEL  # Fallback default

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
    """Get cross-encoder reranker model via ModelManager singleton.

    This function provides backward compatibility while using the new
    ModelManager singleton for efficient model lifecycle management.

    Args:
        params: Dictionary with reranker configuration
        device: Device to load model on ('cuda', 'cpu', or 'mps')

    Returns:
        SentenceTransformerRerank instance (managed by ModelManager)
    """
    from tensortruth.services.model_manager import ModelManager

    # Default to the high-precision BGE-M3 v2 if not specified
    model = params.get("reranker_model", "BAAI/bge-reranker-v2-m3")
    top_n = params.get("reranker_top_n", 3)

    manager = ModelManager.get_instance()
    return manager.get_reranker(model_name=model, top_n=top_n, device=device)


# Operator mapping for filter expressions
_FILTER_OPERATORS = {
    "$eq": FilterOperator.EQ,
    "$ne": FilterOperator.NE,
    "$gt": FilterOperator.GT,
    "$gte": FilterOperator.GTE,
    "$lt": FilterOperator.LT,
    "$lte": FilterOperator.LTE,
    "$in": FilterOperator.IN,
    "$nin": FilterOperator.NIN,
    "$contains": FilterOperator.CONTAINS,
    "$text_match": FilterOperator.TEXT_MATCH,
}


def _build_metadata_filters(
    filter_spec: Optional[Dict[str, Any]],
) -> Optional[MetadataFilters]:
    """Build LlamaIndex MetadataFilters from a filter specification.

    Supports simple equality filters and operator-based filters.

    Args:
        filter_spec: Dictionary mapping field names to values or operator dicts.
            Simple: {"doc_type": "library"}
            With operator: {"version": {"$gte": "2.0"}}
            List values: {"doc_type": ["library", "book"]} (uses IN operator)

    Returns:
        MetadataFilters object, or None if filter_spec is empty/None

    Examples:
        >>> _build_metadata_filters({"doc_type": "library"})
        MetadataFilters(filters=[MetadataFilter(key="doc_type", value="library")])

        >>> _build_metadata_filters({"version": {"$gte": "2.0"}})
        MetadataFilters(filters=[MetadataFilter(key="version", value="2.0", operator=GTE)])
    """
    if not filter_spec:
        return None

    filters = []

    for key, value in filter_spec.items():
        if isinstance(value, dict):
            # Operator syntax: {"field": {"$op": "value"}}
            for op_key, op_value in value.items():
                if op_key in _FILTER_OPERATORS:
                    filters.append(
                        MetadataFilter(
                            key=key,
                            value=op_value,
                            operator=_FILTER_OPERATORS[op_key],
                        )
                    )
                else:
                    # Unknown operator, skip this entry (could log a warning)
                    pass
                break  # Only process first key in operator dict
        elif isinstance(value, list):
            # List values use IN operator
            filters.append(
                MetadataFilter(
                    key=key,
                    value=value,
                    operator=FilterOperator.IN,
                )
            )
        else:
            # Simple equality
            filters.append(MetadataFilter(key=key, value=value))

    if not filters:
        return None

    return MetadataFilters(
        filters=filters,  # type: ignore[arg-type]
        condition=FilterCondition.AND,
    )


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
        balance_strategy: str = "top_k_per_index",
    ) -> None:
        """Initialize multi-index retriever.

        Args:
            retrievers: List of retriever instances
            max_workers: Maximum parallel workers (default: min(len(retrievers), 8))
            enable_cache: Whether to cache retrieval results
            cache_size: LRU cache size
            balance_strategy: Balancing strategy ("none", "top_k_per_index")
        """
        self.retrievers = retrievers
        self.max_workers = max_workers or min(len(retrievers), 8)
        self.enable_cache = enable_cache
        self.balance_strategy = balance_strategy
        super().__init__()

        # Create LRU cache for retrieve operations if enabled
        if self.enable_cache:
            # lru_cache wrapper compatibility
            self._retrieve_cached = lru_cache(maxsize=cache_size)(
                self._retrieve_impl
            )  # type: ignore[assignment]
        else:
            self._retrieve_cached = self._retrieve_impl  # type: ignore[assignment]

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
                executor.submit(r.retrieve, query_bundle): (r, idx)
                for idx, r in enumerate(self.retrievers)
            }

            for future in as_completed(future_to_retriever):
                try:
                    nodes = future.result()
                    retriever, idx = future_to_retriever[future]

                    # Tag nodes with source index for balancing
                    for node in nodes:
                        # Access the underlying node's metadata dict directly
                        # Try NodeWithScore structure first (.node.metadata)
                        try:
                            inner_node = getattr(node, "node", None)
                            if inner_node is not None and isinstance(
                                getattr(inner_node, "metadata", None), dict
                            ):
                                inner_node.metadata["_source_index"] = idx
                                continue
                        except (AttributeError, TypeError):
                            pass

                        # Fall back to direct .metadata attribute
                        try:
                            if isinstance(getattr(node, "metadata", None), dict):
                                node.metadata["_source_index"] = idx
                        except (AttributeError, TypeError):
                            pass

                    combined_nodes.extend(nodes)
                except Exception as e:
                    # Log error but continue with other retrievers
                    print(f"Retriever failed: {e}")

        # Apply balancing if configured and multiple indexes
        if len(self.retrievers) > 1 and self.balance_strategy == "top_k_per_index":
            combined_nodes = self._balance_top_k_per_index(combined_nodes)

        return combined_nodes

    def _balance_top_k_per_index(
        self, nodes: List[NodeWithScore]
    ) -> List[NodeWithScore]:
        """Balance nodes by taking top-k from each index.

        Ensures each index contributes fairly to the candidate pool before reranking.

        Args:
            nodes: Combined nodes from all retrievers (tagged with _source_index)

        Returns:
            Balanced list with equal representation per index
        """
        from collections import defaultdict

        # Group by source index
        by_index = defaultdict(list)
        for node in nodes:
            # Try to get metadata from NodeWithScore structure
            metadata = None
            if hasattr(node, "node") and hasattr(node.node, "metadata"):
                metadata = node.node.metadata
            elif hasattr(node, "metadata"):
                metadata = node.metadata

            idx = metadata.get("_source_index", 0) if metadata else 0
            by_index[idx].append(node)

        if not by_index:
            return []

        # Determine per-index limit (ensure fair representation)
        num_indexes = len(by_index)
        total_nodes = len(nodes)
        per_index_limit = max(1, total_nodes // num_indexes)

        # Take top nodes from each index (already sorted by score)
        balanced = []
        for idx_nodes in by_index.values():
            balanced.extend(idx_nodes[:per_index_limit])

        # Re-sort by score for postprocessor chain
        balanced.sort(key=lambda n: n.score if n.score else 0.0, reverse=True)

        return balanced

    def _retrieve(self, query_bundle: QueryBundle):
        """Public retrieve method that leverages caching.

        Args:
            query_bundle: Query bundle with query string and embeddings

        Returns:
            List of retrieved nodes
        """
        return self._retrieve_cached(query_bundle.query_str)

    def clear_cache(self) -> None:
        """Clear the LRU cache to release GPU tensor references.

        Should be called before deleting the retriever to prevent VRAM leaks.
        """
        if self.enable_cache and hasattr(self._retrieve_cached, "cache_clear"):
            self._retrieve_cached.cache_clear()


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

    # Determine devices - use session param, fallback to config default, then auto-detect
    config = None
    if "rag_device" not in engine_params:
        try:
            # Try to get default from config
            config = load_config()
            rag_device = config.rag.default_device
        except (ImportError, Exception):
            # Fallback to auto-detection if config unavailable
            try:
                from tensortruth.app_utils.helpers import get_system_devices

                available_devices = get_system_devices()
                rag_device = available_devices[0] if available_devices else "cpu"
            except (ImportError, Exception):
                rag_device = "cpu"
    else:
        rag_device = engine_params["rag_device"]

    # Determine embedding model - use session param, fallback to config default
    embedding_model = engine_params.get("embedding_model")
    if embedding_model is None:
        try:
            if config is None:
                config = load_config()
            embedding_model = config.rag.default_embedding_model
        except (ImportError, Exception):
            embedding_model = "BAAI/bge-m3"

    # Calculate adaptive similarity_top_k based on reranker_top_n
    # Retrieve 2-3x more candidates than final target to ensure quality
    reranker_top_n = engine_params.get("reranker_top_n", 3)
    similarity_top_k = max(5, reranker_top_n * 2)

    # Set Global Settings for this session (Embedder via ModelManager)
    embed_model = get_embed_model(rag_device, model_name=embedding_model)
    Settings.embed_model = embed_model

    # Get embedding-aware index directory
    from tensortruth.app_utils.paths import get_indexes_dir_for_model

    embedding_indexes_dir = str(get_indexes_dir_for_model(embedding_model))

    active_retrievers: list[BaseRetriever] = []
    print(
        f"--- MOUNTING: {selected_modules} | MODEL: {engine_params.get('model')} | "
        f"EMBEDDER: {embedding_model} | RAG DEVICE: {rag_device} | "
        f"RETRIEVAL: {similarity_top_k} per index → RERANK: top {reranker_top_n} ---"
    )

    for module in selected_modules:
        # First try embedding-aware path: indexes/{model_id}/{module}
        path = os.path.join(embedding_indexes_dir, module)

        # Fall back to legacy path: indexes/{module} (for backward compatibility)
        if not os.path.exists(path):
            legacy_path = os.path.join(get_base_index_dir(), module)
            if os.path.exists(legacy_path):
                print(f"  Using legacy index path for {module}")
                path = legacy_path
            else:
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
        am_retriever = AutoMergingRetriever(
            base, index.storage_context, verbose=False  # type: ignore[arg-type]
        )
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
                base, index.storage_context, verbose=False  # type: ignore[arg-type]
            )
            active_retrievers.append(am_retriever)
            print("Session index loaded successfully")
        except Exception as e:
            print(f"Failed to load session index: {e}")

    if not active_retrievers:
        raise FileNotFoundError("No valid indices loaded.")

    # Get balance strategy from params (default to "top_k_per_index")
    balance_strategy = engine_params.get("balance_strategy", "top_k_per_index")

    composite_retriever = MultiIndexRetriever(
        active_retrievers, balance_strategy=balance_strategy
    )

    memory = ChatMemoryBuffer.from_defaults(token_limit=3000)

    # Restore chat history from previous engine if provided
    if preserved_chat_history:
        for msg in preserved_chat_history:
            # Restore each message to the memory buffer
            memory.put(msg)

    llm = get_llm(engine_params)

    # Build node postprocessors chain
    # Order: Reranker first, then hard cutoff filter on reranked scores
    node_postprocessors: list[Any] = []

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
