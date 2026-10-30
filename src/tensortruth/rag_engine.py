import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from typing import Any, Dict, List, Optional

import chromadb
from llama_index.core import (
    QueryBundle,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.llms import LLM
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
from llama_index.vector_stores.chroma import ChromaVectorStore

# Config import for model defaults
from tensortruth.app_utils.config import load_config
from tensortruth.core.providers import create_llm, resolve_model_from_params

logger = logging.getLogger(__name__)

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
    "4. FORMATTING: Use Markdown headers for structure. Use LaTeX for math.\n"
    "5. SOURCE ATTRIBUTION: Every excerpt is labelled with its own document and "
    "section in its header. Cite only the document named in that excerpt's header; "
    "never attribute a passage to a document that is not named in its own header. "
    "When two documents disagree, name both and say which says what.\n\n"
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
    "2. PRIORITIZATION: If the Chat History contains the answer, ignore the "
    "retrieved context entirely.\n"
    "3. NO HALLUCINATION: If neither History nor Context supports a factual answer, "
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


def get_llm(params: Dict[str, Any]) -> LLM:
    """Initialize LLM with configuration parameters.

    Supports both Ollama and OpenAI-compatible providers via the
    ``provider_id`` session parameter.

    Args:
        params: Dictionary with model configuration

    Returns:
        LlamaIndex LLM instance
    """
    # Try to get model from params, then from config, then use default
    model_name = params.get("model")
    if model_name is None:
        try:
            config = load_config()
            model_name = config.llm.default_model
        except Exception:
            from tensortruth.core.constants import DEFAULT_MODEL

            model_name = DEFAULT_MODEL  # Fallback default

    # Resolve model reference via provider registry
    model_ref = resolve_model_from_params(params, model_name)

    user_system_prompt = params.get("system_prompt", "").strip()
    device_mode = params.get("llm_device", "gpu")  # 'gpu' or 'cpu'

    ctx_window = params.get("context_window", 16384)

    # Resolve thinking
    thinking_preference = params.get("thinking")
    from tensortruth.core.providers import resolve_thinking as provider_resolve

    thinking_enabled = provider_resolve(model_ref, thinking_preference)

    # Build additional kwargs (Ollama-specific options are handled internally)
    additional: dict = {"num_ctx": ctx_window}
    if device_mode == "cpu" and model_ref.provider_type == "ollama":
        print(f"Loading LLM {model_name} on: CPU (Forced)")
        additional["num_gpu"] = 0

    if thinking_enabled:
        additional["num_predict"] = params.get("max_tokens", 4096)
    else:
        additional["num_predict"] = -1

    extra_kwargs: Dict[str, Any] = {
        "temperature": params.get("temperature", 0.3),
        "context_window": ctx_window,
        "request_timeout": 300.0,
        "additional_kwargs": additional,
    }
    if model_ref.provider_type in ("ollama", "llama_cpp"):
        extra_kwargs["thinking"] = thinking_enabled
    if user_system_prompt:
        extra_kwargs["system_prompt"] = user_system_prompt

    return create_llm(model_ref, **extra_kwargs)


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

    # LlamaIndex stubs incomplete: List[MetadataFilter] valid but not recognized
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
        # Known mypy limitation: cannot track lru_cache wrapper types
        if self.enable_cache:
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


def resolve_engine_settings(
    engine_params: Optional[Dict[str, Any]] = None,
) -> tuple[str, str]:
    """Resolve the embedding model id and RAG device for an engine build.

    Args:
        engine_params: Engine configuration parameters.

    Returns:
        (embedding_model, rag_device) with the embedding model name healed to
        its canonical HuggingFace id.
    """
    params = engine_params or {}
    config = None

    if "rag_device" not in params:
        try:
            config = load_config()
            rag_device = config.rag.default_device
        except (ImportError, Exception):
            try:
                from tensortruth.app_utils.helpers import get_system_devices

                available_devices = get_system_devices()
                rag_device = available_devices[0] if available_devices else "cpu"
            except (ImportError, Exception):
                rag_device = "cpu"
    else:
        rag_device = params["rag_device"]

    embedding_model = params.get("embedding_model")
    if embedding_model is None:
        try:
            if config is None:
                config = load_config()
            embedding_model = config.rag.default_embedding_model
        except (ImportError, Exception):
            embedding_model = "BAAI/bge-m3"

    from tensortruth.indexing.metadata import resolve_embedding_model_name

    return resolve_embedding_model_name(embedding_model), rag_device


def get_similarity_top_k(engine_params: Optional[Dict[str, Any]] = None) -> int:
    """Adaptive candidate pool size: 2x the reranker target, at least 5."""
    params = engine_params or {}
    reranker_top_n = params.get("reranker_top_n", 3)
    return max(5, reranker_top_n * 2)


def _open_index(
    persist_dir: str,
    embed_model: Any,
    similarity_top_k: int,
    use_lexical: bool = True,
) -> tuple[BaseRetriever, Any]:
    """Open one persisted Chroma index and build its retriever.

    The embedder is passed explicitly so retrieval never depends on the
    process-global ``Settings.embed_model`` (which concurrent sessions with
    different embedding models would otherwise clobber).

    When a lexical side index exists next to the vector store, the dense
    auto-merging retriever is fused with a leaf-level lexical retriever so
    exact-term queries are not lost to embedding similarity alone.
    """
    db = chromadb.PersistentClient(path=persist_dir)
    collection = db.get_or_create_collection("data")
    vector_store = ChromaVectorStore(chroma_collection=collection)

    storage_context = StorageContext.from_defaults(
        persist_dir=persist_dir, vector_store=vector_store
    )
    index = load_index_from_storage(storage_context, embed_model=embed_model)

    base = index.as_retriever(similarity_top_k=similarity_top_k)
    auto_merging = AutoMergingRetriever(
        base, index.storage_context, verbose=False  # type: ignore[arg-type]
    )
    retriever: BaseRetriever = auto_merging

    if use_lexical:
        from tensortruth.indexing.lexical import LEXICAL_DB_FILENAME, LexicalIndex
        from tensortruth.retrieval.fusion import LexicalRetriever, RRFusionRetriever

        if os.path.exists(os.path.join(persist_dir, LEXICAL_DB_FILENAME)):
            try:
                lexical = LexicalIndex(persist_dir)
                lexical_retriever = LexicalRetriever(
                    lexical=lexical,
                    top_k=similarity_top_k,
                    node_lookup=index.storage_context.docstore.get_node,
                )
                retriever = RRFusionRetriever(
                    [auto_merging, lexical_retriever], rrf_k=60
                )
                logger.info(
                    f"Fused 2 retrieval backends (dense+auto-merge, lexical) for "
                    f"index '{persist_dir}' with {lexical.count()} lexical chunks"
                )
            except Exception as e:
                # A broken lexical side index must not cost the dense retriever.
                logger.warning(
                    f"Lexical index at '{persist_dir}' unusable, "
                    f"falling back to dense retrieval: {e}"
                )

    return retriever, index


def resolve_module_index_path(module_name: str, embedding_model: str) -> Optional[str]:
    """Resolve the on-disk index directory for a module, or None if missing.

    Prefers the embedding-aware path (indexes/{model_id}/{module}) and falls
    back to the legacy path (indexes/{module}) for older installs.
    """
    from tensortruth.app_utils.paths import get_indexes_dir_for_model

    embedding_indexes_dir = str(get_indexes_dir_for_model(embedding_model))
    path = os.path.join(embedding_indexes_dir, module_name)
    if os.path.exists(path):
        return path

    legacy_path = os.path.join(get_base_index_dir(), module_name)
    if os.path.exists(legacy_path):
        logger.info(f"Using legacy index path for module '{module_name}'")
        print(f"  Using legacy index path for {module_name}")
        return legacy_path

    logger.warning(
        f"Module '{module_name}' not found at '{path}' or legacy path "
        f"'{legacy_path}' - skipping. Check if indexed with "
        f"embedding model '{embedding_model}'."
    )
    return None


def load_module_index(
    module_name: str,
    embedding_model: str,
    embed_model: Any,
    similarity_top_k: int,
    use_lexical: bool = True,
) -> Optional[tuple[BaseRetriever, Any]]:
    """Load a single module index.

    Args:
        module_name: Indexed module name.
        embedding_model: Canonical embedding model id (used for the path).
        embed_model: Embedder instance shared through ModelManager.
        similarity_top_k: Candidate pool size for the inner retriever.
        use_lexical: Fuse a lexical side index when one is present.

    Returns:
        (retriever, index) or None when the module has no index on disk.
    """
    path = resolve_module_index_path(module_name, embedding_model)
    if path is None:
        return None
    return _open_index(path, embed_model, similarity_top_k, use_lexical=use_lexical)


def load_index_from_path(
    index_path: str,
    embed_model: Any,
    similarity_top_k: int,
    use_lexical: bool = True,
) -> Optional[tuple[BaseRetriever, Any]]:
    """Load an explicit index path (session PDFs, project indexes)."""
    if not os.path.exists(index_path):
        logger.warning(f"Additional index path not found, skipping: {index_path}")
        return None
    try:
        return _open_index(
            index_path, embed_model, similarity_top_k, use_lexical=use_lexical
        )
    except Exception as e:
        logger.warning(f"Failed to load additional index {index_path}: {e}")
        return None


def compose_chat_engine(
    retrievers: List[BaseRetriever],
    engine_params: Optional[Dict[str, Any]] = None,
    rag_device: Optional[str] = None,
) -> CondensePlusContextChatEngine:
    """Wire retrievers, memory, LLM and postprocessors into a chat engine.

    Args:
        retrievers: Per-index retrievers already loaded.
        engine_params: Engine configuration parameters.
        rag_device: Device used for the reranker (resolved if omitted).

    Returns:
        Configured CondensePlusContextChatEngine.

    Raises:
        FileNotFoundError: If no retrievers were supplied.
    """
    params = engine_params or {}
    if not retrievers:
        raise FileNotFoundError("No valid indices loaded.")
    if rag_device is None:
        _, rag_device = resolve_engine_settings(params)

    balance_strategy = params.get("balance_strategy", "top_k_per_index")
    composite_retriever = MultiIndexRetriever(
        retrievers, balance_strategy=balance_strategy
    )

    memory_token_limit = params.get("memory_token_limit")
    if memory_token_limit is None:
        try:
            memory_token_limit = load_config().conversation.memory_token_limit
        except (ImportError, Exception):
            memory_token_limit = 4000
    memory = ChatMemoryBuffer.from_defaults(token_limit=memory_token_limit)

    # Chat history is passed to query methods from session storage; the buffer
    # exists for LlamaIndex internal state only and is not pre-populated.

    llm = get_llm(params)

    # Reranker first, then the hard cutoff filter on reranked scores.
    node_postprocessors: list[Any] = [get_reranker(params, device=rag_device)]

    confidence_cutoff_hard = params.get("confidence_cutoff_hard", 0.0)
    if confidence_cutoff_hard > 0.0:
        print(
            f"--- HARD CUTOFF: Filtering reranked nodes below "
            f"{confidence_cutoff_hard} ---"
        )
        node_postprocessors.append(
            SimilarityPostprocessor(similarity_cutoff=confidence_cutoff_hard)
        )

    return CondensePlusContextChatEngine.from_defaults(
        retriever=composite_retriever,
        node_postprocessors=node_postprocessors,
        llm=llm,
        memory=memory,
        context_prompt=CUSTOM_CONTEXT_PROMPT_TEMPLATE,
        condense_prompt=CUSTOM_CONDENSE_PROMPT_TEMPLATE,
        verbose=False,
    )


def load_engine_for_modules(
    selected_modules: List[str],
    engine_params: Optional[Dict[str, Any]] = None,
    additional_index_paths: Optional[List[str]] = None,
) -> CondensePlusContextChatEngine:
    """Load a RAG chat engine directly (compatibility wrapper).

    Prefer the ModuleIndexRegistry + RAGService per-owner sets in the app: this
    function loads every index itself and is kept for tests and standalone use.

    Args:
        selected_modules: Module names to load.
        engine_params: Engine configuration parameters.
        additional_index_paths: Optional extra index paths (session PDFs,
            project indexes).

    Returns:
        Configured CondensePlusContextChatEngine instance.

    Raises:
        FileNotFoundError: If no valid indices loaded.
    """
    params = engine_params or {}
    embedding_model, rag_device = resolve_engine_settings(params)
    similarity_top_k = get_similarity_top_k(params)
    embed_model = get_embed_model(rag_device, model_name=embedding_model)

    active_retrievers: list[BaseRetriever] = []
    loaded_modules: list[str] = []
    skipped_modules: list[str] = []

    for module in selected_modules:
        loaded = load_module_index(
            module, embedding_model, embed_model, similarity_top_k
        )
        if loaded is None:
            skipped_modules.append(module)
            continue
        retriever, _ = loaded
        active_retrievers.append(retriever)
        loaded_modules.append(module)

    if skipped_modules:
        logger.warning(
            f"Skipped {len(skipped_modules)}/{len(selected_modules)} modules "
            f"(not found): {skipped_modules}"
        )
    logger.info(
        f"Loaded {len(loaded_modules)}/{len(selected_modules)} modules: "
        f"{loaded_modules}"
    )

    for idx_path in additional_index_paths or []:
        loaded = load_index_from_path(idx_path, embed_model, similarity_top_k)
        if loaded is None:
            continue
        retriever, _ = loaded
        active_retrievers.append(retriever)
        logger.info(f"Additional index loaded: {idx_path}")

    return compose_chat_engine(active_retrievers, params, rag_device=rag_device)
