"""RAG (Retrieval-Augmented Generation) service.

This service wraps the existing rag_engine module with lifecycle management
and a cleaner interface for the UI layer.
"""

import base64
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generator,
    List,
    Optional,
    Tuple,
    Union,
    cast,
)

from llama_index.core.base.llms.types import (
    ChatMessage,
    ImageBlock,
    MessageRole,
    TextBlock,
)
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.llms import LLM

if TYPE_CHECKING:
    from llama_index.core.retrievers import BaseRetriever

from tensortruth.app_utils.config_schema import TensorTruthConfig
from tensortruth.core.prompts import current_date_context
from tensortruth.rag_engine import (
    CUSTOM_CONTEXT_PROMPT_LOW_CONFIDENCE,
    CUSTOM_CONTEXT_PROMPT_NO_SOURCES,
    CUSTOM_CONTEXT_PROMPT_TEMPLATE,
    LLM_ONLY_SYSTEM_PROMPT,
    MultiIndexRetriever,
    compose_chat_engine,
    get_base_index_dir,
    get_embed_model,
    get_llm,
    get_similarity_top_k,
    load_index_from_path,
    resolve_engine_settings,
)
from tensortruth.utils.history_condenser import (
    condense_query,
    create_condenser_llm,
)

from .chat_history import ChatHistoryService
from .index_registry import ModuleIndexRegistry
from .index_registry import get_instance as get_index_registry
from .models import RAGChunk, RAGResponse, RAGRetrievalResult, ToolProgress
from .retrieval_metrics import compute_retrieval_metrics

logger = logging.getLogger(__name__)

# Owner key used when a caller does not identify a session (legacy call sites,
# tests, direct library use). Real sessions always pass their session id.
DEFAULT_OWNER = "__default__"


@dataclass
class SessionRagSet:
    """Per-owner retrieval set on top of the shared module indexes.

    Holds only cheap per-session objects: the module names and params it was
    built from, the composite retriever and chat engine, and the set of
    registry handles (module names) it refcounts.
    """

    owner: str
    modules: List[str]
    params: Dict[str, Any]
    config_hash: Optional[Tuple[Any, ...]]
    embedding_model: str
    chat_engine: CondensePlusContextChatEngine
    llm: Optional[LLM]
    retriever: Optional["BaseRetriever"]
    acquired_modules: List[str] = field(default_factory=list)
    additional_index_paths: List[str] = field(default_factory=list)
    last_used: float = field(default_factory=time.time)


class RAGService:
    """Service for RAG query operations with lifecycle management.

    Manages the chat engine instance, handling loading, reloading,
    and cleanup of GPU resources.

    Chat history is now managed via ChatHistoryService and passed
    explicitly to query methods from session storage.
    """

    def __init__(
        self,
        config: Optional[TensorTruthConfig] = None,
        indexes_dir: Optional[Union[str, Path]] = None,
        chat_history_service: Optional[ChatHistoryService] = None,
        index_registry: Optional[ModuleIndexRegistry] = None,
    ):
        """Initialize RAG service.

        Args:
            config: TensorTruth configuration. If None, loads from default.
            indexes_dir: Base directory for vector indexes.
            chat_history_service: Optional ChatHistoryService instance.
                If None, one is created lazily when needed.
            index_registry: Module index registry. Defaults to the process
                singleton, which is what makes module handles shared.
        """
        if config is None:
            from tensortruth.app_utils.config import load_config

            config = load_config()

        self.config = config
        self.indexes_dir = (
            Path(indexes_dir) if indexes_dir else Path(get_base_index_dir())
        )

        # One thin retrieval set per owner (session id). Index handles are
        # shared through the registry and refcounted, never duplicated here.
        self._sets: Dict[str, SessionRagSet] = {}
        self._sets_lock = threading.RLock()
        self._registry = index_registry or get_index_registry()
        self._chat_history_service = chat_history_service

    # --- Backwards-compatible single-engine views ---------------------------
    # Older call sites (and tests) read/write the engine attributes directly.
    # They map onto the default owner's set; real sessions use their own.

    def _default_set(self) -> Optional[SessionRagSet]:
        return self._sets.get(DEFAULT_OWNER)

    def _ensure_default_set(self) -> SessionRagSet:
        rag_set = self._sets.get(DEFAULT_OWNER)
        if rag_set is None:
            rag_set = SessionRagSet(
                owner=DEFAULT_OWNER,
                modules=[],
                params={},
                config_hash=None,
                embedding_model="",
                chat_engine=cast(CondensePlusContextChatEngine, None),
                llm=None,
                retriever=None,
            )
            self._sets[DEFAULT_OWNER] = rag_set
        return rag_set

    @property
    def _engine(self) -> Optional[CondensePlusContextChatEngine]:
        rag_set = self._default_set()
        return rag_set.chat_engine if rag_set else None

    @_engine.setter
    def _engine(self, engine: Optional[CondensePlusContextChatEngine]) -> None:
        if engine is None:
            self._sets.pop(DEFAULT_OWNER, None)
            return
        rag_set = self._ensure_default_set()
        rag_set.chat_engine = engine

    @property
    def _llm(self) -> Optional[LLM]:
        rag_set = self._default_set()
        return rag_set.llm if rag_set else None

    @_llm.setter
    def _llm(self, llm: Optional[LLM]) -> None:
        self._ensure_default_set().llm = llm

    @property
    def _retriever(self) -> Optional["BaseRetriever"]:
        rag_set = self._default_set()
        return rag_set.retriever if rag_set else None

    @_retriever.setter
    def _retriever(self, retriever: Optional["BaseRetriever"]) -> None:
        self._ensure_default_set().retriever = retriever

    @property
    def _current_params(self) -> Optional[Dict[str, Any]]:
        rag_set = self._default_set()
        return rag_set.params if rag_set else None

    @_current_params.setter
    def _current_params(self, params: Optional[Dict[str, Any]]) -> None:
        self._ensure_default_set().params = params or {}

    @property
    def _current_modules(self) -> Optional[List[str]]:
        rag_set = self._default_set()
        return rag_set.modules if rag_set else None

    @_current_modules.setter
    def _current_modules(self, modules: Optional[List[str]]) -> None:
        self._ensure_default_set().modules = modules or []

    @property
    def _current_config_hash(self) -> Optional[Tuple[Any, ...]]:
        rag_set = self._default_set()
        return rag_set.config_hash if rag_set else None

    @_current_config_hash.setter
    def _current_config_hash(self, value: Optional[Tuple[Any, ...]]) -> None:
        self._ensure_default_set().config_hash = value

    @property
    def chat_history_service(self) -> ChatHistoryService:
        """Get or create the ChatHistoryService instance (lazy initialization)."""
        if self._chat_history_service is None:
            self._chat_history_service = ChatHistoryService(self.config)
        return self._chat_history_service

    def _compute_config_hash(
        self,
        modules: Optional[List[str]],
        params: Dict[str, Any],
        additional_index_paths: Optional[List[str]] = None,
    ) -> Optional[Tuple[Any, ...]]:
        """Compute configuration hash for cache invalidation.

        Args:
            modules: List of module names.
            params: Engine parameters (may contain nested dicts/lists).
            additional_index_paths: Optional list of additional index paths.

        Returns:
            Hashable tuple for comparison.
        """
        modules_tuple = tuple(sorted(modules)) if modules else None
        # Use JSON for consistent hashing of nested structures
        # sort_keys ensures {"a":1,"b":2} == {"b":2,"a":1}
        # default=str handles non-serializable objects gracefully
        param_hash = json.dumps(params, sort_keys=True, default=str)
        paths_tuple = (
            tuple(sorted(additional_index_paths)) if additional_index_paths else None
        )

        if modules_tuple or paths_tuple:
            return (modules_tuple, param_hash, paths_tuple)
        return None

    def load_engine(
        self,
        modules: List[str],
        params: Dict[str, Any],
        additional_index_paths: Optional[List[str]] = None,
        owner: str = DEFAULT_OWNER,
    ) -> None:
        """Load or reload the retrieval set for one owner.

        Only this owner's set is replaced: other sessions keep their modules,
        retrievers and chat engine. Module index handles come from the shared
        registry (refcounted), so a module used by several owners is loaded
        once. Additional index paths are per-session and always loaded here.

        Chat history is passed to query()/retrieve() at call time, not here.

        Args:
            modules: List of module names to load.
            params: Engine parameters (model, temperature, etc).
            additional_index_paths: Optional list of additional index paths
                (session PDFs, project indexes).
            owner: Session id owning this set (default for legacy callers).
        """
        logger.info(f"Loading RAG set for owner={owner} modules: {modules}")
        new_set = self._build_set(owner, modules, params, additional_index_paths)

        with self._sets_lock:
            previous = self._sets.get(owner)
            self._sets[owner] = new_set
        if previous is not None:
            self._release_set(previous)

        logger.info(
            f"RAG set loaded for owner={owner}: "
            f"{len(new_set.acquired_modules)} module(s) "
            f"+ {len(new_set.additional_index_paths)} extra index(es)"
        )

    def _build_set(
        self,
        owner: str,
        modules: List[str],
        params: Dict[str, Any],
        additional_index_paths: Optional[List[str]],
    ) -> SessionRagSet:
        """Acquire indexes and compose the owner's chat engine."""
        embedding_model, rag_device = resolve_engine_settings(params)
        similarity_top_k = get_similarity_top_k(params)
        embed_model = get_embed_model(rag_device, model_name=embedding_model)

        skipped: List[str] = []
        retrievers = self._registry.acquire(
            list(modules or []),
            embedding_model,
            embed_model,
            similarity_top_k,
            skipped=skipped,
        )
        acquired_modules = [
            name for name in dict.fromkeys(modules or []) if name not in skipped
        ]

        additional_loaded: List[str] = []
        try:
            for idx_path in additional_index_paths or []:
                loaded = load_index_from_path(idx_path, embed_model, similarity_top_k)
                if loaded is None:
                    continue
                retriever, _ = loaded
                retrievers.append(retriever)
                additional_loaded.append(idx_path)

            chat_engine = compose_chat_engine(retrievers, params, rag_device=rag_device)
        except Exception:
            # Do not leak refcounts when the set cannot be composed.
            self._registry.release(acquired_modules, embedding_model)
            raise

        return SessionRagSet(
            owner=owner,
            modules=list(modules or []),
            params=dict(params or {}),
            config_hash=self._compute_config_hash(
                modules, params, additional_index_paths
            ),
            embedding_model=embedding_model,
            chat_engine=chat_engine,
            llm=cast(LLM, chat_engine._llm),
            retriever=chat_engine._retriever,
            acquired_modules=acquired_modules,
            additional_index_paths=additional_loaded,
        )

    def _release_set(self, rag_set: SessionRagSet) -> None:
        """Drop one owner's retrievers/engine and give back registry refs."""
        retriever = rag_set.retriever
        if isinstance(retriever, MultiIndexRetriever):
            retriever.clear_cache()
        memory = getattr(rag_set.chat_engine, "memory", None)
        if memory is not None:
            try:
                memory.reset()
            except Exception:
                logger.debug("Failed to reset chat memory", exc_info=True)
        if rag_set.acquired_modules and rag_set.embedding_model:
            self._registry.release(rag_set.acquired_modules, rag_set.embedding_model)

    def needs_reload(
        self,
        modules: List[str],
        params: Dict[str, Any],
        additional_index_paths: Optional[List[str]] = None,
        owner: str = DEFAULT_OWNER,
    ) -> bool:
        """Check if this owner's set needs rebuilding due to config changes.

        Args:
            modules: List of module names.
            params: Engine parameters.
            additional_index_paths: Optional list of additional index paths.
            owner: Session id owning the set.

        Returns:
            True if the set should be reloaded.
        """
        rag_set = self._sets.get(owner)
        new_hash = self._compute_config_hash(modules, params, additional_index_paths)
        if rag_set is None or rag_set.chat_engine is None:
            return new_hash is not None  # Only reload if there is something to load
        return new_hash != rag_set.config_hash

    def is_loaded(self, owner: Optional[str] = None) -> bool:
        """Check if a retrieval set is loaded.

        Args:
            owner: Session id to check. When None, reports whether *any* owner
                has a set (used by tool registration).

        Returns:
            True if a set exists for the owner (or for any owner when None).
        """
        if owner is None:
            return any(s.chat_engine is not None for s in self._sets.values())
        rag_set = self._sets.get(owner)
        return rag_set is not None and rag_set.chat_engine is not None

    def get_loaded_modules(self, owner: str = DEFAULT_OWNER) -> List[str]:
        """Modules loaded for one owner (empty if no set)."""
        rag_set = self._sets.get(owner)
        return list(rag_set.modules) if rag_set else []

    def release(self, owner: str) -> None:
        """Release one owner's set and decrement its registry refcounts."""
        with self._sets_lock:
            rag_set = self._sets.pop(owner, None)
        if rag_set is None:
            return
        self._release_set(rag_set)
        logger.info(f"Released RAG set for owner={owner}")

    def sweep_idle(self, max_idle_seconds: float) -> List[str]:
        """Release sets untouched for longer than ``max_idle_seconds``.

        Called from the lifespan background task; returns released owners.
        """
        cutoff = time.time() - max_idle_seconds
        with self._sets_lock:
            idle = [
                owner
                for owner, rag_set in self._sets.items()
                if rag_set.last_used < cutoff
            ]
        for owner in idle:
            self.release(owner)
        if idle:
            logger.info(f"Swept {len(idle)} idle RAG set(s): {idle}")
        return idle

    def query(
        self,
        prompt: str,
        params: Optional[Dict[str, Any]] = None,
        session_messages: Optional[List[Dict[str, Any]]] = None,
        images: Optional[List[Dict[str, str]]] = None,
        owner: str = DEFAULT_OWNER,
    ) -> Generator[RAGChunk, None, RAGResponse]:
        """Execute a streaming query through the unified pipeline.

        Handles both RAG (with retriever) and LLM-only (no retriever) cases.
        When no engine is loaded, creates an ad-hoc LLM from params and skips
        retrieval. Prompt selection adapts based on whether retrieval was
        performed and what it returned.

        Args:
            prompt: User's query prompt.
            params: Engine parameters (model, temperature, etc). Used to create
                an ad-hoc LLM when no engine is loaded.
            session_messages: Chat history from session storage.
                Pass session["messages"] or session_service.get_messages().
                If None, query runs without history context.
            owner: Session id whose retrieval set to use.

        Yields:
            RAGChunk with status, thinking, or text content.

        Returns:
            Final RAGResponse with complete text and sources.
        """
        # Resolve this owner's set; never fall back to another session's engine.
        rag_set = self._sets.get(owner)
        engine = rag_set.chat_engine if rag_set is not None else None
        if rag_set is not None and engine is not None:
            rag_set.last_used = time.time()
            llm = rag_set.llm
            retriever = rag_set.retriever
            effective_params = rag_set.params or params or {}
        else:
            llm = get_llm(params or {})
            retriever = None
            effective_params = params or {}

        assert llm is not None, "LLM must be available"

        logger.info(f"=== RAG Query Start: '{prompt}' ===")
        logger.info(
            f"Session has {len(session_messages) if session_messages else 0} messages"
        )
        logger.info(f"Retriever available: {retriever is not None}")

        # Build chat history from session messages using ChatHistoryService
        max_turns = effective_params.get("max_history_turns")
        history = self.chat_history_service.build_history(
            session_messages,
            max_turns=max_turns,
            apply_cleaning=self.config.history_cleaning.enabled,
        )

        # Convert to LlamaIndex format for LLM
        chat_history = history.to_llama_messages()
        chat_history_str = history.to_prompt_string()

        source_nodes: list = []
        metrics_dict: Optional[Dict[str, Any]] = None

        if retriever is not None:
            # Phase 1: Retrieval
            yield RAGChunk(
                status="retrieving",
                progress=ToolProgress(
                    tool_id="rag",
                    phase="retrieving",
                    message="Searching knowledge base...",
                ),
            )

            condenser = getattr(engine, "_condense_prompt_template", None)
            logger.info(f"Condenser value: {condenser}")
            logger.info(f"Condenser type: {type(condenser)}")
            logger.info(f"_skip_condense: {getattr(engine, '_skip_condense', None)}")

            # Condense the question with chat history if we have a condenser
            condensed_question = prompt
            if condenser and not history.is_empty:
                logger.info(f"Original query: {prompt}")
                logger.info(f"History length: {len(history.messages)} messages")

                template_str = getattr(condenser, "template", str(condenser))
                condenser_llm = create_condenser_llm(
                    llm,
                    provider_id=effective_params.get("provider_id"),
                )

                condensed_question = condense_query(
                    llm=condenser_llm,
                    chat_history=chat_history_str,
                    question=prompt,
                    prompt_template=template_str,
                    fallback_on_error=True,
                )
                logger.info(f"Condensed query: {condensed_question}")
            else:
                if not condenser:
                    logger.debug("No condenser configured, using original query")
                elif history.is_empty:
                    logger.debug("History is empty, skipping condensation")

            # Retrieve context nodes
            source_nodes = retriever.retrieve(condensed_question)
            logger.info(f"Retrieved {len(source_nodes)} nodes before reranking")

            # Phase 2: Reranking (if postprocessors exist)
            assert engine is not None  # guaranteed when retriever exists
            if hasattr(engine, "_node_postprocessors") and engine._node_postprocessors:
                yield RAGChunk(
                    status="reranking",
                    progress=ToolProgress(
                        tool_id="rag",
                        phase="reranking",
                        message="Ranking results...",
                    ),
                )

                from llama_index.core.schema import QueryBundle

                query_bundle = QueryBundle(query_str=condensed_question)

                try:
                    for postprocessor in engine._node_postprocessors:
                        source_nodes = postprocessor.postprocess_nodes(
                            source_nodes, query_bundle=query_bundle
                        )
                except Exception as e:
                    logger.warning(
                        f"Postprocessor failed, using unprocessed nodes: {e}"
                    )

            # Enforce reranker_top_n limit as a safeguard
            reranker_top_n = effective_params.get("reranker_top_n")
            logger.info(
                f"After reranking: {len(source_nodes)} nodes "
                f"(reranker_top_n={reranker_top_n})"
            )
            if reranker_top_n and len(source_nodes) > reranker_top_n:
                source_nodes = source_nodes[:reranker_top_n]
                logger.info(f"Truncated to {len(source_nodes)} nodes")

            # Compute retrieval metrics from final reranked nodes
            metrics = compute_retrieval_metrics(source_nodes)
            configured_top_n = effective_params.get("reranker_top_n")
            metrics.configured_top_n = configured_top_n
            metrics_dict = metrics.to_dict()

        # Phase 3: Prompt selection
        date_prefix = current_date_context() + "\n\n"
        is_low_confidence = False
        if retriever is None:
            # No retriever — LLM-only mode with system prompt
            messages = [
                ChatMessage(
                    role=MessageRole.SYSTEM,
                    content=date_prefix + LLM_ONLY_SYSTEM_PROMPT,
                )
            ]
            if not history.is_empty:
                messages.extend(chat_history)
            messages.append(ChatMessage(role=MessageRole.USER, content=prompt))
        elif not source_nodes:
            # Retrieval ran but returned nothing
            formatted_prompt = date_prefix + CUSTOM_CONTEXT_PROMPT_NO_SOURCES.format(
                chat_history=chat_history_str,
                query_str=prompt,
            )
            messages = chat_history + [
                ChatMessage(role=MessageRole.USER, content=formatted_prompt)
            ]
        else:
            # Retrieval returned sources — check confidence
            confidence_threshold = effective_params.get("confidence_cutoff", 0.0)
            context_str = "\n\n".join([n.get_content() for n in source_nodes])

            if confidence_threshold > 0:
                best_score = max(
                    (node.score for node in source_nodes if node.score is not None),
                    default=0.0,
                )
                if best_score < confidence_threshold:
                    is_low_confidence = True
                    formatted_prompt = (
                        date_prefix
                        + CUSTOM_CONTEXT_PROMPT_LOW_CONFIDENCE.format(
                            context_str=context_str,
                            chat_history=chat_history_str,
                            query_str=prompt,
                        )
                    )
                else:
                    formatted_prompt = (
                        date_prefix
                        + CUSTOM_CONTEXT_PROMPT_TEMPLATE.format(
                            context_str=context_str,
                            chat_history=chat_history_str,
                            query_str=prompt,
                        )
                    )
            else:
                formatted_prompt = date_prefix + CUSTOM_CONTEXT_PROMPT_TEMPLATE.format(
                    context_str=context_str,
                    chat_history=chat_history_str,
                    query_str=prompt,
                )

            messages = chat_history + [
                ChatMessage(role=MessageRole.USER, content=formatted_prompt)
            ]

        # Attach image blocks to the last user message if images are present
        if images and messages:
            last_msg = messages[-1]
            if last_msg.role == MessageRole.USER:
                blocks: list = [TextBlock(text=last_msg.content or "")]
                for img in images:
                    blocks.append(
                        ImageBlock(
                            image=base64.b64decode(img["data"]),
                            image_mimetype=img["mimetype"],
                        )
                    )
                messages[-1] = ChatMessage(role=MessageRole.USER, blocks=blocks)

        # Phase 4: Check if model supports thinking and start generation
        thinking_enabled = getattr(llm, "thinking", False)

        if thinking_enabled:
            yield RAGChunk(
                status="thinking",
                progress=ToolProgress(
                    tool_id="rag",
                    phase="thinking",
                    message="Thinking...",
                ),
            )
        else:
            yield RAGChunk(
                status="generating",
                progress=ToolProgress(
                    tool_id="rag",
                    phase="generating",
                    message="Generating response...",
                ),
            )

        # Stream directly from LLM to access thinking tokens
        full_response = ""
        full_thinking = ""
        sent_generating_status = not thinking_enabled

        try:
            for chunk in llm.stream_chat(messages):
                # Extract thinking delta if present
                thinking_delta = chunk.additional_kwargs.get("thinking_delta")
                if thinking_delta:
                    full_thinking += thinking_delta
                    yield RAGChunk(thinking=thinking_delta)
                elif not sent_generating_status and thinking_enabled:
                    # Transition from thinking to generating
                    yield RAGChunk(
                        status="generating",
                        progress=ToolProgress(
                            tool_id="rag",
                            phase="generating",
                            message="Generating response...",
                        ),
                    )
                    sent_generating_status = True

                # Extract content delta
                if chunk.delta:
                    full_response += chunk.delta
                    yield RAGChunk(text=chunk.delta)

        except Exception as e:
            logger.error("LLM stream_chat failed: %s", e, exc_info=True)
            error_text = (
                f"**Error during generation:** `{e}`\n\n"
                "The model returned an error while generating a response. "
                "Check the server logs for details."
            )
            yield RAGChunk(text=error_text)
            full_response = error_text

        # Yield final complete chunk with sources and metrics
        yield RAGChunk(
            source_nodes=source_nodes,
            is_complete=True,
            metrics=metrics_dict,
            confidence_level="low" if is_low_confidence else "normal",
        )

        return RAGResponse(
            text=full_response, source_nodes=source_nodes, metrics=metrics_dict
        )

    def retrieve(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
        session_messages: Optional[List[Dict[str, Any]]] = None,
        progress_callback: Optional[Any] = None,
        owner: str = DEFAULT_OWNER,
    ) -> RAGRetrievalResult:
        """Perform retrieval, reranking, and confidence scoring without LLM generation.

        Extracts the retrieval pipeline from query() into a non-streaming method.
        Returns sources and metrics without calling the LLM for synthesis.
        Used by the orchestrator's rag_query tool.

        Args:
            query: User's query string.
            params: Engine parameters. Used to resolve effective params when
                engine params are not available.
            session_messages: Chat history from session storage for query condensation.
            progress_callback: Optional callable that accepts ToolProgress objects
                for reporting phase progress.
            owner: Session id whose retrieval set to use.

        Returns:
            RAGRetrievalResult with source nodes, confidence level, and metrics.
        """
        rag_set = self._sets.get(owner)
        engine = rag_set.chat_engine if rag_set is not None else None
        if rag_set is None or engine is None:
            logger.warning(
                "retrieve() called with no engine loaded for owner=%s", owner
            )
            return RAGRetrievalResult(confidence_level="none")

        retriever = rag_set.retriever
        if retriever is None:
            logger.warning("retrieve() called but no retriever available")
            return RAGRetrievalResult(confidence_level="none")

        rag_set.last_used = time.time()
        llm = rag_set.llm
        effective_params = rag_set.params or params or {}

        logger.info(f"=== RAG Retrieve Start: '{query}' ===")

        # Build chat history for query condensation
        max_turns = effective_params.get("max_history_turns")
        history = self.chat_history_service.build_history(
            session_messages,
            max_turns=max_turns,
            apply_cleaning=self.config.history_cleaning.enabled,
        )
        chat_history_str = history.to_prompt_string()

        # Phase 1: Retrieval
        if progress_callback:
            progress_callback(
                ToolProgress(
                    tool_id="rag",
                    phase="retrieving",
                    message="Searching knowledge base...",
                )
            )

        # Condense query with chat history if condenser is available
        condensed_question = query
        condenser = getattr(engine, "_condense_prompt_template", None)
        if condenser and not history.is_empty and llm is not None:
            template_str = getattr(condenser, "template", str(condenser))
            condenser_llm = create_condenser_llm(
                llm,
                provider_id=effective_params.get("provider_id"),
            )
            condensed_question = condense_query(
                llm=condenser_llm,
                chat_history=chat_history_str,
                question=query,
                prompt_template=template_str,
                fallback_on_error=True,
            )
            logger.info(f"Condensed query: {condensed_question}")

        # Retrieve context nodes
        source_nodes = retriever.retrieve(condensed_question)
        logger.info(f"Retrieved {len(source_nodes)} nodes before reranking")

        # Phase 2: Reranking
        if hasattr(engine, "_node_postprocessors") and engine._node_postprocessors:
            if progress_callback:
                progress_callback(
                    ToolProgress(
                        tool_id="rag",
                        phase="reranking",
                        message="Ranking results...",
                    )
                )

            from llama_index.core.schema import QueryBundle

            query_bundle = QueryBundle(query_str=condensed_question)

            try:
                for postprocessor in engine._node_postprocessors:
                    source_nodes = postprocessor.postprocess_nodes(
                        source_nodes, query_bundle=query_bundle
                    )
            except Exception as e:
                logger.warning(f"Postprocessor failed, using unprocessed nodes: {e}")

        # Enforce reranker_top_n limit
        reranker_top_n = effective_params.get("reranker_top_n")
        if reranker_top_n and len(source_nodes) > reranker_top_n:
            source_nodes = source_nodes[:reranker_top_n]

        logger.info(f"After reranking: {len(source_nodes)} nodes")

        # Compute retrieval metrics
        metrics = compute_retrieval_metrics(source_nodes)
        metrics.configured_top_n = effective_params.get("reranker_top_n")
        metrics_dict = metrics.to_dict()

        # Confidence scoring
        confidence_level = "normal"
        if not source_nodes:
            confidence_level = "none"
        else:
            confidence_threshold = effective_params.get("confidence_cutoff", 0.0)
            if confidence_threshold > 0:
                best_score = max(
                    (node.score for node in source_nodes if node.score is not None),
                    default=0.0,
                )
                if best_score < confidence_threshold:
                    confidence_level = "low"

        logger.info(
            f"=== RAG Retrieve Done: {len(source_nodes)} sources, "
            f"confidence={confidence_level} ==="
        )

        return RAGRetrievalResult(
            source_nodes=source_nodes,
            confidence_level=confidence_level,
            metrics=metrics_dict,
            condensed_query=condensed_question,
            num_sources=len(source_nodes),
        )

    def query_simple(
        self,
        prompt: str,
        chat_history: Optional[List[Any]] = None,
        owner: str = DEFAULT_OWNER,
    ) -> Generator[str, None, str]:
        """Execute a streaming query without source tracking.

        Simpler interface for basic chat without RAG metadata.

        Args:
            prompt: User's query prompt.
            chat_history: Optional chat history context (LlamaIndex ChatMessage list).
            owner: Session id whose retrieval set to use.

        Yields:
            Response tokens.

        Returns:
            Complete response text.

        Raises:
            RuntimeError: If engine is not loaded.
        """
        rag_set = self._sets.get(owner)
        if rag_set is None or rag_set.chat_engine is None:
            raise RuntimeError("RAG engine not loaded. Call load_engine() first.")

        response_stream = rag_set.chat_engine.stream_chat(prompt)

        full_response = ""
        for token in response_stream.response_gen:
            full_response += token
            yield token

        return full_response

    def get_llm(self, owner: str = DEFAULT_OWNER) -> Optional[LLM]:
        """Get the underlying LLM instance for an owner's set.

        Useful for direct LLM access without the full RAG pipeline.

        Args:
            owner: Session id whose set to read.

        Returns:
            LLM instance or None if no set is loaded for the owner.
        """
        rag_set = self._sets.get(owner)
        return rag_set.llm if rag_set else None

    def get_llm_from_params(self, params: Dict[str, Any]) -> LLM:
        """Create an LLM instance from parameters without loading full engine.

        Useful for operations that only need the LLM without loading the full engine.

        Args:
            params: Engine parameters.

        Returns:
            LLM instance.
        """
        return get_llm(params)

    def clear(self) -> None:
        """Release every owner's set and drop all cached module indexes.

        Used on shutdown and by tests; normal per-session churn goes through
        release(owner) or the idle sweeper.
        """
        with self._sets_lock:
            sets = list(self._sets.values())
            self._sets.clear()
        for rag_set in sets:
            self._release_set(rag_set)
        self._registry.clear()

    def __enter__(self) -> "RAGService":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit - ensures cleanup."""
        self.clear()
