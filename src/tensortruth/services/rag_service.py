"""RAG (Retrieval-Augmented Generation) service.

This service wraps the existing rag_engine module with lifecycle management
and a cleaner interface for the UI layer.
"""

import json
import logging
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

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.llms.ollama import Ollama

if TYPE_CHECKING:
    from llama_index.core.retrievers import BaseRetriever

from tensortruth.app_utils.config_schema import TensorTruthConfig
from tensortruth.rag_engine import (
    CUSTOM_CONTEXT_PROMPT_LOW_CONFIDENCE,
    CUSTOM_CONTEXT_PROMPT_NO_SOURCES,
    CUSTOM_CONTEXT_PROMPT_TEMPLATE,
    LLM_ONLY_SYSTEM_PROMPT,
    MultiIndexRetriever,
    get_base_index_dir,
    get_llm,
    load_engine_for_modules,
)
from tensortruth.utils.history_condenser import (
    condense_query,
    create_condenser_llm,
)

from .chat_history import ChatHistoryService
from .models import RAGChunk, RAGResponse
from .retrieval_metrics import compute_retrieval_metrics

logger = logging.getLogger(__name__)


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
    ):
        """Initialize RAG service.

        Args:
            config: TensorTruth configuration. If None, loads from default.
            indexes_dir: Base directory for vector indexes.
            chat_history_service: Optional ChatHistoryService instance.
                If None, one is created lazily when needed.
        """
        if config is None:
            from tensortruth.app_utils.config import load_config

            config = load_config()

        self.config = config
        self.indexes_dir = (
            Path(indexes_dir) if indexes_dir else Path(get_base_index_dir())
        )

        self._engine: Optional[CondensePlusContextChatEngine] = None
        self._current_config_hash: Optional[Tuple[Any, ...]] = None
        self._current_modules: Optional[List[str]] = None
        self._current_params: Optional[Dict[str, Any]] = None
        self._chat_history_service = chat_history_service

        # Cache frequently accessed components to avoid repeated private access
        self._llm: Optional[Ollama] = None
        self._retriever: Optional[BaseRetriever] = None

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
    ) -> None:
        """Load or reload the RAG engine with specified configuration.

        Note: Chat history is now passed to query() methods at query time,
        not here. This simplifies engine lifecycle management.

        Args:
            modules: List of module names to load.
            params: Engine parameters (model, temperature, etc).
            additional_index_paths: Optional list of additional index paths
                (session PDFs, project indexes).
        """
        # Clear existing engine first to free GPU memory
        self.clear()

        logger.info(f"Loading RAG engine for modules: {modules}")

        self._engine = load_engine_for_modules(
            selected_modules=modules,
            engine_params=params,
            additional_index_paths=additional_index_paths,
        )

        # Cache components - single point of private access (acceptable during init)
        # LlamaIndex CondensePlusContextChatEngine uses private attrs
        # Cast to expected types since we know the engine configuration
        self._llm = cast(Ollama, self._engine._llm)
        self._retriever = self._engine._retriever

        self._current_modules = modules
        self._current_params = params
        self._current_config_hash = self._compute_config_hash(
            modules, params, additional_index_paths
        )

        logger.info(f"RAG engine loaded successfully for {len(modules)} modules")

    def needs_reload(
        self,
        modules: List[str],
        params: Dict[str, Any],
        additional_index_paths: Optional[List[str]] = None,
    ) -> bool:
        """Check if engine needs to be reloaded due to config changes.

        Args:
            modules: List of module names.
            params: Engine parameters.
            additional_index_paths: Optional list of additional index paths.

        Returns:
            True if engine should be reloaded.
        """
        if self._engine is None:
            new_hash = self._compute_config_hash(
                modules, params, additional_index_paths
            )
            return new_hash is not None  # Only reload if there's something to load

        new_hash = self._compute_config_hash(modules, params, additional_index_paths)
        return new_hash != self._current_config_hash

    def is_loaded(self) -> bool:
        """Check if an engine is currently loaded.

        Returns:
            True if engine is loaded.
        """
        return self._engine is not None

    def query(
        self,
        prompt: str,
        params: Optional[Dict[str, Any]] = None,
        session_messages: Optional[List[Dict[str, Any]]] = None,
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

        Yields:
            RAGChunk with status, thinking, or text content.

        Returns:
            Final RAGResponse with complete text and sources.
        """
        # Determine LLM and retriever based on engine state
        if self._engine is not None:
            llm = self._llm
            retriever = self._retriever
            effective_params = self._current_params or params or {}
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
            yield RAGChunk(status="retrieving")

            condenser = getattr(self._engine, "_condense_prompt_template", None)
            logger.info(f"Condenser value: {condenser}")
            logger.info(f"Condenser type: {type(condenser)}")
            logger.info(
                f"_skip_condense: {getattr(self._engine, '_skip_condense', None)}"
            )

            # Condense the question with chat history if we have a condenser
            condensed_question = prompt
            if condenser and not history.is_empty:
                logger.info(f"Original query: {prompt}")
                logger.info(f"History length: {len(history.messages)} messages")

                template_str = getattr(condenser, "template", str(condenser))
                condenser_llm = create_condenser_llm(llm)

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
            assert self._engine is not None  # guaranteed when retriever exists
            if (
                hasattr(self._engine, "_node_postprocessors")
                and self._engine._node_postprocessors
            ):
                yield RAGChunk(status="reranking")

                from llama_index.core.schema import QueryBundle

                query_bundle = QueryBundle(query_str=condensed_question)

                try:
                    for postprocessor in self._engine._node_postprocessors:
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
        is_low_confidence = False
        if retriever is None:
            # No retriever — LLM-only mode with system prompt
            messages = [
                ChatMessage(role=MessageRole.SYSTEM, content=LLM_ONLY_SYSTEM_PROMPT)
            ]
            if not history.is_empty:
                messages.extend(chat_history)
            messages.append(ChatMessage(role=MessageRole.USER, content=prompt))
        elif not source_nodes:
            # Retrieval ran but returned nothing
            formatted_prompt = CUSTOM_CONTEXT_PROMPT_NO_SOURCES.format(
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
                    formatted_prompt = CUSTOM_CONTEXT_PROMPT_LOW_CONFIDENCE.format(
                        context_str=context_str,
                        chat_history=chat_history_str,
                        query_str=prompt,
                    )
                else:
                    formatted_prompt = CUSTOM_CONTEXT_PROMPT_TEMPLATE.format(
                        context_str=context_str,
                        chat_history=chat_history_str,
                        query_str=prompt,
                    )
            else:
                formatted_prompt = CUSTOM_CONTEXT_PROMPT_TEMPLATE.format(
                    context_str=context_str,
                    chat_history=chat_history_str,
                    query_str=prompt,
                )

            messages = chat_history + [
                ChatMessage(role=MessageRole.USER, content=formatted_prompt)
            ]

        # Phase 4: Check if model supports thinking and start generation
        thinking_enabled = getattr(llm, "thinking", False)

        if thinking_enabled:
            yield RAGChunk(status="thinking")
        else:
            yield RAGChunk(status="generating")

        # Stream directly from LLM to access thinking tokens
        full_response = ""
        full_thinking = ""
        sent_generating_status = not thinking_enabled

        for chunk in llm.stream_chat(messages):
            # Extract thinking delta if present
            thinking_delta = chunk.additional_kwargs.get("thinking_delta")
            if thinking_delta:
                full_thinking += thinking_delta
                yield RAGChunk(thinking=thinking_delta)
            elif not sent_generating_status and thinking_enabled:
                # Transition from thinking to generating
                yield RAGChunk(status="generating")
                sent_generating_status = True

            # Extract content delta
            if chunk.delta:
                full_response += chunk.delta
                yield RAGChunk(text=chunk.delta)

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

    def query_simple(
        self,
        prompt: str,
        chat_history: Optional[List[Any]] = None,
    ) -> Generator[str, None, str]:
        """Execute a streaming query without source tracking.

        Simpler interface for basic chat without RAG metadata.

        Args:
            prompt: User's query prompt.
            chat_history: Optional chat history context (LlamaIndex ChatMessage list).

        Yields:
            Response tokens.

        Returns:
            Complete response text.

        Raises:
            RuntimeError: If engine is not loaded.
        """
        if self._engine is None:
            raise RuntimeError("RAG engine not loaded. Call load_engine() first.")

        response_stream = self._engine.stream_chat(prompt)

        full_response = ""
        for token in response_stream.response_gen:
            full_response += token
            yield token

        return full_response

    def get_llm(self) -> Optional[Ollama]:
        """Get the underlying LLM instance from the engine.

        Useful for intent classification reuse.

        Returns:
            Ollama LLM instance or None if engine not loaded.
        """
        return self._llm

    def get_llm_from_params(self, params: Dict[str, Any]) -> Ollama:
        """Create an LLM instance from parameters without loading full engine.

        Useful for operations that only need LLM (like intent classification).

        Args:
            params: Engine parameters.

        Returns:
            Ollama LLM instance.
        """
        return get_llm(params)

    def clear(self) -> None:
        """Clear the engine and free GPU memory.

        Should be called before loading a new engine or when done.
        """
        if self._engine is not None:
            # Clear retriever cache to release GPU tensor references
            if self._retriever is not None:
                if isinstance(self._retriever, MultiIndexRetriever):
                    self._retriever.clear_cache()

            # Clear chat memory
            if hasattr(self._engine, "memory"):
                self._engine.memory.reset()

            self._engine = None
            self._llm = None
            self._retriever = None
            self._current_config_hash = None
            self._current_modules = None
            self._current_params = None

    def __enter__(self) -> "RAGService":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit - ensures cleanup."""
        self.clear()
