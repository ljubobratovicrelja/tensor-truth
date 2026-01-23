"""RAG (Retrieval-Augmented Generation) service.

This service wraps the existing rag_engine module with lifecycle management
and a cleaner interface for the UI layer.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.llms.ollama import Ollama

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
        self._current_config_hash: Optional[Tuple] = None
        self._current_modules: Optional[List[str]] = None
        self._current_params: Optional[Dict[str, Any]] = None
        self._chat_history_service = chat_history_service

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
        session_index_path: Optional[str] = None,
    ) -> Optional[Tuple]:
        """Compute configuration hash for cache invalidation.

        Args:
            modules: List of module names.
            params: Engine parameters (may contain nested dicts/lists).
            session_index_path: Optional session index path.

        Returns:
            Hashable tuple for comparison.
        """
        modules_tuple = tuple(sorted(modules)) if modules else None
        # Use JSON for consistent hashing of nested structures
        # sort_keys ensures {"a":1,"b":2} == {"b":2,"a":1}
        # default=str handles non-serializable objects gracefully
        param_hash = json.dumps(params, sort_keys=True, default=str)
        has_session_index = bool(session_index_path)

        if modules_tuple or has_session_index:
            return (modules_tuple, param_hash, has_session_index)
        return None

    def load_engine(
        self,
        modules: List[str],
        params: Dict[str, Any],
        session_index_path: Optional[str] = None,
    ) -> None:
        """Load or reload the RAG engine with specified configuration.

        Note: Chat history is now passed to query() methods at query time,
        not here. This simplifies engine lifecycle management.

        Args:
            modules: List of module names to load.
            params: Engine parameters (model, temperature, etc).
            session_index_path: Optional path to session-specific PDF index.
        """
        # Clear existing engine first to free GPU memory
        self.clear()

        logger.info(f"Loading RAG engine for modules: {modules}")

        self._engine = load_engine_for_modules(
            selected_modules=modules,
            engine_params=params,
            session_index_path=session_index_path,
        )

        self._current_modules = modules
        self._current_params = params
        self._current_config_hash = self._compute_config_hash(
            modules, params, session_index_path
        )

        logger.info(f"RAG engine loaded successfully for {len(modules)} modules")

    def needs_reload(
        self,
        modules: List[str],
        params: Dict[str, Any],
        session_index_path: Optional[str] = None,
    ) -> bool:
        """Check if engine needs to be reloaded due to config changes.

        Args:
            modules: List of module names.
            params: Engine parameters.
            session_index_path: Optional session index path.

        Returns:
            True if engine should be reloaded.
        """
        if self._engine is None:
            return True

        new_hash = self._compute_config_hash(modules, params, session_index_path)
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
        session_messages: Optional[List[Dict[str, Any]]] = None,
    ) -> Generator[RAGChunk, None, RAGResponse]:
        """Execute a streaming RAG query with thinking token support.

        Yields status updates during retrieval and thinking phases, then
        streams content tokens (and thinking tokens if model supports them).

        Args:
            prompt: User's query prompt.
            session_messages: Chat history from session storage.
                Pass session["messages"] or session_service.get_messages().
                If None, query runs without history context.

        Yields:
            RAGChunk with status, thinking, or text content.

        Returns:
            Final RAGResponse with complete text and sources.

        Raises:
            RuntimeError: If engine is not loaded.
        """
        if self._engine is None:
            raise RuntimeError("RAG engine not loaded. Call load_engine() first.")

        # Phase 1: Retrieval
        yield RAGChunk(status="retrieving")

        # Get retriever and condenser
        retriever = self._engine._retriever
        condenser = getattr(self._engine, "_condenser_prompt_template", None)

        # Build chat history from session messages using ChatHistoryService
        # Session params can override max_history_turns
        max_turns = (self._current_params or {}).get("max_history_turns")
        history = self.chat_history_service.build_history(
            session_messages,
            max_turns=max_turns,
            apply_cleaning=self.config.history_cleaning.enabled,
        )

        # Convert to LlamaIndex format for LLM
        chat_history = history.to_llama_messages()
        chat_history_str = history.to_prompt_string()

        # Condense the question with chat history if we have a condenser
        condensed_question = prompt
        if condenser and not history.is_empty:
            try:
                # Build condensed question using LLM
                condenser_prompt = condenser.format(
                    chat_history=chat_history_str,
                    question=prompt,
                )
                condensed_response = self._engine._llm.complete(condenser_prompt)
                condensed_question = str(condensed_response)
            except Exception:
                # Fall back to original prompt if condensation fails
                condensed_question = prompt

        # Retrieve context nodes
        source_nodes = retriever.retrieve(condensed_question)

        # Phase 2: Reranking (if postprocessors exist)
        if (
            hasattr(self._engine, "_node_postprocessors")
            and self._engine._node_postprocessors
        ):
            yield RAGChunk(status="reranking")

            import logging

            from llama_index.core.schema import QueryBundle

            logger = logging.getLogger(__name__)
            query_bundle = QueryBundle(query_str=condensed_question)

            try:
                for postprocessor in self._engine._node_postprocessors:
                    source_nodes = postprocessor.postprocess_nodes(
                        source_nodes, query_bundle=query_bundle
                    )
            except Exception as e:
                # Log but don't break streaming if postprocessor fails
                logger.warning(f"Postprocessor failed, using unprocessed nodes: {e}")

        # Enforce reranker_top_n limit as a safeguard
        # LlamaIndex's SentenceTransformerRerank may not always respect top_n
        reranker_top_n = (self._current_params or {}).get("reranker_top_n")
        if reranker_top_n and len(source_nodes) > reranker_top_n:
            source_nodes = source_nodes[:reranker_top_n]

        # Compute retrieval metrics from final reranked nodes
        metrics = compute_retrieval_metrics(source_nodes)
        # Add configured top_n for debugging/verification
        configured_top_n = (self._current_params or {}).get("reranker_top_n")
        metrics.configured_top_n = configured_top_n
        metrics_dict = metrics.to_dict()

        # Phase 3: Determine prompt template based on source quality
        confidence_threshold = (self._current_params or {}).get(
            "confidence_cutoff", 0.0
        )

        # Select appropriate prompt based on source availability and confidence
        if not source_nodes:
            # No sources after filtering - use no-sources prompt
            formatted_prompt = CUSTOM_CONTEXT_PROMPT_NO_SOURCES.format(
                chat_history=chat_history_str,
                query_str=prompt,
            )
        elif confidence_threshold > 0:
            # Check best score against threshold
            best_score = max(
                (node.score for node in source_nodes if node.score is not None),
                default=0.0,
            )
            context_str = "\n\n".join([n.get_content() for n in source_nodes])

            if best_score < confidence_threshold:
                # Low confidence - use low-confidence prompt
                formatted_prompt = CUSTOM_CONTEXT_PROMPT_LOW_CONFIDENCE.format(
                    context_str=context_str,
                    chat_history=chat_history_str,
                    query_str=prompt,
                )
            else:
                # Good confidence - use normal prompt
                formatted_prompt = CUSTOM_CONTEXT_PROMPT_TEMPLATE.format(
                    context_str=context_str,
                    chat_history=chat_history_str,
                    query_str=prompt,
                )
        else:
            # No threshold configured - use normal prompt
            context_str = "\n\n".join([n.get_content() for n in source_nodes])
            formatted_prompt = CUSTOM_CONTEXT_PROMPT_TEMPLATE.format(
                context_str=context_str,
                chat_history=chat_history_str,
                query_str=prompt,
            )

        # Phase 4: Check if model supports thinking and start generation
        llm = self._engine._llm
        thinking_enabled = getattr(llm, "thinking", False)

        if thinking_enabled:
            yield RAGChunk(status="thinking")
        else:
            yield RAGChunk(status="generating")

        # Build messages with the formatted prompt
        messages = chat_history + [
            ChatMessage(role=MessageRole.USER, content=formatted_prompt)
        ]

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
            source_nodes=source_nodes, is_complete=True, metrics=metrics_dict
        )

        return RAGResponse(
            text=full_response, source_nodes=source_nodes, metrics=metrics_dict
        )

    def query_simple(
        self,
        prompt: str,
        chat_history: Optional[List] = None,
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

    def query_llm_only(
        self,
        prompt: str,
        params: Dict[str, Any],
        session_messages: Optional[List[Dict[str, Any]]] = None,
    ) -> Generator[RAGChunk, None, RAGResponse]:
        """Execute a streaming query using LLM only, without RAG retrieval.

        Used when no modules or PDFs are attached to the session.
        Includes appropriate disclaimers about lack of document verification.

        Args:
            prompt: User's query prompt.
            params: Engine parameters (model, temperature, etc).
            session_messages: Chat history from session storage.
                If None, query runs without history context.

        Yields:
            RAGChunk with status or text content.

        Returns:
            Final RAGResponse with complete text (no sources).
        """
        # Create LLM instance
        llm = get_llm(params)

        # Check if model supports thinking
        thinking_enabled = getattr(llm, "thinking", False)

        if thinking_enabled:
            yield RAGChunk(status="thinking")
        else:
            yield RAGChunk(status="generating")

        # Build messages with system prompt
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=LLM_ONLY_SYSTEM_PROMPT)
        ]

        # Build chat history from session messages using ChatHistoryService
        # Session params can override max_history_turns
        max_turns = params.get("max_history_turns")
        history = self.chat_history_service.build_history(
            session_messages,
            max_turns=max_turns,
            apply_cleaning=self.config.history_cleaning.enabled,
        )

        # Add history to messages
        if not history.is_empty:
            messages.extend(history.to_llama_messages())

        # Add user prompt
        messages.append(ChatMessage(role=MessageRole.USER, content=prompt))

        # Stream from LLM
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

        # Yield final complete chunk (no sources in LLM-only mode)
        yield RAGChunk(source_nodes=[], is_complete=True)

        return RAGResponse(text=full_response, source_nodes=[], metrics=None)

    def get_llm(self) -> Optional[Ollama]:
        """Get the underlying LLM instance from the engine.

        Useful for intent classification reuse.

        Returns:
            Ollama LLM instance or None if engine not loaded.
        """
        if self._engine is None:
            return None
        # Cast to Ollama since we know it's the LLM type used in this project
        return self._engine._llm  # type: ignore[return-value]

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
            if hasattr(self._engine, "_retriever"):
                retriever = self._engine._retriever
                if isinstance(retriever, MultiIndexRetriever):
                    retriever.clear_cache()

            # Clear chat memory
            if hasattr(self._engine, "memory"):
                self._engine.memory.reset()

            self._engine = None
            self._current_config_hash = None
            self._current_modules = None
            self._current_params = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures cleanup."""
        self.clear()
        return False
