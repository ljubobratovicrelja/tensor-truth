"""Unit tests for OrchestratorService (Story 11).

Tests the core orchestration flow, system prompt composition, context window
budgeting, config toggle, graceful degradation, and LLM singleton behavior.

Existing coverage in other test files (NOT duplicated here):
- test_orchestrator_tool_wrappers.py: tool wrapper factories, progress emission
- test_orchestrator_tool_discovery.py: _build_tools(), tool filtering, MCP tools
- test_orchestrator_stream.py: translate_event, stream translator, source accumulation
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tensortruth.services.models import RAGRetrievalResult, ToolProgress
from tensortruth.services.orchestrator_service import (
    CHARS_PER_TOKEN,
    CHAT_HISTORY_PCT,
    RESPONSE_BUFFER_PCT,
    ModuleDescription,
    OrchestratorEvent,
    OrchestratorService,
    build_source_reference,
)

# ---------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------


@pytest.fixture
def tool_service():
    """Mock ToolService with no MCP tools."""
    service = MagicMock()
    service.tools = []
    service.execute_tool = AsyncMock()
    return service


@pytest.fixture
def rag_service_loaded():
    """Mock RAGService that reports is_loaded() = True."""
    service = MagicMock()
    service.is_loaded.return_value = True
    service.retrieve = MagicMock(
        return_value=RAGRetrievalResult(
            source_nodes=[],
            confidence_level="none",
            condensed_query="test",
            num_sources=0,
        )
    )
    return service


@pytest.fixture
def rag_service_not_loaded():
    """Mock RAGService that reports is_loaded() = False."""
    service = MagicMock()
    service.is_loaded.return_value = False
    return service


def _create_service(
    tool_service,
    rag_service,
    module_descriptions=None,
    custom_instructions=None,
    project_metadata=None,
    context_window=4096,
    **kwargs,
) -> OrchestratorService:
    """Helper to create an OrchestratorService with default params."""
    return OrchestratorService(
        tool_service=tool_service,
        rag_service=rag_service,
        model="test-model",
        base_url="http://localhost:11434",
        context_window=context_window,
        module_descriptions=module_descriptions,
        custom_instructions=custom_instructions,
        project_metadata=project_metadata,
        **kwargs,
    )


# ---------------------------------------------------------------
# System prompt composition
# ---------------------------------------------------------------


class TestSystemPromptComposition:
    """Tests for _build_system_prompt()."""

    def test_base_prompt_always_present(self, tool_service, rag_service_not_loaded):
        """The role description should always appear in the system prompt."""
        svc = _create_service(tool_service, rag_service_not_loaded)
        tools = svc._build_tools(MagicMock())
        prompt = svc._build_system_prompt(tools)

        assert "intelligent assistant" in prompt.lower()
        assert "tools" in prompt.lower()

    def test_modules_appear_in_prompt(self, tool_service, rag_service_loaded):
        """When module descriptions are provided, they should appear in the prompt."""
        modules = [
            ModuleDescription(
                name="pytorch_docs",
                display_name="PyTorch Documentation",
                doc_type="library_doc",
            ),
            ModuleDescription(
                name="attention_paper",
                display_name="Attention Is All You Need",
                doc_type="paper",
            ),
        ]
        svc = _create_service(
            tool_service, rag_service_loaded, module_descriptions=modules
        )
        tools = svc._build_tools(MagicMock())
        prompt = svc._build_system_prompt(tools)

        assert "pytorch_docs" in prompt
        assert "PyTorch Documentation" in prompt
        assert "library_doc" in prompt
        assert "attention_paper" in prompt
        assert "Attention Is All You Need" in prompt
        assert "rag_query" in prompt.lower()

    def test_no_modules_section_when_empty(self, tool_service, rag_service_not_loaded):
        """When no modules are provided, the explicit module listing should be absent."""
        svc = _create_service(tool_service, rag_service_not_loaded)
        tools = svc._build_tools(MagicMock())
        prompt = svc._build_system_prompt(tools)

        # The explicit module block with "You have access to a knowledge base with
        # the following indexed modules:" should NOT be present
        assert "following indexed modules" not in prompt.lower()

    def test_custom_instructions_in_prompt(self, tool_service, rag_service_not_loaded):
        """Custom instructions should appear in the system prompt."""
        svc = _create_service(
            tool_service,
            rag_service_not_loaded,
            custom_instructions="Always respond in French.",
        )
        tools = svc._build_tools(MagicMock())
        prompt = svc._build_system_prompt(tools)

        assert "Always respond in French." in prompt
        assert "additional instructions" in prompt.lower()

    def test_project_metadata_in_prompt(self, tool_service, rag_service_not_loaded):
        """Project metadata should appear in the system prompt."""
        svc = _create_service(
            tool_service,
            rag_service_not_loaded,
            project_metadata="Project: ML Research\nFocus on deep learning papers.",
        )
        tools = svc._build_tools(MagicMock())
        prompt = svc._build_system_prompt(tools)

        assert "ML Research" in prompt
        assert "deep learning papers" in prompt
        assert "project context" in prompt.lower()

    def test_tool_names_listed_in_prompt(self, tool_service, rag_service_loaded):
        """The explicit tool list should be present to prevent hallucination."""
        svc = _create_service(tool_service, rag_service_loaded)
        tools = svc._build_tools(MagicMock())
        prompt = svc._build_system_prompt(tools)

        # All built-in tool names should be listed
        assert "web_search" in prompt
        assert "fetch_page" in prompt
        assert "rag_query" in prompt
        assert "ONLY these tools" in prompt


# ---------------------------------------------------------------
# Context window budgeting
# ---------------------------------------------------------------


class TestContextWindowBudgeting:
    """Tests for _budget_history()."""

    def test_short_history_passes_through(self, tool_service, rag_service_not_loaded):
        """Short history within budget should pass through unchanged."""
        svc = _create_service(tool_service, rag_service_not_loaded, context_window=8192)
        history = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
        result = svc._budget_history(
            history, "short system prompt", "short user prompt"
        )

        assert len(result) == 2
        assert result[0]["content"] == "Hi"
        assert result[1]["content"] == "Hello!"

    def test_long_history_is_truncated(self, tool_service, rag_service_not_loaded):
        """History that exceeds the budget should be truncated (oldest dropped)."""
        # Small context window to force truncation
        svc = _create_service(tool_service, rag_service_not_loaded, context_window=256)
        # Create many messages that exceed the budget
        history = []
        for i in range(20):
            history.append({"role": "user", "content": f"Message {i} " * 20})
            history.append({"role": "assistant", "content": f"Response {i} " * 20})

        result = svc._budget_history(history, "System prompt " * 10, "User prompt")

        # Should have fewer messages than the original
        assert len(result) < len(history)
        # Most recent messages should be kept (last message in result is recent)
        if result:
            # The kept messages should be from the end of the original list
            assert result[-1]["content"] in [m["content"] for m in history[-4:]]

    def test_empty_history_returns_empty(self, tool_service, rag_service_not_loaded):
        """Empty history should return empty."""
        svc = _create_service(tool_service, rag_service_not_loaded, context_window=4096)
        result = svc._budget_history([], "System prompt", "User prompt")
        assert result == []

    def test_orphaned_assistant_message_dropped(
        self, tool_service, rag_service_not_loaded
    ):
        """If truncation results in an orphaned assistant message first, drop it."""
        # Very small context window
        svc = _create_service(tool_service, rag_service_not_loaded, context_window=128)
        # History where budget would cut mid-turn
        history = [
            {"role": "user", "content": "First question " * 50},
            {"role": "assistant", "content": "First answer " * 50},
            {"role": "user", "content": "Short"},
            {"role": "assistant", "content": "Reply"},
        ]

        result = svc._budget_history(history, "Sys", "Usr")

        # If anything is kept, first message should not be an orphaned assistant
        if result:
            assert result[0]["role"] != "assistant" or len(result) <= 1

    def test_budget_respects_response_buffer(
        self, tool_service, rag_service_not_loaded
    ):
        """History budget should not eat into the response buffer allocation."""
        # 512 tokens context window
        svc = _create_service(tool_service, rag_service_not_loaded, context_window=512)
        total_chars = 512 * CHARS_PER_TOKEN
        response_buffer_chars = int(total_chars * RESPONSE_BUFFER_PCT)

        # System prompt + user prompt that takes up space
        system_prompt = "S" * 200
        user_prompt = "U" * 200

        # History that would exceed available space
        history = [
            {"role": "user", "content": "H" * 1000},
            {"role": "assistant", "content": "R" * 1000},
        ]

        result = svc._budget_history(history, system_prompt, user_prompt)

        # Calculate what's left after system + user + response buffer
        available = (
            total_chars - len(system_prompt) - len(user_prompt) - response_buffer_chars
        )
        kept_chars = sum(len(str(m.get("content", ""))) for m in result)
        # Kept chars should fit within the available or history budget
        history_budget = int(total_chars * CHAT_HISTORY_PCT)
        effective = min(history_budget, max(0, available))
        assert kept_chars <= effective + 50  # small tolerance for rounding


# ---------------------------------------------------------------
# Config toggle (_is_orchestrator_enabled)
# ---------------------------------------------------------------


class TestOrchestratorConfigToggle:
    """Tests for _is_orchestrator_enabled()."""

    def test_enabled_when_config_and_model_support(self):
        """Should return True when config enabled and model supports tools."""
        from tensortruth.api.routes.chat import _is_orchestrator_enabled

        session = {"params": {"orchestrator_enabled": True}}
        with patch(
            "tensortruth.api.routes.chat.check_tool_call_support", return_value=True
        ):
            assert _is_orchestrator_enabled(session, "qwen3:32b") is True

    def test_disabled_when_config_false(self):
        """Should return False when orchestrator_enabled is False in config."""
        from tensortruth.api.routes.chat import _is_orchestrator_enabled

        session = {"params": {"orchestrator_enabled": False}}
        assert _is_orchestrator_enabled(session, "qwen3:32b") is False

    def test_disabled_when_model_lacks_tools(self):
        """Should return False when model does not support tool-calling."""
        from tensortruth.api.routes.chat import _is_orchestrator_enabled

        session = {"params": {"orchestrator_enabled": True}}
        with patch(
            "tensortruth.api.routes.chat.check_tool_call_support", return_value=False
        ):
            assert _is_orchestrator_enabled(session, "llama3.1:8b") is False

    def test_disabled_when_no_model_name(self):
        """Should return False when model_name is None."""
        from tensortruth.api.routes.chat import _is_orchestrator_enabled

        session = {"params": {"orchestrator_enabled": True}}
        assert _is_orchestrator_enabled(session, None) is False

    def test_default_enabled_when_param_missing(self):
        """Should default to True when orchestrator_enabled key is absent."""
        from tensortruth.api.routes.chat import _is_orchestrator_enabled

        session = {"params": {}}
        with patch(
            "tensortruth.api.routes.chat.check_tool_call_support", return_value=True
        ):
            assert _is_orchestrator_enabled(session, "qwen3:32b") is True

    def test_disabled_on_capability_check_exception(self):
        """Should return False when the capability check throws an exception."""
        from tensortruth.api.routes.chat import _is_orchestrator_enabled

        session = {"params": {"orchestrator_enabled": True}}
        with patch(
            "tensortruth.api.routes.chat.check_tool_call_support",
            side_effect=Exception("Network error"),
        ):
            assert _is_orchestrator_enabled(session, "qwen3:32b") is False


# ---------------------------------------------------------------
# LLM singleton behavior
# ---------------------------------------------------------------


class TestLLMSingleton:
    """Tests for get_orchestrator_llm() caching behavior."""

    def test_returns_same_instance_for_same_params(self):
        """Should return the same LLM instance for the same model/base_url."""
        import tensortruth.core.ollama as ollama_mod
        from tensortruth.core.ollama import get_orchestrator_llm

        # Reset singleton state
        ollama_mod._orchestrator_llm_instance = None
        ollama_mod._orchestrator_llm_key = None

        with patch("llama_index.llms.ollama.Ollama") as MockOllama:
            mock_instance = MagicMock()
            MockOllama.return_value = mock_instance

            llm1 = get_orchestrator_llm("test-model", "http://localhost:11434")
            llm2 = get_orchestrator_llm("test-model", "http://localhost:11434")

            # Should be the exact same instance
            assert llm1 is llm2
            # Ollama constructor should only be called once
            assert MockOllama.call_count == 1

        # Cleanup
        ollama_mod._orchestrator_llm_instance = None
        ollama_mod._orchestrator_llm_key = None

    def test_creates_new_instance_on_model_change(self):
        """Should create a new instance when the model name changes."""
        import tensortruth.core.ollama as ollama_mod
        from tensortruth.core.ollama import get_orchestrator_llm

        # Reset singleton state
        ollama_mod._orchestrator_llm_instance = None
        ollama_mod._orchestrator_llm_key = None

        with patch("llama_index.llms.ollama.Ollama") as MockOllama:
            mock1 = MagicMock()
            mock2 = MagicMock()
            MockOllama.side_effect = [mock1, mock2]

            llm1 = get_orchestrator_llm("model-a", "http://localhost:11434")
            llm2 = get_orchestrator_llm("model-b", "http://localhost:11434")

            assert llm1 is not llm2
            assert MockOllama.call_count == 2

        # Cleanup
        ollama_mod._orchestrator_llm_instance = None
        ollama_mod._orchestrator_llm_key = None

    def test_creates_new_instance_on_url_change(self):
        """Should create a new instance when the base_url changes."""
        import tensortruth.core.ollama as ollama_mod
        from tensortruth.core.ollama import get_orchestrator_llm

        ollama_mod._orchestrator_llm_instance = None
        ollama_mod._orchestrator_llm_key = None

        with patch("llama_index.llms.ollama.Ollama") as MockOllama:
            mock1 = MagicMock()
            mock2 = MagicMock()
            MockOllama.side_effect = [mock1, mock2]

            llm1 = get_orchestrator_llm("model-a", "http://localhost:11434")
            llm2 = get_orchestrator_llm("model-a", "http://remote:11434")

            assert llm1 is not llm2
            assert MockOllama.call_count == 2

        ollama_mod._orchestrator_llm_instance = None
        ollama_mod._orchestrator_llm_key = None


# ---------------------------------------------------------------
# OrchestratorService.execute() — main orchestration flow
# ---------------------------------------------------------------


class _AwaitableHandler:
    """A handler mock that supports both stream_events() and await."""

    def __init__(self, stream_events_fn, final_result_str="Result"):
        self._stream_events_fn = stream_events_fn
        self._final_result_str = final_result_str

    def stream_events(self):
        return self._stream_events_fn()

    def __await__(self):
        async def _resolve():
            result = MagicMock()
            result.__str__ = lambda self: self._result_str
            result._result_str = self._final_result_str
            return result

        return _resolve().__await__()


def _make_awaitable_handler(stream_events_fn, final_result_str="Result"):
    """Create a mock handler that supports stream_events() and is awaitable.

    Args:
        stream_events_fn: A callable (no args) returning an async generator.
        final_result_str: The string representation of the final response.

    Returns:
        A handler object with stream_events() and __await__.
    """
    return _AwaitableHandler(stream_events_fn, final_result_str)


class TestExecuteFlow:
    """Tests for OrchestratorService.execute() with mocked FunctionAgent."""

    @pytest.mark.asyncio
    async def test_yields_token_events(self, tool_service, rag_service_not_loaded):
        """Token deltas from FunctionAgent should yield OrchestratorEvent(token=...)."""
        from llama_index.core.agent.workflow import AgentStream

        svc = _create_service(tool_service, rag_service_not_loaded)

        # AgentStream requires response and current_agent_name fields
        event1 = AgentStream(
            delta="Hello", response="", current_agent_name="orchestrator"
        )
        event2 = AgentStream(
            delta=" world", response="", current_agent_name="orchestrator"
        )

        async def stream_events():
            yield event1
            yield event2

        handler = _make_awaitable_handler(stream_events, "Hello world")

        with (
            patch(
                "tensortruth.services.orchestrator_service.LIFunctionAgent",
                return_value=MagicMock(run=MagicMock(return_value=handler)),
            ),
            patch(
                "tensortruth.services.orchestrator_service.get_orchestrator_llm",
                return_value=MagicMock(),
            ),
        ):
            events = []
            async for event in svc.execute("Hi"):
                events.append(event)

        token_events = [e for e in events if e.token is not None]
        assert len(token_events) >= 1
        token_text = "".join(e.token for e in token_events)
        assert "Hello" in token_text

    @pytest.mark.asyncio
    async def test_yields_tool_call_events(self, tool_service, rag_service_not_loaded):
        """ToolCall events should yield OrchestratorEvent(tool_call=...)."""
        from llama_index.core.agent.workflow import ToolCall

        svc = _create_service(tool_service, rag_service_not_loaded)

        tc = ToolCall(
            tool_name="web_search",
            tool_kwargs={"query": "test"},
            tool_id="tc_1",
        )

        async def stream_events():
            yield tc

        handler = _make_awaitable_handler(stream_events)

        with (
            patch(
                "tensortruth.services.orchestrator_service.LIFunctionAgent",
                return_value=MagicMock(run=MagicMock(return_value=handler)),
            ),
            patch(
                "tensortruth.services.orchestrator_service.get_orchestrator_llm",
                return_value=MagicMock(),
            ),
        ):
            events = []
            async for event in svc.execute("Search the web"):
                events.append(event)

        tool_call_events = [e for e in events if e.tool_call is not None]
        assert len(tool_call_events) >= 1
        assert tool_call_events[0].tool_call["tool"] == "web_search"
        assert tool_call_events[0].tool_call["params"] == {"query": "test"}

    @pytest.mark.asyncio
    async def test_yields_tool_call_result_events(
        self, tool_service, rag_service_not_loaded
    ):
        """ToolCallResult events should yield OrchestratorEvent(tool_call_result=...)."""
        from llama_index.core.agent.workflow import ToolCallResult as LIToolCallResult
        from llama_index.core.tools.types import ToolOutput

        svc = _create_service(tool_service, rag_service_not_loaded)

        tool_output = ToolOutput(
            tool_name="web_search",
            content="search results",
            raw_input={"query": "test"},
            raw_output="search results",
            is_error=False,
        )
        tcr = LIToolCallResult(
            tool_name="web_search",
            tool_kwargs={"query": "test"},
            tool_id="tc_1",
            tool_output=tool_output,
            return_direct=False,
        )

        async def stream_events():
            yield tcr

        handler = _make_awaitable_handler(stream_events)

        with (
            patch(
                "tensortruth.services.orchestrator_service.LIFunctionAgent",
                return_value=MagicMock(run=MagicMock(return_value=handler)),
            ),
            patch(
                "tensortruth.services.orchestrator_service.get_orchestrator_llm",
                return_value=MagicMock(),
            ),
        ):
            events = []
            async for event in svc.execute("Search the web"):
                events.append(event)

        tcr_events = [e for e in events if e.tool_call_result is not None]
        assert len(tcr_events) >= 1
        assert tcr_events[0].tool_call_result["tool"] == "web_search"
        assert tcr_events[0].tool_call_result["is_error"] is False

    @pytest.mark.asyncio
    async def test_yields_analyzing_phase_after_tool_call_result(
        self, tool_service, rag_service_not_loaded
    ):
        """After a ToolCallResult, an 'analyzing' phase should be emitted."""
        from llama_index.core.agent.workflow import ToolCallResult as LIToolCallResult
        from llama_index.core.tools.types import ToolOutput

        svc = _create_service(tool_service, rag_service_not_loaded)

        tool_output = ToolOutput(
            tool_name="fetch_page",
            content="page content",
            raw_input={"url": "https://example.com"},
            raw_output="page content",
            is_error=False,
        )
        tcr = LIToolCallResult(
            tool_name="fetch_page",
            tool_kwargs={"url": "https://example.com"},
            tool_id="tc_1",
            tool_output=tool_output,
            return_direct=False,
        )

        async def stream_events():
            yield tcr

        handler = _make_awaitable_handler(stream_events)

        with (
            patch(
                "tensortruth.services.orchestrator_service.LIFunctionAgent",
                return_value=MagicMock(run=MagicMock(return_value=handler)),
            ),
            patch(
                "tensortruth.services.orchestrator_service.get_orchestrator_llm",
                return_value=MagicMock(),
            ),
        ):
            events = []
            async for event in svc.execute("Fetch the page"):
                events.append(event)

        # Find the analyzing phase event after the tool_call_result
        phase_events = [e for e in events if e.tool_phase is not None]
        analyzing = [e for e in phase_events if e.tool_phase.phase == "analyzing"]
        assert len(analyzing) >= 1
        assert "page content" in analyzing[0].tool_phase.message.lower()

    @pytest.mark.asyncio
    async def test_error_handling_yields_error_token(
        self, tool_service, rag_service_not_loaded
    ):
        """When execution fails, should yield an error message token."""
        svc = _create_service(tool_service, rag_service_not_loaded)

        async def stream_events_error():
            raise RuntimeError("Agent crashed")
            yield  # pragma: no cover  # Make it an async generator

        handler = _make_awaitable_handler(stream_events_error)

        with (
            patch(
                "tensortruth.services.orchestrator_service.LIFunctionAgent",
                return_value=MagicMock(run=MagicMock(return_value=handler)),
            ),
            patch(
                "tensortruth.services.orchestrator_service.get_orchestrator_llm",
                return_value=MagicMock(),
            ),
        ):
            events = []
            async for event in svc.execute("Crash please"):
                events.append(event)

        token_events = [e for e in events if e.token is not None]
        assert len(token_events) >= 1
        error_text = "".join(e.token for e in token_events)
        assert "error" in error_text.lower()

    @pytest.mark.asyncio
    async def test_pending_phases_drained_after_tool_call(
        self, tool_service, rag_service_not_loaded
    ):
        """Tool phase events from wrappers should be drained after ToolCall events."""
        from llama_index.core.agent.workflow import ToolCall

        svc = _create_service(tool_service, rag_service_not_loaded)

        tc = ToolCall(
            tool_name="web_search",
            tool_kwargs={"query": "test"},
            tool_id="tc_1",
        )

        async def stream_events():
            # Before yielding the tool call, inject a pending phase
            svc._pending_phases.append(
                ToolProgress(
                    tool_id="web_search",
                    phase="searching",
                    message="Searching...",
                )
            )
            yield tc

        handler = _make_awaitable_handler(stream_events, "Done")

        with (
            patch(
                "tensortruth.services.orchestrator_service.LIFunctionAgent",
                return_value=MagicMock(run=MagicMock(return_value=handler)),
            ),
            patch(
                "tensortruth.services.orchestrator_service.get_orchestrator_llm",
                return_value=MagicMock(),
            ),
        ):
            events = []
            async for event in svc.execute("Search"):
                events.append(event)

        phase_events = [e for e in events if e.tool_phase is not None]
        # At least the orchestrator "thinking" phase + the injected "searching" phase
        assert len(phase_events) >= 2
        phase_ids = [e.tool_phase.tool_id for e in phase_events]
        assert "web_search" in phase_ids


# ---------------------------------------------------------------
# Helper method tests
# ---------------------------------------------------------------


class TestHelperMethods:
    """Tests for static/utility methods."""

    def test_to_chat_messages_converts_roles(
        self, tool_service, rag_service_not_loaded
    ):
        """Should convert role strings to LlamaIndex MessageRole."""
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "system", "content": "Be helpful"},
        ]
        messages = OrchestratorService._to_chat_messages(history)

        assert len(messages) == 3
        assert str(messages[0].role) == "MessageRole.USER"
        assert str(messages[1].role) == "MessageRole.ASSISTANT"
        assert str(messages[2].role) == "MessageRole.SYSTEM"

    def test_to_chat_messages_skips_empty_content(
        self, tool_service, rag_service_not_loaded
    ):
        """Should skip messages with empty content."""
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": ""},
            {"role": "user", "content": "World"},
        ]
        messages = OrchestratorService._to_chat_messages(history)

        # The empty content message should be skipped
        assert len(messages) == 2
        assert messages[0].content == "Hello"
        assert messages[1].content == "World"

    def test_to_chat_messages_empty_history(self, tool_service, rag_service_not_loaded):
        """Should return empty list for empty history."""
        messages = OrchestratorService._to_chat_messages([])
        assert messages == []

    def test_properties(self, tool_service, rag_service_not_loaded):
        """Verify service properties work correctly."""
        svc = _create_service(tool_service, rag_service_not_loaded)

        assert svc.model == "test-model"
        assert svc.tools == []  # No tools built yet
        assert svc.last_rag_result is None
        assert svc.last_web_sources is None


# ---------------------------------------------------------------
# OrchestratorEvent dataclass
# ---------------------------------------------------------------


class TestOrchestratorEvent:
    """Basic tests for OrchestratorEvent."""

    def test_token_event(self):
        event = OrchestratorEvent(token="hello")
        assert event.token == "hello"
        assert event.tool_call is None
        assert event.tool_call_result is None
        assert event.tool_phase is None

    def test_tool_call_event(self):
        event = OrchestratorEvent(tool_call={"tool": "web_search"})
        assert event.tool_call == {"tool": "web_search"}
        assert event.token is None

    def test_tool_phase_event(self):
        tp = ToolProgress(tool_id="rag", phase="retrieving", message="Searching...")
        event = OrchestratorEvent(tool_phase=tp)
        assert event.tool_phase is tp

    def test_empty_event(self):
        event = OrchestratorEvent()
        assert event.token is None
        assert event.tool_call is None
        assert event.tool_call_result is None
        assert event.tool_phase is None


# ---------------------------------------------------------------
# ModuleDescription and load_module_descriptions
# ---------------------------------------------------------------


class TestModuleDescription:
    """Tests for ModuleDescription dataclass."""

    def test_module_description_fields(self):
        md = ModuleDescription(
            name="pytorch_docs",
            display_name="PyTorch Documentation",
            doc_type="library_doc",
        )
        assert md.name == "pytorch_docs"
        assert md.display_name == "PyTorch Documentation"
        assert md.doc_type == "library_doc"


class TestBuildSourceReference:
    """Tests for build_source_reference()."""

    def test_empty_when_no_sources(self):
        """Should return empty string when no sources available."""
        result = build_source_reference([])
        assert result == ""

    def test_includes_rag_sources(self):
        """Should include RAG sources with knowledge base label."""
        from llama_index.core.schema import NodeWithScore, TextNode

        node = NodeWithScore(
            node=TextNode(text="Content", metadata={"display_name": "PyTorch Docs"}),
            score=0.92,
        )
        rag_result = RAGRetrievalResult(
            source_nodes=[node],
            confidence_level="normal",
            condensed_query="test",
            num_sources=1,
        )

        result = build_source_reference([], rag_result=rag_result)
        assert "[1]" in result
        assert "PyTorch Docs" in result
        assert "knowledge base" in result
        assert "0.92" in result
        assert "Source Reference" in result

    def test_includes_web_sources(self):
        """Should include web sources with URL."""
        from tensortruth.core.source import SourceNode, SourceStatus, SourceType

        web_node = SourceNode(
            id="w1",
            title="Example Page",
            source_type=SourceType.WEB,
            url="https://example.com",
            score=0.87,
            status=SourceStatus.SUCCESS,
        )

        result = build_source_reference([], web_sources=[web_node])
        assert "[1]" in result
        assert "Example Page" in result
        assert "web" in result
        assert "https://example.com" in result

    def test_combined_rag_and_web_numbering(self):
        """RAG and web sources should be numbered sequentially."""
        from llama_index.core.schema import NodeWithScore, TextNode

        from tensortruth.core.source import SourceNode, SourceStatus, SourceType

        rag_node = NodeWithScore(
            node=TextNode(text="Content", metadata={"display_name": "Docs"}),
            score=0.9,
        )
        rag_result = RAGRetrievalResult(
            source_nodes=[rag_node],
            confidence_level="normal",
            condensed_query="test",
            num_sources=1,
        )

        web_node = SourceNode(
            id="w1",
            title="Web Page",
            source_type=SourceType.WEB,
            url="https://web.com",
            score=0.8,
            status=SourceStatus.SUCCESS,
        )

        result = build_source_reference(
            [], rag_result=rag_result, web_sources=[web_node]
        )
        assert "[1]" in result  # RAG source
        assert "[2]" in result  # Web source
        assert "Docs" in result
        assert "Web Page" in result

    def test_skips_failed_web_sources(self):
        """Should skip web sources with FAILED status."""
        from tensortruth.core.source import SourceNode, SourceStatus, SourceType

        failed = SourceNode(
            id="w1",
            title="Failed",
            source_type=SourceType.WEB,
            url="https://failed.com",
            score=None,
            status=SourceStatus.FAILED,
        )
        success = SourceNode(
            id="w2",
            title="Success",
            source_type=SourceType.WEB,
            url="https://success.com",
            score=0.8,
            status=SourceStatus.SUCCESS,
        )

        result = build_source_reference([], web_sources=[failed, success])
        assert "Failed" not in result
        assert "Success" in result
        assert "[1]" in result
        assert "[2]" not in result  # Only one source should be listed

    def test_fallback_from_search_results(self):
        """When no web_sources exist, should fall back to web_search_results."""
        search_results = [
            {"title": "Page A", "url": "https://a.com", "snippet": "..."},
            {"title": "Page B", "url": "https://b.com", "snippet": "..."},
        ]
        result = build_source_reference([], web_search_results=search_results)
        assert "[1]" in result
        assert "[2]" in result
        assert "Page A" in result
        assert "snippet only" in result

    def test_snippet_only_preamble_when_unfetched(self):
        """Source reference block should include a preamble when snippet-only sources exist."""
        search_results = [
            {"title": "Page A", "url": "https://a.com", "snippet": "..."},
        ]
        result = build_source_reference([], web_search_results=search_results)
        assert "Note:" in result
        assert "snippet only" in result.lower()

    def test_no_snippet_preamble_for_fetched_sources(self):
        """Should NOT include snippet preamble when all sources are fetched."""
        from tensortruth.core.source import SourceNode, SourceStatus, SourceType

        web_node = SourceNode(
            id="w1",
            title="Fetched Page",
            source_type=SourceType.WEB,
            url="https://fetched.com",
            score=0.9,
            status=SourceStatus.SUCCESS,
        )
        result = build_source_reference([], web_sources=[web_node])
        assert "Note:" not in result

    def test_prefers_web_sources_over_search_results(self):
        """When web_sources exist, should use them instead of search_results fallback."""
        from tensortruth.core.source import SourceNode, SourceStatus, SourceType

        web_node = SourceNode(
            id="w1",
            title="Fetched Page",
            source_type=SourceType.WEB,
            url="https://fetched.com",
            score=0.9,
            status=SourceStatus.SUCCESS,
        )
        search_results = [
            {"title": "Search Result", "url": "https://search.com"},
        ]
        result = build_source_reference(
            [], web_sources=[web_node], web_search_results=search_results
        )
        assert "Fetched Page" in result
        assert "Search Result" not in result  # fallback not used


class TestLoadModuleDescriptions:
    """Tests for load_module_descriptions()."""

    def test_returns_empty_for_no_modules(self):
        """Should return empty list when no modules provided."""
        from tensortruth.services.orchestrator_service import load_module_descriptions

        result = load_module_descriptions([], MagicMock())
        assert result == []

    def test_fallback_on_import_error(self):
        """Should fall back to bare names when imports fail."""
        from tensortruth.services.orchestrator_service import load_module_descriptions

        # Mock config
        config = MagicMock()

        # The function imports get_module_display_name from app_utils.helpers.
        # If get_module_display_name raises, it falls back to bare names.
        with (
            patch(
                "tensortruth.app_utils.helpers.get_module_display_name",
                side_effect=Exception("Not found"),
            ),
            patch(
                "tensortruth.app_utils.paths.get_indexes_dir",
                return_value=MagicMock(__truediv__=MagicMock(return_value=MagicMock())),
            ),
            patch(
                "tensortruth.indexing.metadata.sanitize_model_id",
                return_value="test",
            ),
        ):
            result = load_module_descriptions(["module_a"], config)

        assert len(result) == 1
        assert result[0].name == "module_a"
        assert result[0].doc_type == "unknown"


# ---------------------------------------------------------------
# Graceful max-iterations handling
# ---------------------------------------------------------------


class TestMaxIterationsHandling:
    """Tests for graceful handling when max_iterations is reached."""

    @pytest.mark.asyncio
    async def test_max_iterations_with_results_proceeds_to_synthesis(
        self, tool_service, rag_service_not_loaded
    ):
        """When max_iterations is hit but tool results exist, synthesis should run."""
        from llama_index.core.agent.workflow import ToolCall as LIToolCall
        from llama_index.core.agent.workflow import ToolCallResult as LIToolCallResult
        from llama_index.core.tools.types import ToolOutput
        from llama_index.core.workflow.errors import WorkflowRuntimeError

        svc = _create_service(tool_service, rag_service_not_loaded)

        tool_output = ToolOutput(
            tool_name="web_search",
            content="search results about AI",
            raw_input={"query": "AI"},
            raw_output="search results about AI",
            is_error=False,
        )
        tc = LIToolCall(
            tool_name="web_search",
            tool_kwargs={"query": "AI"},
            tool_id="tc_1",
        )
        tcr = LIToolCallResult(
            tool_name="web_search",
            tool_kwargs={"query": "AI"},
            tool_id="tc_1",
            tool_output=tool_output,
            return_direct=False,
        )

        async def stream_events():
            yield tc
            yield tcr
            raise WorkflowRuntimeError("Max iterations of 10 reached!")

        handler = _make_awaitable_handler(stream_events)

        # Mock synthesis service to return a known token
        mock_synth = MagicMock()

        async def fake_synthesize(**kwargs):
            yield OrchestratorEvent(token="Synthesized answer from tool results.")

        mock_synth.synthesize = MagicMock(side_effect=fake_synthesize)

        with (
            patch(
                "tensortruth.services.orchestrator_service.LIFunctionAgent",
                return_value=MagicMock(run=MagicMock(return_value=handler)),
            ),
            patch(
                "tensortruth.services.orchestrator_service.get_orchestrator_llm",
                return_value=MagicMock(),
            ),
            patch(
                "tensortruth.services.synthesis_service.get_synthesis_service",
                return_value=mock_synth,
            ),
        ):
            events = []
            async for event in svc.execute("Tell me about AI"):
                events.append(event)

        # Should get synthesis tokens, not error tokens
        token_events = [e for e in events if e.token is not None]
        token_text = "".join(e.token for e in token_events)
        assert "Synthesized answer" in token_text
        assert "error" not in token_text.lower()

    @pytest.mark.asyncio
    async def test_max_iterations_without_results_yields_friendly_message(
        self, tool_service, rag_service_not_loaded
    ):
        """When max_iterations is hit with no tool results, show a friendly message."""
        from llama_index.core.workflow.errors import WorkflowRuntimeError

        svc = _create_service(tool_service, rag_service_not_loaded)

        async def stream_events():
            raise WorkflowRuntimeError("Max iterations of 10 reached!")
            yield  # pragma: no cover

        handler = _make_awaitable_handler(stream_events)

        with (
            patch(
                "tensortruth.services.orchestrator_service.LIFunctionAgent",
                return_value=MagicMock(run=MagicMock(return_value=handler)),
            ),
            patch(
                "tensortruth.services.orchestrator_service.get_orchestrator_llm",
                return_value=MagicMock(),
            ),
        ):
            events = []
            async for event in svc.execute("Something"):
                events.append(event)

        token_events = [e for e in events if e.token is not None]
        assert len(token_events) >= 1
        token_text = "".join(e.token for e in token_events)
        assert "iteration limit" in token_text
        # Should NOT contain the raw LlamaIndex error
        assert "Max iterations of 10 reached" not in token_text

    def test_budget_guidance_in_system_prompt(
        self, tool_service, rag_service_not_loaded
    ):
        """System prompt should contain budget guidance with max_iterations value."""
        svc = _create_service(tool_service, rag_service_not_loaded, max_iterations=15)
        tools = svc._build_tools(MagicMock())
        prompt = svc._build_system_prompt(tools)

        assert "budget" in prompt.lower()
        assert "15" in prompt
        assert "iterations" in prompt.lower()
        assert "1-2" in prompt  # Specific allocation guidance
        # Old prescriptive recipes should be gone
        assert "standard web research pattern" not in prompt
        assert "2-3 tool calls" not in prompt

    def test_default_budget_value_in_prompt(self, tool_service, rag_service_not_loaded):
        """Default max_iterations (10) should appear in the system prompt."""
        svc = _create_service(tool_service, rag_service_not_loaded)
        tools = svc._build_tools(MagicMock())
        prompt = svc._build_system_prompt(tools)

        assert "budget of 10 iterations" in prompt

    def test_tool_routing_is_not_prescriptive(
        self, tool_service, rag_service_not_loaded
    ):
        """Tool routing guidance should be generic, not prescribe tool chains."""
        svc = _create_service(tool_service, rag_service_not_loaded)
        tools = svc._build_tools(MagicMock())
        prompt = svc._build_system_prompt(tools)

        # Old prescriptive directives should be gone
        assert "Do NOT skip fetching" not in prompt
        assert "standard web research pattern" not in prompt
        assert "then call fetch_pages_batch" not in prompt
        # General routing guidance still present
        assert "knowledge base" in prompt.lower()
        assert "current events" in prompt.lower()
        # Snippet warning and fetch instruction present
        assert "snippet" in prompt.lower()
        assert "fetch" in prompt.lower()

    def test_orchestrator_llm_no_nested_options(self):
        """Orchestrator LLM additional_kwargs must not nest options inside 'options'."""
        import tensortruth.core.ollama as ollama_mod
        from tensortruth.core.ollama import get_orchestrator_llm

        ollama_mod._orchestrator_llm_instance = None
        ollama_mod._orchestrator_llm_key = None

        with patch("llama_index.llms.ollama.Ollama") as MockOllama:
            MockOllama.return_value = MagicMock()
            get_orchestrator_llm("test-model", "http://localhost:11434")

            kwargs = MockOllama.call_args[1]
            assert "options" not in kwargs["additional_kwargs"]
            assert "num_predict" in kwargs["additional_kwargs"]

        ollama_mod._orchestrator_llm_instance = None
        ollama_mod._orchestrator_llm_key = None

    def test_orchestrator_llm_no_redundant_num_ctx(self):
        """Orchestrator LLM additional_kwargs must not contain num_ctx."""
        import tensortruth.core.ollama as ollama_mod
        from tensortruth.core.ollama import get_orchestrator_llm

        ollama_mod._orchestrator_llm_instance = None
        ollama_mod._orchestrator_llm_key = None

        with patch("llama_index.llms.ollama.Ollama") as MockOllama:
            MockOllama.return_value = MagicMock()
            get_orchestrator_llm("test-model", "http://localhost:11434")

            kwargs = MockOllama.call_args[1]
            assert "num_ctx" not in kwargs["additional_kwargs"]

        ollama_mod._orchestrator_llm_instance = None
        ollama_mod._orchestrator_llm_key = None
