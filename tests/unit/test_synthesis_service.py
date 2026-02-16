"""Unit tests for SynthesisService.

Tests the synthesis system prompt composition, singleton caching,
and both thinking/non-thinking model handling.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tensortruth.services.orchestrator_service import ModuleDescription


def _mock_astream_chat(chunks):
    """Build a mock for astream_chat that returns an async generator.

    ``astream_chat`` is an async method that returns an async generator,
    so we need ``await astream_chat(...)`` → async-iterable-of-chunks.
    """

    async def _async_gen(*_args, **_kwargs):
        for c in chunks:
            yield c

    # astream_chat is awaited first, then iterated:
    #   async for chunk in await llm.astream_chat(msgs)
    # So the mock must be an AsyncMock whose return value is the generator.
    mock = AsyncMock()
    mock.return_value = _async_gen()
    return mock


# ---------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------


def _make_synthesis_service(thinking=False):
    """Create a SynthesisService with mocked LLM and thinking support."""
    with (
        patch(
            "tensortruth.services.synthesis_service.check_thinking_support",
            return_value=thinking,
        ),
        patch("llama_index.llms.ollama.Ollama", return_value=MagicMock()),
    ):
        from tensortruth.services.synthesis_service import SynthesisService

        return SynthesisService(
            model="test-model",
            base_url="http://localhost:11434",
            context_window=8192,
        )


# ---------------------------------------------------------------
# System prompt composition
# ---------------------------------------------------------------


class TestSynthesisSystemPrompt:
    """Tests for SynthesisService._build_system_prompt()."""

    def test_synthesis_prompt_has_role(self):
        svc = _make_synthesis_service()
        prompt = svc._build_system_prompt()

        assert "synthesize" in prompt.lower() or "synthesis" in prompt.lower()
        assert "comprehensive" in prompt.lower()

    def test_synthesis_prompt_excludes_brief(self):
        """Synthesis prompt must NOT contain 'be brief' or tool routing instructions."""
        svc = _make_synthesis_service()
        prompt = svc._build_system_prompt()

        assert "very brief one-line summary" not in prompt
        assert "ONLY these tools" not in prompt
        assert "Do NOT call any tool" not in prompt

    def test_synthesis_prompt_excludes_tool_routing(self):
        """Synthesis prompt must NOT contain tool routing guidance."""
        svc = _make_synthesis_service()
        prompt = svc._build_system_prompt()

        assert "use rag_query" not in prompt.lower()
        assert "use web_search" not in prompt.lower()
        assert "tool usage guidance" not in prompt.lower()

    def test_synthesis_prompt_includes_modules(self):
        svc = _make_synthesis_service()
        modules = [
            ModuleDescription(
                name="pytorch_docs",
                display_name="PyTorch Documentation",
                doc_type="library_doc",
            ),
        ]
        prompt = svc._build_system_prompt(module_descriptions=modules)

        assert "pytorch_docs" in prompt
        assert "PyTorch Documentation" in prompt
        assert "library_doc" in prompt

    def test_synthesis_prompt_includes_custom_instructions(self):
        svc = _make_synthesis_service()
        prompt = svc._build_system_prompt(
            custom_instructions="Always respond in French."
        )

        assert "Always respond in French." in prompt

    def test_synthesis_prompt_includes_project_metadata(self):
        svc = _make_synthesis_service()
        prompt = svc._build_system_prompt(project_metadata="Project: ML Research")

        assert "ML Research" in prompt

    def test_synthesis_prompt_includes_synthesis_guidance(self):
        svc = _make_synthesis_service()
        prompt = svc._build_system_prompt()

        assert "cite" in prompt.lower()
        assert "markdown" in prompt.lower()

    def test_synthesis_prompt_includes_snippet_only_guidance(self):
        """Synthesis prompt must instruct LLM about snippet-only source handling."""
        svc = _make_synthesis_service()
        prompt = svc._build_system_prompt()

        assert "snippet only" in prompt.lower()
        assert "not retrieved" in prompt.lower() or "not available" in prompt.lower()

    def test_synthesis_prompt_no_modules_when_empty(self):
        svc = _make_synthesis_service()
        prompt = svc._build_system_prompt(module_descriptions=[])

        assert "knowledge modules" not in prompt.lower()


# ---------------------------------------------------------------
# Singleton caching
# ---------------------------------------------------------------


class TestSynthesisSingleton:
    """Tests for get_synthesis_service() caching behavior."""

    def test_returns_same_instance_for_same_params(self):
        import tensortruth.services.synthesis_service as mod

        # Reset singleton state
        mod._synthesis_service = None
        mod._synthesis_service_key = None

        with (
            patch(
                "tensortruth.services.synthesis_service.check_thinking_support",
                return_value=False,
            ),
            patch("llama_index.llms.ollama.Ollama", return_value=MagicMock()),
        ):
            svc1 = mod.get_synthesis_service(
                "test-model", "http://localhost:11434", 8192
            )
            svc2 = mod.get_synthesis_service(
                "test-model", "http://localhost:11434", 8192
            )

            assert svc1 is svc2

        # Cleanup
        mod._synthesis_service = None
        mod._synthesis_service_key = None

    def test_creates_new_instance_on_model_change(self):
        import tensortruth.services.synthesis_service as mod

        mod._synthesis_service = None
        mod._synthesis_service_key = None

        with (
            patch(
                "tensortruth.services.synthesis_service.check_thinking_support",
                return_value=False,
            ),
            patch("llama_index.llms.ollama.Ollama", return_value=MagicMock()),
        ):
            svc1 = mod.get_synthesis_service("model-a", "http://localhost:11434", 8192)
            svc2 = mod.get_synthesis_service("model-b", "http://localhost:11434", 8192)

            assert svc1 is not svc2

        mod._synthesis_service = None
        mod._synthesis_service_key = None

    def test_creates_new_instance_on_url_change(self):
        import tensortruth.services.synthesis_service as mod

        mod._synthesis_service = None
        mod._synthesis_service_key = None

        with (
            patch(
                "tensortruth.services.synthesis_service.check_thinking_support",
                return_value=False,
            ),
            patch("llama_index.llms.ollama.Ollama", return_value=MagicMock()),
        ):
            svc1 = mod.get_synthesis_service("model-a", "http://localhost:11434", 8192)
            svc2 = mod.get_synthesis_service("model-a", "http://remote:11434", 8192)

            assert svc1 is not svc2

        mod._synthesis_service = None
        mod._synthesis_service_key = None

    def test_synthesis_llm_no_nested_options(self):
        """Synthesis LLM additional_kwargs must not nest options inside 'options'."""
        with (
            patch(
                "tensortruth.services.synthesis_service.check_thinking_support",
                return_value=False,
            ),
            patch("llama_index.llms.ollama.Ollama") as MockOllama,
        ):
            MockOllama.return_value = MagicMock()
            from tensortruth.services.synthesis_service import SynthesisService

            SynthesisService(
                model="test-model",
                base_url="http://localhost:11434",
                context_window=8192,
            )

            kwargs = MockOllama.call_args[1]
            assert "options" not in kwargs["additional_kwargs"]
            assert "num_predict" in kwargs["additional_kwargs"]

    def test_synthesis_llm_no_redundant_num_ctx(self):
        """Synthesis LLM additional_kwargs must not contain num_ctx."""
        with (
            patch(
                "tensortruth.services.synthesis_service.check_thinking_support",
                return_value=False,
            ),
            patch("llama_index.llms.ollama.Ollama") as MockOllama,
        ):
            MockOllama.return_value = MagicMock()
            from tensortruth.services.synthesis_service import SynthesisService

            SynthesisService(
                model="test-model",
                base_url="http://localhost:11434",
                context_window=8192,
            )

            kwargs = MockOllama.call_args[1]
            assert "num_ctx" not in kwargs["additional_kwargs"]

    def test_creates_new_instance_on_context_window_change(self):
        """context_window is part of the cache key — changing it rebuilds."""
        import tensortruth.services.synthesis_service as mod

        mod._synthesis_service = None
        mod._synthesis_service_key = None

        with (
            patch(
                "tensortruth.services.synthesis_service.check_thinking_support",
                return_value=False,
            ),
            patch("llama_index.llms.ollama.Ollama", return_value=MagicMock()),
        ):
            svc1 = mod.get_synthesis_service("model-a", "http://localhost:11434", 8192)
            svc2 = mod.get_synthesis_service("model-a", "http://localhost:11434", 32768)

            assert svc1 is not svc2

        mod._synthesis_service = None
        mod._synthesis_service_key = None


# ---------------------------------------------------------------
# Thinking / non-thinking model support
# ---------------------------------------------------------------


class TestThinkingSupport:
    """Tests for thinking model handling."""

    def test_thinking_supported_property_true(self):
        svc = _make_synthesis_service(thinking=True)
        assert svc.thinking_supported is True

    def test_thinking_supported_property_false(self):
        svc = _make_synthesis_service(thinking=False)
        assert svc.thinking_supported is False


# ---------------------------------------------------------------
# Synthesize streaming
# ---------------------------------------------------------------


class TestSynthesize:
    """Tests for SynthesisService.synthesize()."""

    @pytest.mark.asyncio
    async def test_synthesize_yields_token_events(self):
        """Non-thinking model should yield token events via async streaming."""
        svc = _make_synthesis_service(thinking=False)

        chunk1 = MagicMock()
        chunk1.additional_kwargs = {}
        chunk1.delta = "Hello"

        chunk2 = MagicMock()
        chunk2.additional_kwargs = {}
        chunk2.delta = " world"

        svc._llm.astream_chat = _mock_astream_chat([chunk1, chunk2])

        events = []
        async for event in svc.synthesize(
            prompt="What is backpropagation?",
            chat_history=[],
            tool_results=["[rag_query (OK)]\nBackpropagation is..."],
        ):
            events.append(event)

        token_events = [e for e in events if e.token is not None]
        assert len(token_events) == 2
        assert token_events[0].token == "Hello"
        assert token_events[1].token == " world"

    @pytest.mark.asyncio
    async def test_synthesize_yields_thinking_events(self):
        """Thinking model should yield thinking events followed by token events."""
        svc = _make_synthesis_service(thinking=True)

        think_chunk = MagicMock()
        think_chunk.additional_kwargs = {"thinking_delta": "Let me reason..."}
        think_chunk.delta = ""

        content_chunk = MagicMock()
        content_chunk.additional_kwargs = {}
        content_chunk.delta = "The answer is..."

        svc._llm.astream_chat = _mock_astream_chat([think_chunk, content_chunk])

        events = []
        async for event in svc.synthesize(
            prompt="What is backpropagation?",
            chat_history=[],
            tool_results=["[rag_query (OK)]\nBackpropagation is..."],
        ):
            events.append(event)

        thinking_events = [e for e in events if e.thinking is not None]
        token_events = [e for e in events if e.token is not None]

        assert len(thinking_events) == 1
        assert thinking_events[0].thinking == "Let me reason..."
        assert len(token_events) == 1
        assert token_events[0].token == "The answer is..."

    @pytest.mark.asyncio
    async def test_synthesize_emits_progress_thinking(self):
        """Should emit 'thinking' progress when model supports thinking."""
        svc = _make_synthesis_service(thinking=True)
        svc._llm.astream_chat = _mock_astream_chat([])

        progress_events = []

        def emitter(tp):
            progress_events.append(tp)

        async for _ in svc.synthesize(
            prompt="test",
            chat_history=[],
            tool_results=[],
            progress_emitter=emitter,
        ):
            pass

        assert any(p.phase == "thinking" for p in progress_events)

    @pytest.mark.asyncio
    async def test_synthesize_emits_progress_generating(self):
        """Should emit 'generating' progress when model does NOT support thinking."""
        svc = _make_synthesis_service(thinking=False)
        svc._llm.astream_chat = _mock_astream_chat([])

        progress_events = []

        def emitter(tp):
            progress_events.append(tp)

        async for _ in svc.synthesize(
            prompt="test",
            chat_history=[],
            tool_results=[],
            progress_emitter=emitter,
        ):
            pass

        assert any(p.phase == "generating" for p in progress_events)

    @pytest.mark.asyncio
    async def test_synthesize_handles_error(self):
        """Should yield error token on LLM failure."""
        svc = _make_synthesis_service(thinking=False)
        svc._llm.astream_chat = AsyncMock(side_effect=RuntimeError("LLM crashed"))

        events = []
        async for event in svc.synthesize(
            prompt="test",
            chat_history=[],
            tool_results=[],
        ):
            events.append(event)

        token_events = [e for e in events if e.token is not None]
        assert len(token_events) == 1
        assert "error" in token_events[0].token.lower()

    @pytest.mark.asyncio
    async def test_synthesize_passes_context_to_system_prompt(self):
        """Module descriptions, custom instructions, and project metadata
        should be passed through to the system prompt and into the messages."""
        svc = _make_synthesis_service(thinking=False)

        # Capture the messages passed to astream_chat
        captured_messages = []

        async def _capture_gen(*_a, **_k):
            return
            yield  # pragma: no cover — makes this an async generator

        async def _capture_and_return(messages):
            captured_messages.extend(messages)
            return _capture_gen()

        svc._llm.astream_chat = _capture_and_return

        modules = [
            ModuleDescription(
                name="test_mod",
                display_name="Test Module",
                doc_type="paper",
            )
        ]

        async for _ in svc.synthesize(
            prompt="test",
            chat_history=[],
            tool_results=["[rag_query (OK)]\nSome results"],
            module_descriptions=modules,
            custom_instructions="Be concise.",
            project_metadata="Project: Test",
        ):
            pass

        # The first message should be the system prompt
        assert len(captured_messages) >= 1
        system_content = str(captured_messages[0].content)

        assert "test_mod" in system_content
        assert "Test Module" in system_content
        assert "Be concise." in system_content
        assert "Project: Test" in system_content

    @pytest.mark.asyncio
    async def test_synthesize_includes_source_reference_in_user_message(self):
        """Source reference block should appear in the synthesis user message."""
        svc = _make_synthesis_service(thinking=False)

        captured_messages = []

        async def _capture_gen(*_a, **_k):
            return
            yield  # pragma: no cover

        async def _capture_and_return(messages):
            captured_messages.extend(messages)
            return _capture_gen()

        svc._llm.astream_chat = _capture_and_return

        source_ref = (
            "--- Source Reference ---\n"
            '[1] "Docs" (knowledge base, score: 0.92)\n'
            "--- End Source Reference ---"
        )

        async for _ in svc.synthesize(
            prompt="test",
            chat_history=[],
            tool_results=["[rag_query (OK)]\nResults"],
            source_reference=source_ref,
        ):
            pass

        # Find the user message (last one)
        user_msgs = [m for m in captured_messages if str(m.role) == "MessageRole.USER"]
        assert len(user_msgs) >= 1
        user_content = str(user_msgs[-1].content)
        assert "Source Reference" in user_content
        assert '[1] "Docs"' in user_content

    @pytest.mark.asyncio
    async def test_synthesize_omits_source_reference_when_none(self):
        """When no source_reference is provided, the block should not appear."""
        svc = _make_synthesis_service(thinking=False)

        captured_messages = []

        async def _capture_gen(*_a, **_k):
            return
            yield  # pragma: no cover

        async def _capture_and_return(messages):
            captured_messages.extend(messages)
            return _capture_gen()

        svc._llm.astream_chat = _capture_and_return

        async for _ in svc.synthesize(
            prompt="test",
            chat_history=[],
            tool_results=["Some results"],
        ):
            pass

        user_msgs = [m for m in captured_messages if str(m.role) == "MessageRole.USER"]
        assert len(user_msgs) >= 1
        user_content = str(user_msgs[-1].content)
        assert "Source Reference ---" not in user_content
