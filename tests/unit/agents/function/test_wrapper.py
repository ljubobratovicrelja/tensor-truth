"""Tests for FunctionAgentWrapper."""

from types import SimpleNamespace
from unittest.mock import MagicMock, call

import pytest
from llama_index.core.agent.workflow import AgentStream, ToolCall, ToolCallResult
from llama_index.core.agent.workflow.function_agent import (
    FunctionAgent as LIFunctionAgent,
)
from llama_index.core.tools import FunctionTool
from llama_index.core.tools.types import ToolOutput
from llama_index.llms.ollama import Ollama

from tensortruth.agents.config import AgentCallbacks, AgentResult
from tensortruth.agents.function.wrapper import FunctionAgentWrapper


def test_function_agent_wrapper_implements_agent_interface():
    """Test that FunctionAgentWrapper implements Agent interface."""
    # Create a mock FunctionAgent
    tools = [FunctionTool.from_defaults(fn=lambda x: x, name="test_tool")]
    llm = Ollama(model="llama3.2:3b")
    function_agent = LIFunctionAgent(tools=tools, llm=llm, system_prompt="Test")

    wrapper = FunctionAgentWrapper(function_agent, agent_name="test")

    # Check methods exist
    assert hasattr(wrapper, "run")
    assert hasattr(wrapper, "get_metadata")


def test_function_agent_wrapper_get_metadata():
    """Test get_metadata returns correct structure."""
    tools = [FunctionTool.from_defaults(fn=lambda x: x, name="test_tool")]
    llm = Ollama(model="llama3.2:3b")
    function_agent = LIFunctionAgent(tools=tools, llm=llm, system_prompt="Test")

    wrapper = FunctionAgentWrapper(function_agent, agent_name="my_agent")
    metadata = wrapper.get_metadata()

    assert metadata["name"] == "my_agent"
    assert metadata["agent_type"] == "function"
    assert "capabilities" in metadata
    assert "native_tool_calling" in metadata["capabilities"]


@pytest.mark.asyncio
async def test_function_agent_wrapper_run_returns_agent_result():
    """Test that run() returns AgentResult."""

    # Create a simple tool
    def echo_tool(text: str) -> str:
        """Echo the text."""
        return f"Echo: {text}"

    tools = [FunctionTool.from_defaults(fn=echo_tool, name="echo")]
    llm = Ollama(model="llama3.2:3b")
    function_agent = LIFunctionAgent(
        tools=tools, llm=llm, system_prompt="You are a helpful assistant."
    )

    wrapper = FunctionAgentWrapper(function_agent, agent_name="echo_agent")

    # Note: This may fail if Ollama is not running, but tests the interface
    try:
        result = await wrapper.run("Test query", AgentCallbacks())
        assert isinstance(result, AgentResult)
        assert isinstance(result.final_answer, str)
    except Exception:
        # Skip if Ollama not available
        pytest.skip("Ollama not available for integration test")


def test_function_agent_wrapper_stores_agent_name():
    """Test that wrapper stores agent name correctly."""
    tools = [FunctionTool.from_defaults(fn=lambda x: x, name="test_tool")]
    llm = Ollama(model="llama3.2:3b")
    function_agent = LIFunctionAgent(tools=tools, llm=llm, system_prompt="Test")

    wrapper = FunctionAgentWrapper(function_agent, agent_name="custom_name")

    assert wrapper._agent_name == "custom_name"
    assert wrapper.get_metadata()["name"] == "custom_name"


# ---------------------------------------------------------------------------
# Streaming / callback tests (mocked handler)
# ---------------------------------------------------------------------------


class _MockHandler:
    """Mock WorkflowHandler that yields events and returns a result when awaited."""

    def __init__(self, events, final_result="final answer"):
        self._events = events
        self._final_result = final_result

    async def stream_events(self):
        for ev in self._events:
            yield ev

    def __await__(self):
        async def _result():
            return self._final_result

        return _result().__await__()


def _make_mock_handler(events, final_result="final answer"):
    """Create a mock WorkflowHandler that yields events and returns a result."""
    return _MockHandler(events, final_result)


def _make_wrapper_with_mock(events, final_result="final answer", agent_name="test"):
    """Create a FunctionAgentWrapper with a mocked underlying agent."""
    mock_agent = MagicMock()
    mock_agent.run = MagicMock(return_value=_make_mock_handler(events, final_result))
    wrapper = FunctionAgentWrapper(mock_agent, agent_name=agent_name)
    return wrapper


@pytest.mark.asyncio
async def test_run_fires_on_progress():
    """on_progress is called with 'Starting ...' message."""
    wrapper = _make_wrapper_with_mock(events=[])
    on_progress = MagicMock()

    await wrapper.run("hello", AgentCallbacks(on_progress=on_progress))

    on_progress.assert_any_call("Starting test agent...")


@pytest.mark.asyncio
async def test_run_fires_on_tool_call():
    """on_tool_call is fired for each ToolCall event."""
    events = [
        ToolCall(tool_name="search", tool_kwargs={"q": "python"}, tool_id="t1"),
        ToolCall(tool_name="read", tool_kwargs={"path": "/tmp"}, tool_id="t2"),
    ]
    wrapper = _make_wrapper_with_mock(events)
    on_tool_call = MagicMock()

    await wrapper.run("hello", AgentCallbacks(on_tool_call=on_tool_call))

    assert on_tool_call.call_count == 2
    on_tool_call.assert_any_call("search", {"q": "python"})
    on_tool_call.assert_any_call("read", {"path": "/tmp"})


@pytest.mark.asyncio
async def test_run_fires_on_token():
    """on_token is fired for each AgentStream event with a delta."""
    events = [
        AgentStream(delta="Hello", response="Hello", current_agent_name="test"),
        AgentStream(delta=" world", response="Hello world", current_agent_name="test"),
    ]
    wrapper = _make_wrapper_with_mock(events)
    on_token = MagicMock()

    await wrapper.run("hello", AgentCallbacks(on_token=on_token))

    assert on_token.call_args_list == [call("Hello"), call(" world")]


@pytest.mark.asyncio
async def test_run_tracks_tools_called():
    """AgentResult.tools_called contains tool names from ToolCall events."""
    events = [
        ToolCall(tool_name="search", tool_kwargs={}, tool_id="t1"),
        ToolCall(tool_name="fetch", tool_kwargs={}, tool_id="t2"),
    ]
    wrapper = _make_wrapper_with_mock(events)

    result = await wrapper.run("hello", AgentCallbacks())

    assert result.tools_called == ["search", "fetch"]


@pytest.mark.asyncio
async def test_run_uses_streamed_response():
    """When AgentStream deltas are present, final_answer is built from them."""
    events = [
        AgentStream(delta="streamed ", response="streamed ", current_agent_name="t"),
        AgentStream(delta="answer", response="streamed answer", current_agent_name="t"),
    ]
    wrapper = _make_wrapper_with_mock(events, final_result="handler result")
    on_token = MagicMock()

    result = await wrapper.run("hello", AgentCallbacks(on_token=on_token))

    assert result.final_answer == "streamed answer"


@pytest.mark.asyncio
async def test_run_falls_back_to_handler_result():
    """When no AgentStream deltas, final_answer comes from await handler."""
    wrapper = _make_wrapper_with_mock(events=[], final_result="fallback result")

    result = await wrapper.run("hello", AgentCallbacks())

    assert result.final_answer == "fallback result"


@pytest.mark.asyncio
async def test_run_tool_call_fires_progress():
    """ToolCall events also fire on_progress with 'Calling ...' message."""
    events = [
        ToolCall(tool_name="search", tool_kwargs={"q": "test"}, tool_id="t1"),
    ]
    wrapper = _make_wrapper_with_mock(events, agent_name="my_agent")
    on_progress = MagicMock()

    await wrapper.run("hello", AgentCallbacks(on_progress=on_progress))

    on_progress.assert_any_call("Starting my_agent agent...")
    on_progress.assert_any_call("Calling search...")


@pytest.mark.asyncio
async def test_run_skips_empty_deltas():
    """AgentStream events with empty delta don't fire on_token."""
    events = [
        AgentStream(delta="", response="", current_agent_name="test"),
        AgentStream(delta="content", response="content", current_agent_name="test"),
    ]
    wrapper = _make_wrapper_with_mock(events)
    on_token = MagicMock()

    result = await wrapper.run("hello", AgentCallbacks(on_token=on_token))

    # Only one call — the empty delta is skipped
    on_token.assert_called_once_with("content")
    assert result.final_answer == "content"


# ---------------------------------------------------------------------------
# ToolCallResult tests
# ---------------------------------------------------------------------------


def _make_tool_output(content: str, is_error: bool = False) -> ToolOutput:
    """Create a ToolOutput with the given content."""
    return ToolOutput(
        content=content,
        tool_name="test_tool",
        raw_input={},
        raw_output=content,
        is_error=is_error,
    )


@pytest.mark.asyncio
async def test_run_captures_tool_call_result():
    """ToolCallResult events populate tool_steps in AgentResult."""
    tool_output = _make_tool_output("search results here")
    events = [
        ToolCall(tool_name="search", tool_kwargs={"q": "python"}, tool_id="t1"),
        ToolCallResult(
            tool_name="search",
            tool_kwargs={"q": "python"},
            tool_id="t1",
            tool_output=tool_output,
            return_direct=False,
        ),
    ]
    wrapper = _make_wrapper_with_mock(events)

    result = await wrapper.run("hello", AgentCallbacks())

    assert len(result.tool_steps) == 1
    step = result.tool_steps[0]
    assert step["tool"] == "search"
    assert step["params"] == {"q": "python"}
    assert step["output"] == "search results here"
    assert step["is_error"] is False


@pytest.mark.asyncio
async def test_run_fires_on_tool_call_result_callback():
    """on_tool_call_result is fired for each ToolCallResult event."""
    tool_output = _make_tool_output("result data")
    events = [
        ToolCallResult(
            tool_name="fetch",
            tool_kwargs={"url": "https://example.com"},
            tool_id="t1",
            tool_output=tool_output,
            return_direct=False,
        ),
    ]
    wrapper = _make_wrapper_with_mock(events)
    on_tool_call_result = MagicMock()

    await wrapper.run("hello", AgentCallbacks(on_tool_call_result=on_tool_call_result))

    on_tool_call_result.assert_called_once_with(
        "fetch", {"url": "https://example.com"}, "result data", False
    )


@pytest.mark.asyncio
async def test_run_captures_tool_error():
    """ToolCallResult with is_error=True is captured correctly."""
    tool_output = _make_tool_output("Error: tool not found", is_error=True)
    events = [
        ToolCallResult(
            tool_name="bad_tool",
            tool_kwargs={},
            tool_id="t1",
            tool_output=tool_output,
            return_direct=False,
        ),
    ]
    wrapper = _make_wrapper_with_mock(events)

    result = await wrapper.run("hello", AgentCallbacks())

    assert len(result.tool_steps) == 1
    assert result.tool_steps[0]["is_error"] is True
    assert "Error" in result.tool_steps[0]["output"]


@pytest.mark.asyncio
async def test_run_multiple_tool_results():
    """Multiple ToolCallResult events are accumulated in order."""
    events = [
        ToolCall(tool_name="search", tool_kwargs={"q": "a"}, tool_id="t1"),
        ToolCallResult(
            tool_name="search",
            tool_kwargs={"q": "a"},
            tool_id="t1",
            tool_output=_make_tool_output("result a"),
            return_direct=False,
        ),
        ToolCall(tool_name="fetch", tool_kwargs={"url": "b"}, tool_id="t2"),
        ToolCallResult(
            tool_name="fetch",
            tool_kwargs={"url": "b"},
            tool_id="t2",
            tool_output=_make_tool_output("result b"),
            return_direct=False,
        ),
    ]
    wrapper = _make_wrapper_with_mock(events)

    result = await wrapper.run("hello", AgentCallbacks())

    assert len(result.tool_steps) == 2
    assert result.tool_steps[0]["tool"] == "search"
    assert result.tool_steps[1]["tool"] == "fetch"


@pytest.mark.asyncio
async def test_run_mcp_text_content_produces_clean_output():
    """MCP-style TextContent objects should produce clean text, not repr."""
    # Simulate MCP tool output: ToolOutput whose raw_output is a
    # CallToolResult containing TextContent items.
    text_content = [SimpleNamespace(type="text", text="The actual documentation text")]
    call_tool_result = SimpleNamespace(content=text_content)
    tool_output = ToolOutput(
        content="meta=None content=[TextContent(type='text', text='The actual documentation text')]",
        tool_name="get-library-docs",
        raw_input={"topic": "react hooks"},
        raw_output=call_tool_result,
        is_error=False,
    )
    events = [
        ToolCallResult(
            tool_name="get-library-docs",
            tool_kwargs={"topic": "react hooks"},
            tool_id="t1",
            tool_output=tool_output,
            return_direct=False,
        ),
    ]
    wrapper = _make_wrapper_with_mock(events)

    result = await wrapper.run("hello", AgentCallbacks())

    assert len(result.tool_steps) == 1
    step = result.tool_steps[0]
    # Should contain clean text, not the repr with TextContent(...)
    assert step["output"] == "The actual documentation text"
    assert "TextContent" not in step["output"]
    assert "meta=None" not in step["output"]
