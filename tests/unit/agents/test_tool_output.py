"""Tests for tensortruth.agents.tool_output extraction helpers."""

from types import SimpleNamespace

import pytest
from llama_index.core.tools import FunctionTool
from mcp.types import CallToolResult, TextContent

from tensortruth.agents.tool_output import describe_tool_call, extract_tool_text


class TestExtractToolText:
    """Tests for extract_tool_text()."""

    def test_plain_string(self):
        assert extract_tool_text("hello world") == "hello world"

    def test_empty_string(self):
        assert extract_tool_text("") == ""

    def test_list_of_text_content(self):
        """MCP tools return [TextContent(type='text', text='...')] lists."""
        items = [
            SimpleNamespace(type="text", text="first block"),
            SimpleNamespace(type="text", text="second block"),
        ]
        assert extract_tool_text(items) == "first block\nsecond block"

    def test_single_text_content_in_list(self):
        items = [SimpleNamespace(type="text", text="only one")]
        assert extract_tool_text(items) == "only one"

    def test_tool_output_with_raw_output(self):
        """ToolOutput.raw_output → CallToolResult → [TextContent]."""
        text_content = [SimpleNamespace(type="text", text="the actual text")]
        call_result = SimpleNamespace(content=text_content)
        tool_output = SimpleNamespace(raw_output=call_result, content="repr junk")
        assert extract_tool_text(tool_output) == "the actual text"

    def test_tool_output_with_string_content(self):
        """ToolOutput where .content is already a clean string."""
        tool_output = SimpleNamespace(
            raw_output="direct string", content="direct string"
        )
        assert extract_tool_text(tool_output) == "direct string"

    def test_object_with_content_string(self):
        """Object with .content that is a plain string."""
        obj = SimpleNamespace(content="some text")
        assert extract_tool_text(obj) == "some text"

    def test_object_with_text_attr(self):
        """Single TextContent-like object with .text."""
        obj = SimpleNamespace(text="single text")
        assert extract_tool_text(obj) == "single text"

    def test_depth_limit_fallback(self):
        """Deeply nested objects fall back to str() after depth > 5."""
        # Build a chain of .content nesting deeper than 5
        obj = "deep value"
        for _ in range(7):
            obj = SimpleNamespace(content=obj)
        # Should fall back to str() since depth limit is exceeded
        result = extract_tool_text(obj)
        # str(SimpleNamespace(...)) produces "namespace(...)" — not the clean string
        assert "namespace(" in result

    def test_fallback_to_str(self):
        """Non-traversable objects fall back to str()."""
        assert extract_tool_text(42) == "42"
        assert extract_tool_text(None) == "None"

    def test_list_without_text_attr_fallback(self):
        """A list of items without .text falls back to str()."""
        items = [{"key": "value"}, {"key": "value2"}]
        result = extract_tool_text(items)
        assert "key" in result  # str() representation

    def test_raw_output_preferred_over_content(self):
        """raw_output path is tried before .content path."""
        text_items = [SimpleNamespace(type="text", text="from raw")]
        tool_output = SimpleNamespace(
            raw_output=SimpleNamespace(content=text_items),
            content="meta=None content=[TextContent(type='text', text='from raw')]",
        )
        assert extract_tool_text(tool_output) == "from raw"


class TestMCPToolOutputForLLM:
    """Tests that MCP tool results produce clean text for the LLM.

    Root cause: LlamaIndex's FunctionTool._parse_tool_output doesn't know
    about MCP's CallToolResult, so it falls to str() which gives ugly repr
    like 'meta=None content=[TextContent(...)]'. This repr goes into the
    ToolOutput.blocks and is what the LLM receives — confusing it.

    The fix: wrap MCP tool functions to return clean text strings.
    """

    @pytest.mark.asyncio
    async def test_unwrapped_mcp_tool_sends_ugly_repr_to_llm(self):
        """WITHOUT wrapping, FunctionTool converts CallToolResult to ugly repr."""

        async def mock_mcp_fn(**kwargs):
            return CallToolResult(
                content=[TextContent(type="text", text="Documentation about PyTorch")],
                isError=False,
            )

        tool = FunctionTool.from_defaults(async_fn=mock_mcp_fn, name="test-mcp")
        output = await tool.acall(query="test")

        # This documents the bug: the LLM sees repr, not clean text
        assert (
            "TextContent" in output.content
        ), "Expected ugly repr (documenting the bug)"

    @pytest.mark.asyncio
    async def test_wrapped_mcp_tool_sends_clean_text_to_llm(self):
        """WITH wrapping, the LLM receives clean text instead of repr."""
        from tensortruth.agents.tool_output import wrap_mcp_tool_fn

        async def mock_mcp_fn(**kwargs):
            return CallToolResult(
                content=[TextContent(type="text", text="Documentation about PyTorch")],
                isError=False,
            )

        wrapped = wrap_mcp_tool_fn(mock_mcp_fn)
        tool = FunctionTool.from_defaults(async_fn=wrapped, name="test-mcp")
        output = await tool.acall(query="test")

        # The LLM should see clean text, not Python repr
        assert "TextContent" not in output.content
        assert output.content == "Documentation about PyTorch"

    @pytest.mark.asyncio
    async def test_wrapped_mcp_tool_handles_multiple_text_blocks(self):
        """Wrapped tool joins multiple TextContent blocks with newlines."""
        from tensortruth.agents.tool_output import wrap_mcp_tool_fn

        async def mock_mcp_fn(**kwargs):
            return CallToolResult(
                content=[
                    TextContent(type="text", text="First section"),
                    TextContent(type="text", text="Second section"),
                ],
            )

        wrapped = wrap_mcp_tool_fn(mock_mcp_fn)
        tool = FunctionTool.from_defaults(async_fn=wrapped, name="test-mcp")
        output = await tool.acall(query="test")

        assert output.content == "First section\nSecond section"

    @pytest.mark.asyncio
    async def test_wrapped_mcp_tool_passes_through_strings(self):
        """Wrapped tool passes through plain string results unchanged."""
        from tensortruth.agents.tool_output import wrap_mcp_tool_fn

        async def string_fn(**kwargs):
            return "plain string result"

        wrapped = wrap_mcp_tool_fn(string_fn)
        tool = FunctionTool.from_defaults(async_fn=wrapped, name="test-builtin")
        output = await tool.acall(query="test")

        assert output.content == "plain string result"


class TestDescribeToolCall:
    """Tests for describe_tool_call()."""

    def test_url_param_shows_domain(self):
        result = describe_tool_call(
            "fetch_page", {"url": "https://docs.pytorch.org/stable/nn.html"}
        )
        assert "Fetching" in result
        assert "docs.pytorch.org" in result

    def test_url_strips_www(self):
        result = describe_tool_call("fetch", {"url": "https://www.example.com/page"})
        assert "www." not in result
        assert "example.com" in result

    def test_urls_list_shows_count(self):
        result = describe_tool_call(
            "fetch_pages",
            {
                "urls": [
                    "https://a.com",
                    "https://b.com",
                    "https://c.com",
                    "https://d.com",
                ]
            },
        )
        assert "4" in result
        assert "Fetching" in result

    def test_query_shows_searching(self):
        result = describe_tool_call(
            "search_web", {"query": "attention is all you need"}
        )
        assert "Searching" in result
        assert "attention" in result

    def test_queries_list(self):
        result = describe_tool_call(
            "search_web", {"queries": ["pytorch batch norm", "layer norm"]}
        )
        assert "2" in result or "Searching" in result

    def test_topic_shows_looking_up(self):
        result = describe_tool_call("query-docs", {"topic": "batch normalization"})
        assert "Looking up" in result

    def test_name_shows_resolving(self):
        result = describe_tool_call("resolve-library-id", {"libraryName": "pytorch"})
        assert "Resolving" in result
        assert "pytorch" in result

    def test_empty_kwargs_fallback(self):
        result = describe_tool_call("my-tool", {})
        assert result == "Calling my-tool..."

    def test_first_short_string_catchall(self):
        result = describe_tool_call("process", {"data": "some value"})
        assert "some value" in result

    def test_long_values_truncated(self):
        long_query = "x" * 200
        result = describe_tool_call("search", {"query": long_query})
        assert len(result) < 200

    def test_only_numeric_kwargs_fallback(self):
        result = describe_tool_call("counter", {"count": 5})
        assert result == "Calling counter..."
