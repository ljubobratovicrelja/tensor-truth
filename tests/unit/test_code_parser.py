"""Tests for code block parser."""

from tensortruth.code_execution.parser import CodeBlockParser


def test_code_block_with_trailing_text():
    """Test that code blocks are detected when followed by text."""
    parser = CodeBlockParser()

    # Simulate streaming tokens
    response = "```python\nprint('hello')\n```\nThis is some explanation."

    for token in response:
        parser.feed_token(token)

    parser.finalize()
    blocks = parser.get_all_blocks()

    assert len(blocks) == 1
    assert blocks[0].language == "python"
    assert blocks[0].code == "print('hello')"


def test_code_block_without_trailing_text():
    """Test that code blocks are detected even when stream ends right after closing fence.

    This is the bug fix - when the LLM response contains only a code block with no
    follow-up text, the stream ends immediately after the closing backticks without
    a trailing newline or space. The parser must handle this case.
    """
    parser = CodeBlockParser()

    # Simulate streaming tokens - no trailing text after closing backticks
    response = "```python\nprint('hello')\n```"

    for token in response:
        parser.feed_token(token)

    parser.finalize()
    blocks = parser.get_all_blocks()

    assert len(blocks) == 1, "Code block should be detected even without trailing text"
    assert blocks[0].language == "python"
    assert blocks[0].code == "print('hello')"


def test_multiple_code_blocks():
    """Test parsing multiple code blocks in one response."""
    parser = CodeBlockParser()

    response = "```python\nprint('first')\n```\nSome text\n```py\nprint('second')\n```"

    for token in response:
        parser.feed_token(token)

    parser.finalize()
    blocks = parser.get_all_blocks()

    assert len(blocks) == 2
    assert blocks[0].code == "print('first')"
    assert blocks[1].code == "print('second')"


def test_non_python_blocks_ignored():
    """Test that non-Python code blocks are not detected."""
    parser = CodeBlockParser()

    response = "```javascript\nconsole.log('hello')\n```"

    for token in response:
        parser.feed_token(token)

    parser.finalize()
    blocks = parser.get_all_blocks()

    assert len(blocks) == 0, "Non-Python blocks should be ignored"


def test_incomplete_block_without_closing_fence():
    """Test that incomplete blocks (no closing fence) are still captured."""
    parser = CodeBlockParser()

    # Stream ends without closing fence
    response = "```python\nprint('incomplete')"

    for token in response:
        parser.feed_token(token)

    incomplete = parser.finalize()

    assert len(incomplete) == 1
    assert incomplete[0].code == "print('incomplete')"


def test_code_block_with_newline_after_fence():
    """Test code block followed by newline (common case)."""
    parser = CodeBlockParser()

    response = "```python\nprint('hello')\n```\n"

    for token in response:
        parser.feed_token(token)

    parser.finalize()
    blocks = parser.get_all_blocks()

    assert len(blocks) == 1
    assert blocks[0].code == "print('hello')"
