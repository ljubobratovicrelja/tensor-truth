"""Code block parser for detecting Python code in streaming LLM responses."""

from dataclasses import dataclass
from enum import Enum
from typing import List


class ParserState(Enum):
    """Parser state machine states."""

    OUTSIDE = "outside"  # Outside any code block
    IN_OPENING_FENCE = "in_opening_fence"  # Parsing opening ```
    ACCUMULATING = "accumulating"  # Inside code block, accumulating code
    IN_CLOSING_FENCE = "in_closing_fence"  # Parsing closing ```


@dataclass
class CodeBlock:
    """Represents a detected code block."""

    language: str
    code: str
    start_position: int  # Character position where block started

    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return {
            "language": self.language,
            "code": self.code,
            "start_position": self.start_position,
        }


class CodeBlockParser:
    """Incrementally parse markdown code blocks from streaming tokens.

    This parser uses a state machine to detect code blocks as tokens arrive
    from the LLM streaming response. It can handle:
    - Multiple code blocks in one response
    - Incomplete blocks during streaming
    - Various language specifiers (python, py, etc.)
    - Nested backticks within code

    Example:
        parser = CodeBlockParser()
        for token in stream:
            completed_blocks = parser.feed_token(token)
            for block in completed_blocks:
                print(f"Found {block.language} code: {block.code}")

        # Get any remaining blocks after stream ends
        final_blocks = parser.finalize()
    """

    def __init__(self):
        """Initialize parser state."""
        self.state = ParserState.OUTSIDE
        self.buffer = ""
        self.current_language = ""
        self.current_code = ""
        self.current_start_pos = 0
        self.position = 0
        self.completed_blocks: List[CodeBlock] = []

        # Backtick counters for fence detection
        self.backtick_count = 0
        self.fence_backticks = 0  # Number of backticks in opening fence

    def feed_token(self, token: str) -> List[CodeBlock]:
        """Feed a token from the stream and return any completed code blocks.

        Args:
            token: String token from LLM stream

        Returns:
            List of completed CodeBlock objects (empty if no blocks completed)
        """
        completed = []

        for char in token:
            self.buffer += char

            if self.state == ParserState.OUTSIDE:
                completed.extend(self._handle_outside(char))

            elif self.state == ParserState.IN_OPENING_FENCE:
                completed.extend(self._handle_opening_fence(char))

            elif self.state == ParserState.ACCUMULATING:
                completed.extend(self._handle_accumulating(char))

            elif self.state == ParserState.IN_CLOSING_FENCE:
                completed.extend(self._handle_closing_fence(char))

            self.position += 1

        return completed

    def _handle_outside(self, char: str) -> List[CodeBlock]:
        """Handle character when outside any code block."""
        if char == "`":
            self.backtick_count += 1
            if self.backtick_count == 3:
                # Start of opening fence
                self.state = ParserState.IN_OPENING_FENCE
                self.current_start_pos = self.position - 2
                self.fence_backticks = 3
                self.current_language = ""
                self.current_code = ""
        else:
            self.backtick_count = 0

        return []

    def _handle_opening_fence(self, char: str) -> List[CodeBlock]:
        """Handle character while parsing opening fence."""
        if char == "`":
            # More backticks in fence (`````, etc.)
            self.fence_backticks += 1
            return []

        if char == "\n":
            # End of opening fence line, start accumulating code
            self.state = ParserState.ACCUMULATING
            # Extract language from buffer after the opening fence backticks
            # The buffer contains everything up to and including the newline
            # We need to find where the backticks end and the language starts
            # Find the last position of fence backticks
            fence_str = "`" * self.fence_backticks
            fence_pos = self.buffer.rfind(fence_str)
            if fence_pos >= 0:
                # Get everything after the fence backticks, before the newline
                language_part = self.buffer[fence_pos + len(fence_str) :].strip()
                self.current_language = (
                    language_part.split()[0] if language_part else ""
                )
            else:
                self.current_language = ""
            self.buffer = ""
            self.backtick_count = (
                0  # Reset backtick counter when starting to accumulate code
            )
            return []

        # Language identifier character (python, js, etc.) - just accumulate in buffer
        return []

    def _handle_accumulating(self, char: str) -> List[CodeBlock]:
        """Handle character while accumulating code inside block."""
        if char == "`":
            self.backtick_count += 1
            if self.backtick_count == self.fence_backticks:
                # Potential closing fence
                self.state = ParserState.IN_CLOSING_FENCE
        else:
            if self.backtick_count > 0:
                # False alarm, add the backticks we counted
                self.current_code += "`" * self.backtick_count
                self.backtick_count = 0
            self.current_code += char

        return []

    def _handle_closing_fence(self, char: str) -> List[CodeBlock]:
        """Handle character while potentially in closing fence."""
        completed = []

        if char == "`":
            # More backticks
            self.backtick_count += 1
            return []

        # Check if we have a confirmed closing fence
        is_closing_fence = (
            char == "\n" or char == " " or self.backtick_count == self.fence_backticks
        )

        if is_closing_fence:
            # Confirmed closing fence - create completed block
            code = self.current_code

            # Only create block if language is Python-related
            if self._is_python_language(self.current_language):
                block = CodeBlock(
                    language=self.current_language or "python",
                    code=code.rstrip(),  # Remove trailing whitespace
                    start_position=self.current_start_pos,
                )
                completed.append(block)
                self.completed_blocks.append(block)

            # Reset state for next code block
            self._reset_state()
            return completed

        # False alarm - not a closing fence, back to accumulating
        self.current_code += "`" * self.backtick_count + char
        self.backtick_count = 0
        self.state = ParserState.ACCUMULATING
        return []

    def _reset_state(self):
        """Reset parser state for next code block."""
        self.state = ParserState.OUTSIDE
        self.current_code = ""
        self.current_language = ""
        self.backtick_count = 0
        self.buffer = ""

    def _is_python_language(self, language: str) -> bool:
        """Check if language identifier indicates Python code."""
        if not language:
            return False  # No language specified - don't execute
        python_identifiers = {"python", "py", "python3", "py3"}
        return language.lower() in python_identifiers

    def finalize(self) -> List[CodeBlock]:
        """Finalize parsing and return any incomplete blocks.

        Call this after the stream ends to get blocks that may not have
        had a closing fence yet.

        Returns:
            List of any incomplete CodeBlock objects
        """
        incomplete = []

        # If we're accumulating code but stream ended, treat as incomplete
        if self.state == ParserState.ACCUMULATING and self.current_code.strip():
            if self._is_python_language(self.current_language):
                block = CodeBlock(
                    language=self.current_language or "python",
                    code=self.current_code.rstrip(),
                    start_position=self.current_start_pos,
                )
                incomplete.append(block)
                self.completed_blocks.append(block)

        # If we're in closing fence state when stream ends, the block is complete
        elif self.state == ParserState.IN_CLOSING_FENCE:
            if self._is_python_language(self.current_language):
                block = CodeBlock(
                    language=self.current_language or "python",
                    code=self.current_code.rstrip(),
                    start_position=self.current_start_pos,
                )
                incomplete.append(block)
                self.completed_blocks.append(block)

        return incomplete

    def get_all_blocks(self) -> List[CodeBlock]:
        """Get all code blocks detected so far.

        Returns:
            List of all CodeBlock objects detected during parsing
        """
        return self.completed_blocks.copy()

    def reset(self):
        """Reset parser state for reuse."""
        self.state = ParserState.OUTSIDE
        self.buffer = ""
        self.current_language = ""
        self.current_code = ""
        self.current_start_pos = 0
        self.position = 0
        self.completed_blocks = []
        self.backtick_count = 0
        self.fence_backticks = 0
