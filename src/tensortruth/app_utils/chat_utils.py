"""Chat history and message conversion utilities."""

from typing import List, Optional

from llama_index.core.base.llms.types import ChatMessage, MessageRole


def create_execution_message(execution_results: list) -> ChatMessage:
    """Create a ChatMessage from code execution results for LLM context.

    Args:
        execution_results: List of ExecutionResult objects

    Returns:
        ChatMessage with SYSTEM role containing formatted execution results
    """
    exec_content = "Code execution results:\n"
    for i, result in enumerate(execution_results):
        exec_content += f"\nBlock {i+1}:\n"
        if result.success:
            if result.stdout:
                exec_content += f"stdout: {result.stdout}\n"
            if result.stderr:
                exec_content += f"stderr: {result.stderr}\n"
        else:
            exec_content += f"ERROR: {result.error_message}\n"
            if result.stderr:
                exec_content += f"stderr: {result.stderr}\n"

    return ChatMessage(content=exec_content, role=MessageRole.SYSTEM)


def build_chat_history(
    session_messages: List[dict], max_messages: Optional[int] = None
) -> List[ChatMessage]:
    """Convert session messages to LlamaIndex ChatMessage format.

    Args:
        session_messages: List of message dicts from session history
        max_messages: Optional limit on number of messages to include

    Returns:
        List of ChatMessage objects (user, assistant, and code_execution)
    """
    chat_messages = []

    for msg in session_messages:
        if msg["role"] == "user":
            chat_messages.append(
                ChatMessage(content=msg["content"], role=MessageRole.USER)
            )
        elif msg["role"] == "assistant":
            chat_messages.append(
                ChatMessage(content=msg["content"], role=MessageRole.ASSISTANT)
            )
        elif msg["role"] == "code_execution":
            # Code execution results as system message for LLM context
            chat_messages.append(
                ChatMessage(content=msg["content"], role=MessageRole.SYSTEM)
            )
        # Skip command messages and other roles

    # Apply max_messages limit if specified
    if max_messages is not None and len(chat_messages) > max_messages:
        return chat_messages[-max_messages:]

    return chat_messages if chat_messages else []


def preserve_chat_history(
    session_messages: List[dict], max_messages: int = 4
) -> Optional[List[ChatMessage]]:
    """Extract and preserve recent chat history for engine loading.

    This preserves only the last N messages (default 4 = 2 conversation turns)
    to maintain immediate context without causing hallucinations.

    Args:
        session_messages: List of message dicts from session history
        max_messages: Maximum number of recent messages to preserve (default 4)

    Returns:
        List of ChatMessage objects or None if no valid messages
    """
    if not session_messages:
        return None

    try:
        chat_messages = build_chat_history(session_messages)

        if not chat_messages:
            return None

        # Preserve only the last N messages
        if len(chat_messages) > max_messages:
            return chat_messages[-max_messages:]
        else:
            return chat_messages

    except Exception as e:
        print(f"Error preserving chat history: {e}")
        return None
