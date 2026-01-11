"""Shared response handling logic for both RAG and non-RAG chat modes."""

from typing import Any, Dict, List, Optional

import streamlit as st
from llama_index.core.base.llms.types import ChatMessage, MessageRole

from tensortruth.app_utils.chat_utils import create_execution_message
from tensortruth.app_utils.session import save_sessions


def execute_code_blocks(current_id: str, code_blocks: List[Any]) -> List[Any]:
    """Execute code blocks and return results.

    Args:
        current_id: Current session ID
        code_blocks: List of CodeBlock objects to execute

    Returns:
        List of ExecutionResult objects
    """
    if not code_blocks or not st.session_state.get("code_execution_enabled", True):
        return []

    try:
        from tensortruth.code_execution import ExecutionOrchestrator

        orchestrator = ExecutionOrchestrator()
        return orchestrator.execute_blocks(
            session_id=current_id,
            code_blocks=code_blocks,
            timeout=st.session_state.get("code_exec_timeout", 30),
            enabled=True,
            reset_session=True,  # Reset session for each new chat message
        )
    except Exception as exec_error:
        st.warning(f"Code execution error: {exec_error}")
        return []


def build_message_data(
    full_response: str,
    elapsed: float,
    thinking: Optional[str] = None,
    code_blocks: Optional[List[Any]] = None,
    execution_results: Optional[List[Any]] = None,
    sources: Optional[List[Dict]] = None,
    low_confidence: bool = False,
) -> Dict[str, Any]:
    """Build message data dictionary for session storage.

    Args:
        full_response: The assistant's response text
        elapsed: Time taken for the response
        thinking: Optional thinking/reasoning text
        code_blocks: Optional list of CodeBlock objects
        execution_results: Optional list of ExecutionResult objects
        sources: Optional list of source metadata (RAG only)
        low_confidence: Whether confidence was low (RAG only)

    Returns:
        Dictionary containing message data for session storage
    """
    message_data = {
        "role": "assistant",
        "content": full_response,
        "time_taken": elapsed,
    }

    if thinking:
        message_data["thinking"] = thinking

    if code_blocks:
        message_data["code_blocks"] = [block.to_dict() for block in code_blocks]

    if execution_results:
        message_data["execution_results"] = [
            result.to_dict() for result in execution_results
        ]

    # RAG-specific fields
    if sources is not None:
        message_data["sources"] = sources
        message_data["low_confidence"] = low_confidence

    return message_data


def save_response_to_session(
    session: Dict[str, Any],
    prompt: str,
    message_data: Dict[str, Any],
    execution_results: Optional[List[Any]],
    sessions_file: str,
    engine: Optional[Any] = None,
) -> None:
    """Save assistant response and execution results to session.

    This function handles:
    1. Adding the assistant message to session
    2. Creating and adding code execution system message
    3. Updating engine memory (if engine provided)
    4. Persisting session to disk

    Args:
        session: Current session dictionary
        prompt: Original user prompt
        message_data: Assistant message data dictionary
        execution_results: Optional list of ExecutionResult objects
        sessions_file: Path to sessions file
        engine: Optional chat engine (for memory updates in RAG mode)
    """
    # Add assistant message to session
    session["messages"].append(message_data)

    # Add code execution results as separate message for LLM context
    if execution_results:
        exec_message = create_execution_message(execution_results)

        # Update engine memory if available (RAG mode)
        if engine:
            engine._memory.put(exec_message)
            print("\n=== ADDED CODE EXECUTION TO MEMORY ===")
            print(f"Memory now has {len(list(engine._memory.get()))} messages")
            print("=" * 80)

        # Add to session history (won't be rendered, just for LLM context)
        session["messages"].append(
            {"role": "code_execution", "content": exec_message.content}
        )

    # Persist session to disk
    save_sessions(sessions_file)


def update_engine_memory(engine: Any, prompt: str, full_response: str) -> None:
    """Update engine's chat memory with user and assistant messages.

    Args:
        engine: Chat engine instance
        prompt: User's prompt
        full_response: Assistant's response
    """
    user_message = ChatMessage(content=prompt, role=MessageRole.USER)
    assistant_message = ChatMessage(content=full_response, role=MessageRole.ASSISTANT)
    engine._memory.put(user_message)
    engine._memory.put(assistant_message)

    print("\n=== UPDATED ENGINE MEMORY ===")
    print(f"Memory now has {len(list(engine._memory.get()))} messages")
    print("=" * 80)


def maybe_update_title(
    should_update: bool,
    current_id: str,
    prompt: str,
    model: str,
    sessions_file: str,
) -> None:
    """Update session title if needed.

    Args:
        should_update: Whether title needs updating
        current_id: Current session ID
        prompt: User's prompt
        model: Model name
        sessions_file: Path to sessions file
    """
    if should_update:
        from tensortruth.app_utils.session import update_title

        with st.spinner("Generating title..."):
            update_title(current_id, prompt, model, sessions_file)
