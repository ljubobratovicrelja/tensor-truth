"""Debug rendering functions for RAG context visualization."""

import streamlit as st

from .rendering import extract_source_metadata, get_doc_type_icon


def render_debug_context(
    context_nodes,
    best_score: float,
    confidence_threshold: float,
    synthesizer,
    low_confidence_warning: bool,
    has_real_sources: bool,
    actual_context_str: str,
    actual_formatted_prompt: str,
    complete_conversation: str,
    user_query: str,
    condensed_query: str,
):
    """Render debug context showing EXACTLY what the LLM receives.

    Shows raw context string and full prompt as plain text - no HTML.

    Args:
        context_nodes: List of NodeWithScore objects
        best_score: Best similarity score from retrieval
        confidence_threshold: Configured confidence threshold
        synthesizer: RAG synthesizer with prompt template
        low_confidence_warning: Whether low confidence warning was triggered
        has_real_sources: Whether real sources were retrieved (not fallback)
        actual_context_str: The ACTUAL raw context string sent to LLM
        actual_formatted_prompt: The ACTUAL formatted prompt with context
        complete_conversation: Complete conversation with chat history + current prompt
        user_query: The original user input
        condensed_query: The query after LLM condensing (actually used for retrieval)
    """
    st.markdown("### 🔍 DEBUG: RAG Context (What the LLM Sees)")

    # Section 1: Retrieval Query
    st.markdown("#### 🔎 Retrieval Query")

    # Show user's original input
    st.markdown("**User Input:**")
    st.text_area(
        "Original Query (read-only)",
        value=user_query,
        height=60,
        disabled=True,
        key=f"debug_user_query_{hash(user_query[:100])}",
    )

    # Show condensed query if different
    if user_query != condensed_query:
        st.markdown("**Condensed Query (used for retrieval):**")
        st.info(
            "🔄 Engine condensed the query with chat history to create a standalone search query"
        )
        st.text_area(
            "Condensed Query (read-only)",
            value=condensed_query,
            height=80,
            disabled=True,
            key=f"debug_condensed_query_{hash(condensed_query[:100])}",
        )
    else:
        st.markdown(
            "*No condensing needed (first message or query already standalone)*"
        )

    st.markdown("---")

    # Section 2: Retrieval Summary
    st.markdown("#### 📊 Retrieval Summary")
    num_nodes = len(context_nodes) if has_real_sources else 0
    status_icon = "✅" if best_score >= confidence_threshold else "❌"
    status_text = "Good" if best_score >= confidence_threshold else "Low Confidence"
    if not has_real_sources:
        status_icon = "⚠️"
        status_text = "No Sources (Fallback Mode)"

    st.markdown(f"""
- **Nodes Retrieved:** {num_nodes}
- **Best Score:** {best_score:.4f}
- **Threshold:** {confidence_threshold:.4f}
- **Status:** {status_icon} {status_text}
""")

    if has_real_sources and context_nodes:
        # Show source list
        st.markdown("**Sources:**")
        for idx, node in enumerate(context_nodes, 1):
            metadata = extract_source_metadata(node, is_node=True)
            icon = get_doc_type_icon(metadata["doc_type"])
            score = node.score if node.score else 0.0
            st.markdown(
                f"{idx}. {icon} {metadata['display_name']} (score: {score:.4f})"
            )

    st.markdown("---")

    # Section 3: ACTUAL CONTEXT STRING (RAW TEXT)
    st.markdown("#### 📝 Context String Sent to LLM")
    st.markdown(
        "*This is the EXACT raw text passed as context (nodes joined by `\\n\\n`):*"
    )

    if has_real_sources:
        st.text_area(
            "Raw Context (read-only)",
            value=actual_context_str,
            height=300,
            disabled=True,
            key=f"debug_context_{hash(actual_context_str[:100])}",
        )
    else:
        st.info("No context (fallback mode - LLM has no retrieved documents)")

    st.markdown("---")

    # Section 4: CURRENT PROMPT (just this turn)
    st.markdown("#### 📝 Current Prompt (This Turn)")
    st.markdown("*The formatted prompt for this specific query (context + query):*")

    st.text_area(
        "Current Prompt (read-only)",
        value=actual_formatted_prompt,
        height=150,
        disabled=True,
        key=f"debug_prompt_{hash(actual_formatted_prompt[:100])}",
    )

    st.markdown("---")

    # Section 5: COMPLETE CONVERSATION (history + current)
    st.markdown("#### 💬 Complete Conversation Sent to LLM")
    st.markdown("*The FULL conversation including chat history + current prompt:*")

    st.text_area(
        "Complete Conversation (read-only)",
        value=complete_conversation,
        height=300,
        disabled=True,
        key=f"debug_conversation_{hash(complete_conversation[:100])}",
    )


def render_debug_simple_llm(
    model_name: str,
    temperature: float,
    max_tokens: int,
    device: str,
    chat_history_summary: list,
    prompt: str,
    system_prompt: str = None,
):
    """Render debug information for simple LLM mode showing what the LLM receives.

    Shows model config, chat history, and current prompt as plain text.

    Args:
        model_name: Model name
        temperature: Temperature setting
        max_tokens: Max tokens setting
        device: Device setting
        chat_history_summary: List of dicts with role/content
        prompt: Current user prompt
        system_prompt: System prompt if available
    """
    st.markdown("### 🔍 DEBUG: Simple LLM Mode (No RAG)")

    # Section 1: Mode Info
    st.info(
        "**Mode:** Simple LLM (No RAG) - No document retrieval, pure LLM response based on chat history only"
    )

    # Section 2: Model Configuration
    st.markdown("#### ⚙️ Model Configuration")
    st.markdown(f"""
- **Model:** {model_name}
- **Temperature:** {temperature}
- **Max Tokens:** {max_tokens}
- **Device:** {device}
""")

    st.markdown("---")

    # Section 3: Chat History
    st.markdown("#### 💬 Chat History Sent to LLM")
    st.markdown(f"*Total messages: {len(chat_history_summary)}*")

    if chat_history_summary:
        # Build a text representation of chat history
        history_text = ""
        for idx, msg in enumerate(chat_history_summary, 1):
            role_icon = "👤" if "USER" in msg["role"].upper() else "🤖"
            history_text += (
                f"Message {idx} [{role_icon} {msg['role']}]:\n{msg['content']}\n\n"
            )

        st.text_area(
            "Chat History (read-only)",
            value=history_text.strip(),
            height=200,
            disabled=True,
            key=f"debug_llm_history_{hash(history_text[:100])}",
        )
    else:
        st.info("No chat history (first message in session)")

    st.markdown("---")

    # Section 4: Current Prompt
    st.markdown("#### 📝 Current User Prompt")
    st.text_area(
        "User Prompt (read-only)",
        value=prompt,
        height=100,
        disabled=True,
        key=f"debug_llm_prompt_{hash(prompt[:100])}",
    )

    # Section 5: System Prompt (if available)
    if system_prompt:
        st.markdown("---")
        st.markdown("#### 🔧 System Prompt")
        st.text_area(
            "System Prompt (read-only)",
            value=system_prompt,
            height=150,
            disabled=True,
            key=f"debug_llm_system_{hash(system_prompt[:100])}",
        )
