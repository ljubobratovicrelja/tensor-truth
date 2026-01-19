"""Rendering utilities for Streamlit UI components."""

import html

import streamlit as st

from tensortruth import convert_latex_delimiters


def _create_scrollable_box(
    content: str, label: str, css_class: str, escape_html: bool = True
) -> str:
    """Create HTML for a scrollable box with fixed height.

    Args:
        content: The content to display inside the box
        label: The label/title for the box
        css_class: CSS class name for styling
        escape_html: Whether to escape HTML in content (default: True for security)

    Returns:
        HTML string with styled scrollable container
    """
    # Escape content by default to prevent XSS
    if escape_html:
        content = html.escape(content)

    return f"""
<div class="tt-scrollable-box {css_class}">
<strong>{label}</strong>
<div style="margin-top: 0.5rem;">
{content}
</div>
</div>"""


def render_thinking(thinking_text: str, placeholder=None):
    """Render thinking/reasoning content with consistent formatting.

    Args:
        thinking_text: The thinking content to display
        placeholder: Optional Streamlit placeholder to render into (uses st.markdown if None)
    """
    html_content = _create_scrollable_box(
        content=convert_latex_delimiters(thinking_text),
        label="🧠 Reasoning:",
        css_class="tt-thinking-box",
        escape_html=False,  # LaTeX/markdown content, already sanitized
    )

    if placeholder:
        placeholder.markdown(html_content, unsafe_allow_html=True)
    else:
        st.markdown(html_content, unsafe_allow_html=True)


def render_web_search_progress(progress_text: str, placeholder=None):
    """Render web search progress with consistent formatting.

    Args:
        progress_text: The progress updates to display
        placeholder: Optional Streamlit placeholder to render into (uses st.markdown if None)
    """
    html_content = _create_scrollable_box(
        content=progress_text,
        label="🔍 Web Search Progress:",
        css_class="tt-web-search-box",
    )

    if placeholder:
        placeholder.markdown(html_content, unsafe_allow_html=True)
    else:
        st.markdown(html_content, unsafe_allow_html=True)


def get_doc_type_icon(doc_type: str) -> str:
    """Get icon emoji for document type.

    Args:
        doc_type: Type of document (paper, library_doc, uploaded_pdf, book, etc.)

    Returns:
        Emoji string for the document type
    """
    icon_map = {
        "paper": "📄",
        "library_doc": "📚",
        "uploaded_pdf": "📎",
        "book": "📖",
    }
    return icon_map.get(doc_type, "📄")


def extract_source_metadata(source_or_node, is_node: bool = False) -> dict:
    """Extract metadata from a source dict or NodeWithScore.

    Args:
        source_or_node: Either a source dict from message history or NodeWithScore
        is_node: If True, extract from node.metadata, else from dict directly

    Returns:
        Dictionary with standardized metadata fields
    """
    if is_node:
        # Extract from NodeWithScore
        metadata = source_or_node.metadata
        score = float(source_or_node.score) if source_or_node.score else 0.0
        fname = metadata.get("file_name", "Unknown")
        display_name = metadata.get("display_name", fname)
        source_url = metadata.get("source_url")
        authors = metadata.get("authors")
        doc_type = metadata.get("doc_type", "unknown")
    else:
        # Extract from dict (message history)
        fname = source_or_node.get("file", "Unknown")
        display_name = source_or_node.get("display_name", fname)
        source_url = source_or_node.get("source_url")
        authors = source_or_node.get("authors")
        doc_type = source_or_node.get("doc_type", "unknown")
        score = source_or_node.get("score", 0.0)

    return {
        "file": fname,
        "display_name": display_name,
        "source_url": source_url,
        "authors": authors,
        "doc_type": doc_type,
        "score": score,
    }


def render_source_item(metadata: dict):
    """Render a single source item with icon, name, and score.

    Args:
        metadata: Source metadata dict from extract_source_metadata()
    """
    icon = get_doc_type_icon(metadata["doc_type"])
    label = metadata["display_name"]
    score = metadata["score"]
    source_url = metadata["source_url"]

    if source_url:
        st.caption(f"{icon} [{label}]({source_url}) ({score:.2f})")
    else:
        st.caption(f"{icon} {label} ({score:.2f})")


def render_source_expander(sources_or_nodes, is_nodes: bool = False):
    """Render sources in an expander widget.

    Args:
        sources_or_nodes: List of source dicts or NodeWithScore objects
        is_nodes: If True, treat as NodeWithScore objects, else as dicts
    """
    if not sources_or_nodes:
        return

    with st.expander("📚 Sources"):
        for item in sources_or_nodes:
            metadata = extract_source_metadata(item, is_node=is_nodes)
            render_source_item(metadata)


def render_message_metadata(
    message: dict, params: dict, modules: list, has_pdf_index: bool = False
) -> str:
    """Generate metadata caption for a message.

    Shows time and status indicators (PDF mode, soft fallback).

    Args:
        message: Message dictionary from session history
        params: Session parameters dictionary
        modules: List of active module names
        has_pdf_index: Whether session has PDF documents indexed

    Returns:
        Formatted metadata string
    """
    if "time_taken" not in message:
        return ""

    time_str = f"⏱️ {message['time_taken']:.2f}s"
    has_sources = "sources" in message and message.get("sources")
    has_rag = bool(modules) or has_pdf_index

    # Check different response types
    if message.get("low_confidence", False) and not has_sources and has_rag:
        # Low confidence with no sources in RAG mode
        return f"{time_str} | ⚠️ No Sources"
    elif message.get("low_confidence", False):
        # Low confidence with sources
        return f"{time_str} | ⚠️ Low Confidence"
    elif message["role"] == "assistant" and not has_sources and has_rag:
        # Has RAG but no sources = RAG failure
        return f"{time_str} | ⚠️ No Sources"
    elif message["role"] == "assistant" and not has_rag:
        # No RAG mode
        return f"{time_str} | 🔴 No RAG"
    else:
        return time_str


def render_message_footer(
    sources_or_nodes=None,
    is_nodes: bool = False,
    time_taken: float = None,
    low_confidence: bool = False,
    modules: list = None,
    has_pdf_index: bool = False,
):
    """Render the footer section of a message with sources and metadata.

    Args:
        sources_or_nodes: List of source dicts or NodeWithScore objects (optional)
        is_nodes: If True, treat as NodeWithScore objects
        time_taken: Time in seconds (optional)
        low_confidence: Whether this is a low confidence response
        modules: List of active module names (for determining No RAG mode)
        has_pdf_index: Whether session has PDF documents indexed
    """
    meta_cols = st.columns([3, 1])
    has_rag = bool(modules) or has_pdf_index

    with meta_cols[0]:
        if sources_or_nodes:
            render_source_expander(sources_or_nodes, is_nodes=is_nodes)

    with meta_cols[1]:
        if time_taken is not None:
            if low_confidence and not sources_or_nodes and has_rag:
                # Low confidence with no sources in RAG mode
                st.caption(f"⏱️ {time_taken:.2f}s | ⚠️ No Sources")
            elif low_confidence:
                # Low confidence with sources
                st.caption(f"⏱️ {time_taken:.2f}s | ⚠️ Low Confidence")
            elif not sources_or_nodes and has_rag:
                # Has RAG but no sources = RAG failure (not low confidence mode)
                st.caption(f"⏱️ {time_taken:.2f}s | ⚠️ No Sources")
            elif not has_rag:
                # No RAG mode
                st.caption(f"⏱️ {time_taken:.2f}s | 🔴 No RAG")
            else:
                st.caption(f"⏱️ {time_taken:.2f}s")


def render_low_confidence_warning(
    best_score: float, confidence_threshold: float, has_sources: bool = True
):
    """Render low confidence warning banner.

    Args:
        best_score: Best similarity score from sources
        confidence_threshold: Configured threshold
        has_sources: Whether any sources were retrieved
    """
    if not has_sources:
        st.warning(
            "⚠️ **NO SOURCES RETRIEVED** - "
            "Response based on general knowledge only, "
            "not your indexed documents."
        )
    else:
        st.warning(
            f"⚠️ **Low Confidence Match** - Best similarity score ({best_score:.2f}) "
            f"is below your threshold ({confidence_threshold:.2f}). "
            "The answer may not be reliable. Consider lowering the threshold "
            "or rephrasing your query."
        )


def render_debug_from_stored_data(debug_data: dict):
    """Render debug information from stored message data.

    Args:
        debug_data: Debug data dict stored in message history
    """
    if not debug_data:
        return

    mode = debug_data.get("mode", "unknown")

    if mode == "rag":
        # Render RAG debug data
        st.markdown("### 🔍 DEBUG: RAG Context (What the LLM Sees)")

        # Section 1: Retrieval Query
        user_query = debug_data.get("user_query") or debug_data.get(
            "retrieval_query"
        )  # Backward compatibility
        condensed_query = debug_data.get("condensed_query")

        if user_query:
            st.markdown("#### 🔎 Retrieval Query")

            # Show user's original input
            st.markdown("**User Input:**")
            st.text_area(
                "Original Query (read-only)",
                value=user_query,
                height=60,
                disabled=True,
                key=f"debug_stored_user_query_{hash(user_query[:100])}",
            )

            # Show condensed query if different
            if condensed_query and user_query != condensed_query:
                st.markdown("**Condensed Query (used for retrieval):**")
                st.info(
                    "🔄 Engine condensed the query with chat history to create a standalone search query"
                )
                st.text_area(
                    "Condensed Query (read-only)",
                    value=condensed_query,
                    height=80,
                    disabled=True,
                    key=f"debug_stored_condensed_query_{hash(condensed_query[:100])}",
                )
            else:
                st.markdown(
                    "*No condensing needed (first message or query already standalone)*"
                )

            st.markdown("---")

        # Section 2: Retrieval Summary
        st.markdown("#### 📊 Retrieval Summary")
        num_nodes = debug_data.get("num_nodes", 0)
        best_score = debug_data.get("best_score", 0.0)
        confidence_threshold = debug_data.get("confidence_threshold", 0.0)
        has_real_sources = debug_data.get("has_real_sources", True)

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

        if has_real_sources and debug_data.get("node_scores"):
            st.markdown("**Sources:**")
            for idx, node_info in enumerate(debug_data["node_scores"], 1):
                st.markdown(
                    f"{idx}. {node_info['display_name']} (score: {node_info['score']:.4f})"
                )

        st.markdown("---")

        # Section 3: ACTUAL CONTEXT STRING
        st.markdown("#### 📝 Context String Sent to LLM")
        st.markdown(
            "*This is the EXACT raw text passed as context (nodes joined by `\\n\\n`):*"
        )

        if has_real_sources and debug_data.get("actual_context_str"):
            st.text_area(
                "Raw Context (read-only)",
                value=debug_data["actual_context_str"],
                height=300,
                disabled=True,
                key=f"debug_stored_context_{hash(debug_data['actual_context_str'][:100])}",
            )
        else:
            st.info("No context (fallback mode - LLM had no retrieved documents)")

        st.markdown("---")

        # Section 4: CURRENT PROMPT (just this turn)
        st.markdown("#### 📝 Current Prompt (This Turn)")
        st.markdown("*The formatted prompt for this specific query (context + query):*")

        if debug_data.get("actual_formatted_prompt"):
            st.text_area(
                "Current Prompt (read-only)",
                value=debug_data["actual_formatted_prompt"],
                height=150,
                disabled=True,
                key=f"debug_stored_prompt_{hash(debug_data['actual_formatted_prompt'][:100])}",
            )

        st.markdown("---")

        # Section 5: COMPLETE CONVERSATION (history + current)
        st.markdown("#### 💬 Complete Conversation Sent to LLM")
        st.markdown("*The FULL conversation including chat history + current prompt:*")

        if debug_data.get("complete_conversation"):
            st.text_area(
                "Complete Conversation (read-only)",
                value=debug_data["complete_conversation"],
                height=300,
                disabled=True,
                key=f"debug_stored_conversation_{hash(debug_data['complete_conversation'][:100])}",
            )

    elif mode == "simple_llm":
        # Render Simple LLM debug data (clean, no HTML)
        st.markdown("### 🔍 DEBUG: Simple LLM Mode (No RAG)")

        # Section 1: Mode Info
        st.info(
            "**Mode:** Simple LLM (No RAG) - No document retrieval, pure LLM response based on chat history only"
        )

        # Section 2: Model Configuration
        st.markdown("#### ⚙️ Model Configuration")
        st.markdown(f"""
- **Model:** {debug_data.get("model", "Unknown")}
- **Temperature:** {debug_data.get("temperature", 0.7)}
- **Max Tokens:** {debug_data.get("max_tokens", 2048)}
- **Device:** {debug_data.get("device", "auto")}
""")

        st.markdown("---")

        # Section 3: Chat History
        st.markdown("#### 💬 Chat History Sent to LLM")
        chat_history = debug_data.get("chat_history", [])
        st.markdown(f"*Total messages: {len(chat_history)}*")

        if chat_history:
            history_text = ""
            for idx, msg in enumerate(chat_history, 1):
                role_icon = "👤" if "USER" in msg.get("role", "").upper() else "🤖"
                history_text += f"Message {idx} [{role_icon} {msg.get('role', 'unknown')}]:\n{msg.get('content', '')}\n\n"

            st.text_area(
                "Chat History (read-only)",
                value=history_text.strip(),
                height=200,
                disabled=True,
                key=f"debug_stored_llm_history_{hash(history_text[:100])}",
            )
        else:
            st.info("No chat history (first message in session)")

        st.markdown("---")

        # Section 4: Current Prompt
        if debug_data.get("prompt"):
            st.markdown("#### 📝 Current User Prompt")
            st.text_area(
                "User Prompt (read-only)",
                value=debug_data["prompt"],
                height=100,
                disabled=True,
                key=f"debug_stored_llm_prompt_{hash(debug_data['prompt'][:100])}",
            )

        # Section 5: System Prompt
        if debug_data.get("system_prompt"):
            st.markdown("---")
            st.markdown("#### 🔧 System Prompt")
            st.text_area(
                "System Prompt (read-only)",
                value=debug_data["system_prompt"],
                height=150,
                disabled=True,
                key=f"debug_stored_llm_system_{hash(debug_data['system_prompt'][:100])}",
            )


def render_chat_message(
    message: dict, params: dict, modules: list, has_pdf_index: bool = False
):
    """Render a complete chat message with content, sources, and metadata.

    Args:
        message: Message dict from session history
        params: Session parameters dict
        modules: List of active module names
        has_pdf_index: Whether session has PDF documents indexed
    """
    avatar = ":material/settings:" if message["role"] == "command" else None
    has_rag = bool(modules) or has_pdf_index

    with st.chat_message(message["role"], avatar=avatar):
        # Show low confidence warning BEFORE content
        if message.get("low_confidence", False) and has_rag:
            confidence_threshold = params.get("confidence_cutoff", 0.0)

            if message.get("sources") and len(message["sources"]) > 0:
                best_score = max(
                    (src["score"] for src in message["sources"]), default=0.0
                )
                render_low_confidence_warning(
                    best_score, confidence_threshold, has_sources=True
                )
            else:
                render_low_confidence_warning(
                    0.0, confidence_threshold, has_sources=False
                )

        # Render debug data if present (BEFORE other content)
        if message.get("debug_data"):
            render_debug_from_stored_data(message["debug_data"])

        # Render thinking if present (for RAG responses)
        if message.get("thinking"):
            render_thinking(message["thinking"])

        # Render agent thinking if present (for agent responses)
        if message.get("agent_thinking"):
            from tensortruth.app_utils.rendering_agent import render_agent_thinking

            render_agent_thinking(message["agent_thinking"])

        # Render message content
        st.markdown(convert_latex_delimiters(message["content"]))

        # Render footer (sources + metadata)
        meta_cols = st.columns([3, 1])
        with meta_cols[0]:
            if "sources" in message and message["sources"]:
                render_source_expander(message["sources"], is_nodes=False)
        with meta_cols[1]:
            caption = render_message_metadata(message, params, modules, has_pdf_index)
            if caption:
                st.caption(caption)
