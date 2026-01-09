"""Rendering utilities for Streamlit UI components."""

import re

import streamlit as st

from tensortruth import convert_latex_delimiters


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


def render_execution_result(result: dict):
    """Render a single code execution result.

    Args:
        result: ExecutionResult dict with success, stdout, stderr, etc.
    """
    if result["success"]:
        with st.expander("✅ Code executed successfully", expanded=True):
            # Render output files (images, plots, etc.)
            if result.get("output_files"):
                from pathlib import Path

                for file_path in result["output_files"]:
                    file_path = Path(file_path)
                    if file_path.exists():
                        # Determine file type and render appropriately
                        if file_path.suffix.lower() in {
                            ".png",
                            ".jpg",
                            ".jpeg",
                            ".gif",
                        }:
                            st.image(str(file_path), use_container_width=True)
                        elif file_path.suffix.lower() == ".svg":
                            with open(file_path, "r") as f:
                                st.markdown(f.read(), unsafe_allow_html=True)
                        elif file_path.suffix.lower() == ".pdf":
                            st.caption(f"📄 Generated: {file_path.name}")
                            with open(file_path, "rb") as f:
                                st.download_button(
                                    label=f"Download {file_path.name}",
                                    data=f.read(),
                                    file_name=file_path.name,
                                    mime="application/pdf",
                                )
                        elif file_path.suffix.lower() in {
                            ".csv",
                            ".json",
                            ".txt",
                            ".html",
                        }:
                            st.caption(f"📊 Generated: {file_path.name}")
                            with open(file_path, "rb") as f:
                                st.download_button(
                                    label=f"Download {file_path.name}",
                                    data=f.read(),
                                    file_name=file_path.name,
                                )

            if result["stdout"]:
                st.markdown("**Output:**")
                st.code(result["stdout"], language="text")
            if result["stderr"]:
                st.markdown("**Warnings:**")
                st.code(result["stderr"], language="text")
            st.caption(f"⏱️ {result['execution_time']:.2f}s")
    else:
        with st.expander("❌ Execution failed", expanded=True):
            st.error(result["error_message"])
            if result["stderr"]:
                st.code(result["stderr"], language="text")
            st.caption(f"⏱️ {result['execution_time']:.2f}s")


def render_content_with_execution_results(content: str, execution_results: list = None):
    """Render markdown content with execution results injected after code blocks.

    Args:
        content: Markdown content from message
        execution_results: List of execution result dicts (one per code block)
    """
    if not execution_results:
        # No execution results, render content normally
        st.markdown(convert_latex_delimiters(content))
        return

    # Pattern to match code blocks (both ```python and ``` generic)
    # Matches: ```language\ncode\n``` or ```language\ncode```
    code_block_pattern = re.compile(r"(```[\w]*\n.*?```)", re.DOTALL)

    # Split content by code blocks while keeping the code blocks
    parts = code_block_pattern.split(content)

    result_index = 0

    for part in parts:
        if part.startswith("```"):
            # This is a code block - render it
            st.markdown(convert_latex_delimiters(part))

            # Inject execution result if available
            if result_index < len(execution_results):
                render_execution_result(execution_results[result_index])
                result_index += 1
        else:
            # This is regular text - render it
            if part.strip():  # Only render non-empty parts
                st.markdown(convert_latex_delimiters(part))


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
    # Skip rendering code_execution messages - they're shown via expander boxes
    if message["role"] == "code_execution":
        return

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

        # Render thinking if present
        if message.get("thinking"):
            st.markdown(
                f"""<div class="thinking-content">

**🧠 Reasoning:**

{convert_latex_delimiters(message['thinking'])}

</div>""",
                unsafe_allow_html=True,
            )

        # Render message content with inline execution results
        execution_results = message.get("execution_results", [])
        render_content_with_execution_results(message["content"], execution_results)

        # Render footer (sources + metadata)
        meta_cols = st.columns([3, 1])
        with meta_cols[0]:
            if "sources" in message and message["sources"]:
                render_source_expander(message["sources"], is_nodes=False)
        with meta_cols[1]:
            caption = render_message_metadata(message, params, modules, has_pdf_index)
            if caption:
                st.caption(caption)
