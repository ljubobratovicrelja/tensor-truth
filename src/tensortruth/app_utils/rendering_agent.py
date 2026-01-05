"""Rendering utilities for agent-specific UI components.

This module provides rendering functions for displaying agent thinking,
progress updates, and execution history in the Streamlit UI.
"""

import html

import streamlit as st


def _create_agent_scrollable_box(
    content: str,
    label: str,
    bg_color: str,
    border_color: str,
    escape_html: bool = True,
) -> str:
    """Create HTML for a scrollable box with fixed height.

    Args:
        content: The content to display inside the box
        label: The label/title for the box
        bg_color: Background color (hex)
        border_color: Border color (hex)
        escape_html: Whether to escape HTML in content (default: True for security)

    Returns:
        HTML string with styled scrollable container
    """
    # Escape content by default to prevent XSS
    if escape_html:
        content = html.escape(content)

    style = (
        "max-height: 200px; "
        "overflow-y: auto; "
        "overflow-x: hidden; "
        "padding: 0.75rem; "
        "margin: 0.5rem 0; "
        f"background-color: {bg_color}; "
        f"border-left: 3px solid {border_color}; "
        "border-radius: 4px;"
    )
    return f"""
<div style="{style}">
<strong>{label}</strong>
<br>
<div style="margin-top: 0.5rem;">
{content}
</div>
</div>"""


def render_agent_progress(progress_text: str, placeholder=None):
    """Render agent action progress updates.

    This displays ephemeral progress updates during agent execution,
    such as "Searching...", "Fetching page...", "Reasoning...".
    Should be cleared after agent completes.

    Args:
        progress_text: The progress updates to display (markdown formatted)
        placeholder: Optional Streamlit placeholder to render into (uses st.markdown if None)
    """
    html_content = _create_agent_scrollable_box(
        content=progress_text,
        label="🤖 Agent Progress:",
        bg_color="#FFF4E5",
        border_color="#FF9800",
    )

    if placeholder:
        placeholder.markdown(html_content, unsafe_allow_html=True)
    else:
        st.markdown(html_content, unsafe_allow_html=True)


def render_agent_thinking(thinking_sections: list, placeholder=None):
    """Render multi-iteration agent thinking as collapsible sections.

    Displays the agent's reasoning process across iterations with each
    iteration in a collapsible <details> element.

    Args:
        thinking_sections: List of dicts with keys:
            - iteration: int (iteration number)
            - thinking: str (agent's thinking text)
            - action: str (action type taken)
            - reasoning: str (reasoning for action)
        placeholder: Optional Streamlit placeholder to render into
    """
    if not thinking_sections:
        return

    # Build HTML for collapsible sections
    html_parts = ['<div style="margin: 0.5rem 0;">']

    for section in thinking_sections:
        iteration = section.get("iteration", "?")
        thinking = html.escape(section.get("thinking", ""))
        action = html.escape(section.get("action", ""))
        reasoning = html.escape(section.get("reasoning", ""))

        # Create collapsible section for this iteration
        html_parts.append(
            f"""
<details style="margin: 0.5rem 0; padding: 0.5rem; background-color: #F0F4F8; border-left: 3px solid #2C5282; border-radius: 4px;">
    <summary style="cursor: pointer; font-weight: bold; color: #2C5282;">
        Iteration {iteration}: {action}
    </summary>
    <div style="padding: 0.75rem; margin-top: 0.5rem; background-color: #FFFFFF; border-radius: 4px;">
        <div style="margin-bottom: 0.5rem;">
            <strong>Thinking:</strong>
            <pre style="white-space: pre-wrap; word-wrap: break-word; background-color: #F7FAFC; padding: 0.5rem; border-radius: 3px; font-family: monospace; font-size: 0.9em;">{thinking}</pre>
        </div>
        {f'<div><strong>Reasoning:</strong> {reasoning}</div>' if reasoning else ''}
    </div>
</details>
"""
        )

    html_parts.append("</div>")

    # Wrap in outer container with label
    outer_container = f"""
<div style="max-height: 300px; overflow-y: auto; overflow-x: hidden; \
padding: 0.75rem; margin: 0.5rem 0; background-color: #EDF2F7; \
border-left: 3px solid #2C5282; border-radius: 4px;">
<strong>🧠 Agent Thinking:</strong>
<div style="margin-top: 0.5rem;">
{''.join(html_parts)}
</div>
</div>
"""

    if placeholder:
        placeholder.markdown(outer_container, unsafe_allow_html=True)
    else:
        st.markdown(outer_container, unsafe_allow_html=True)


def render_agent_summary(state, placeholder=None):
    """Render a summary of agent execution.

    Shows high-level metrics about agent execution:
    - Total iterations
    - Searches performed
    - Pages visited
    - Termination reason

    Args:
        state: AgentState object with execution history
        placeholder: Optional Streamlit placeholder
    """
    summary_lines = [
        f"**Iterations:** {state.current_iteration}",
        f"**Searches:** {len(state.searches_performed)}",
        f"**Pages Visited:** {len(state.pages_visited)}",
    ]

    if state.termination_reason:
        reason_display = {
            "goal_satisfied": "✅ Goal satisfied",
            "max_iterations": "🔄 Max iterations reached",
            "timeout": "⏱️ Timeout",
            "error": "❌ Error occurred",
        }.get(state.termination_reason, state.termination_reason)
        summary_lines.append(f"**Status:** {reason_display}")

    summary_text = " | ".join(summary_lines)

    if placeholder:
        placeholder.caption(summary_text)
    else:
        st.caption(summary_text)
