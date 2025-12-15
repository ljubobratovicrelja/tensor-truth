"""Slash command processing for chat interface."""

import streamlit as st

from .helpers import free_memory, get_system_devices


def process_command(prompt, session, available_mods):
    """
    Handles /slash commands.

    Returns:
        tuple: (is_command, response_message, state_modifier_fn)
        The state_modifier_fn (if not None) should be called to apply state changes.
    """
    cmd_parts = prompt.strip().split()
    command = cmd_parts[0].lower()
    args = cmd_parts[1:] if len(cmd_parts) > 1 else []

    active_mods = session.get("modules", [])
    current_params = session.get("params", {})
    available_devices = get_system_devices()

    response_msg = ""
    state_modifier = None

    if command in ["/list", "/ls", "/status"]:
        lines = ["### Knowledge Base & System Status"]
        for mod in available_mods:
            lines.append(f"- {'✅' if mod in active_mods else '⚪'} {mod}")

        lines.append(
            f"\n**Pipeline Device:** `{current_params.get('rag_device', 'cuda')}`"
        )
        lines.append(f"**LLM Device:** `{current_params.get('llm_device', 'gpu')}`")
        lines.append(
            f"**Confidence Cutoff:** `{current_params.get('confidence_cutoff', 0.3)}`"
        )
        lines.append(
            (
                "\n**Usage:** `/load <name>`, `/device rag <cpu|cuda|mps>`, "
                "`/device llm <cpu|gpu>`, `/conf <val>`"
            )
        )
        response_msg = "\n".join(lines)
        return True, response_msg, None

    elif command == "/help":
        lines = [
            "###  Command Reference",
            "- **/list** / **/status**: Show active indices & hardware usage.",
            "- **/load <index>**: Load a specific knowledge base.",
            "- **/unload <index>**: Unload a knowledge base.",
            "- **/reload**: Flush VRAM and restart the engine.",
            (
                "- **/device rag <cpu|cuda|mps>**: Move RAG pipeline (Embed/Rerank) "
                "to specific hardware."
            ),
            "- **/device llm <cpu|gpu>**: Move LLM (Ollama) to specific hardware.",
            "- **/conf <0.0-1.0>**: Set the confidence score cutoff for retrieval.",
            "- **/help**: Show this list.",
        ]
        response_msg = "\n".join(lines)

    elif command == "/load":
        if not args:
            response_msg = " Usage: `/load <index_name>`"
        else:
            target = args[0]
            if target not in available_mods:
                response_msg = f"Index `{target}` not found."
            elif target in active_mods:
                response_msg = f" Index `{target}` is active."
            else:
                response_msg = f"✅ **Loaded:** `{target}`. Engine restarting..."

                def load_module():
                    session["modules"].append(target)
                    st.session_state.loaded_config = None

                state_modifier = load_module

    elif command == "/unload":
        if not args:
            response_msg = " Usage: `/unload <index_name>`"
        else:
            target = args[0]
            if target not in active_mods:
                response_msg = f"ℹ️ Index `{target}` not active."
            else:
                response_msg = f"✅ **Unloaded:** `{target}`. Engine restarting..."

                def unload_module():
                    session["modules"].remove(target)
                    st.session_state.loaded_config = None

                state_modifier = unload_module

    elif command == "/reload":
        response_msg = "**System Reload:** Memory flushed."

        def reload_system():
            free_memory()
            st.session_state.loaded_config = None

        state_modifier = reload_system

    elif command in ["/conf", "/confidence"]:
        if not args:
            response_msg = " Usage: `/conf <value>` (e.g. 0.2)"
        else:
            try:
                new_conf = float(args[0])
                if 0.0 <= new_conf <= 1.0:
                    response_msg = (
                        f"**Confidence Cutoff:** Set to `{new_conf}`. "
                        f"Engine restarting..."
                    )

                    def update_confidence():
                        session["params"]["confidence_cutoff"] = new_conf
                        st.session_state.loaded_config = (
                            None  # Force reload to apply postprocessor change
                        )

                    state_modifier = update_confidence
                else:
                    response_msg = "Value must be between 0.0 and 1.0."
            except ValueError:
                response_msg = "Invalid number. Example: `/conf 0.3`"

    elif command == "/device":
        if len(args) < 2:
            response_msg = (
                "Usage: `/device rag <cpu|cuda|mps>` OR " "`/device llm <cpu|gpu>`"
            )
        else:
            target_type = args[0].lower()  # 'rag' or 'llm'
            target_dev = args[1].lower()  # 'cpu', 'cuda', ...

            if target_type == "rag":
                if target_dev not in available_devices:
                    response_msg = (
                        f"Device `{target_dev}` not available. Options: "
                        f"{available_devices}"
                    )
                else:
                    response_msg = (
                        f"**Pipeline Switched:** Now running Embed/Rerank on "
                        f"`{target_dev.upper()}`."
                    )

                    def update_rag_device():
                        session["params"]["rag_device"] = target_dev
                        st.session_state.loaded_config = None

                    state_modifier = update_rag_device

            elif target_type == "llm":
                if target_dev not in ["cpu", "gpu"]:
                    response_msg = "LLM Device options: `cpu` or `gpu`"
                else:
                    response_msg = (
                        f"**LLM Switched:** Now running Model on "
                        f"`{target_dev.upper()}`."
                    )

                    def update_llm_device():
                        session["params"]["llm_device"] = target_dev
                        st.session_state.loaded_config = None

                    state_modifier = update_llm_device
            else:
                response_msg = "Unknown target. Use `rag` or `llm`."

    # Return the result - all paths lead here except early returns for errors
    return True, response_msg, state_modifier if response_msg else (False, None, None)
