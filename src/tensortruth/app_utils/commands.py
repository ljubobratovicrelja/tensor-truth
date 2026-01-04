"""Slash command processing for chat interface - Refactored with class-based architecture."""

from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Tuple

import streamlit as st

from .helpers import (
    format_ollama_runtime_info,
    free_memory,
    get_ollama_models,
    get_system_devices,
)

# Type aliases for clarity
StateModifier = Optional[Callable[[], None]]
CommandResult = Tuple[bool, str, StateModifier]


class Command(ABC):
    """Base class for all commands."""

    def __init__(
        self,
        name: str,
        aliases: Optional[List[str]] = None,
        usage: Optional[str] = None,
    ):
        """
        Initialize a command.

        Args:
            name: Primary command name (e.g., "list")
            aliases: Alternative names for the command (e.g., ["ls", "status"])
            usage: Usage string shown in help (e.g., "/list - Show active indices")
        """
        self.name = name
        self.aliases = aliases or []
        self.usage = usage or f"/{name}"

    def matches(self, command: str) -> bool:
        """Check if a command string matches this command."""
        cmd = command.lstrip("/").lower()
        return cmd == self.name or cmd in self.aliases

    @abstractmethod
    def execute(
        self, args: List[str], session: dict, available_mods: List[str]
    ) -> CommandResult:
        """
        Execute the command.

        Args:
            args: Command arguments
            session: Current session state
            available_mods: Available knowledge base modules

        Returns:
            Tuple of (is_command, response_message, state_modifier_fn)
        """
        pass


class ListCommand(Command):
    """Command to show knowledge base and system status."""

    def __init__(self):
        super().__init__(
            name="list",
            aliases=["ls", "status"],
            usage="/list - Show active indices & hardware usage",
        )

    def execute(
        self, args: List[str], session: dict, available_mods: List[str]
    ) -> CommandResult:
        active_mods = session.get("modules", [])
        current_params = session.get("params", {})

        lines = ["### Knowledge Base & System Status"]

        # Knowledge Base Section
        for mod in available_mods:
            lines.append(f"- {'✅' if mod in active_mods else '⚪'} {mod}")

        # Model Configuration Section
        lines.append("\n#### Model Configuration")
        lines.append(f"**Model:** `{current_params.get('model', 'Unknown')}`")
        lines.append(f"**Temperature:** `{current_params.get('temperature', 0.3)}`")
        lines.append(
            f"**Context Window:** `{current_params.get('context_window', 4096)}`"
        )
        lines.append(
            f"**Confidence Warning:** `{current_params.get('confidence_cutoff', 0.3)}`"
        )
        lines.append(
            f"**Confidence Cutoff (Hard):** `{current_params.get('confidence_cutoff_hard', 0.0)}`"
        )

        # Hardware Allocation Section
        lines.append("\n#### Hardware Allocation")
        lines.append(
            f"**Pipeline Device:** `{current_params.get('rag_device', 'cuda')}`"
        )
        lines.append(f"**LLM Device:** `{current_params.get('llm_device', 'gpu')}`")

        # Ollama Runtime Info Section
        runtime_info = format_ollama_runtime_info()
        if runtime_info:
            lines.append("\n#### Ollama Runtime")
            lines.extend(runtime_info)

        lines.append(
            "\n**Commands:** `/load <name>`, `/device rag <cpu|cuda|mps>`, "
            "`/device llm <cpu|gpu>`, `/conf <val>`"
        )
        return True, "\n".join(lines), None


class HelpCommand(Command):
    """Command to show help information."""

    def __init__(self, command_registry: "CommandRegistry"):
        super().__init__(name="help", usage="/help - Show command help")
        self.command_registry = command_registry

    def execute(
        self, args: List[str], session: dict, available_mods: List[str]
    ) -> CommandResult:
        lines = ["###  Command Reference"]
        for cmd in self.command_registry.commands:
            lines.append(f"- **{cmd.usage}**")
        return True, "\n".join(lines), None


class ModelCommand(Command):
    """Command to show or switch models."""

    def __init__(self):
        super().__init__(
            name="model",
            usage="/model [name] - Show current model info or switch to a different model",
        )

    def execute(
        self, args: List[str], session: dict, available_mods: List[str]
    ) -> CommandResult:
        current_params = session.get("params", {})

        if not args:
            # Show current model info and list available models
            return self._show_model_info(current_params)
        else:
            # Switch to a different model
            return self._switch_model(args[0], session)

    def _show_model_info(self, current_params: dict) -> CommandResult:
        """Show current model configuration and available models."""
        lines = ["### Current Model Configuration"]
        lines.append(f"**Active Model:** `{current_params.get('model', 'Unknown')}`")

        # Show Ollama runtime info if available
        runtime_info = format_ollama_runtime_info()
        if runtime_info:
            # Adjust label for VRAM in model info context
            adjusted_info = []
            for line in runtime_info:
                if line.startswith("**VRAM:**"):
                    adjusted_info.append(line.replace("**VRAM:**", "**VRAM Usage:**"))
                elif not line.startswith("**Running:**"):
                    adjusted_info.append(line)
            lines.extend(adjusted_info)

        # List available models
        try:
            available_models = get_ollama_models()
            if available_models:
                lines.append("\n### Available Models")
                for model in available_models:
                    if model == current_params.get("model"):
                        lines.append(f"- ✅ `{model}` (current)")
                    else:
                        lines.append(f"- `{model}`")
                lines.append("\n💡 **Tip:** Use `/model <name>` to switch models")
            else:
                lines.append("\n⚠️ No Ollama models found")
        except Exception:
            lines.append("\n⚠️ Could not fetch available models from Ollama")

        return True, "\n".join(lines), None

    def _switch_model(self, new_model: str, session: dict) -> CommandResult:
        """Switch to a different model."""
        try:
            available_models = get_ollama_models()
            if available_models and new_model in available_models:
                response = (
                    f"✅ **Model switched to:** `{new_model}`\n\n"
                    f"Engine restarting with new model..."
                )

                def update_model():
                    session["params"]["model"] = new_model
                    st.session_state.loaded_config = None

                return True, response, update_model
            else:
                response = (
                    f"❌ Model `{new_model}` not found.\n\n"
                    f"Use `/model` to see available models."
                )
                return True, response, None
        except Exception:
            response = (
                f"⚠️ Could not verify model availability.\n\n"
                f"Attempting to switch to `{new_model}` anyway..."
            )

            def update_model():
                session["params"]["model"] = new_model
                st.session_state.loaded_config = None

            return True, response, update_model


class LoadCommand(Command):
    """Command to load a knowledge base module."""

    def __init__(self):
        super().__init__(
            name="load", usage="/load <index> - Load a specific knowledge base"
        )

    def execute(
        self, args: List[str], session: dict, available_mods: List[str]
    ) -> CommandResult:
        if not args:
            return True, " Usage: `/load <index_name>`", None

        target = args[0]
        active_mods = session.get("modules", [])

        if target not in available_mods:
            return True, f"Index `{target}` not found.", None
        elif target in active_mods:
            return True, f" Index `{target}` is active.", None
        else:
            response = f"✅ **Loaded:** `{target}`. Engine restarting..."

            def load_module():
                session["modules"].append(target)
                st.session_state.loaded_config = None

            return True, response, load_module


class UnloadCommand(Command):
    """Command to unload a knowledge base module."""

    def __init__(self):
        super().__init__(
            name="unload", usage="/unload <index> - Unload a knowledge base"
        )

    def execute(
        self, args: List[str], session: dict, available_mods: List[str]
    ) -> CommandResult:
        if not args:
            return True, " Usage: `/unload <index_name>`", None

        target = args[0]
        active_mods = session.get("modules", [])

        if target not in active_mods:
            return True, f"ℹ️ Index `{target}` not active.", None
        else:
            response = f"✅ **Unloaded:** `{target}`. Engine restarting..."

            def unload_module():
                session["modules"].remove(target)
                st.session_state.loaded_config = None

            return True, response, unload_module


class ReloadCommand(Command):
    """Command to reload the system and flush memory."""

    def __init__(self):
        super().__init__(
            name="reload", usage="/reload - Flush VRAM and restart the engine"
        )

    def execute(
        self, args: List[str], session: dict, available_mods: List[str]
    ) -> CommandResult:
        response = "**System Reload:** Memory flushed."

        def reload_system():
            free_memory()
            st.session_state.loaded_config = None

        return True, response, reload_system


class ConfCommand(Command):
    """Command to set confidence warning threshold and hard cutoff."""

    def __init__(self):
        super().__init__(
            name="conf",
            aliases=["confidence"],
            usage=(
                "/conf <warning> [hard] - "
                "Set confidence warning threshold and optional hard cutoff"
            ),
        )

    def execute(
        self, args: List[str], session: dict, available_mods: List[str]
    ) -> CommandResult:
        if not args:
            return (
                True,
                " Usage: `/conf <warning> [hard]` (e.g. `/conf 0.3 0.15`)",
                None,
            )

        try:
            new_warning = float(args[0])
            if not (0.0 <= new_warning <= 1.0):
                return True, "Warning threshold must be between 0.0 and 1.0.", None

            # Optional hard cutoff parameter
            new_hard = None
            if len(args) > 1:
                new_hard = float(args[1])
                if not (0.0 <= new_hard <= 1.0):
                    return True, "Hard cutoff must be between 0.0 and 1.0.", None
                if new_hard > new_warning:
                    return True, "Hard cutoff must be <= warning threshold.", None

            # Build response message
            if new_hard is not None:
                response = (
                    f"**Confidence Warning:** Set to `{new_warning}`\n"
                    f"**Confidence Cutoff (Hard):** Set to `{new_hard}`\n"
                    "Engine restarting..."
                )
            else:
                response = f"**Confidence Warning:** Set to `{new_warning}`. Engine restarting..."

            def update_confidence():
                session["params"]["confidence_cutoff"] = new_warning
                if new_hard is not None:
                    session["params"]["confidence_cutoff_hard"] = new_hard
                st.session_state.loaded_config = None

            return True, response, update_confidence

        except ValueError:
            return True, "Invalid number. Example: `/conf 0.3 0.15`", None


class DeviceCommand(Command):
    """Command to configure hardware allocation."""

    def __init__(self):
        super().__init__(
            name="device",
            usage=(
                "/device rag <cpu|cuda|mps> OR /device llm <cpu|gpu> - "
                "Configure hardware allocation"
            ),
        )

    def execute(
        self, args: List[str], session: dict, available_mods: List[str]
    ) -> CommandResult:
        if len(args) < 2:
            return (
                True,
                "Usage: `/device rag <cpu|cuda|mps>` OR `/device llm <cpu|gpu>`",
                None,
            )

        target_type = args[0].lower()  # 'rag' or 'llm'
        target_dev = args[1].lower()  # 'cpu', 'cuda', ...

        if target_type == "rag":
            return self._configure_rag_device(target_dev, session)
        elif target_type == "llm":
            return self._configure_llm_device(target_dev, session)
        else:
            return True, "Unknown target. Use `rag` or `llm`.", None

    def _configure_rag_device(self, target_dev: str, session: dict) -> CommandResult:
        """Configure RAG pipeline device."""
        available_devices = get_system_devices()
        if target_dev not in available_devices:
            response = (
                f"Device `{target_dev}` not available. Options: {available_devices}"
            )
            return True, response, None
        else:
            response = f"**Pipeline Switched:** Now running Embed/Rerank on `{target_dev.upper()}`."

            def update_rag_device():
                session["params"]["rag_device"] = target_dev
                st.session_state.loaded_config = None

            return True, response, update_rag_device

    def _configure_llm_device(self, target_dev: str, session: dict) -> CommandResult:
        """Configure LLM device."""
        if target_dev not in ["cpu", "gpu"]:
            return True, "LLM Device options: `cpu` or `gpu`", None
        else:
            response = f"**LLM Switched:** Now running Model on `{target_dev.upper()}`."

            def update_llm_device():
                session["params"]["llm_device"] = target_dev
                st.session_state.loaded_config = None

            return True, response, update_llm_device


class WebSearchCommand(Command):
    """Command to search the web and get AI-generated summary."""

    def __init__(self):
        super().__init__(
            name="websearch",
            aliases=["web"],
            usage="/websearch <query> - Search the web and get a summary",
        )

    def execute(
        self, args: List[str], session: dict, available_mods: List[str]
    ) -> CommandResult:
        if not args:
            return True, "⚠️ Usage: `/search <query>`", None

        query = " ".join(args)

        # Import here to avoid circular deps
        from tensortruth.core.ollama import get_ollama_url
        from tensortruth.utils.web_search import web_search

        try:
            # Import rendering utilities
            from tensortruth.app_utils.rendering import render_web_search_progress

            # Use session's current model (reuses VRAM)
            model_name = session["params"]["model"]
            ollama_url = get_ollama_url()

            # Get config (or use defaults)
            max_results = session["params"].get("web_search_max_results", 5)
            max_pages = session["params"].get("web_search_pages_to_fetch", 3)

            # Create progress placeholder
            progress_placeholder = st.empty()
            progress_updates = []

            def update_progress(message: str):
                """Callback to update progress display."""
                progress_updates.append(message)
                render_web_search_progress(
                    "\n\n".join(progress_updates), placeholder=progress_placeholder
                )

            # Execute search with real-time progress updates
            response = web_search(
                query=query,
                model_name=model_name,
                ollama_url=ollama_url,
                max_results=max_results,
                max_pages=max_pages,
                progress_callback=update_progress,
            )

            # Clear progress display after completion
            progress_placeholder.empty()

            return True, response, None

        except Exception as e:
            error_msg = (
                f"❌ **Web search failed:** {str(e)}\n\n"
                f"This could be due to network issues, rate limiting, "
                f"or unavailable web resources."
            )
            return True, error_msg, None


class CommandRegistry:
    """Registry for managing and executing commands."""

    def __init__(self):
        self.commands: List[Command] = []

    def register(self, command: Command):
        """Register a command."""
        self.commands.append(command)

    def find_command(self, command_str: str) -> Optional[Command]:
        """Find a command by name or alias."""
        for cmd in self.commands:
            if cmd.matches(command_str):
                return cmd
        return None

    def get_help_text(self) -> str:
        """Get formatted help text for unknown commands."""
        lines = [
            "### Available Commands",
            "- **/list** / **/status** - Show active indices & hardware usage",
            "- **/model [name]** - Show current model or switch to different model",
            "- **/load <index>** - Load a knowledge base",
            "- **/unload <index>** - Unload a knowledge base",
            "- **/reload** - Flush VRAM and restart engine",
            "- **/device rag <cpu|cuda|mps>** - Move RAG pipeline to specific hardware",
            "- **/device llm <cpu|gpu>** - Move LLM to specific hardware",
            "- **/conf <warning> [hard]** - Set confidence warning and optional hard cutoff",
            "- **/search <query>** / **/web <query>** - Search the web and get AI summary",
            "- **/help** - Show command help",
        ]
        return "\n".join(lines)


# Initialize global command registry
_registry = CommandRegistry()
_registry.register(ListCommand())
_registry.register(ModelCommand())
_registry.register(LoadCommand())
_registry.register(UnloadCommand())
_registry.register(ReloadCommand())
_registry.register(ConfCommand())
_registry.register(DeviceCommand())
_registry.register(WebSearchCommand())
_registry.register(HelpCommand(_registry))


def process_command(
    prompt: str, session: dict, available_mods: List[str]
) -> CommandResult:
    """
    Process a slash command.

    Args:
        prompt: The command string (e.g., "/list" or "/load pytorch")
        session: Current session state
        available_mods: Available knowledge base modules

    Returns:
        Tuple of (is_command, response_message, state_modifier_fn)
    """
    cmd_parts = prompt.strip().split()
    command = cmd_parts[0].lower()
    args = cmd_parts[1:] if len(cmd_parts) > 1 else []

    # Find and execute command
    cmd_obj = _registry.find_command(command)
    if cmd_obj:
        return cmd_obj.execute(args, session, available_mods)
    else:
        # Unknown command
        response = (
            f"❌ **Unknown command:** `{command}`\n\n" + _registry.get_help_text()
        )
        return True, response, None
