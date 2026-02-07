"""Unified loader for user-defined commands, agents, and Python extensions.

Scans ``~/.tensortruth/commands/`` and ``~/.tensortruth/agents/`` for YAML and
Python files and registers them with the appropriate service registries.
"""

import importlib.util
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml
from pydantic import ValidationError

from tensortruth.agents.config import AgentConfig
from tensortruth.api.routes.commands import CommandRegistry, ToolCommand
from tensortruth.extensions.errors import ExtensionLoadError
from tensortruth.extensions.schema import AgentSpec, CommandSpec
from tensortruth.extensions.yaml_command import YamlAgentCommand, YamlCommand
from tensortruth.services.agent_service import AgentService
from tensortruth.services.tool_service import ToolService

logger = logging.getLogger(__name__)


@dataclass
class ExtensionLoadResult:
    """Summary of a load_all() run."""

    commands_loaded: int = 0
    agents_loaded: int = 0
    errors: List[ExtensionLoadError] = field(default_factory=list)

    def __repr__(self) -> str:
        parts = []
        if self.commands_loaded:
            parts.append(f"{self.commands_loaded} commands")
        if self.agents_loaded:
            parts.append(f"{self.agents_loaded} agents")
        if self.errors:
            parts.append(f"{len(self.errors)} errors")
        return ", ".join(parts) if parts else "no extensions"


class UserExtensionLoader:
    """Scans user directories for YAML/Python extension files."""

    def __init__(self, user_dir: Optional[Path] = None):
        if user_dir is None:
            from tensortruth.app_utils.paths import get_user_data_dir

            user_dir = get_user_data_dir()
        self._user_dir = user_dir

    async def load_all(
        self,
        command_registry: CommandRegistry,
        agent_service: AgentService,
        tool_service: ToolService,
    ) -> ExtensionLoadResult:
        result = ExtensionLoadResult()

        for ext_type, directory in [
            ("command", self._user_dir / "commands"),
            ("agent", self._user_dir / "agents"),
        ]:
            self._load_directory(
                directory,
                ext_type,
                command_registry,
                agent_service,
                tool_service,
                result,
            )

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_directory(
        self,
        dir_path: Path,
        ext_type: str,
        cmd_reg: CommandRegistry,
        agent_svc: AgentService,
        tool_svc: ToolService,
        result: ExtensionLoadResult,
    ) -> None:
        if not dir_path.exists():
            return

        for path in sorted(dir_path.iterdir()):
            if path.suffix in (".yaml", ".yml"):
                self._load_yaml_file(
                    path, ext_type, cmd_reg, agent_svc, tool_svc, result
                )
            elif path.suffix == ".py":
                self._load_python_file(path, cmd_reg, agent_svc, tool_svc, result)

    # --- YAML loading ---------------------------------------------------

    def _load_yaml_file(
        self,
        path: Path,
        ext_type: str,
        cmd_reg: CommandRegistry,
        agent_svc: AgentService,
        tool_svc: ToolService,
        result: ExtensionLoadResult,
    ) -> None:
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("YAML root must be a mapping")
        except Exception as exc:
            logger.warning(f"Skipping {path.name}: invalid YAML ({exc})")
            result.errors.append(ExtensionLoadError(path, exc, ext_type))
            return

        if ext_type == "command":
            self._register_yaml_command(path, raw, cmd_reg, agent_svc, result)
        elif ext_type == "agent":
            self._register_yaml_agent(path, raw, agent_svc, result)

    def _register_yaml_command(
        self,
        path: Path,
        raw: dict,
        cmd_reg: CommandRegistry,
        agent_svc: AgentService,
        result: ExtensionLoadResult,
    ) -> None:
        try:
            spec = CommandSpec(**raw)
        except (ValidationError, ValueError) as exc:
            logger.warning(f"Skipping {path.name}: schema error ({exc})")
            result.errors.append(ExtensionLoadError(path, exc, "command"))
            return

        cmd: ToolCommand
        if spec.steps:
            cmd = YamlCommand(spec)
        else:
            cmd = YamlAgentCommand(spec)

        cmd_reg.register(cmd)
        result.commands_loaded += 1
        logger.info(f"Registered user command: /{spec.name} (from {path.name})")

    def _register_yaml_agent(
        self,
        path: Path,
        raw: dict,
        agent_svc: AgentService,
        result: ExtensionLoadResult,
    ) -> None:
        try:
            spec = AgentSpec(**raw)
        except (ValidationError, ValueError) as exc:
            logger.warning(f"Skipping {path.name}: schema error ({exc})")
            result.errors.append(ExtensionLoadError(path, exc, "agent"))
            return

        agent_config = AgentConfig(
            name=spec.name,
            description=spec.description,
            tools=spec.tools,
            system_prompt=spec.system_prompt or "",
            agent_type=spec.agent_type,
            model=spec.model,
            max_iterations=spec.max_iterations,
            factory_params=spec.factory_params,
        )
        agent_svc.register_agent(agent_config)
        result.agents_loaded += 1
        logger.info(f"Registered user agent: {spec.name} (from {path.name})")

    # --- Python loading --------------------------------------------------

    def _load_python_file(
        self,
        path: Path,
        cmd_reg: CommandRegistry,
        agent_svc: AgentService,
        tool_svc: ToolService,
        result: ExtensionLoadResult,
    ) -> None:
        try:
            module_name = f"_ext_{path.stem}"
            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot load {path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            register_fn = getattr(module, "register", None)
            if register_fn is None:
                raise AttributeError(f"{path.name} has no register() function")

            register_fn(cmd_reg, agent_svc, tool_svc)
            result.commands_loaded += 1
            logger.info(f"Loaded Python extension: {path.name}")

        except Exception as exc:
            logger.warning(f"Skipping {path.name}: Python load error ({exc})")
            result.errors.append(ExtensionLoadError(path, exc, "python"))
