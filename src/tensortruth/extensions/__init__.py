"""User-extensible commands, agents, and tools.

Public API
----------
.. autofunction:: load_user_extensions
"""

import logging
from pathlib import Path
from typing import Optional

from tensortruth.api.routes.commands import CommandRegistry
from tensortruth.extensions.loader import ExtensionLoadResult, UserExtensionLoader
from tensortruth.services.agent_service import AgentService
from tensortruth.services.tool_service import ToolService

logger = logging.getLogger(__name__)


async def load_user_extensions(
    command_registry: CommandRegistry,
    agent_service: AgentService,
    tool_service: ToolService,
    user_dir: Optional[Path] = None,
) -> ExtensionLoadResult:
    """Scan ``~/.tensortruth/{commands,agents}/`` and register extensions.

    This is the single entry-point called from ``lifespan()`` in ``main.py``.
    Every extension file loads inside a try/except so a broken file never
    crashes the application.

    Args:
        command_registry: The global :class:`CommandRegistry`.
        agent_service: The singleton :class:`AgentService`.
        tool_service: The singleton :class:`ToolService`.
        user_dir: Override for ``~/.tensortruth`` (useful in tests).

    Returns:
        :class:`ExtensionLoadResult` summarising what was loaded.
    """
    loader = UserExtensionLoader(user_dir=user_dir)
    result = await loader.load_all(command_registry, agent_service, tool_service)

    if result.errors:
        for err in result.errors:
            logger.warning(f"Extension load error: {err}")

    return result
