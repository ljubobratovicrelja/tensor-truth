"""Pydantic validation models for YAML extension files."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, model_validator


class StepSpec(BaseModel):
    """A single step in a YAML command pipeline.

    Each step calls one tool with templated parameters and optionally
    stores the result in a named variable for later steps.

    If ``result_extract`` is set, it is compiled as a regex and applied
    to the step output.  Named groups become ``{result_var}.{group}``
    in the template context (e.g. ``(?P<libraryId>/\\S+)``).
    If there are no named groups the whole match is stored as the
    result_var value.
    """

    tool: str
    params: Dict[str, Any] = {}
    result_var: Optional[str] = None
    result_extract: Optional[str] = None


class CommandSpec(BaseModel):
    """Schema for a YAML command definition.

    A command either defines a ``steps`` pipeline (sequential tool calls)
    or delegates to a named ``agent``.  The two are mutually exclusive.
    """

    name: str
    description: str
    usage: str = ""
    aliases: List[str] = []
    steps: Optional[List[StepSpec]] = None
    agent: Optional[str] = None
    response: str = "{{_last_result}}"
    requires_mcp: Optional[str] = None

    @model_validator(mode="after")
    def _steps_xor_agent(self) -> "CommandSpec":
        if self.steps and self.agent:
            raise ValueError("'steps' and 'agent' are mutually exclusive")
        if not self.steps and not self.agent:
            raise ValueError("Either 'steps' or 'agent' must be provided")
        return self


class AgentSpec(BaseModel):
    """Schema for a YAML agent definition.

    Maps directly to ``AgentConfig`` fields used by ``AgentService``.
    """

    name: str
    description: str
    tools: List[str]
    agent_type: str = "function"
    system_prompt: Optional[str] = None
    model: Optional[str] = None
    max_iterations: int = 10
    factory_params: Dict[str, Any] = {}
