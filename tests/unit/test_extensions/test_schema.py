"""Tests for Pydantic extension schemas."""

import pytest
from pydantic import ValidationError

from tensortruth.extensions.schema import AgentSpec, CommandSpec, StepSpec


class TestStepSpec:
    def test_minimal(self):
        step = StepSpec(tool="my-tool")
        assert step.tool == "my-tool"
        assert step.params == {}
        assert step.result_var is None

    def test_full(self):
        step = StepSpec(
            tool="resolve-library-id",
            params={"libraryName": "{{args}}"},
            result_var="resolved",
        )
        assert step.result_var == "resolved"
        assert step.params["libraryName"] == "{{args}}"


class TestCommandSpec:
    def test_steps_command(self):
        spec = CommandSpec(
            name="test",
            description="A test command",
            steps=[StepSpec(tool="my-tool")],
        )
        assert spec.name == "test"
        assert spec.steps is not None
        assert spec.agent is None

    def test_agent_command(self):
        spec = CommandSpec(
            name="test",
            description="A test command",
            agent="browse",
        )
        assert spec.agent == "browse"
        assert spec.steps is None

    def test_steps_and_agent_raises(self):
        with pytest.raises(ValidationError, match="mutually exclusive"):
            CommandSpec(
                name="test",
                description="A test command",
                steps=[StepSpec(tool="my-tool")],
                agent="browse",
            )

    def test_neither_steps_nor_agent_raises(self):
        with pytest.raises(ValidationError, match="must be provided"):
            CommandSpec(
                name="test",
                description="A test command",
            )

    def test_defaults(self):
        spec = CommandSpec(
            name="test",
            description="desc",
            steps=[StepSpec(tool="t")],
        )
        assert spec.usage == ""
        assert spec.aliases == []
        assert spec.response == "{{_last_result}}"
        assert spec.requires_mcp is None

    def test_aliases(self):
        spec = CommandSpec(
            name="test",
            description="desc",
            aliases=["t", "tst"],
            steps=[StepSpec(tool="t")],
        )
        assert spec.aliases == ["t", "tst"]


class TestAgentSpec:
    def test_minimal(self):
        spec = AgentSpec(
            name="my_agent",
            description="Does things",
            tools=["search_web"],
        )
        assert spec.agent_type == "function"
        assert spec.max_iterations == 10
        assert spec.model is None

    def test_full(self):
        spec = AgentSpec(
            name="researcher",
            description="Research agent",
            tools=["search_web", "fetch_page"],
            agent_type="router",
            system_prompt="You are a researcher.",
            model="llama3.2:3b",
            max_iterations=5,
            factory_params={"min_pages_required": 2},
        )
        assert spec.model == "llama3.2:3b"
        assert spec.factory_params["min_pages_required"] == 2

    def test_missing_required_fields_raises(self):
        with pytest.raises(ValidationError):
            AgentSpec(name="test")  # missing description and tools
