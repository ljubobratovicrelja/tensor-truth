"""Tests for the session-scoped ``inspect`` repository tool.

The tool must:
- expose a CLOSED action set (no shell, no passthrough to arbitrary commands),
- reject an out-of-enum action or git command before doing any work,
- gate ``clone`` through the confirmation service,
- read only inside the session's workspace jail,
- return content (not a summary) and cap it to the context budget,
- be registered in the orchestrator tool set,
- announce cloned repositories in the system prompt on every turn.
"""

import json
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from tensortruth.services.orchestrator_service import OrchestratorService
from tensortruth.services.orchestrator_tool_wrappers import (
    InspectInput,
    _format_age,
    create_all_tool_wrappers,
    create_inspect_tool,
)
from tensortruth.services.repo_workspace import RepoRecord

# ----------------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------------


def _git(cwd: Path, *args: str) -> str:
    """Run a git command with a deterministic identity and no user config."""
    result = subprocess.run(
        [
            "git",
            "-c",
            "user.email=test@example.com",
            "-c",
            "user.name=Test",
            "-c",
            "init.defaultBranch=main",
            *args,
        ],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _record(repo_dir: Path, name: str = "acme/widget") -> RepoRecord:
    owner, repo = name.split("/")
    return RepoRecord(
        name=name,
        slug=f"{owner}__{repo}",
        owner=owner,
        repo=repo,
        url=f"https://github.com/{name}",
        sha=_git(repo_dir, "rev-parse", "HEAD"),
        branch=_git(repo_dir, "rev-parse", "--abbrev-ref", "HEAD"),
        cloned_at=time.time(),
        shallow=True,
    )


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """A session workspace holding one cloned repository on disk."""
    ws = tmp_path / "sessions" / "s-1"
    repo_dir = ws / "repos" / "acme__widget"
    (repo_dir / "src").mkdir(parents=True)
    (repo_dir / "README.md").write_text("# Widget\n\nA widget.\n")
    (repo_dir / "src" / "main.py").write_text(
        "def main():\n    return 'needle in a haystack'\n"
    )
    (repo_dir / "src" / "binary.bin").write_bytes(b"\x00\x01\x02")
    _git(repo_dir, "init", "-q")
    _git(repo_dir, "add", "-A")
    _git(repo_dir, "commit", "-qm", "initial import")

    record = _record(repo_dir)
    (repo_dir / ".repo_meta.json").write_text(json.dumps(record.to_dict()))
    return ws


@pytest.fixture
def emitter():
    return MagicMock()


def _confirmation(outcome: str = "approved"):
    service = MagicMock()
    service.create_confirmation.return_value = SimpleNamespace(confirmation_id="c-1")
    service.await_resolution = AsyncMock(return_value=outcome)
    return service


# ----------------------------------------------------------------------------
# Schema: the action set is closed
# ----------------------------------------------------------------------------


class TestClosedActionSet:
    @pytest.mark.parametrize(
        "action",
        ["bash", "sh", "exec", "shell", "write", "rm", "git push", ""],
    )
    async def test_unknown_action_is_rejected_by_the_schema(self, action):
        with pytest.raises(ValidationError):
            InspectInput(action=action)

    @pytest.mark.parametrize(
        "action", ["list", "clone", "tree", "read", "grep", "git", "drop"]
    )
    async def test_known_actions_validate(self, action):
        assert InspectInput(action=action).action == action

    @pytest.mark.parametrize("command", ["push", "commit", "clean", "reset"])
    async def test_git_command_is_a_closed_enum(self, command):
        with pytest.raises(ValidationError):
            InspectInput(action="git", git_command=command)

    async def test_direct_call_with_unknown_action_returns_error(
        self, workspace, emitter
    ):
        """The enum is the first gate; the body still refuses politely."""
        tool = create_inspect_tool(emitter, session_id="s-1", scope_dir=workspace)
        result = await tool.acall(action="bash", url=None)

        assert "unknown action" in str(result)


# ----------------------------------------------------------------------------
# clone
# ----------------------------------------------------------------------------


class TestClone:
    async def test_clone_requires_url(self, workspace, emitter):
        tool = create_inspect_tool(emitter, session_id="s-1", scope_dir=workspace)
        result = await tool.acall(action="clone")

        assert "requires url" in str(result)

    async def test_rejected_clone_does_not_clone(self, workspace, emitter, monkeypatch):
        calls = []
        monkeypatch.setattr(
            "tensortruth.services.repo_workspace.clone_repo",
            lambda *a, **k: calls.append((a, k)),
        )
        tool = create_inspect_tool(
            emitter,
            session_id="s-1",
            scope_dir=workspace,
            confirmation_service=_confirmation("rejected"),
        )

        result = await tool.acall(action="clone", url="https://github.com/acme/gizmo")

        assert "rejected" in str(result).lower()
        assert calls == []

    async def test_approved_clone_reports_sha(self, workspace, emitter, monkeypatch):
        record = RepoRecord(
            name="acme/gizmo",
            slug="acme__gizmo",
            owner="acme",
            repo="gizmo",
            url="https://github.com/acme/gizmo",
            sha="a" * 40,
            branch="main",
            cloned_at=time.time(),
            shallow=True,
        )
        monkeypatch.setattr(
            "tensortruth.services.repo_workspace.clone_repo",
            lambda *a, **k: record,
        )
        tool = create_inspect_tool(
            emitter,
            session_id="s-1",
            scope_dir=workspace,
            confirmation_service=_confirmation("approved"),
        )

        result = str(
            await tool.acall(action="clone", url="https://github.com/acme/gizmo")
        )

        assert "acme/gizmo" in result
        assert "aaaaaaaa" in result
        assert "inspect(action=" in result

    async def test_full_history_alternative_is_honoured(
        self, workspace, emitter, monkeypatch
    ):
        seen = {}

        def fake_clone(scope_dir, url, *, full_history=False, **kwargs):
            seen["full_history"] = full_history
            return RepoRecord(
                name="acme/gizmo",
                slug="acme__gizmo",
                owner="acme",
                repo="gizmo",
                url=url,
                sha="b" * 40,
                branch="main",
                cloned_at=time.time(),
                shallow=not full_history,
            )

        monkeypatch.setattr(
            "tensortruth.services.repo_workspace.clone_repo", fake_clone
        )
        tool = create_inspect_tool(
            emitter,
            session_id="s-1",
            scope_dir=workspace,
            confirmation_service=_confirmation("full_history"),
        )

        await tool.acall(action="clone", url="https://github.com/acme/gizmo")

        assert seen["full_history"] is True


# ----------------------------------------------------------------------------
# Read-only actions
# ----------------------------------------------------------------------------


class TestReadOnlyActions:
    async def test_list_reports_cloned_repos(self, workspace, emitter):
        tool = create_inspect_tool(emitter, session_id="s-1", scope_dir=workspace)
        result = str(await tool.acall(action="list"))

        assert "acme/widget" in result
        assert "branch" in result

    async def test_list_with_no_repos_explains_how_to_clone(self, tmp_path, emitter):
        tool = create_inspect_tool(
            emitter, session_id="s-1", scope_dir=tmp_path / "empty"
        )
        result = str(await tool.acall(action="list"))

        assert "No repositories" in result
        assert 'inspect(action="clone"' in result

    async def test_tree_lists_dirs_and_files(self, workspace, emitter):
        tool = create_inspect_tool(emitter, session_id="s-1", scope_dir=workspace)
        result = str(await tool.acall(action="tree"))

        assert "acme/widget" in result
        assert "src/" in result
        assert "README.md" in result

    async def test_read_returns_numbered_content(self, workspace, emitter):
        tool = create_inspect_tool(emitter, session_id="s-1", scope_dir=workspace)
        result = str(await tool.acall(action="read", path="src/main.py"))

        assert "needle in a haystack" in result
        assert "1|" in result.replace("  ", " ")

    async def test_read_honours_start_line(self, workspace, emitter):
        tool = create_inspect_tool(emitter, session_id="s-1", scope_dir=workspace)
        result = str(await tool.acall(action="read", path="src/main.py", start_line=2))

        assert "needle in a haystack" in result
        assert "def main" not in result

    async def test_grep_finds_matches_with_path_and_line(self, workspace, emitter):
        tool = create_inspect_tool(emitter, session_id="s-1", scope_dir=workspace)
        result = str(await tool.acall(action="grep", pattern="needle"))

        assert "src/main.py:2" in result

    async def test_grep_without_pattern_returns_error(self, workspace, emitter):
        tool = create_inspect_tool(emitter, session_id="s-1", scope_dir=workspace)
        result = str(await tool.acall(action="grep"))

        assert "requires pattern" in result

    async def test_git_log_returns_commit(self, workspace, emitter):
        tool = create_inspect_tool(emitter, session_id="s-1", scope_dir=workspace)
        result = str(await tool.acall(action="git", git_command="log"))

        assert "initial import" in result

    async def test_git_without_command_returns_error(self, workspace, emitter):
        tool = create_inspect_tool(emitter, session_id="s-1", scope_dir=workspace)
        result = str(await tool.acall(action="git"))

        assert "requires git_command" in result

    async def test_path_escape_is_refused(self, workspace, emitter):
        tool = create_inspect_tool(emitter, session_id="s-1", scope_dir=workspace)

        for escaping in ("../../../../etc/passwd", "/etc/passwd"):
            result = str(await tool.acall(action="read", path=escaping))
            assert "Error" in result
            assert "root:" not in result

    async def test_drop_removes_the_clone(self, workspace, emitter):
        tool = create_inspect_tool(emitter, session_id="s-1", scope_dir=workspace)

        result = str(await tool.acall(action="drop", repo="acme/widget"))

        assert "Dropped acme/widget" in result
        assert not (workspace / "repos" / "acme__widget").exists()

    async def test_output_is_capped_to_the_context_budget(self, workspace, emitter):
        tool = create_inspect_tool(
            emitter, session_id="s-1", scope_dir=workspace, context_window=1024
        )
        (workspace / "repos" / "acme__widget" / "big.txt").write_text(
            "x" * 50_000 + "\n"
        )

        result = str(await tool.acall(action="read", path="big.txt", max_lines=2000))

        assert len(result) < 50_000
        assert "truncated to fit the context budget" in result


# ----------------------------------------------------------------------------
# Registration and prompt injection
# ----------------------------------------------------------------------------


class TestNoShell:
    """The read-only promise is enforced mechanically, not by convention."""

    async def test_workspace_module_never_spawns_a_shell(self):
        import ast
        import inspect as pyinspect

        from tensortruth.services import repo_workspace

        source = pyinspect.getsource(repo_workspace)
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                for keyword in node.keywords:
                    if keyword.arg == "shell":
                        assert not getattr(keyword.value, "value", False)
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                assert not (
                    node.value.id in {"os", "subprocess"}
                    and node.attr in {"system", "popen", "getoutput", "getstatusoutput"}
                )
            if isinstance(node, ast.Name):
                assert node.id != "eval"

        assert "shell=True" not in source
        # The only subprocess entry point allowed is an argv list.
        assert "subprocess.run(" in source


class TestRegistration:
    async def test_registered_in_the_agent_toolset(self):
        tools = create_all_tool_wrappers(MagicMock(), AsyncMock(), session_id="s-1")

        assert "inspect" in [tool.metadata.name for tool in tools]

    async def test_not_registered_without_a_session(self):
        tools = create_all_tool_wrappers(MagicMock(), AsyncMock())

        assert "inspect" not in [tool.metadata.name for tool in tools]


class TestPromptSection:
    def _service(self, session_id: str = "s-1") -> OrchestratorService:
        service = object.__new__(OrchestratorService)
        service._session_id = session_id
        service._module_descriptions = []
        service._custom_instructions = None
        service._project_metadata = None
        service._max_iterations = 10
        service._context_window = 16384
        return service

    async def test_section_lists_the_cloned_repo(self, workspace, monkeypatch):
        monkeypatch.setattr(
            "tensortruth.app_utils.paths.get_session_dir", lambda _sid: workspace
        )

        prompt = self._service()._build_system_prompt([])

        assert "CLONED REPOSITORIES" in prompt
        assert "acme/widget" in prompt

    async def test_section_is_absent_without_repos(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "tensortruth.app_utils.paths.get_session_dir",
            lambda _sid: tmp_path / "nothing",
        )

        prompt = self._service()._build_system_prompt([])

        assert "CLONED REPOSITORIES" not in prompt

    async def test_section_disappears_after_drop(self, workspace, emitter, monkeypatch):
        monkeypatch.setattr(
            "tensortruth.app_utils.paths.get_session_dir", lambda _sid: workspace
        )
        tool = create_inspect_tool(emitter, session_id="s-1", scope_dir=workspace)

        await tool.acall(action="drop", repo="acme/widget")
        prompt = self._service()._build_system_prompt([])

        assert "CLONED REPOSITORIES" not in prompt


class TestFormatAge:
    async def test_recent_clone_reads_as_just_now(self):
        assert _format_age(time.time()) == "just now"
