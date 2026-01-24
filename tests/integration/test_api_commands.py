"""Tests for command system - Tool/Agent trigger framework.

Following TDD: These tests are written BEFORE implementation.
They will fail initially, then pass once we implement the command system.
"""

import pytest
from fastapi.testclient import TestClient

from tensortruth.api.main import app
from tensortruth.api.routes.commands import (
    CommandRegistry,
    HelpCommand,
    ToolCommand,
    WebSearchCommand,
)


@pytest.fixture
def client():
    """Test client for API."""
    return TestClient(app)


@pytest.fixture
def registry():
    """Fresh command registry for testing."""
    return CommandRegistry()


class TestCommandRegistry:
    """Test the command registration and discovery system."""

    def test_registry_initialization(self, registry):
        """Test that registry initializes empty."""
        assert isinstance(registry.commands, dict)
        assert len(registry.commands) == 0

    def test_register_command(self, registry):
        """Test registering a command."""

        # Create a dummy command
        class DummyCommand(ToolCommand):
            name = "test"
            aliases = ["t", "tst"]
            description = "Test command"
            usage = "/test <args>"

            async def execute(self, args, session, websocket):
                pass

        cmd = DummyCommand()
        registry.register(cmd)

        # Verify command registered under name
        assert "test" in registry.commands
        assert registry.commands["test"] is cmd

        # Verify command registered under aliases
        assert "t" in registry.commands
        assert "tst" in registry.commands
        assert registry.commands["t"] is cmd
        assert registry.commands["tst"] is cmd

    def test_get_command(self, registry):
        """Test retrieving a command by name or alias."""

        class DummyCommand(ToolCommand):
            name = "search"
            aliases = ["s"]
            description = "Search command"
            usage = "/search <query>"

            async def execute(self, args, session, websocket):
                pass

        cmd = DummyCommand()
        registry.register(cmd)

        # Get by name
        assert registry.get("search") is cmd
        # Get by alias
        assert registry.get("s") is cmd
        # Get non-existent
        assert registry.get("nonexistent") is None

    def test_get_command_case_insensitive(self, registry):
        """Test that command lookup is case-insensitive."""

        class DummyCommand(ToolCommand):
            name = "test"
            aliases = []
            description = "Test"
            usage = "/test"

            async def execute(self, args, session, websocket):
                pass

        cmd = DummyCommand()
        registry.register(cmd)

        assert registry.get("test") is cmd
        assert registry.get("TEST") is cmd
        assert registry.get("TeSt") is cmd

    def test_list_all_commands(self, registry):
        """Test listing all registered commands."""

        class Command1(ToolCommand):
            name = "cmd1"
            aliases = ["c1"]
            description = "First command"
            usage = "/cmd1"

            async def execute(self, args, session, websocket):
                pass

        class Command2(ToolCommand):
            name = "cmd2"
            aliases = []
            description = "Second command"
            usage = "/cmd2"

            async def execute(self, args, session, websocket):
                pass

        cmd1 = Command1()
        cmd2 = Command2()
        registry.register(cmd1)
        registry.register(cmd2)

        commands_list = registry.list_all()

        # Should return unique commands (not duplicated by aliases)
        assert len(commands_list) == 2

        # Verify command data structure
        cmd1_data = next(c for c in commands_list if c["name"] == "cmd1")
        assert cmd1_data["aliases"] == ["c1"]
        assert cmd1_data["description"] == "First command"
        assert cmd1_data["usage"] == "/cmd1"

        cmd2_data = next(c for c in commands_list if c["name"] == "cmd2")
        assert cmd2_data["aliases"] == []
        assert cmd2_data["description"] == "Second command"


class TestCommandsEndpoint:
    """Test the GET /api/commands endpoint."""

    def test_get_commands_endpoint_exists(self, client):
        """Test that GET /api/commands endpoint exists."""
        response = client.get("/api/commands")
        assert response.status_code == 200

    def test_get_commands_returns_json(self, client):
        """Test that endpoint returns JSON with commands list."""
        response = client.get("/api/commands")
        data = response.json()

        assert "commands" in data
        assert isinstance(data["commands"], list)

    def test_get_commands_includes_help(self, client):
        """Test that help command is included in response."""
        response = client.get("/api/commands")
        data = response.json()

        commands = data["commands"]
        help_cmd = next((c for c in commands if c["name"] == "help"), None)

        assert help_cmd is not None
        assert help_cmd["description"]
        assert help_cmd["usage"] == "/help"

    def test_get_commands_includes_web_search(self, client):
        """Test that web search command is included."""
        response = client.get("/api/commands")
        data = response.json()

        commands = data["commands"]
        web_cmd = next((c for c in commands if c["name"] == "web"), None)

        assert web_cmd is not None
        assert "search" in web_cmd["aliases"] or "websearch" in web_cmd["aliases"]
        assert web_cmd["description"]
        assert "/web" in web_cmd["usage"]


class TestHelpCommand:
    """Test the help command implementation."""

    @pytest.mark.asyncio
    async def test_help_command_attributes(self):
        """Test that HelpCommand has correct attributes."""
        cmd = HelpCommand()
        assert cmd.name == "help"
        assert cmd.usage == "/help"
        assert cmd.description

    @pytest.mark.asyncio
    async def test_help_command_execute(self):
        """Test that help command generates command list."""

        # Create mock websocket
        class MockWebSocket:
            def __init__(self):
                self.messages = []

            async def send_json(self, data):
                self.messages.append(data)

        ws = MockWebSocket()
        cmd = HelpCommand()

        # Execute help command
        await cmd.execute("", {}, ws)

        # Should have sent one message
        assert len(ws.messages) == 1

        # Message should be type "done" with markdown content
        msg = ws.messages[0]
        assert msg["type"] == "done"
        assert "content" in msg
        assert "Available Commands" in msg["content"] or "Commands" in msg["content"]


class TestWebSearchCommand:
    """Test the web search command implementation."""

    @pytest.mark.asyncio
    async def test_websearch_command_attributes(self):
        """Test that WebSearchCommand has correct attributes."""
        cmd = WebSearchCommand()
        assert cmd.name == "web"
        assert "search" in cmd.aliases or "websearch" in cmd.aliases
        assert cmd.description
        assert "/web" in cmd.usage

    @pytest.mark.asyncio
    @pytest.mark.requires_network
    async def test_websearch_command_execute_sends_status(self):
        """Test that web search sends status updates."""

        class MockWebSocket:
            def __init__(self):
                self.messages = []

            async def send_json(self, data):
                self.messages.append(data)

        ws = MockWebSocket()
        cmd = WebSearchCommand()

        # Execute search
        await cmd.execute("Python testing best practices", {}, ws)

        # Should have sent multiple messages
        assert len(ws.messages) > 0

        # Get message types
        message_types = [m.get("type") for m in ws.messages]

        # Should have status messages (agent_progress or status type)
        status_msgs = [
            m for m in ws.messages if m.get("type") in ["status", "agent_progress"]
        ]
        assert len(status_msgs) > 0, f"No status messages found. Got: {message_types}"

        # Should have a done message at the end
        done_msgs = [m for m in ws.messages if m.get("type") == "done"]
        assert len(done_msgs) == 1

    @pytest.mark.asyncio
    async def test_websearch_command_with_empty_args(self):
        """Test web search with empty arguments."""

        class MockWebSocket:
            def __init__(self):
                self.messages = []

            async def send_json(self, data):
                self.messages.append(data)

        ws = MockWebSocket()
        cmd = WebSearchCommand()

        # Execute with empty query
        await cmd.execute("", {}, ws)

        # Should send error or handle gracefully
        messages = ws.messages
        assert len(messages) > 0

        # Either error message or done with empty/error content
        last_msg = messages[-1]
        assert last_msg["type"] in ["error", "done"]


class TestWebSocketCommandDetection:
    """Test command detection in WebSocket chat handler."""

    @pytest.mark.asyncio
    async def test_command_detection_at_start(self):
        """Test detecting command at start of message."""
        # This will test the actual WebSocket handler once implemented
        # For now, we test the regex pattern that will be used

        import re

        pattern = r"/(\w+)(?:\s+(.+))?"

        # Command at start
        match = re.search(pattern, "/web search query")
        assert match is not None
        assert match.group(1) == "web"
        assert match.group(2) == "search query"

    @pytest.mark.asyncio
    async def test_command_detection_in_middle(self):
        """Test detecting command in middle of message."""
        import re

        pattern = r"/(\w+)(?:\s+(.+))?"

        # Command in middle
        match = re.search(pattern, "I don't know - /web search for it")
        assert match is not None
        assert match.group(1) == "web"
        assert match.group(2) == "search for it"

    @pytest.mark.asyncio
    async def test_command_detection_no_args(self):
        """Test detecting command without arguments."""
        import re

        pattern = r"/(\w+)(?:\s+(.+))?"

        # Command without args
        match = re.search(pattern, "/help")
        assert match is not None
        assert match.group(1) == "help"
        assert match.group(2) is None

    @pytest.mark.asyncio
    async def test_no_command_detection(self):
        """Test that regular messages don't match command pattern."""
        import re

        pattern = r"/(\w+)(?:\s+(.+))?"

        # No command
        match = re.search(pattern, "This is a regular message")
        assert match is None

    @pytest.mark.asyncio
    async def test_command_with_special_chars(self):
        """Test command with special characters in args."""
        import re

        pattern = r"/(\w+)(?:\s+(.+))?"

        # Command with special chars
        match = re.search(pattern, '/web "quoted search" & special')
        assert match is not None
        assert match.group(1) == "web"
        assert match.group(2) == '"quoted search" & special'


class TestCommandIntegration:
    """Integration tests for command execution via WebSocket."""

    @pytest.mark.asyncio
    async def test_help_command_via_websocket(self, client):
        """Test executing help command via WebSocket."""
        # Create a test session first
        session_response = client.post(
            "/api/sessions",
            json={"title": "Test Command Session", "modules": [], "params": {}},
        )
        assert session_response.status_code == 201
        session_id = session_response.json()["session_id"]

        # Connect to WebSocket and send help command
        with client.websocket_connect(f"/ws/chat/{session_id}") as websocket:
            # Send help command
            websocket.send_json({"prompt": "/help"})

            # Collect messages
            messages = []
            while True:
                msg = websocket.receive_json()
                messages.append(msg)
                if msg.get("type") == "done":
                    break

            # Should have received done message with help content
            done_msg = next(m for m in messages if m.get("type") == "done")
            assert "content" in done_msg
            assert len(done_msg["content"]) > 0

    @pytest.mark.asyncio
    async def test_unknown_command_via_websocket(self, client):
        """Test that unknown commands return helpful error."""
        # Create test session
        session_response = client.post(
            "/api/sessions",
            json={"title": "Test Unknown Command", "modules": [], "params": {}},
        )
        assert session_response.status_code == 201
        session_id = session_response.json()["session_id"]

        # Connect and send unknown command
        with client.websocket_connect(f"/ws/chat/{session_id}") as websocket:
            websocket.send_json({"prompt": "/unknowncommand test"})

            # Should receive error message
            msg = websocket.receive_json()
            assert msg.get("type") == "error"
            assert "Unknown command" in msg.get("detail", "")
            assert "/help" in msg.get("detail", "")

    @pytest.mark.asyncio
    async def test_command_in_middle_of_message(self, client):
        """Test that commands anywhere in message are detected."""
        session_response = client.post(
            "/api/sessions",
            json={"title": "Test Mid-Message Command", "modules": [], "params": {}},
        )
        assert session_response.status_code == 201
        session_id = session_response.json()["session_id"]

        with client.websocket_connect(f"/ws/chat/{session_id}") as websocket:
            websocket.send_json({"prompt": "I need help - /help please"})

            # Collect messages until done
            messages = []
            while True:
                msg = websocket.receive_json()
                messages.append(msg)
                if msg.get("type") in ["done", "error"]:
                    break

            # Should have processed as help command
            final_msg = messages[-1]
            assert final_msg["type"] == "done"
