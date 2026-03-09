"""Service for managing MCP server configurations.

CRUD operations on ~/.tensortruth/mcp_servers.json. Does NOT manage live
connections — that stays in MCPServerRegistry.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any

from tensortruth.agents.config import MCPServerConfig, MCPServerType
from tensortruth.agents.server_registry import DEFAULT_SERVERS, get_user_mcp_config_path

logger = logging.getLogger(__name__)

# Pre-configured templates for known MCP servers
MCP_SERVER_PRESETS: dict[str, dict[str, Any]] = {
    "context7": {
        "name": "context7",
        "type": "stdio",
        "command": "npx",
        "args": ["-y", "@upstash/context7-mcp@latest"],
        "description": "Context7 — library documentation lookup",
        "enabled": True,
    },
    "github": {
        "name": "github",
        "type": "stdio",
        "command": "docker",
        "args": [
            "run",
            "-i",
            "--rm",
            "-e",
            "GITHUB_PERSONAL_ACCESS_TOKEN",
            "-e",
            "GITHUB_TOOLSETS=repos,issues,pull_requests,actions",
            "ghcr.io/github/github-mcp-server",
        ],
        "env": {
            "GITHUB_PERSONAL_ACCESS_TOKEN": "$GITHUB_PERSONAL_ACCESS_TOKEN",
        },
        "description": "GitHub — repos, issues, PRs, code search",
        "enabled": True,
    },
    "huggingface": {
        "name": "huggingface",
        "type": "stdio",
        "command": "npx",
        "args": ["-y", "@llmindset/hf-mcp-server"],
        "env": {
            "HF_TOKEN": "$HF_TOKEN",
        },
        "description": "HuggingFace — models, datasets, papers, spaces",
        "enabled": True,
    },
    "playwright": {
        "name": "playwright",
        "type": "stdio",
        "command": "npx",
        "args": ["@playwright/mcp@latest"],
        "description": "Playwright — browser automation for JS-heavy sites",
        "enabled": True,
    },
    "memory": {
        "name": "memory",
        "type": "stdio",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-memory"],
        "description": "Knowledge Graph — persistent entity and relationship memory",
        "enabled": True,
    },
    "scholarly-research": {
        "name": "scholarly-research",
        "type": "stdio",
        "command": "npx",
        "args": ["-y", "scholarly-research-mcp"],
        "description": "Scholarly Research — PubMed, Google Scholar, ArXiv, JSTOR",
        "enabled": True,
    },
    "wikipedia": {
        "name": "wikipedia",
        "type": "stdio",
        "command": "npx",
        "args": ["-y", "wikipedia-mcp"],
        "description": "Wikipedia — direct article search and retrieval",
        "enabled": True,
    },
}


def _check_env_status(env: dict[str, str] | None) -> dict[str, bool]:
    """Check which env vars are resolved in the current environment."""
    if not env:
        return {}
    status: dict[str, bool] = {}
    for key, value in env.items():
        if isinstance(value, str) and value.startswith("$"):
            var_name = value[1:]
            status[var_name] = var_name in os.environ and bool(os.environ[var_name])
        else:
            status[key] = True
    return status


class MCPServerService:
    """Manages MCP server configurations on disk."""

    def __init__(self, config_path: Path | None = None):
        self._config_path = config_path or get_user_mcp_config_path()

    def _read_user_servers(self) -> list[dict[str, Any]]:
        """Read user server configs from JSON file."""
        if not self._config_path.exists():
            return []
        try:
            with open(self._config_path) as f:
                data = json.load(f)
            return data.get("servers", [])
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Corrupted MCP config at {self._config_path}: {e}")
            return []

    def _write_user_servers(self, servers: list[dict[str, Any]]) -> None:
        """Write server configs to JSON file atomically."""
        self._config_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._config_path.with_suffix(".tmp")
        with open(tmp_path, "w") as f:
            json.dump({"servers": servers}, f, indent=2)
        tmp_path.replace(self._config_path)

    def _builtin_names(self) -> set[str]:
        return {s.name for s in DEFAULT_SERVERS}

    def list_all(self) -> list[dict[str, Any]]:
        """List all servers: built-in + user, with metadata."""
        builtin_names = self._builtin_names()
        user_servers = self._read_user_servers()
        user_by_name = {s["name"]: s for s in user_servers}

        results: list[dict[str, Any]] = []

        # Built-in servers (with possible user overrides for enabled state)
        for default in DEFAULT_SERVERS:
            override = user_by_name.get(default.name)
            enabled = (
                override.get("enabled", default.enabled)
                if override
                else default.enabled
            )
            results.append(
                {
                    "name": default.name,
                    "type": default.type.value,
                    "command": default.command,
                    "args": default.args or [],
                    "url": default.url,
                    "description": default.description,
                    "env": default.env,
                    "enabled": enabled,
                    "builtin": True,
                    "env_status": _check_env_status(default.env),
                }
            )

        # User-configured servers (skip any that shadow built-in names)
        for server_data in user_servers:
            server_data = self._coerce_config(server_data)
            name = server_data.get("name", "")
            if name in builtin_names:
                continue  # Already handled above
            env = server_data.get("env")
            results.append(
                {
                    "name": name,
                    "type": server_data.get("type", "stdio"),
                    "command": server_data.get("command"),
                    "args": server_data.get("args", []),
                    "url": server_data.get("url"),
                    "description": server_data.get("description"),
                    "env": env,
                    "enabled": server_data.get("enabled", True),
                    "builtin": False,
                    "env_status": _check_env_status(env),
                }
            )

        return results

    def get_configured_server_names(self) -> set[str]:
        """Get names of all configured and enabled servers."""
        names = set()
        for server in self.list_all():
            if server.get("enabled", True):
                names.add(server["name"])
        return names

    @staticmethod
    def _coerce_config(config: dict[str, Any]) -> dict[str, Any]:
        """Normalize config values that may arrive as wrong types.

        LLMs sometimes pass args as a JSON string instead of a list, or
        enabled as a string instead of a bool.
        """
        config = dict(config)  # shallow copy
        args = config.get("args")
        if isinstance(args, str):
            try:
                parsed = json.loads(args)
                if isinstance(parsed, list):
                    config["args"] = parsed
            except (json.JSONDecodeError, ValueError):
                # Treat as single arg
                config["args"] = [args] if args.strip() else []
        enabled = config.get("enabled")
        if isinstance(enabled, str):
            config["enabled"] = enabled.lower() in ("true", "1", "yes")
        return config

    def add(self, config: dict[str, Any]) -> dict[str, Any]:
        """Add a new user server configuration."""
        config = self._coerce_config(config)
        name = config.get("name", "")
        if not name:
            raise ValueError("Server name is required")

        # Validate config
        server_type = MCPServerType(config.get("type", "stdio"))
        MCPServerConfig(
            name=name,
            type=server_type,
            command=config.get("command"),
            args=config.get("args", []),
            url=config.get("url"),
            description=config.get("description"),
            env=config.get("env"),
            enabled=config.get("enabled", True),
        )

        # Single read: check uniqueness + get list to append to
        servers = self._read_user_servers()
        existing_names = {s["name"] for s in servers}
        builtin_names = self._builtin_names()
        if name in existing_names or name in builtin_names:
            raise ValueError(f"Server '{name}' already exists")

        server_entry: dict[str, Any] = {
            "name": name,
            "type": config.get("type", "stdio"),
            "enabled": config.get("enabled", True),
        }
        if config.get("command"):
            server_entry["command"] = config["command"]
        if config.get("args"):
            server_entry["args"] = config["args"]
        if config.get("url"):
            server_entry["url"] = config["url"]
        if config.get("description"):
            server_entry["description"] = config["description"]
        if config.get("env"):
            server_entry["env"] = config["env"]

        servers.append(server_entry)
        self._write_user_servers(servers)

        env = config.get("env")
        return {
            **server_entry,
            "builtin": False,
            "env_status": _check_env_status(env),
        }

    def update(self, name: str, updates: dict[str, Any]) -> dict[str, Any]:
        """Update a user server configuration."""
        updates = self._coerce_config(updates)
        if name in self._builtin_names():
            raise ValueError(f"Cannot modify built-in server '{name}'")

        servers = self._read_user_servers()
        for i, server in enumerate(servers):
            if server.get("name") == name:
                for key, value in updates.items():
                    if value is not None:
                        server[key] = value
                # Re-validate
                server_type = MCPServerType(server.get("type", "stdio"))
                MCPServerConfig(
                    name=name,
                    type=server_type,
                    command=server.get("command"),
                    args=server.get("args", []),
                    url=server.get("url"),
                    description=server.get("description"),
                    env=server.get("env"),
                    enabled=server.get("enabled", True),
                )
                servers[i] = server
                self._write_user_servers(servers)
                env = server.get("env")
                return {
                    **server,
                    "builtin": False,
                    "env_status": _check_env_status(env),
                }

        raise ValueError(f"Server '{name}' not found")

    def remove(self, name: str) -> None:
        """Remove a user server configuration."""
        if name in self._builtin_names():
            raise ValueError(f"Cannot remove built-in server '{name}'")

        servers = self._read_user_servers()
        original_count = len(servers)
        servers = [s for s in servers if s.get("name") != name]
        if len(servers) == original_count:
            raise ValueError(f"Server '{name}' not found")
        self._write_user_servers(servers)

    def toggle(self, name: str, enabled: bool) -> dict[str, Any]:
        """Toggle a server's enabled state."""
        builtin_names = self._builtin_names()
        servers = self._read_user_servers()

        if name in builtin_names:
            # For built-in servers, store an override entry in user config
            for server in servers:
                if server.get("name") == name:
                    server["enabled"] = enabled
                    self._write_user_servers(servers)
                    # Return the built-in server info with override
                    default = next(s for s in DEFAULT_SERVERS if s.name == name)
                    return {
                        "name": default.name,
                        "type": default.type.value,
                        "command": default.command,
                        "args": default.args or [],
                        "url": default.url,
                        "description": default.description,
                        "env": default.env,
                        "enabled": enabled,
                        "builtin": True,
                        "env_status": _check_env_status(default.env),
                    }

            # No override entry yet — create one
            servers.append({"name": name, "enabled": enabled})
            self._write_user_servers(servers)
            default = next(s for s in DEFAULT_SERVERS if s.name == name)
            return {
                "name": default.name,
                "type": default.type.value,
                "command": default.command,
                "args": default.args or [],
                "url": default.url,
                "description": default.description,
                "env": default.env,
                "enabled": enabled,
                "builtin": True,
                "env_status": _check_env_status(default.env),
            }

        # User server
        for server in servers:
            if server.get("name") == name:
                server["enabled"] = enabled
                self._write_user_servers(servers)
                env = server.get("env")
                return {
                    **server,
                    "builtin": False,
                    "env_status": _check_env_status(env),
                }

        raise ValueError(f"Server '{name}' not found")

    @staticmethod
    def get_presets() -> dict[str, dict[str, Any]]:
        """Get available server presets."""
        return MCP_SERVER_PRESETS
