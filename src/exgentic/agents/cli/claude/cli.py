# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from ..base import BaseCLIConfig, BaseCLIWrapper, ExecutionBackend


class ClaudeCLIConfig(BaseCLIConfig):
    backend_model: str
    claude_model: str
    env_key: str = "ANTHROPIC_API_KEY"
    auth_token_env: str = "ANTHROPIC_AUTH_TOKEN"
    output_format: str = "text"
    skip_permissions: bool = True
    max_turns: int = 150
    mcp_only: bool = True  # New: only allow MCP tools
    allowed_tools: Optional[list[str]] = None
    image: str = "exgentic-claude-code:dev"
    image_workdir: str = "/work"
    mcp_timeout_ms: int = 600_000
    mcp_tool_timeout_ms: int = 1_800_000


class ClaudeCodeCLI(BaseCLIWrapper):
    """Thin wrapper for running Claude Code's CLI in print mode."""

    def __init__(
        self,
        env: Optional[dict[str, str]] = None,
        log_path: Optional[Path] = None,
        config_dir: Optional[Path] = None,
        logger=None,
        runner: ExecutionBackend = ExecutionBackend.PROCESS,
    ) -> None:
        super().__init__(
            env=env,
            log_path=log_path,
            config_dir=config_dir,
            logger=logger,
            runner=runner,
        )
        self._mcp_config_path: Optional[Path] = None
        self.config_prefix = "claude_code_cli_"
        self.spawn_error_message = "Failed to start Claude CLI"

    # Required hooks --------------------------------------------------

    def build_env(self, *, cfg_root: Path, prompt: str, config: ClaudeCLIConfig) -> dict[str, str]:
        env = self.env.copy()
        env["ANTHROPIC_BASE_URL"] = config.provider_url
        if config.env:
            env.update(config.env)

        token = env.get(config.auth_token_env) or env.get(config.env_key) or "dummy-api-key"
        env[config.env_key] = token
        env[config.auth_token_env] = token

        # Isolate config/settings/state under a temp directory
        env["HOME"] = str(cfg_root)

        return env

    def build_command(self, *, cfg_root: Path, prompt: str, config: ClaudeCLIConfig) -> list[str]:
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        mcp_cfg_path = (cfg_root / "mcp.json").absolute()
        self._mcp_config_path = mcp_cfg_path

        # Use host.docker.internal for Docker containers when server binds to 0.0.0.0
        mcp_host = config.mcp_host
        from ..command_runner import DockerRunner, PodmanRunner

        if isinstance(self.runner, (DockerRunner, PodmanRunner)) and mcp_host == "0.0.0.0":
            mcp_host = "host.docker.internal"

        mcp_url = f"http://{mcp_host}:{config.mcp_port}/mcp"
        self._write_mcp_config(mcp_cfg_path, mcp_url)
        self._write_settings_config(cfg_root)

        # Verify config was written
        if not mcp_cfg_path.exists():
            raise RuntimeError(f"Failed to write MCP config to {mcp_cfg_path}")

        cmd: list[str] = [
            "npx",
            "claude",
            "-p",  # print mode, single-shot
            "--model",
            config.claude_model,
            "--mcp-config",
            str(mcp_cfg_path),
            "--strict-mcp-config",
            "--output-format",
            config.output_format,
            "--debug",
            "--mcp-debug",
            "--no-session-persistence",
            "--append-system-prompt",
            "AUTONOMOUS SOLO MODE.\n"
            "First, discover available MCP tools by listing tools from the environment server.\n"
            "Then use those tools to complete the task.\n"
            "Never ask for clarification - make reasonable assumptions.\n"
            "You have NO filesystem access - work exclusively through MCP tools.",
            "--dangerously-skip-permissions",
            "--max-turns",
            str(config.max_turns),
        ]

        # if config.allowed_tools:
        #     cmd.extend(["--allowedTools", ",".join(config.allowed_tools)])

        # Ensure the following argument is treated purely as the prompt, not part of a variadic flag.
        cmd.append("--")
        cmd.append(prompt)
        return cmd

    # Internal helpers -------------------------------------------------

    def _write_mcp_config(self, path: Path, mcp_url: str) -> None:
        """Write MCP server configuration."""
        config = {"mcpServers": {"environment": {"type": "http", "url": mcp_url}}}
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(config, fh, indent=2)

    def _write_settings_config(self, home_dir: Path) -> None:
        """Write settings that completely block filesystem access."""
        settings_path = home_dir / ".claude" / "settings.json"

        settings = {
            "enableAllProjectMcpServers": True,
        }

        settings_path.parent.mkdir(parents=True, exist_ok=True)
        with open(settings_path, "w", encoding="utf-8") as fh:
            json.dump(settings, fh, indent=2)
