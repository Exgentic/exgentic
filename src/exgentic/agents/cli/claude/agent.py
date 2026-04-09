# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

import os
from typing import Any, ClassVar

from ....core.types import ModelSettings
from ..base import ExecutionBackend, ProxyBackedAgent, ProxyBackedMCPAgentInstance
from .cli import ClaudeCLIConfig, ClaudeCodeCLI


class ClaudeCodeAgentInstance(ProxyBackedMCPAgentInstance):
    """Self-contained Claude Code CLI agent routed through a LiteLLM proxy."""

    # All Claude model IDs that Claude Code (and its sub-agents) may request.
    # Every alias here is routed to the single backend model under test.
    _CLAUDE_MODEL_ALIASES: ClassVar[list[str]] = [
        # Aliases
        "claude-sonnet-4-5",
        "claude-sonnet-4-6",
        "claude-opus-4-5",
        "claude-opus-4-6",
        "claude-haiku-4-5",
        # Dated snapshots
        "claude-sonnet-4-5-20250514",
        "claude-sonnet-4-6-20250715",
        "claude-opus-4-5-20250514",
        "claude-opus-4-6-20250715",
        "claude-haiku-4-5-20251001",
        # Legacy models
        "claude-3-5-sonnet-20241022",
        "claude-3-5-sonnet-20240620",
        "claude-3-5-haiku-20241022",
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
    ]

    def __init__(
        self,
        session_id: str,
        model_id: str,
        max_steps: int = 150,
        execution_backend: ExecutionBackend = ExecutionBackend.AUTO,
        model_settings: ModelSettings | None = None,
    ):
        # The alias is what we ask Claude Code CLI for; the proxy maps it to the backend model.
        self._claude_model_alias = "claude-3-5-sonnet-20241022"
        super().__init__(
            session_id,
            model_id,
            max_steps=max_steps,
            model_alias=self._claude_model_alias,
            execution_backend=execution_backend,
            model_settings=model_settings,
        )
        self._claude_log = self.paths.agent_dir / "claude_cli.log"

    @property
    def cli_display_name(self) -> str:
        return "Claude Code CLI"

    def _build_cli(self) -> ClaudeCodeCLI:
        cfg_dir = self.paths.agent_dir / "claude_code_config"
        return ClaudeCodeCLI(
            env=os.environ.copy(),
            log_path=self._claude_log,
            config_dir=cfg_dir,
            logger=self.logger,
            runner=self.execution_backend,
        )

    def _run_cli(
        self,
        cli: ClaudeCodeCLI,
        prompt: str,
        mcp_host: str,
        mcp_port: int,
        proxy: Any,
    ) -> Any:
        # allowed_tools = [f"mcp__environment__{action.name}" for action in self.actions]
        config = ClaudeCLIConfig(
            mcp_host=mcp_host,
            mcp_port=mcp_port,
            provider_url=proxy.base_url,
            backend_model=self.model_id,
            claude_model=self._claude_model_alias,
            # allowed_tools=allowed_tools,
            max_turns=self.max_steps,
        )
        config.env = {
            "MCP_TIMEOUT": str(config.mcp_timeout_ms),
            "MCP_TOOL_TIMEOUT": str(config.mcp_tool_timeout_ms),
        }
        result = cli.run(prompt=prompt, config=config)
        return result.stdout

    def _proxy_alias_map(self) -> dict[str, str]:
        """Route all known Claude models to the backend model under test."""
        return {alias: self.model_id for alias in self._CLAUDE_MODEL_ALIASES}

    def _stringify_empty_output(self) -> bool:
        return True


class ClaudeCodeAgent(ProxyBackedAgent):
    display_name: ClassVar[str] = "Claude Code CLI"
    slug_name: ClassVar[str] = "claude_code"
    execution_backend: ExecutionBackend = ExecutionBackend.AUTO

    @classmethod
    def _get_instance_class(cls):
        return ClaudeCodeAgentInstance

    def get_models_names(self) -> list[str]:  # type: ignore[override]
        return [str(self.model_id)]
