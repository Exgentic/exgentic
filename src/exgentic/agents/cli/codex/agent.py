# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

import os
from typing import Any, ClassVar

from ....core.types import ActionType, ModelSettings
from ..base import ProxyBackedAgent, ProxyBackedMCPAgentInstance
from .cli import CodexCLI, CodexCLIConfig, ExecutionBackend


class CodexAgentInstance(ProxyBackedMCPAgentInstance):
    """Self-contained Codex CLI agent that runs through a LiteLLM proxy."""

    def __init__(
        self,
        session_id: str,
        task: str,
        context: dict[str, Any],
        actions: list[ActionType],
        model_id: str,
        max_steps: int = 150,
        runner: ExecutionBackend = ExecutionBackend.PROCESS,
        model_settings: ModelSettings | None = None,
    ):
        super().__init__(
            session_id,
            task,
            context,
            actions,
            model_id,
            max_steps=max_steps,
            runner=runner,
            model_settings=model_settings,
        )
        self._codex_log = self.paths.agent_dir / "codex_cli.log"

    @property
    def cli_display_name(self) -> str:
        return "Codex CLI"

    def _build_cli(self) -> CodexCLI:
        return CodexCLI(env=os.environ.copy(), log_path=self._codex_log, logger=self.logger)

    def _run_cli(
        self,
        cli: CodexCLI,
        prompt: str,
        mcp_host: str,
        mcp_port: int,
        proxy: Any,
    ) -> Any:
        prev_base = os.environ.get("OPENAI_API_BASE")
        os.environ["OPENAI_API_BASE"] = proxy.base_url
        try:
            config = CodexCLIConfig(
                mcp_host=mcp_host,
                mcp_port=mcp_port,
                model_id=self.model_id,
                provider_url=proxy.base_url,
            )
            result = cli.run(prompt=prompt, config=config)
            return result.stdout
        finally:
            if prev_base is None:
                os.environ.pop("OPENAI_API_BASE", None)
            else:
                os.environ["OPENAI_API_BASE"] = prev_base


class CodexAgent(ProxyBackedAgent):
    display_name: ClassVar[str] = "Codex CLI"
    slug_name: ClassVar[str] = "codex_cli"
    agent_cls: ClassVar[type[ProxyBackedMCPAgentInstance]] = CodexAgentInstance
