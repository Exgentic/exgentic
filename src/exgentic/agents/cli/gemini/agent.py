# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

import os
from typing import Any, ClassVar

from ....core.types import ModelSettings
from ..base import ExecutionBackend, ProxyBackedAgent, ProxyBackedMCPAgentInstance
from .cli import GeminiCLI, GeminiCLIConfig


class GeminiAgentInstance(ProxyBackedMCPAgentInstance):
    """Self-contained Gemini CLI agent that runs through a LiteLLM proxy."""

    def __init__(
        self,
        session_id: str,
        model_id: str,
        max_steps: int = 150,
        execution_backend: ExecutionBackend = ExecutionBackend.DOCKER,
        model_settings: ModelSettings | None = None,
    ):
        self._gemini_model_alias = "gemini-2.5-pro"
        super().__init__(
            session_id,
            model_id,
            max_steps=max_steps,
            model_alias=self._gemini_model_alias,
            execution_backend=execution_backend,
            model_settings=model_settings,
        )
        self._gemini_log = self.paths.agent_dir / "gemini_cli.log"

    @property
    def cli_display_name(self) -> str:
        return "Gemini CLI"

    def _build_cli(self) -> GeminiCLI:
        cfg_dir = self.paths.agent_dir / "gemini_config"
        return GeminiCLI(
            env=os.environ.copy(),
            log_path=self._gemini_log,
            config_dir=cfg_dir,
            logger=self.logger,
            runner=self.execution_backend,
        )

    def _run_cli(
        self,
        cli: GeminiCLI,
        prompt: str,
        mcp_host: str,
        mcp_port: int,
        proxy: Any,
    ) -> Any:
        config = GeminiCLIConfig(
            mcp_host=mcp_host,
            mcp_port=mcp_port,
            provider_url=proxy.base_url,
            backend_model=self.model_id,
            gemini_model=self._gemini_model_alias,
            allowed_mcp_server_names=["environment"],
            allowed_tools=[action.name for action in self.actions],
        )
        result = cli.run(prompt=prompt, config=config)
        return result.stdout


class GeminiAgent(ProxyBackedAgent):
    display_name: ClassVar[str] = "Gemini CLI"
    slug_name: ClassVar[str] = "gemini_cli"
    execution_backend: ExecutionBackend = ExecutionBackend.DOCKER

    @classmethod
    def _get_instance_class(cls):
        return GeminiAgentInstance

    def get_models_names(self) -> list[str]:  # type: ignore[override]
        return [str(self.model_id)]
