# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..base import BaseCLIConfig, BaseCLIWrapper, ExecutionBackend


class CodexCLIConfig(BaseCLIConfig):
    model_id: str
    env_key: str = "OPENAI_API_KEY"
    profile: str = "temp_model"
    provider_name: str = "Provider"


class CodexCLI(BaseCLIWrapper):
    """Standalone wrapper for launching Codex CLI."""

    def __init__(
        self,
        env: Optional[dict[str, str]] = None,
        log_path: Optional[Path] = None,
        logger=None,
        runner: ExecutionBackend = ExecutionBackend.PROCESS,
    ) -> None:
        super().__init__(env=env, log_path=log_path, config_dir=None, logger=logger, runner=runner)
        self.config_prefix = "codex_cli_"
        self.spawn_error_message = "Failed to start Codex CLI"

    def build_env(self, *, cfg_root: Path, prompt: str, config: CodexCLIConfig) -> dict[str, str]:
        return self.env.copy()

    def build_command(self, *, cfg_root: Path, prompt: str, config: CodexCLIConfig) -> list[str]:
        mcp_url = f"http://{config.mcp_host}:{config.mcp_port}/mcp"
        overrides = [
            f'mcp_servers.environment.url="{mcp_url}"',
            f'model_providers.temp.name="{config.provider_name}"',
            f'model_providers.temp.base_url="{config.provider_url}"',
            f'model_providers.temp.env_key="{config.env_key}"',
            f'profiles.{config.profile}.model_provider="temp"',
            f'profiles.{config.profile}.model="{config.model_id}"',
        ]

        cmd: list[str] = ["codex", "exec", "--profile", config.profile]
        for override in overrides:
            cmd.extend(["-c", override])
        cmd.append(prompt)

        return cmd
