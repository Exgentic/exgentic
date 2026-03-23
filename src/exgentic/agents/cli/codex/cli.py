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
    image: str = "exgentic-codex:dev"


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
        env = self.env.copy()
        env["OPENAI_API_BASE"] = config.provider_url
        api_key = env.get(config.env_key) or "dummy-api-key"
        env[config.env_key] = api_key
        return env

    def build_command(self, *, cfg_root: Path, prompt: str, config: CodexCLIConfig) -> list[str]:
        # Rewrite localhost addresses to host gateway for container runners
        mcp_host = config.mcp_host
        provider_url = config.provider_url
        from ..command_runner import ContainerRunner

        if isinstance(self.runner, ContainerRunner):
            if mcp_host in ("0.0.0.0", "127.0.0.1", "localhost"):
                mcp_host = self.runner.host_gateway
            for local in ("://127.0.0.1:", "://localhost:"):
                if local in provider_url:
                    provider_url = provider_url.replace(local, f"://{self.runner.host_gateway}:")
                    break

        mcp_url = f"http://{mcp_host}:{config.mcp_port}/mcp"
        overrides = [
            f'mcp_servers.environment.url="{mcp_url}"',
            f'model_providers.temp.name="{config.provider_name}"',
            f'model_providers.temp.base_url="{provider_url}"',
            f'model_providers.temp.env_key="{config.env_key}"',
            f'profiles.{config.profile}.model_provider="temp"',
            f'profiles.{config.profile}.model="{config.model_id}"',
        ]

        cmd: list[str] = ["codex", "exec", "--skip-git-repo-check", "--profile", config.profile]
        for override in overrides:
            cmd.extend(["-c", override])
        cmd.append(prompt)

        return cmd
