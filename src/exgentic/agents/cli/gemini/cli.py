# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from ..base import BaseCLIConfig, BaseCLIWrapper, ExecutionBackend


class GeminiCLIConfig(BaseCLIConfig):
    backend_model: str
    gemini_model: str = "gemini-2.5-pro"
    env_key: str = "GEMINI_API_KEY"
    server_name: str = "environment"
    output_format: str = "text"
    approval_mode: str = "yolo"
    allowed_mcp_server_names: Optional[list[str]] = None
    allowed_tools: Optional[list[str]] = None
    image: str = "exgentic-gemini:dev"


class GeminiCLI(BaseCLIWrapper):
    """Lightweight wrapper for running the Gemini CLI headlessly."""

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
        self._settings_path: Optional[Path] = None
        self.config_prefix = "gemini_cli_"
        self.spawn_error_message = "Failed to start Gemini CLI"

    # Required hooks --------------------------------------------------

    def build_env(self, *, cfg_root: Path, prompt: str, config: GeminiCLIConfig) -> dict[str, str]:
        env = self.env.copy()
        env["GOOGLE_GEMINI_BASE_URL"] = config.provider_url

        api_key = env.get(config.env_key) or env.get("GEMINI_API_KEY") or "dummy-api-key"
        env[config.env_key] = api_key
        env["GEMINI_API_KEY"] = api_key

        env["HOME"] = str(cfg_root)
        return env

    def build_command(self, *, cfg_root: Path, prompt: str, config: GeminiCLIConfig) -> list[str]:
        gemini_cfg_dir = cfg_root / ".gemini"
        settings_path = gemini_cfg_dir / "settings.json"
        self._settings_path = settings_path

        # Rewrite localhost addresses to host gateway for container runners
        mcp_host = config.mcp_host
        from ..command_runner import ContainerRunner

        if isinstance(self.runner, ContainerRunner) and mcp_host in ("0.0.0.0", "127.0.0.1", "localhost"):
            mcp_host = self.runner.host_gateway

        gemini_cfg_dir.mkdir(parents=True, exist_ok=True)
        mcp_url = f"http://{mcp_host}:{config.mcp_port}/mcp"
        self._ensure_settings(settings_path, config.server_name, mcp_url)

        cmd: list[str] = [
            "gemini",
            "--model",
            config.gemini_model,
            "--output-format",
            config.output_format,
            "--approval-mode",
            config.approval_mode,
        ]
        if config.allowed_mcp_server_names:
            cmd.extend(["--allowed-mcp-server-names", *config.allowed_mcp_server_names])
        if config.allowed_tools:
            cmd.extend(["--allowed-tools", *config.allowed_tools])
        cmd.append(prompt)
        return cmd

    # Internal helpers -------------------------------------------------

    def _ensure_settings(self, settings_path: Path, server_name: str, mcp_url: str) -> None:
        settings: dict[str, object] = {}
        if settings_path.exists():
            try:
                with open(settings_path, encoding="utf-8-sig") as fh:
                    settings = json.load(fh)
            except Exception:
                settings = {}

        servers = settings.setdefault("mcpServers", {})
        if not isinstance(servers, dict):
            servers = {}
            settings["mcpServers"] = servers

        servers[server_name] = {"httpUrl": mcp_url, "trust": True}

        settings_path.parent.mkdir(parents=True, exist_ok=True)
        with open(settings_path, "w", encoding="utf-8") as fh:
            json.dump(settings, fh, indent=2)
