# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

from pathlib import Path

from exgentic.agents.cli.base import (
    BaseCLIConfig,
    BaseCLIWrapper,
    CLIResult,
    ExecutionBackend,
)
from exgentic.core.context import Context, save_runtime, set_context


class _DummyRunner:
    def __init__(self):
        self.env = None

    def run(self, *, cmd, env, cfg_root, config, spawn_error_message, stdin_devnull=False):
        self.env = env
        return CLIResult(stdout="", stderr="", code=0)

    def close(self) -> None:
        return None


class _DummyCLI(BaseCLIWrapper):
    def build_env(self, *, cfg_root: Path, prompt: str, config: BaseCLIConfig):
        return {}

    def build_command(self, *, cfg_root: Path, prompt: str, config: BaseCLIConfig):
        return ["echo", "ok"]


def test_cli_includes_context_env(tmp_path):
    ctx = Context(run_id="run-cli", output_dir=str(tmp_path), cache_dir="/tmp/cache")
    set_context(ctx)

    # Write runtime.json so _derive_runtime_dir resolves.
    runtime_dir = tmp_path / "run-cli"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    save_runtime(runtime_dir)

    runner = _DummyRunner()
    cli = _DummyCLI(runner=ExecutionBackend.PROCESS)
    cli.runner = runner
    cli.run(
        prompt="hi",
        config=BaseCLIConfig(
            mcp_host="127.0.0.1",
            mcp_port=1234,
            provider_url="http://example.com",
            image="img",
        ),
    )

    assert runner.env["EXGENTIC_RUNTIME_FILE"] == str(runtime_dir / "runtime.json")
