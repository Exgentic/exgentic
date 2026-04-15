# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

import os

from exgentic.agents.cli.base import (
    BaseCLIConfig,
    BaseCLIWrapper,
    CLIResult,
    ExecutionBackend,
)
from exgentic.core.context import Context, Role, save_service_runtime, set_context


class _DummyRunner:
    def __init__(self):
        self.env = None

    def run(self, *, cmd, env, cfg_root, config, spawn_error_message, stdin_devnull=False):
        self.env = env
        return CLIResult(stdout="", stderr="", code=0)

    def close(self) -> None:
        return None


class _DummyCLI(BaseCLIWrapper):
    def build_env(self, *, cfg_root, prompt, config):
        return {}

    def build_command(self, *, cfg_root, prompt, config):
        return ["echo", "ok"]


def test_cli_includes_context_env(tmp_path):
    ctx = Context(
        run_id="run-cli",
        output_dir=str(tmp_path),
        cache_dir="/tmp/cache",
        session_id="sess-1",
    )
    set_context(ctx)

    # Write a per-service runtime.json for the agent role so the CLI
    # wrapper inherits EXGENTIC_RUNTIME_FILE from its parent process.
    runtime_path = save_service_runtime(Role.AGENT)
    old = os.environ.get("EXGENTIC_RUNTIME_FILE")
    os.environ["EXGENTIC_RUNTIME_FILE"] = str(runtime_path)
    try:
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

        assert runner.env["EXGENTIC_RUNTIME_FILE"] == str(runtime_path)
    finally:
        if old is None:
            os.environ.pop("EXGENTIC_RUNTIME_FILE", None)
        else:
            os.environ["EXGENTIC_RUNTIME_FILE"] = old
