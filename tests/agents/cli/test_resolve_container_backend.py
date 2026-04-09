# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

from exgentic.agents.cli.command_runner import ExecutionBackend


def test_execution_backend_values():
    assert ExecutionBackend.PROCESS == "process"
    assert ExecutionBackend.DOCKER == "docker"


def test_execution_backend_has_no_podman():
    values = {e.value for e in ExecutionBackend}
    assert "podman" not in values
    assert "auto" not in values
