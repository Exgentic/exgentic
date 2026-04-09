# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

import subprocess
from unittest.mock import patch

import pytest
from exgentic.agents.cli.command_runner import (
    ExecutionBackend,
    _podman_works,
    resolve_container_backend,
)

# -- _podman_works -----------------------------------------------------------


def test_podman_works_when_info_succeeds():
    with patch("exgentic.agents.cli.command_runner.shutil.which", return_value="/usr/bin/podman"):
        with patch("exgentic.agents.cli.command_runner.subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess([], 0)
            assert _podman_works() is True


def test_podman_works_false_when_not_installed():
    with patch("exgentic.agents.cli.command_runner.shutil.which", return_value=None):
        assert _podman_works() is False


def test_podman_works_false_when_info_fails():
    with patch("exgentic.agents.cli.command_runner.shutil.which", return_value="/usr/bin/podman"):
        with patch(
            "exgentic.agents.cli.command_runner.subprocess.run",
            side_effect=subprocess.CalledProcessError(125, "podman"),
        ):
            assert _podman_works() is False


# -- resolve_container_backend: env-var override ------------------------------


def test_env_override_docker(monkeypatch):
    monkeypatch.setenv("EXGENTIC_CONTAINER_CMD", "docker")
    with patch("exgentic.agents.cli.command_runner.shutil.which", return_value="/usr/bin/docker"):
        assert resolve_container_backend() == ExecutionBackend.DOCKER


def test_env_override_podman(monkeypatch):
    monkeypatch.setenv("EXGENTIC_CONTAINER_CMD", "podman")
    with patch("exgentic.agents.cli.command_runner.shutil.which", return_value="/usr/bin/podman"):
        assert resolve_container_backend() == ExecutionBackend.PODMAN


def test_env_override_invalid_value(monkeypatch):
    monkeypatch.setenv("EXGENTIC_CONTAINER_CMD", "nerdctl")
    with pytest.raises(RuntimeError, match="not supported"):
        resolve_container_backend()


def test_env_override_not_on_path(monkeypatch):
    monkeypatch.setenv("EXGENTIC_CONTAINER_CMD", "docker")
    with patch("exgentic.agents.cli.command_runner.shutil.which", return_value=None):
        with pytest.raises(RuntimeError, match="not found on PATH"):
            resolve_container_backend()


# -- resolve_container_backend: auto-detect -----------------------------------


def test_auto_detect_podman_when_working(monkeypatch):
    monkeypatch.delenv("EXGENTIC_CONTAINER_CMD", raising=False)
    with patch("exgentic.agents.cli.command_runner._podman_works", return_value=True):
        assert resolve_container_backend() == ExecutionBackend.PODMAN


def test_auto_detect_falls_back_to_docker_when_podman_broken(monkeypatch):
    monkeypatch.delenv("EXGENTIC_CONTAINER_CMD", raising=False)
    with patch("exgentic.agents.cli.command_runner._podman_works", return_value=False):
        with patch("exgentic.agents.cli.command_runner.shutil.which", return_value="/usr/bin/docker"):
            assert resolve_container_backend() == ExecutionBackend.DOCKER


def test_auto_detect_raises_when_nothing_available(monkeypatch):
    monkeypatch.delenv("EXGENTIC_CONTAINER_CMD", raising=False)
    with patch("exgentic.agents.cli.command_runner._podman_works", return_value=False):
        with patch("exgentic.agents.cli.command_runner.shutil.which", return_value=None):
            with pytest.raises(RuntimeError, match="Neither podman nor docker found"):
                resolve_container_backend()
