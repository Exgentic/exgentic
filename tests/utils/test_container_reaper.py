# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Unit tests for container_reaper: orphaned-container cleanup."""

from __future__ import annotations

import logging as _logging
import os
import signal
import subprocess
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from exgentic.adapters.runners.docker import DockerRunner
from exgentic.agents.cli.command_runner import BaseCLIConfig
from exgentic.agents.cli.command_runner import DockerRunner as CLIDockerRunner
from exgentic.benchmarks.swebench import swebench_eval, swebench_evaluation
from exgentic.utils import container_reaper
from exgentic.utils.container_reaper import (
    LABEL_OWNER_PID,
    LABEL_OWNER_TOKEN,
    OWN_TOKEN,
    docker_run_label_args,
    docker_sdk_label_injection,
    install_cleanup_handlers,
    reap_orphaned_containers,
    reap_own_containers,
)


def _fake_ps(rows: list[tuple[str, int, str]]) -> MagicMock:
    stdout = "\n".join(f"{cid}\t{pid}\t{tok}" for cid, pid, tok in rows) + ("\n" if rows else "")
    ps_result = MagicMock(returncode=0, stdout=stdout, stderr="")
    rm_result = MagicMock(returncode=0, stdout="", stderr="")

    def side_effect(cmd, **kwargs):
        if len(cmd) >= 2 and cmd[1] == "ps":
            return ps_result
        if len(cmd) >= 2 and cmd[1] == "rm":
            return rm_result
        return MagicMock(returncode=1, stdout="", stderr="unexpected")

    return MagicMock(side_effect=side_effect)


@pytest.mark.parametrize(
    ("kwargs", "expected_pid", "expected_token"),
    [
        ({}, os.getpid(), OWN_TOKEN),
        ({"pid": 12345, "token": "abc"}, 12345, "abc"),
    ],
)
def test_docker_run_label_args_emits_both_labels(kwargs, expected_pid, expected_token) -> None:
    args = docker_run_label_args(**kwargs)
    assert args == [
        "--label",
        f"{LABEL_OWNER_PID}={expected_pid}",
        "--label",
        f"{LABEL_OWNER_TOKEN}={expected_token}",
    ]


@pytest.mark.parametrize(
    ("rows", "expected_removed"),
    [
        ([], 0),
        ([("cidA", os.getpid(), OWN_TOKEN)], 0),
        (
            [
                ("cidDead", 999_999_999, "dead-token"),
                ("cidLive", os.getpid(), OWN_TOKEN),
            ],
            1,
        ),
        (
            [(f"minisweagent-{i:08x}", 999_999_998, "x") for i in range(5)] + [("active-cid", os.getpid(), OWN_TOKEN)],
            5,
        ),
    ],
)
def test_reap_orphaned_removes_only_dead_owners(rows: list[tuple[str, int, str]], expected_removed: int) -> None:
    runner = _fake_ps(rows)
    with patch.object(container_reaper, "_docker_bin", return_value="/usr/bin/docker"):
        removed = reap_orphaned_containers(runner=runner)
    assert removed == expected_removed
    rm_ids = [c.args[0][-1] for c in runner.call_args_list if c.args[0][1] == "rm"]
    assert len(rm_ids) == expected_removed


def test_reap_orphaned_removes_dead_even_when_pid_recycled() -> None:
    """PID-reuse mitigation: a live PID with a token that isn't ours."""
    recycled_pid = 4242
    runner = _fake_ps([("orphan-cid", recycled_pid, "some-dead-exgentic-token")])

    with (
        patch.object(container_reaper, "_docker_bin", return_value="/usr/bin/docker"),
        patch.object(container_reaper, "_pid_alive", return_value=True),
    ):
        removed = reap_orphaned_containers(runner=runner)

    assert removed == 0


def test_reap_own_requires_both_pid_and_token() -> None:
    """PID match without our token is NOT reaped (label-collision DoS)."""
    pid = os.getpid()
    runner = _fake_ps(
        [
            ("mine1", pid, OWN_TOKEN),
            ("spoofed", pid, "attacker-token"),
            ("sibling", 11111, "sibling-token"),
            ("mine2", pid, OWN_TOKEN),
        ]
    )
    with patch.object(container_reaper, "_docker_bin", return_value="/usr/bin/docker"):
        removed = reap_own_containers(runner=runner)
    assert removed == 2
    rm_ids = sorted(c.args[0][-1] for c in runner.call_args_list if c.args[0][1] == "rm")
    assert rm_ids == ["mine1", "mine2"]


@pytest.mark.parametrize(
    ("bin_return", "runner"),
    [
        (None, None),
        (
            "/usr/bin/docker",
            MagicMock(return_value=MagicMock(returncode=1, stdout="", stderr="cannot connect")),
        ),
    ],
)
def test_reap_degraded_docker_states_are_noops(bin_return, runner) -> None:
    with patch.object(container_reaper, "_docker_bin", return_value=bin_return):
        assert reap_orphaned_containers(runner=runner) == 0
        assert reap_own_containers(runner=runner) == 0


def test_install_cleanup_handlers_idempotent_and_dispatches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Registers atexit once; chains callables; handles SIG_DFL / SIG_IGN."""
    monkeypatch.setattr(container_reaper, "_handlers_installed", False)

    registered_atexit: list = []
    monkeypatch.setattr(container_reaper.atexit, "register", lambda fn: registered_atexit.append(fn))

    prev_calls: list = []

    def _prev_callable(signum, frame):
        prev_calls.append(signum)

    prev_for = {signal.SIGTERM: _prev_callable, signal.SIGINT: signal.SIG_DFL}
    monkeypatch.setattr(container_reaper.signal, "getsignal", lambda sig: prev_for[sig])

    installed: dict = {}
    monkeypatch.setattr(container_reaper.signal, "signal", lambda sig, h: installed.__setitem__(sig, h))

    reap_calls: list = []
    monkeypatch.setattr(container_reaper, "reap_own_containers", lambda **kw: reap_calls.append(kw) or 0)

    killed: list = []
    monkeypatch.setattr(container_reaper.os, "kill", lambda pid, sig: killed.append((pid, sig)))

    install_cleanup_handlers()
    install_cleanup_handlers()  # Idempotent.

    assert len(registered_atexit) == 1
    registered_atexit[0]()
    assert len(reap_calls) == 1

    installed[signal.SIGTERM](signal.SIGTERM, None)
    assert prev_calls == [signal.SIGTERM]
    assert len(reap_calls) == 2

    installed[signal.SIGINT](signal.SIGINT, None)
    assert killed == [(os.getpid(), signal.SIGINT)]
    assert len(reap_calls) == 3


def test_docker_sdk_label_injection_patches_create_and_restores() -> None:
    original = MagicMock(return_value="created")

    class _FakeCollection:
        create = original

    fake_mod = types.ModuleType("docker")
    fake_models = types.ModuleType("docker.models")
    fake_containers = types.ModuleType("docker.models.containers")
    fake_containers.ContainerCollection = _FakeCollection
    fake_models.containers = fake_containers
    fake_mod.models = fake_models

    with patch.dict(
        "sys.modules",
        {"docker": fake_mod, "docker.models": fake_models, "docker.models.containers": fake_containers},
    ):
        with docker_sdk_label_injection():
            _FakeCollection.create(MagicMock(), "some-image", command="run")

    call = original.call_args
    assert call.kwargs["labels"][LABEL_OWNER_PID] == str(os.getpid())
    assert call.kwargs["labels"][LABEL_OWNER_TOKEN] == OWN_TOKEN
    assert _FakeCollection.create is original

    # Missing docker SDK: context manager is a no-op.
    with patch.dict("sys.modules", {"docker.models.containers": None}):
        with docker_sdk_label_injection():
            pass


def test_run_harness_wraps_docker_sdk_create(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """run_harness invokes docker_sdk_label_injection around the harness call."""
    enters: list = []
    exits: list = []

    class _Ctx:
        def __enter__(self):
            enters.append(True)
            return self

        def __exit__(self, *a):
            exits.append(True)
            return False

    monkeypatch.setattr(swebench_evaluation, "docker_sdk_label_injection", lambda: _Ctx())

    fake_harness = types.ModuleType("swebench.harness.run_evaluation")
    fake_harness.main = MagicMock(return_value=None)
    fake_pkg = types.ModuleType("swebench")
    fake_sub = types.ModuleType("swebench.harness")
    fake_sub.run_evaluation = fake_harness

    paths = types.SimpleNamespace(benchmark_dir=tmp_path)

    with (
        patch.dict(
            "sys.modules",
            {"swebench": fake_pkg, "swebench.harness": fake_sub, "swebench.harness.run_evaluation": fake_harness},
        ),
        patch.object(swebench_evaluation, "capture_stdio_to_session", lambda log: _Ctx()),
    ):
        swebench_evaluation.run_harness(
            patch="diff --git a/x b/x\n--- a/x\n+++ a/x\n@@ +1,1 @@\n+x\n",
            instance_id="inst",
            subset="test",
            paths=paths,
            eval_config={"max_workers": 1, "cache_level": "env", "open_file_limit": 4096, "harness_timeout": 60},
            logger=_logging.getLogger("t"),
        )

    assert enters and exits, "docker_sdk_label_injection context was not entered"


def test_docker_runner_tags_containers_with_owner_labels() -> None:
    runner = DockerRunner(
        "exgentic.testing.calculator:Calculator",
        env_name="tests/calculator",
        module_path="exgentic.testing.calculator",
        image="stub-image",
        port=12345,
    )

    captured: list[list[str]] = []

    def _fake_docker(*args, **kwargs):
        captured.append(list(args))
        return subprocess.CompletedProcess(args, 0, "stub-cid\n", "")

    def _raise(*a, **k):
        raise RuntimeError("stop-after-docker-run")

    with (
        patch("exgentic.adapters.runners.docker._docker", side_effect=_fake_docker),
        patch("exgentic.adapters.runners.docker._wait_for_health", side_effect=_raise),
        patch("exgentic.environment.instance.get_manager") as get_mgr,
    ):
        get_mgr.return_value.base_dir = "/tmp/exgentic-cache"
        try:
            runner.start()
        except Exception:
            pass

    run_call = next((c for c in captured if c and c[0] == "run"), None)
    assert run_call is not None
    assert run_call.count("--label") == 2
    assert f"{LABEL_OWNER_PID}={os.getpid()}" in run_call
    assert f"{LABEL_OWNER_TOKEN}={OWN_TOKEN}" in run_call


def test_claude_code_docker_runner_tags_containers() -> None:
    runner = CLIDockerRunner(log_path=None, logger=_logging.getLogger("test"))

    captured_cmds: list[list[str]] = []

    def _fake_popen(cmd, **kwargs):
        captured_cmds.append(list(cmd))
        proc = MagicMock()
        proc.communicate.return_value = ("", "")
        proc.returncode = 0
        proc.poll.return_value = 0
        return proc

    cfg = BaseCLIConfig(
        mcp_host="127.0.0.1",
        mcp_port=5000,
        provider_url="http://example",
        image="stub-image",
    )

    with (
        patch(
            "exgentic.agents.cli.command_runner._detect_container_runtime",
            return_value=("docker", []),
        ),
        patch(
            "exgentic.agents.cli.command_runner.subprocess.Popen",
            side_effect=_fake_popen,
        ),
    ):
        runner.run(
            cmd=["claude", "-p", "hello"],
            env={},
            cfg_root=Path("/tmp"),
            config=cfg,
            spawn_error_message="boom",
        )

    assert captured_cmds
    wrapped = captured_cmds[0]
    assert "run" in wrapped
    assert wrapped.count("--label") == 2
    assert f"{LABEL_OWNER_PID}={os.getpid()}" in wrapped
    assert f"{LABEL_OWNER_TOKEN}={OWN_TOKEN}" in wrapped


def test_swebench_session_injects_reaper_labels_into_minisweagent_config() -> None:
    captured: dict = {}

    class _FakeEnv:
        def execute(self, command, cwd=""):
            return {"output": "abc123\n", "returncode": 0}

    def _fake_get_sb_env(config, instance):
        captured["config"] = config
        return _FakeEnv()

    fake_ms = types.ModuleType("minisweagent")
    fake_ms.__file__ = "/tmp/minisweagent/__init__.py"
    fake_submod = types.ModuleType("minisweagent.run.extra.swebench")
    fake_submod.get_sb_environment = _fake_get_sb_env

    with patch.dict(
        "sys.modules",
        {
            "minisweagent": fake_ms,
            "minisweagent.run": types.ModuleType("minisweagent.run"),
            "minisweagent.run.extra": types.ModuleType("minisweagent.run.extra"),
            "minisweagent.run.extra.swebench": fake_submod,
        },
    ):
        yaml_path = MagicMock()
        yaml_path.read_text.return_value = "environment:\n  cwd: /testbed\n  run_args:\n    - '--rm'\n"

        def _fake_path(arg):
            mock = MagicMock()
            mock.parent.__truediv__.return_value.__truediv__.return_value.__truediv__.return_value = yaml_path
            return mock

        stub = types.SimpleNamespace(
            logger=MagicMock(),
            container_repo_dir="/testbed",
            container_base_commit=None,
            _environment_pull_timeout=600,
            _instance={"base_commit": "abc123"},
            env=None,
        )

        with (
            patch.object(swebench_eval, "Path", side_effect=_fake_path),
            patch("exgentic.utils.logging.capture_stdio_to_session"),
        ):
            swebench_eval.SWEBenchSession._setup_environment(stub)

    run_args = captured["config"]["environment"]["run_args"]
    assert run_args.count("--label") == 2
    assert f"{LABEL_OWNER_PID}={os.getpid()}" in run_args
    assert f"{LABEL_OWNER_TOKEN}={OWN_TOKEN}" in run_args
