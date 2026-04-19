# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Unit tests for container_reaper — orphaned-container cleanup."""

from __future__ import annotations

import os
import signal
import subprocess
from unittest.mock import MagicMock, patch

import pytest
from exgentic.utils import container_reaper
from exgentic.utils.container_reaper import (
    LABEL_OWNER_PID,
    docker_run_label_args,
    install_cleanup_handlers,
    reap_orphaned_containers,
    reap_own_containers,
)


def _fake_ps(lines: list[tuple[str, int]]) -> MagicMock:
    """Build a subprocess.run mock for ``docker ps`` + ``docker rm``."""
    stdout = "\n".join(f"{cid}\t{pid}" for cid, pid in lines) + ("\n" if lines else "")

    ps_result = MagicMock(returncode=0, stdout=stdout, stderr="")
    rm_result = MagicMock(returncode=0, stdout="", stderr="")

    def side_effect(cmd, **kwargs):
        if len(cmd) >= 2 and cmd[1] == "ps":
            return ps_result
        if len(cmd) >= 2 and cmd[1] == "rm":
            return rm_result
        return MagicMock(returncode=1, stdout="", stderr="unexpected")

    runner = MagicMock(side_effect=side_effect)
    return runner


def test_docker_run_label_args_uses_current_pid() -> None:
    args = docker_run_label_args()
    assert args[0] == "--label"
    assert args[1] == f"{LABEL_OWNER_PID}={os.getpid()}"


def test_docker_run_label_args_explicit_pid() -> None:
    args = docker_run_label_args(pid=12345)
    assert args == ["--label", f"{LABEL_OWNER_PID}=12345"]


def test_reap_orphaned_removes_dead_owner_containers() -> None:
    dead_pid = 999_999_999  # PID is effectively guaranteed not alive
    live_pid = os.getpid()  # our PID is always alive
    runner = _fake_ps([("cidA", dead_pid), ("cidB", live_pid)])

    with patch.object(container_reaper, "_docker_bin", return_value="/usr/bin/docker"):
        removed = reap_orphaned_containers(runner=runner)

    assert removed == 1
    rm_calls = [c for c in runner.call_args_list if c.args[0][1] == "rm"]
    assert len(rm_calls) == 1
    assert rm_calls[0].args[0] == ["/usr/bin/docker", "rm", "-f", "cidA"]


def test_reap_orphaned_noop_when_all_alive() -> None:
    runner = _fake_ps([("cidA", os.getpid())])
    with patch.object(container_reaper, "_docker_bin", return_value="/usr/bin/docker"):
        removed = reap_orphaned_containers(runner=runner)
    assert removed == 0
    rm_calls = [c for c in runner.call_args_list if c.args[0][1] == "rm"]
    assert rm_calls == []


def test_reap_own_containers_removes_by_current_pid() -> None:
    pid = os.getpid()
    runner = _fake_ps([("mine1", pid), ("other", 12345), ("mine2", pid)])
    with patch.object(container_reaper, "_docker_bin", return_value="/usr/bin/docker"):
        removed = reap_own_containers(runner=runner)
    assert removed == 2
    rm_args = [c.args[0] for c in runner.call_args_list if c.args[0][1] == "rm"]
    removed_ids = sorted(cmd[3] for cmd in rm_args)
    assert removed_ids == ["mine1", "mine2"]


def test_reap_own_containers_skips_siblings() -> None:
    runner = _fake_ps([("other1", 11111), ("other2", 22222)])
    with patch.object(container_reaper, "_docker_bin", return_value="/usr/bin/docker"):
        removed = reap_own_containers(runner=runner)
    assert removed == 0


def test_reap_handles_missing_docker_binary() -> None:
    with patch.object(container_reaper, "_docker_bin", return_value=None):
        assert reap_orphaned_containers() == 0
        assert reap_own_containers() == 0


def test_reap_handles_docker_ps_failure() -> None:
    runner = MagicMock(return_value=MagicMock(returncode=1, stdout="", stderr="cannot connect"))
    with patch.object(container_reaper, "_docker_bin", return_value="/usr/bin/docker"):
        assert reap_orphaned_containers(runner=runner) == 0


def test_install_cleanup_handlers_runs_at_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Atexit handler must invoke reap_own_containers on shutdown."""
    # Reset the idempotency guard so the test sees a fresh install.
    monkeypatch.setattr(container_reaper, "_handlers_installed", False)

    registered: list = []
    monkeypatch.setattr(
        container_reaper.atexit,
        "register",
        lambda fn: registered.append(fn),
    )
    # Skip signal registration in test to avoid mutating process state.
    monkeypatch.setattr(container_reaper.signal, "signal", lambda *a, **k: None)

    reap_calls: list = []

    def _fake_reap(**kwargs):
        reap_calls.append(kwargs)
        return 0

    monkeypatch.setattr(container_reaper, "reap_own_containers", _fake_reap)

    install_cleanup_handlers()

    assert len(registered) == 1
    # Invoke the registered atexit callback manually — simulates shutdown.
    registered[0]()
    assert len(reap_calls) == 1


def test_install_cleanup_handlers_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(container_reaper, "_handlers_installed", False)
    registered: list = []
    monkeypatch.setattr(
        container_reaper.atexit,
        "register",
        lambda fn: registered.append(fn),
    )
    monkeypatch.setattr(container_reaper.signal, "signal", lambda *a, **k: None)

    install_cleanup_handlers()
    install_cleanup_handlers()
    install_cleanup_handlers()

    assert len(registered) == 1


def test_sigterm_handler_cleans_up_then_chains(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SIGTERM handler must reap containers then invoke previous handler."""
    monkeypatch.setattr(container_reaper, "_handlers_installed", False)
    monkeypatch.setattr(container_reaper.atexit, "register", lambda fn: None)

    prev_called: list = []

    def _prev_handler(signum, frame):
        prev_called.append(signum)

    installed: dict = {}

    def _fake_getsignal(sig):
        return _prev_handler

    def _fake_signal(sig, handler):
        installed[sig] = handler

    monkeypatch.setattr(container_reaper.signal, "getsignal", _fake_getsignal)
    monkeypatch.setattr(container_reaper.signal, "signal", _fake_signal)

    reap_calls: list = []
    monkeypatch.setattr(
        container_reaper,
        "reap_own_containers",
        lambda **kw: reap_calls.append(kw) or 0,
    )

    install_cleanup_handlers()

    assert signal.SIGTERM in installed
    # Invoke the installed SIGTERM handler.
    installed[signal.SIGTERM](signal.SIGTERM, None)

    # Cleanup must have run AND previous handler must have been invoked.
    assert len(reap_calls) == 1
    assert prev_called == [signal.SIGTERM]


# ---------------------------------------------------------------------------
# Integration-style test: simulate the full unclean-termination scenario
# ---------------------------------------------------------------------------


def test_end_to_end_orphan_scenario_reaps_on_batch_start() -> None:
    """Regression test for issue #192.

    Scenario: a previous batch crashed leaving orphaned minisweagent
    containers labeled with a now-dead owner PID.  The reaper invoked at
    the start of a new batch must remove them.
    """
    dead_pid = 999_999_998
    orphans = [(f"minisweagent-{i:08x}", dead_pid) for i in range(5)]
    still_alive = [("active-cid", os.getpid())]

    calls: list[list[str]] = []
    rm_targets: list[str] = []

    def _run(cmd, **kwargs):
        calls.append(list(cmd))
        if len(cmd) >= 2 and cmd[1] == "ps":
            lines = [f"{cid}\t{pid}" for cid, pid in orphans + still_alive]
            return subprocess.CompletedProcess(cmd, 0, "\n".join(lines) + "\n", "")
        if len(cmd) >= 2 and cmd[1] == "rm":
            rm_targets.append(cmd[-1])
            return subprocess.CompletedProcess(cmd, 0, "", "")
        return subprocess.CompletedProcess(cmd, 1, "", "unexpected")

    with patch.object(container_reaper, "_docker_bin", return_value="/usr/bin/docker"):
        removed = reap_orphaned_containers(runner=_run)

    assert removed == 5
    assert sorted(rm_targets) == sorted(cid for cid, _ in orphans)
    assert "active-cid" not in rm_targets


# ---------------------------------------------------------------------------
# Wiring tests: container-spawning callsites must emit the owner-pid label
# ---------------------------------------------------------------------------


def test_docker_runner_tags_containers_with_owner_pid() -> None:
    """adapters/runners/docker.py::DockerRunner.start must label containers."""
    from exgentic.adapters.runners.docker import DockerRunner

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

    # Avoid the health check / transport wiring — the test only validates
    # the docker run argv.
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

    # The first docker call is the `run` (image build is skipped because
    # ``image`` was provided).  Assert it contains our label flag.
    run_call = next((c for c in captured if c and c[0] == "run"), None)
    assert run_call is not None, f"no docker run call captured: {captured}"
    assert "--label" in run_call
    idx = run_call.index("--label")
    assert run_call[idx + 1] == f"{LABEL_OWNER_PID}={os.getpid()}"


def test_claude_code_docker_runner_tags_containers() -> None:
    """agents/cli/command_runner.py::DockerRunner.run must label containers."""
    import logging as _logging
    from pathlib import Path

    from exgentic.agents.cli.command_runner import BaseCLIConfig
    from exgentic.agents.cli.command_runner import DockerRunner as CLIDockerRunner

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

    assert captured_cmds, "no docker command was captured"
    # The wrapped cmd begins with the runtime binary then 'run'.
    wrapped = captured_cmds[0]
    assert "run" in wrapped
    assert "--label" in wrapped
    idx = wrapped.index("--label")
    assert wrapped[idx + 1] == f"{LABEL_OWNER_PID}={os.getpid()}"


def test_swebench_session_injects_reaper_label_into_minisweagent_config() -> None:
    """Swebench session injects owner-pid label into minisweagent run_args.

    SWEBenchSession._setup_environment must inject an ``exgentic.owner_pid``
    label into the minisweagent environment config's ``run_args`` so
    mini-swe-agent's ``docker run`` invocation carries the reaper label
    (regression guard for issue #192).
    """
    import types

    # Capture the config that would be handed to minisweagent.
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
        from exgentic.benchmarks.swebench import swebench_eval

        yaml_path = MagicMock()
        yaml_path.read_text.return_value = "environment:\n  cwd: /testbed\n  run_args:\n    - '--rm'\n"

        def _fake_path(arg):
            # Intercept ``Path(minisweagent.__file__)`` traversal.
            mock = MagicMock()
            mock.parent.__truediv__.return_value.__truediv__.return_value.__truediv__.return_value = yaml_path
            return mock

        # Build a minimal stub session that invokes the real method.
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

    assert captured.get("config"), "minisweagent config was not captured"
    run_args = captured["config"]["environment"]["run_args"]
    assert "--label" in run_args, f"run_args missing --label: {run_args}"
    idx = run_args.index("--label")
    assert run_args[idx + 1] == f"{LABEL_OWNER_PID}={os.getpid()}"
