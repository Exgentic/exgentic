# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""End-to-end tests — run a full session lifecycle through every transport.

Uses the test fixtures (TestSession, TestAgent) to verify that the complete
benchmark→session→agent loop works over each runner/transport layer.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest
from exgentic.adapters.runners import with_runner
from exgentic.testing import (
    BadAction,
    DockerSession,
    EmptyArgs,
    FinishAction,
    GoodAction,
    TestAgent,
    TestSession,
)

# Detect Docker availability for conditional tests.
_docker_available = shutil.which("docker") is not None
if _docker_available:
    try:
        subprocess.run(["docker", "info"], check=True, capture_output=True, timeout=5)
    except Exception:
        _docker_available = False

_RUNNERS = ["direct", "thread", "process", "service"]


@pytest.fixture(params=_RUNNERS)
def runner_name(request):
    return request.param


@pytest.fixture
def session_proxy(runner_name, tmp_path, monkeypatch):
    """Create a TestSession wrapped in the specified runner."""
    monkeypatch.setenv("EXGENTIC_OUTPUT_DIR", str(tmp_path))
    proxy = with_runner(
        TestSession,
        runner=runner_name,
        task_id="task-1",
        session_id="sess-e2e-001",
        stop_on_step=False,
        invalid_observation=False,
    )
    yield proxy
    try:
        proxy.close()
    except Exception:
        pass


# ── basic lifecycle ──────────────────────────────────────────────────


class TestSessionLifecycle:
    """Full start → step → done → score lifecycle across transports."""

    def test_start_returns_observation(self, session_proxy):
        obs = session_proxy.start()
        assert obs.result == "start"

    def test_step_good_action(self, session_proxy):
        session_proxy.start()
        obs = session_proxy.step(GoodAction(arguments=EmptyArgs()))
        assert obs.result == "step"

    def test_done_false_before_finish(self, session_proxy):
        session_proxy.start()
        session_proxy.step(GoodAction(arguments=EmptyArgs()))
        assert session_proxy.done() is False

    def test_finish_action_marks_done(self, session_proxy):
        session_proxy.start()
        session_proxy.step(GoodAction(arguments=EmptyArgs()))
        obs = session_proxy.step(FinishAction(arguments=EmptyArgs()))
        assert obs.result == "finish"
        assert session_proxy.done() is True

    def test_score_after_good_and_finish(self, session_proxy):
        session_proxy.start()
        session_proxy.step(GoodAction(arguments=EmptyArgs()))
        session_proxy.step(FinishAction(arguments=EmptyArgs()))
        result = session_proxy.score()
        assert result.score == 1.0
        assert result.success is True

    def test_score_no_actions(self, session_proxy):
        session_proxy.start()
        session_proxy.step(FinishAction(arguments=EmptyArgs()))
        result = session_proxy.score()
        assert result.score == 0.0
        assert result.success is False


# ── property access over transports ──────────────────────────────────


class TestPropertyAccess:
    """Verify that property reads work transparently across transports."""

    def test_task_property(self, session_proxy):
        assert session_proxy.task == "Task task-1"

    def test_task_id_property(self, session_proxy):
        assert session_proxy.task_id == "task-1"

    def test_context_property(self, session_proxy):
        ctx = session_proxy.context
        assert ctx == {"task_id": "task-1"}

    def test_actions_property(self, session_proxy):
        actions = session_proxy.actions
        assert len(actions) == 3
        names = {a.name for a in actions}
        assert names == {"good", "bad", "finish"}


# ── agent integration ────────────────────────────────────────────────


class TestAgentWithRunnerSession:
    """Run a TestAgent against a session through each transport."""

    def test_good_then_finish_policy(self, session_proxy):
        agent = TestAgent(policy="good_then_finish", finish_after=3)
        instance = agent.get_instance_class()(
            **agent.get_instance_kwargs(session_id="sess-e2e-001"),
        )
        instance.start(
            task=session_proxy.task,
            context=session_proxy.context,
            actions=session_proxy.actions,
        )

        obs = session_proxy.start()
        steps = 0
        while not session_proxy.done() and steps < 10:
            action = instance.react(obs)
            if action is None:
                break
            obs = session_proxy.step(action)
            steps += 1

        assert session_proxy.done() is True
        result = session_proxy.score()
        assert result.success is True
        assert result.score == 1.0

    def test_finish_immediately_policy(self, session_proxy):
        agent = TestAgent(policy="finish_immediately")
        instance = agent.get_instance_class()(
            **agent.get_instance_kwargs(session_id="sess-e2e-001"),
        )
        instance.start(
            task=session_proxy.task,
            context=session_proxy.context,
            actions=session_proxy.actions,
        )

        obs = session_proxy.start()
        action = instance.react(obs)
        session_proxy.step(action)

        assert session_proxy.done() is True
        result = session_proxy.score()
        assert result.score == 0.0


# ── stateful consistency ─────────────────────────────────────────────


class TestStatefulConsistency:
    """Multiple steps keep consistent state across transports."""

    def test_multiple_good_actions(self, session_proxy):
        session_proxy.start()
        for _ in range(5):
            obs = session_proxy.step(GoodAction(arguments=EmptyArgs()))
            assert obs.result == "step"
        session_proxy.step(FinishAction(arguments=EmptyArgs()))
        result = session_proxy.score()
        assert result.score == 1.0
        assert result.session_metrics["good"] == 5
        assert result.session_metrics["total"] == 5

    def test_mixed_actions(self, session_proxy):
        session_proxy.start()
        session_proxy.step(GoodAction(arguments=EmptyArgs()))
        session_proxy.step(BadAction(arguments=EmptyArgs()))
        session_proxy.step(GoodAction(arguments=EmptyArgs()))
        session_proxy.step(FinishAction(arguments=EmptyArgs()))
        result = session_proxy.score()
        assert result.score == pytest.approx(2 / 3)
        assert result.success is False  # had bad actions


# ── Docker transport (skipped when Docker unavailable) ───────────────
#
# Uses DockerSession from exgentic.testing — a minimal session-like
# object that is part of the installed package and therefore importable
# inside the Docker container.


@pytest.mark.skipif(not _docker_available, reason="Docker not available")
class TestDockerSessionE2E:
    """Full session lifecycle over the Docker transport.

    Uses a class-scoped fixture so the image is built only once.
    """

    @pytest.fixture(scope="class")
    def docker_session(self):
        # Rancher Desktop / Docker Desktop on macOS only share /Users/ by
        # default via reverse-sshfs.  pytest's tmp_path lives under
        # /var/folders/ which is NOT shared, so volume mounts silently fail.
        # Use a temp dir under $HOME to ensure Docker can mount it.
        if platform.system() == "Darwin":
            out = Path(tempfile.mkdtemp(prefix=".exgentic_test_", dir=Path.home()))
        else:
            out = Path(tempfile.mkdtemp(prefix="exgentic_test_"))
        import os

        old_output_dir = os.environ.get("EXGENTIC_OUTPUT_DIR")
        os.environ["EXGENTIC_OUTPUT_DIR"] = str(out)
        proxy = with_runner(
            DockerSession,
            runner="docker",
            env_name="test/docker-e2e",
            module_path="",
            task_id="task-1",
            output_dir=str(out),
            volumes={str(out): str(out)},
        )
        yield proxy, out
        try:
            proxy.close()
        except Exception:
            pass
        if old_output_dir is None:
            os.environ.pop("EXGENTIC_OUTPUT_DIR", None)
        else:
            os.environ["EXGENTIC_OUTPUT_DIR"] = old_output_dir
        shutil.rmtree(out, ignore_errors=True)

    def test_start(self, docker_session):
        proxy, _ = docker_session
        obs = proxy.start()
        assert obs["result"] == "start"

    def test_step_and_finish(self, docker_session):
        proxy, _ = docker_session
        obs = proxy.step("good")
        assert obs["result"] == "step"
        obs = proxy.step("finish")
        assert obs["result"] == "finish"
        assert proxy.done() is True

    def test_score(self, docker_session):
        proxy, _ = docker_session
        result = proxy.score()
        assert result["score"] == 1.0
        assert result["success"] is True

    def test_properties(self, docker_session):
        proxy, _ = docker_session
        assert proxy.task_id == "task-1"
        assert proxy.task == "Task task-1"
        assert proxy.context == {"task_id": "task-1"}

    def test_volume_mount_output_visible_on_host(self, docker_session):
        """Verify that files written inside the container are visible on the host."""
        proxy, out = docker_session
        proxy.write_output("test_result.txt", "hello from docker")
        result_file = out / "test_result.txt"
        assert result_file.exists(), "Output file written in container not visible on host"
        assert result_file.read_text() == "hello from docker"
