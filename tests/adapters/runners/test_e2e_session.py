# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""End-to-end tests — run a full session lifecycle through every transport.

Uses the test fixtures (TestSession, TestAgent) to verify that the complete
benchmark→session→agent loop works over each runner/transport layer.
"""

from __future__ import annotations

import shutil
import subprocess
import sys

import pytest

from exgentic.adapters.runners import with_runner
from exgentic.core.types import SessionIndex

from tests.api.fixtures.test_benchmark import TestSession
from tests.api.fixtures.test_agent import (
    TestAgent,
    GoodAction,
    BadAction,
    FinishAction,
    EmptyArgs,
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
    index = SessionIndex(task_id="task-1", session_id="sess-e2e-001")
    proxy = with_runner(
        TestSession,
        runner=runner_name,
        index=index,
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
        instance = agent.assign(
            task=session_proxy.task,
            context=session_proxy.context,
            actions=session_proxy.actions,
            session_id="sess-e2e-001",
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
        instance = agent.assign(
            task=session_proxy.task,
            context=session_proxy.context,
            actions=session_proxy.actions,
            session_id="sess-e2e-001",
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
# TestSession can't be used inside Docker because it imports from
# ``tests.*`` which isn't installed in the container image.  We define
# a self-contained ``_DockerSession`` here that only depends on the
# installed ``exgentic`` package.  cloudpickle embeds the class by value
# so the container never tries to import ``tests.*``.


class _DockerSession:
    """Minimal session-like object for Docker e2e tests.

    Not a real Session subclass — avoids pulling in Session.__init__
    which writes files.  We only need the method/property interface that
    the ObjectProxy will forward over HTTP.
    """

    def __init__(self, task_id: str, output_dir: str | None = None) -> None:
        self._task_id = task_id
        self._done = False
        self._good = 0
        self._steps = 0
        self._output_dir = output_dir

    @property
    def task_id(self) -> str:
        return self._task_id

    @property
    def task(self) -> str:
        return f"Task {self._task_id}"

    @property
    def context(self) -> dict:
        return {"task_id": self._task_id}

    def start(self) -> dict:
        return {"result": "start"}

    def step(self, action_name: str) -> dict:
        self._steps += 1
        if action_name == "good":
            self._good += 1
            return {"result": "step"}
        if action_name == "finish":
            self._done = True
            return {"result": "finish"}
        return {"result": "step"}

    def done(self) -> bool:
        return self._done

    def score(self) -> dict:
        total = self._good
        return {"score": 1.0 if total > 0 else 0.0, "success": self._done and total > 0}

    def write_output(self, filename: str, content: str) -> str:
        """Write a file to the output dir.  Used to verify volume mounts."""
        import os
        out = self._output_dir or os.environ.get("EXGENTIC_OUTPUT_DIR", "/tmp")
        path = os.path.join(out, filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(content)
        return path

    def close(self) -> None:
        pass


@pytest.mark.skipif(not _docker_available, reason="Docker not available")
@pytest.mark.skipif(sys.version_info[:2] != (3, 12), reason="Docker image uses Python 3.12")
class TestDockerSessionE2E:
    """Full session lifecycle over the Docker transport.

    Uses a class-scoped fixture so the image is built only once.
    """

    @pytest.fixture(scope="class")
    def docker_session(self, tmp_path_factory):
        out = tmp_path_factory.mktemp("docker_e2e")
        import os
        os.environ["EXGENTIC_OUTPUT_DIR"] = str(out)
        proxy = with_runner(
            _DockerSession,
            runner="docker",
            task_id="task-1",
            output_dir=str(out),
            volumes={str(out): str(out)},
        )
        yield proxy, out
        try:
            proxy.close()
        except Exception:
            pass

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
