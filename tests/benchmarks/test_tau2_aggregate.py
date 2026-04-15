# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Tests for TAU2Evaluator.aggregate_sessions edge-case handling (issue #136).

The tau2 package is a heavy external dependency that is not available in the
test environment, so we install lightweight stubs into ``sys.modules`` before
importing the module under test.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest import mock

import pytest

# ---------------------------------------------------------------------------
# Lightweight stand-ins for tau2 types
# ---------------------------------------------------------------------------


class _FakeTask:
    def __init__(self, **kwargs: Any):
        self.id = kwargs.get("id", "")


class _FakeResults:
    """Minimal stand-in for ``tau2.data_model.simulation.Results``."""

    def __init__(
        self,
        tasks: list[Any] | None = None,
        simulations: list[Any] | None = None,
        info: Any = None,
    ):
        self.tasks = [_FakeTask(**t) if isinstance(t, dict) else t for t in (tasks or [])]
        self.simulations = simulations or []
        self.info = info or SimpleNamespace()

    @classmethod
    def load(cls, path: Path) -> _FakeResults:
        data = json.loads(path.read_text())
        return cls(
            tasks=data.get("tasks", []),
            simulations=data.get("simulations", []),
            info=SimpleNamespace(**data.get("info", {})),
        )


class _FakeMetrics:
    def __init__(self, avg_reward: float):
        self.avg_reward = avg_reward

    def as_dict(self) -> dict[str, Any]:
        return {"avg_reward": self.avg_reward}


def _fake_compute_metrics(combined: _FakeResults) -> _FakeMetrics:
    if not combined.simulations:
        return _FakeMetrics(avg_reward=0.0)
    rewards = [s.get("reward", 0.0) if isinstance(s, dict) else 0.0 for s in combined.simulations]
    return _FakeMetrics(avg_reward=sum(rewards) / len(rewards))


# ---------------------------------------------------------------------------
# Module-level tau2 stub installation
# ---------------------------------------------------------------------------


def _install_tau2_stubs() -> dict[str, ModuleType]:
    """Insert fake tau2 modules into sys.modules so tau2_shim can be imported."""
    stubs: dict[str, ModuleType] = {}
    names = [
        "tau2",
        "tau2.agent",
        "tau2.agent.llm_agent",
        "tau2.data_model",
        "tau2.data_model.message",
        "tau2.data_model.simulation",
        "tau2.environment",
        "tau2.environment.tool",
        "tau2.metrics",
        "tau2.metrics.agent_metrics",
        "tau2.registry",
        "tau2.run",
        "tau2.utils",
        "tau2.utils.display",
    ]
    for name in names:
        mod = ModuleType(name)
        stubs[name] = mod
        sys.modules.setdefault(name, mod)

    # Populate attributes that tau2_shim expects
    sim_mod = sys.modules["tau2.data_model.simulation"]
    sim_mod.Results = _FakeResults  # type: ignore[attr-defined]
    sim_mod.RunConfig = object  # type: ignore[attr-defined]
    sim_mod.TerminationReason = object  # type: ignore[attr-defined]

    msg_mod = sys.modules["tau2.data_model.message"]
    for attr in ("AssistantMessage", "MultiToolMessage", "ToolCall", "ToolMessage", "UserMessage"):
        setattr(msg_mod, attr, object)

    from abc import ABC, abstractmethod

    class _FakeLLMAgent(ABC):
        @abstractmethod
        def _placeholder(self) -> None:
            ...

    sys.modules["tau2.agent.llm_agent"].LLMAgent = _FakeLLMAgent  # type: ignore[attr-defined]
    sys.modules["tau2.environment.tool"].Tool = object  # type: ignore[attr-defined]

    metrics_mod = sys.modules["tau2.metrics.agent_metrics"]
    metrics_mod.compute_metrics = _fake_compute_metrics  # type: ignore[attr-defined]
    metrics_mod.is_successful = lambda x: False  # type: ignore[attr-defined]

    sys.modules["tau2.registry"].registry = object  # type: ignore[attr-defined]

    run_mod = sys.modules["tau2.run"]
    run_mod.load_tasks = lambda **kw: []  # type: ignore[attr-defined]
    run_mod.run_domain = object  # type: ignore[attr-defined]

    sys.modules["tau2.utils.display"].ConsoleDisplay = object  # type: ignore[attr-defined]

    return stubs


_stubs = _install_tau2_stubs()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_result(path: Path, tasks: list[dict], simulations: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"tasks": tasks, "simulations": simulations, "info": {}}))


def _make_session_paths(result_path: Path, session_id: str) -> Any:
    return SimpleNamespace(benchmark_results=result_path, session_id=session_id)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def evaluator(tmp_path: Path):
    """Return a TAU2Evaluator with mocked session paths."""
    # Patch get_tau2_data_dir so the shim __init__ doesn't try to resolve real dirs
    with mock.patch("exgentic.benchmarks.tau2.get_tau2_data_dir", return_value="/fake"):
        # Force re-import of tau2_shim and tau2_eval with stubs in place
        shim_name = "exgentic.benchmarks.tau2.tau2_shim"
        eval_name = "exgentic.benchmarks.tau2.tau2_eval"

        # Remove cached modules to force re-import with our stubs
        for mod_name in (shim_name, eval_name):
            sys.modules.pop(mod_name, None)

        # Patch configure_litellm and get_settings that tau2_shim calls at import time
        with (
            mock.patch("exgentic.integrations.litellm.config.configure_litellm"),
            mock.patch("exgentic.utils.settings.get_settings"),
        ):
            mod = importlib.import_module(eval_name)

    cls = mod.TAU2Evaluator
    ev = cls.__new__(cls)
    ev._subset = "test"
    ev._score_path = None
    return ev


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAggregateSessions:
    """Tests for TAU2Evaluator.aggregate_sessions."""

    def test_skips_errored_sessions_with_no_simulations(self, evaluator, tmp_path: Path):
        """Sessions with 0 simulations (errored) should be skipped gracefully."""
        ok_path = tmp_path / "ok" / "results.json"
        err_path = tmp_path / "err" / "results.json"

        _write_result(ok_path, tasks=[{"id": "t1"}], simulations=[{"reward": 0.8}])
        _write_result(err_path, tasks=[{"id": "t2"}], simulations=[])

        sessions = [SimpleNamespace(session_id="ok"), SimpleNamespace(session_id="err")]

        with mock.patch.object(
            type(evaluator),
            "get_sessions_paths",
            return_value=[
                _make_session_paths(ok_path, "ok"),
                _make_session_paths(err_path, "err"),
            ],
        ):
            result = evaluator.aggregate_sessions(sessions)

        assert result.score == 0.8
        assert result.total_tasks == 2

    def test_all_sessions_empty_returns_zero_score(self, evaluator, tmp_path: Path):
        """When every session has no simulations, return score=0 instead of crashing."""
        err1 = tmp_path / "err1" / "results.json"
        err2 = tmp_path / "err2" / "results.json"

        _write_result(err1, tasks=[{"id": "t1"}], simulations=[])
        _write_result(err2, tasks=[{"id": "t2"}], simulations=[])

        sessions = [
            SimpleNamespace(session_id="err1"),
            SimpleNamespace(session_id="err2"),
        ]

        with mock.patch.object(
            type(evaluator),
            "get_sessions_paths",
            return_value=[
                _make_session_paths(err1, "err1"),
                _make_session_paths(err2, "err2"),
            ],
        ):
            result = evaluator.aggregate_sessions(sessions)
            assert result.score == 0.0
            assert result.metrics["avg_reward"] == 0.0

    def test_rejects_file_with_multiple_tasks(self, evaluator, tmp_path: Path):
        """A result file with != 1 task should raise ValueError."""
        bad_path = tmp_path / "bad" / "results.json"
        _write_result(bad_path, tasks=[{"id": "t1"}, {"id": "t2"}], simulations=[{"reward": 1.0}])

        sessions = [SimpleNamespace(session_id="bad")]
        with mock.patch.object(
            type(evaluator),
            "get_sessions_paths",
            return_value=[_make_session_paths(bad_path, "bad")],
        ):
            with pytest.raises(ValueError, match="Expected exactly 1 task"):
                evaluator.aggregate_sessions(sessions)

    def test_rejects_file_with_multiple_simulations(self, evaluator, tmp_path: Path):
        """A result file with >1 simulation should raise ValueError."""
        bad_path = tmp_path / "bad" / "results.json"
        _write_result(bad_path, tasks=[{"id": "t1"}], simulations=[{"reward": 0.5}, {"reward": 0.9}])

        sessions = [SimpleNamespace(session_id="bad")]
        with mock.patch.object(
            type(evaluator),
            "get_sessions_paths",
            return_value=[_make_session_paths(bad_path, "bad")],
        ):
            with pytest.raises(ValueError, match="Expected at most 1 simulation"):
                evaluator.aggregate_sessions(sessions)
