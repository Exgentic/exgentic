# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Regression test for Exgentic/exgentic#202.

``_plan_config`` builds a Tracker for batch-mode session execution.
A prior change set ``use_defaults=False`` on that Tracker to suppress
console noise during planning, which silently removed **all** default
observers -- including ``ResultsObserver``, the one that writes
``sessions/{id}/results.json``. The result: every ``exgentic batch
evaluate`` run appeared to complete but recorded zero session-level
results.

These tests verify that the Tracker returned by ``_plan_config``
has the essential observers registered, regardless of how console
noise is suppressed.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from exgentic.core.orchestrator.run import _plan_config
from exgentic.core.types import RunConfig
from exgentic.observers.handlers.logger import ConsoleLoggerObserver
from exgentic.observers.handlers.results import ResultsObserver


@pytest.fixture
def run_config(tmp_path):
    return RunConfig(
        benchmark="test_benchmark",
        agent="test_agent",
        output_dir=str(tmp_path / "output"),
        cache_dir=str(tmp_path / "cache"),
        num_tasks=1,
        benchmark_kwargs={"tasks": ["task-1"]},
        agent_kwargs={"policy": "good_then_finish", "finish_after": 2},
    )


def test_plan_config_tracker_has_results_observer(run_config):
    """The Tracker returned by _plan_config must include ResultsObserver.

    Without it, sessions complete but no results.json is written --
    batch status shows zero progress forever.
    """
    _, tracker, _ = _plan_config(run_config)

    results_observers = [o for o in tracker.observers if isinstance(o, ResultsObserver)]
    assert len(results_observers) == 1, (
        f"Expected exactly 1 ResultsObserver, found {len(results_observers)}. "
        f"Observers: {[type(o).__name__ for o in tracker.observers]}"
    )


def test_plan_config_tracker_has_nonzero_observers(run_config):
    """The Tracker must have at least the core observers (results, configs, warnings, file logger)."""
    _, tracker, _ = _plan_config(run_config)

    assert len(tracker.observers) >= 4, (
        f"Expected at least 4 observers, found {len(tracker.observers)}: "
        f"{[type(o).__name__ for o in tracker.observers]}"
    )


def test_plan_config_tracker_excludes_console_logger(run_config):
    """ConsoleLoggerObserver should NOT be registered in batch mode.

    That's the whole reason for the 'quiet tracker' -- 50 console
    panels during planning is noisy. But suppressing it must not
    suppress ResultsObserver.
    """
    _, tracker, _ = _plan_config(run_config)

    console_observers = [o for o in tracker.observers if isinstance(o, ConsoleLoggerObserver)]
    assert len(console_observers) == 0, (
        "ConsoleLoggerObserver should not be registered in batch mode "
        "(it produces a panel per config during planning)"
    )


def test_plan_config_tracker_registers_otel_when_enabled(run_config):
    """OtelTracingObserver must be registered when ``otel_enabled`` is True."""
    from exgentic.observers.handlers.otel import OtelTracingObserver

    with patch(
        "exgentic.utils.settings.get_settings",
        return_value=MagicMock(otel_enabled=True),
    ):
        _, tracker, _ = _plan_config(run_config)

    otel_observers = [o for o in tracker.observers if isinstance(o, OtelTracingObserver)]
    assert len(otel_observers) == 1, (
        f"Expected OtelTracingObserver when otel_enabled=True, got " f"{[type(o).__name__ for o in tracker.observers]}"
    )


def test_plan_config_tracker_skips_otel_when_disabled(run_config):
    """OtelTracingObserver must NOT be registered when ``otel_enabled`` is False."""
    from exgentic.observers.handlers.otel import OtelTracingObserver

    with patch(
        "exgentic.utils.settings.get_settings",
        return_value=MagicMock(otel_enabled=False),
    ):
        _, tracker, _ = _plan_config(run_config)

    otel_observers = [o for o in tracker.observers if isinstance(o, OtelTracingObserver)]
    assert otel_observers == [], (
        f"OtelTracingObserver should not be registered when otel_enabled=False, got "
        f"{[type(o).__name__ for o in tracker.observers]}"
    )
