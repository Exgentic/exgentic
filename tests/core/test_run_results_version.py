# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Anonymous Authors.

from framework import __version__
from framework.core.types import RunResults


def test_run_results_framework_version():
    """RunResults accepts and round-trips the framework_version field."""
    results = RunResults(
        benchmark_name="test",
        agent_name="test",
        total_sessions=0,
        successful_sessions=0,
        session_results=[],
        framework_version=__version__,
    )
    assert results.framework_version == __version__
    dumped = results.model_dump()
    assert dumped["framework_version"] == __version__
