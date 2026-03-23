# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from exgentic import __version__
from exgentic.core.types import RunResults


def test_run_results_exgentic_version():
    """RunResults accepts and round-trips the exgentic_version field."""
    results = RunResults(
        benchmark_name="test",
        agent_name="test",
        total_sessions=0,
        successful_sessions=0,
        session_results=[],
        exgentic_version=__version__,
    )
    assert results.exgentic_version == __version__
    dumped = results.model_dump()
    assert dumped["exgentic_version"] == __version__
