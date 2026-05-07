# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Anonymous Authors.

from __future__ import annotations

from typing import Iterator

import pytest
from framework.interfaces import registry
from framework.interfaces.registry import RegistryEntry


@pytest.fixture(autouse=True)
def register_test_components() -> Iterator[None]:
    from framework.environment.instance import get_manager

    original_benchmarks = dict(registry.BENCHMARKS)
    original_agents = dict(registry.AGENTS)
    registry.BENCHMARKS["test_benchmark"] = RegistryEntry(
        slug_name="test_benchmark",
        display_name="Test Benchmark",
        module="framework.testing.benchmark",
        attr="TestBenchmark",
        kind="benchmark",
    )
    registry.AGENTS["test_agent"] = RegistryEntry(
        slug_name="test_agent",
        display_name="Test Agent",
        module="framework.testing.agent",
        attr="TestAgent",
        kind="agent",
    )
    # Clear cached task IDs so tests with different task lists
    # don't interfere with each other.
    cache_dir = get_manager().env_path("benchmarks/test_benchmark")
    for f in cache_dir.glob("task_ids_*.json"):
        f.unlink(missing_ok=True)
    try:
        yield
    finally:
        registry.BENCHMARKS.clear()
        registry.BENCHMARKS.update(original_benchmarks)
        registry.AGENTS.clear()
        registry.AGENTS.update(original_agents)
