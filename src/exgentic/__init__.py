# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

from typing import Any

from .interfaces.lib.api import (
    aggregate,
    evaluate,
    execute,
    list_agents,
    list_benchmarks,
    list_subsets,
    list_tasks,
    preview,
    results,
    status,
)
from .interfaces.registry import get_agent_entries, get_benchmark_entries

__all__ = [
    "aggregate",
    "evaluate",
    "execute",
    "list_agents",
    "list_benchmarks",
    "list_subsets",
    "list_tasks",
    "preview",
    "results",
    "status",
]


def _find_component_export(name: str):
    matches = [
        entry
        for entries in (get_benchmark_entries(), get_agent_entries())
        for entry in entries.values()
        if entry.attr == name
    ]
    if not matches:
        return None
    if len(matches) > 1:
        slugs = ", ".join(sorted(entry.slug_name for entry in matches))
        raise AttributeError(f"Ambiguous exgentic export '{name}' found in registry entries: {slugs}.")
    return matches[0]


def __getattr__(name: str) -> Any:
    entry = _find_component_export(name)
    if entry is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = entry.load()
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    # Keep dir() side-effect free.
    # Some introspection libraries (e.g. freezegun) iterate over dir(module)
    # and then call getattr() for each name. Exposing lazy registry exports here
    # can trigger expensive imports during unrelated initialization paths.
    return sorted(set(globals()))
