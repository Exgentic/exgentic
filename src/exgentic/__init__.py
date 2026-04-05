# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from typing import Any

try:
    from ._version import version as __version__
except ImportError:
    try:
        __version__ = package_version("exgentic")
    except PackageNotFoundError:
        __version__ = "0+unknown"

from .environment.manager import EnvironmentManager, EnvType
from .interfaces.registry import get_agent_entries, get_benchmark_entries

_API_EXPORTS = {
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
}

__all__ = [
    "__version__",
    "EnvironmentManager",
    "EnvType",
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
    if name in _API_EXPORTS:
        value = getattr(import_module(".interfaces.lib.api", __name__), name)
        globals()[name] = value
        return value

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
    return sorted(set(globals()) | _API_EXPORTS)
