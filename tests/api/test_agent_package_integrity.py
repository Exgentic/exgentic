# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Tests for agent/benchmark package integrity.

These tests catch two classes of bugs that are easy to introduce when
splitting agent/benchmark modules into separate config and instance files:

1. Missing ``__init__.py`` — without it, ``importlib.resources.files()``
   cannot discover ``requirements.txt`` / ``setup.sh``, so the CLI's
   ``needs_setup()`` returns False and dependencies are never installed.

2. Host-side import of heavy deps — if ``_get_instance_class_ref()`` falls
   back to ``_get_instance_class()`` (which does a lazy import), it will
   pull heavy third-party packages (litellm, smolagents, openai-agents)
   into the host process.  Agents with heavy deps must override
   ``_get_instance_class_ref()`` to return a string directly.
"""

from __future__ import annotations

import importlib
from pathlib import Path

from exgentic.interfaces.registry import get_agent_entries, get_benchmark_entries, load_agent


def test_all_agent_packages_have_init():
    """Every registered agent's package directory must contain __init__.py.

    Without ``__init__.py``, ``importlib.resources.files()`` cannot
    locate ``requirements.txt``, breaking auto-setup.
    """
    entries = get_agent_entries()
    for slug, entry in entries.items():
        # entry.module is like "exgentic.agents.openai.openai_mcp_agent"
        parts = entry.module.split(".")
        # Check every package directory from the agent package up
        for depth in range(3, len(parts)):
            package = ".".join(parts[:depth])
            try:
                mod = importlib.import_module(package)
            except ImportError:
                continue
            mod_file = getattr(mod, "__file__", None)
            if mod_file is None:
                # Namespace package — missing __init__.py
                raise AssertionError(
                    f"Agent '{slug}': package '{package}' is a namespace "
                    f"package (no __init__.py). This breaks "
                    f"importlib.resources.files() and prevents "
                    f"requirements.txt discovery."
                )


def test_all_benchmark_packages_have_init():
    """Every registered benchmark's package directory must contain __init__.py."""
    entries = get_benchmark_entries()
    for slug, entry in entries.items():
        parts = entry.module.split(".")
        for depth in range(3, len(parts)):
            package = ".".join(parts[:depth])
            try:
                mod = importlib.import_module(package)
            except ImportError:
                continue
            mod_file = getattr(mod, "__file__", None)
            if mod_file is None:
                raise AssertionError(
                    f"Benchmark '{slug}': package '{package}' is a namespace "
                    f"package (no __init__.py). This breaks "
                    f"importlib.resources.files() and prevents "
                    f"requirements.txt discovery."
                )


def test_agent_instance_class_ref_is_valid_string():
    """Every agent's _get_instance_class_ref() must return a 'module:class' string.

    This verifies the ref is well-formed; it does NOT import the module
    (which would defeat the purpose of the string ref).
    """
    entries = get_agent_entries()
    for slug, _entry in entries.items():
        agent_cls = load_agent(slug)
        ref = agent_cls._get_instance_class_ref()
        assert isinstance(ref, str), (
            f"Agent '{slug}': _get_instance_class_ref() returned " f"{type(ref).__name__}, expected str"
        )
        assert ":" in ref, (
            f"Agent '{slug}': _get_instance_class_ref() returned '{ref}', " f"expected 'module:qualname' format"
        )
        module_path, qualname = ref.rsplit(":", 1)
        assert module_path, f"Agent '{slug}': empty module path in ref '{ref}'"
        assert qualname, f"Agent '{slug}': empty qualname in ref '{ref}'"


def test_agent_instance_class_ref_module_file_exists():
    """The module referenced by _get_instance_class_ref() must exist on disk.

    This catches typos in string refs without importing the module.
    """
    entries = get_agent_entries()
    for slug, entry in entries.items():
        agent_cls = load_agent(slug)
        ref = agent_cls._get_instance_class_ref()
        module_path, _ = ref.rsplit(":", 1)
        # Convert module path to file path
        parts = module_path.split(".")
        # Find the source root by looking at the agent module's file
        agent_mod = importlib.import_module(entry.module)
        agent_file = Path(agent_mod.__file__)
        # Walk up from the agent file to find the source root
        src_root = agent_file
        module_parts = entry.module.split(".")
        for _ in module_parts:
            src_root = src_root.parent
        # Now resolve the ref module path
        expected_file = src_root / Path(*parts[:-1]) / f"{parts[-1]}.py"
        assert expected_file.exists(), (
            f"Agent '{slug}': _get_instance_class_ref() points to "
            f"'{module_path}' but {expected_file} does not exist."
        )


def test_with_runner_accepts_string_cls():
    """with_runner() must accept a 'module:class' string for the direct runner."""
    from exgentic.adapters.runners import with_runner

    ref = "exgentic.testing.agent:TestAgentInstance"
    proxy = with_runner(
        ref,
        runner="direct",
        session_id="test-string-ref",
        seed=42,
        policy="good_then_finish",
        finish_after=2,
        max_steps=10,
    )
    # Should successfully create the instance
    assert proxy is not None
    proxy.close()
