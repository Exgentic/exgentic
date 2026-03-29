# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Tests for tau2 data directory resolution (issue #74)."""

from __future__ import annotations

import os
from pathlib import Path
from unittest import mock

import pytest


def test_resolve_tau2_data_dir_prefers_cache(tmp_path: Path) -> None:
    """When the cache directory contains data, it should be preferred."""
    cache_data = tmp_path / "benchmarks" / "tau2" / "data"
    cache_data.mkdir(parents=True)

    fake_mgr = mock.MagicMock()
    fake_mgr.env_path.return_value = tmp_path / "benchmarks" / "tau2"

    with mock.patch(
        "exgentic.environment.instance.get_manager",
        return_value=fake_mgr,
    ):
        from exgentic.benchmarks.tau2 import _resolve_tau2_data_dir

        result = _resolve_tau2_data_dir()

    assert result == str(cache_data)


def test_resolve_tau2_data_dir_falls_back_to_legacy(tmp_path: Path) -> None:
    """When cache data is missing, fall back to the legacy installation path."""
    empty_cache = tmp_path / "benchmarks" / "tau2"
    empty_cache.mkdir(parents=True)
    # No data/ subdirectory created — cache dir exists but data/ does not

    fake_mgr = mock.MagicMock()
    fake_mgr.env_path.return_value = empty_cache

    with mock.patch(
        "exgentic.environment.instance.get_manager",
        return_value=fake_mgr,
    ):
        from exgentic.benchmarks.tau2 import _resolve_tau2_data_dir

        result = _resolve_tau2_data_dir()

    # Should fall back to the legacy path under the package directory
    expected_suffix = os.path.join("benchmarks", "tau2", "installation", "tau2-bench", "data")
    assert result.endswith(expected_suffix)


def test_env_var_takes_precedence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """If TAU2_DATA_DIR is already set, the module should not overwrite it."""
    custom_dir = str(tmp_path / "my_custom_data")
    monkeypatch.setenv("TAU2_DATA_DIR", custom_dir)

    # Re-import to trigger the module-level guard
    import importlib

    import exgentic.benchmarks.tau2 as tau2_mod

    importlib.reload(tau2_mod)

    assert os.environ["TAU2_DATA_DIR"] == custom_dir
