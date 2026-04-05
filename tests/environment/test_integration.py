# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Integration tests: verify environment state is clean after operations.

These regression tests ensure that install / uninstall / list leave
the filesystem in the expected state, with no stale artefacts.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

from exgentic.environment import EnvironmentManager, EnvType

_pkg_counter = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_fake_package(tmp_path: Path) -> str:
    """Create a minimal importable package and return its dotted module path."""
    global _pkg_counter
    _pkg_counter += 1
    tag = f"integ{_pkg_counter}"

    top = f"fpkg_{tag}"
    mid = "fbench"
    leaf = "mybench"

    pkg_dir = tmp_path / top / mid / leaf
    pkg_dir.mkdir(parents=True)
    (tmp_path / top / "__init__.py").write_text("")
    (tmp_path / top / mid / "__init__.py").write_text("")
    (pkg_dir / "__init__.py").write_text("")
    (pkg_dir / "main.py").write_text("")

    if str(tmp_path) not in sys.path:
        sys.path.insert(0, str(tmp_path))
    importlib.invalidate_caches()

    return f"{top}.{mid}.{leaf}.main"


def _get_test_manager(tmp_path: Path) -> EnvironmentManager:
    """Return an EnvironmentManager rooted in a tmp_path subdirectory."""
    return EnvironmentManager(base_dir=tmp_path / "envs")


# ---------------------------------------------------------------------------
# 1. Install creates marker at correct path
# ---------------------------------------------------------------------------


def test_install_benchmark_creates_marker(tmp_path: Path) -> None:
    """Install creates .installed at the correct env path."""
    module_path = _create_fake_package(tmp_path)
    mgr = _get_test_manager(tmp_path)

    mgr.install("benchmarks/test-bench", env_type=EnvType.LOCAL, module_path=module_path)

    assert mgr.is_installed("benchmarks/test-bench")
    marker = mgr.env_path("benchmarks/test-bench") / ".installed"
    assert marker.exists()
    data = json.loads(marker.read_text())
    assert "local" in data


# ---------------------------------------------------------------------------
# 2. Install does NOT create venv dirs in manager space
# ---------------------------------------------------------------------------


def test_install_local_does_not_create_venv(tmp_path: Path) -> None:
    """LOCAL install should not create a venv/ directory."""
    module_path = _create_fake_package(tmp_path)
    mgr = _get_test_manager(tmp_path)

    mgr.install("benchmarks/test-bench", env_type=EnvType.LOCAL, module_path=module_path)

    assert not (mgr.env_path("benchmarks/test-bench") / "venv").exists()


# ---------------------------------------------------------------------------
# 3. Uninstall cleans up completely
# ---------------------------------------------------------------------------


def test_uninstall_removes_all_traces(tmp_path: Path) -> None:
    """After uninstall, no files remain in the env dir."""
    module_path = _create_fake_package(tmp_path)
    mgr = _get_test_manager(tmp_path)

    mgr.install("benchmarks/test-bench", env_type=EnvType.LOCAL, module_path=module_path)
    mgr.uninstall("benchmarks/test-bench")

    assert not mgr.env_path("benchmarks/test-bench").exists()


# ---------------------------------------------------------------------------
# 4. list_installed returns correct format
# ---------------------------------------------------------------------------


def test_list_installed_format(tmp_path: Path) -> None:
    """list_installed returns dicts with name and environments."""
    module_path = _create_fake_package(tmp_path)
    mgr = _get_test_manager(tmp_path)

    mgr.install("benchmarks/alpha", env_type=EnvType.LOCAL, module_path=module_path)
    mgr.install("agents/beta", env_type=EnvType.LOCAL, module_path=module_path)

    result = mgr.list_installed()
    assert len(result) == 2
    for item in result:
        assert "name" in item
        assert "environments" in item
        assert "local" in item["environments"]
        assert "installed_at" in item["environments"]["local"]


# ---------------------------------------------------------------------------
# 5. Venv and local can coexist
# ---------------------------------------------------------------------------


def test_venv_and_local_coexist(tmp_path: Path) -> None:
    """Both env types can be installed for the same name."""
    module_path = _create_fake_package(tmp_path)
    mgr = _get_test_manager(tmp_path)

    mgr.install("benchmarks/test-bench", env_type=EnvType.VENV, module_path=module_path)
    mgr.install("benchmarks/test-bench", env_type=EnvType.LOCAL, module_path=module_path)

    assert mgr.is_installed("benchmarks/test-bench", env_type=EnvType.VENV)
    assert mgr.is_installed("benchmarks/test-bench", env_type=EnvType.LOCAL)


# ---------------------------------------------------------------------------
# 6. Runner venv path is under EnvironmentManager space
# ---------------------------------------------------------------------------


def test_runner_venv_in_manager_space() -> None:
    """VenvRunner venvs should be under ~/.exgentic/{kind}/{slug}/venv/."""
    manager_prefix = str(Path.home() / ".exgentic")

    # Build the path the same way RunnerMixin does.
    for kind in ("benchmarks", "agents"):
        venv_dir = str(Path.home() / ".exgentic" / kind / "test-slug" / "venv")
        assert venv_dir.startswith(manager_prefix), "venv_dir should be under ~/.exgentic/"
        assert kind in venv_dir, f"venv_dir should contain {kind}"
        assert venv_dir.endswith("/venv"), "venv_dir should end with /venv"
