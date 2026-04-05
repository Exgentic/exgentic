# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Tests for exgentic.environment.manager.

Tests are organized by the assumptions the rest of the repo makes about
the manager's capabilities.  Every public method and every env type
is tested in isolation.
"""

from __future__ import annotations

import importlib
import json
import os
import shutil
import stat
import subprocess
import sys
import textwrap
from pathlib import Path
from unittest import mock

import pytest
from exgentic.environment import EnvironmentManager, EnvType
from exgentic.environment.helpers import build_subprocess_env, find_package_file, require_uv

_pkg_counter = 0

_real_subprocess_run = subprocess.run


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_fake_package(
    tmp_path: Path,
    *,
    with_requirements: bool = True,
    with_setup: bool = True,
    with_system_deps: bool = False,
) -> str:
    """Create a minimal importable package with optional resource files."""
    global _pkg_counter
    _pkg_counter += 1
    tag = f"p{_pkg_counter}"

    top = f"fpkg_{tag}"
    mid = "fbench"
    leaf = "mybench"

    pkg_dir = tmp_path / top / mid / leaf
    pkg_dir.mkdir(parents=True)
    (tmp_path / top / "__init__.py").write_text("")
    (tmp_path / top / mid / "__init__.py").write_text("")
    (pkg_dir / "__init__.py").write_text("")
    (pkg_dir / "main.py").write_text("")

    if with_requirements:
        (pkg_dir / "requirements.txt").write_text("requests\n")

    if with_setup:
        script = textwrap.dedent(
            """\
            #!/usr/bin/env bash
            mkdir -p "data"
            touch "data/setup_ran.txt"
        """
        )
        setup_sh = pkg_dir / "setup.sh"
        setup_sh.write_text(script)
        setup_sh.chmod(setup_sh.stat().st_mode | stat.S_IEXEC)

    if with_system_deps:
        (pkg_dir / "system-deps.txt").write_text("curl\nwget\n")

    if str(tmp_path) not in sys.path:
        sys.path.insert(0, str(tmp_path))
    importlib.invalidate_caches()

    return f"{top}.{mid}.{leaf}.main"


def _docker_mock_result(**overrides):
    result = mock.MagicMock()
    result.returncode = overrides.get("returncode", 0)
    result.stdout = overrides.get("stdout", "")
    result.stderr = overrides.get("stderr", "")
    return result


def _create_fake_project(tmp_path: Path, *, name: str = "myproject") -> Path:
    """Create a minimal Python project with pyproject.toml and src layout."""
    project = tmp_path / f"project_{name}"
    project.mkdir(parents=True)
    pkg_name = name.replace("-", "_")
    src_dir = project / "src" / pkg_name
    src_dir.mkdir(parents=True)
    (src_dir / "__init__.py").write_text('__version__ = "0.1.0"\n')
    (project / "README.md").write_text(f"# {name}\n")
    (project / "pyproject.toml").write_text(
        textwrap.dedent(
            f"""\
        [project]
        name = "{name}"
        version = "0.1.0"
        requires-python = ">=3.10"
        dependencies = []

        [build-system]
        requires = ["hatchling"]
        build-backend = "hatchling.build"
        """
        )
    )
    return project


# ---------------------------------------------------------------------------
# Venv install
# ---------------------------------------------------------------------------


class TestVenvInstall:
    """Venv is the default env type.

    The evaluate flow and venv runner depend on: venv/ dir existing,
    .installed marker with installed_at, and setup.sh having been run.
    """

    def test_creates_venv_and_marker(self, tmp_path: Path) -> None:
        module_path = _create_fake_package(tmp_path, with_requirements=False, with_setup=False)
        mgr = EnvironmentManager(base_dir=tmp_path / "envs")

        env_dir = mgr.install("mybench", module_path=module_path)

        assert env_dir.is_dir()
        assert (env_dir / "venv").is_dir()
        assert (env_dir / "venv" / "bin" / "python").exists()
        marker = json.loads((env_dir / ".installed").read_text())
        assert "venv" in marker
        assert "installed_at" in marker["venv"]

    def test_skips_if_already_installed(self, tmp_path: Path) -> None:
        module_path = _create_fake_package(tmp_path, with_requirements=False, with_setup=False)
        mgr = EnvironmentManager(base_dir=tmp_path / "envs")

        mgr.install("mybench", module_path=module_path)
        mtime = (mgr.env_path("mybench") / ".installed").stat().st_mtime

        mgr.install("mybench", module_path=module_path)
        assert (mgr.env_path("mybench") / ".installed").stat().st_mtime == mtime

    def test_force_reinstalls(self, tmp_path: Path) -> None:
        module_path = _create_fake_package(tmp_path, with_requirements=False, with_setup=False)
        mgr = EnvironmentManager(base_dir=tmp_path / "envs")

        mgr.install("mybench", module_path=module_path)
        sentinel = mgr.env_path("mybench") / "venv" / "sentinel.txt"
        sentinel.write_text("old")

        mgr.install("mybench", force=True, module_path=module_path)

        assert not sentinel.exists()
        assert mgr.is_installed("mybench", env_type=EnvType.VENV)

    def test_runs_setup_sh(self, tmp_path: Path) -> None:
        module_path = _create_fake_package(tmp_path, with_requirements=False, with_setup=True)
        mgr = EnvironmentManager(base_dir=tmp_path / "envs")

        env_dir = mgr.install("mybench", module_path=module_path)

        assert (env_dir / "data" / "setup_ran.txt").is_file()

    def test_cleanup_on_failure(self, tmp_path: Path) -> None:
        module_path = _create_fake_package(tmp_path, with_requirements=False, with_setup=False)
        mgr = EnvironmentManager(base_dir=tmp_path / "envs")

        def fail_pip(cmd, **kwargs):
            if isinstance(cmd, list) and "pip" in cmd and "install" in cmd:
                raise subprocess.CalledProcessError(1, cmd)
            return _real_subprocess_run(cmd, **kwargs)

        with mock.patch("subprocess.run", side_effect=fail_pip):
            with pytest.raises(subprocess.CalledProcessError):
                mgr.install("mybench", packages=["some-pkg"], module_path=module_path)

        venv_dir = mgr.env_path("mybench") / "venv"
        assert not venv_dir.exists()
        assert not mgr.is_installed("mybench", env_type=EnvType.VENV)

    def test_venv_python_path(self, tmp_path: Path) -> None:
        module_path = _create_fake_package(tmp_path, with_requirements=False, with_setup=False)
        mgr = EnvironmentManager(base_dir=tmp_path / "envs")

        mgr.install("mybench", module_path=module_path)

        python_path = mgr.venv_python("mybench")
        assert python_path == str(tmp_path / "envs" / "mybench" / "venv" / "bin" / "python")
        assert Path(python_path).exists()


# ---------------------------------------------------------------------------
# Local install
# ---------------------------------------------------------------------------


class TestLocalInstall:
    """Local install uses the current Python (sys.executable).

    Used for debugging/development. The manager must record which
    Python was used so runners can find it.
    """

    def test_installs_without_venv(self, tmp_path: Path) -> None:
        module_path = _create_fake_package(tmp_path, with_requirements=False, with_setup=False)
        mgr = EnvironmentManager(base_dir=tmp_path / "envs")

        env_dir = mgr.install("mybench", env_type=EnvType.LOCAL, module_path=module_path)

        assert env_dir.is_dir()
        assert not (env_dir / "venv").exists()
        assert mgr.is_installed("mybench", env_type=EnvType.LOCAL)

    def test_marker_has_python_path(self, tmp_path: Path) -> None:
        module_path = _create_fake_package(tmp_path, with_requirements=False, with_setup=False)
        mgr = EnvironmentManager(base_dir=tmp_path / "envs")

        mgr.install("mybench", env_type=EnvType.LOCAL, module_path=module_path)

        marker = json.loads((mgr.env_path("mybench") / ".installed").read_text())
        assert marker["local"]["python"] == sys.executable
        assert "installed_at" in marker["local"]

    def test_skips_if_already_installed(self, tmp_path: Path) -> None:
        module_path = _create_fake_package(tmp_path, with_requirements=False, with_setup=False)
        mgr = EnvironmentManager(base_dir=tmp_path / "envs")

        mgr.install("mybench", env_type=EnvType.LOCAL, module_path=module_path)
        mtime = (mgr.env_path("mybench") / ".installed").stat().st_mtime

        mgr.install("mybench", env_type=EnvType.LOCAL, module_path=module_path)
        assert (mgr.env_path("mybench") / ".installed").stat().st_mtime == mtime

    def test_force_reinstalls(self, tmp_path: Path) -> None:
        module_path = _create_fake_package(tmp_path, with_requirements=False, with_setup=False)
        mgr = EnvironmentManager(base_dir=tmp_path / "envs")

        mgr.install("mybench", env_type=EnvType.LOCAL, module_path=module_path)
        old_marker = json.loads((mgr.env_path("mybench") / ".installed").read_text())

        mgr.install("mybench", env_type=EnvType.LOCAL, force=True, module_path=module_path)
        new_marker = json.loads((mgr.env_path("mybench") / ".installed").read_text())

        assert new_marker["local"]["installed_at"] >= old_marker["local"]["installed_at"]

    def test_runs_setup_sh_without_virtual_env(self, tmp_path: Path) -> None:
        """setup.sh runs with cwd=env_dir but NOT VIRTUAL_ENV."""
        module_path = _create_fake_package(tmp_path, with_requirements=False, with_setup=True)
        mgr = EnvironmentManager(base_dir=tmp_path / "envs")

        env_dir = mgr.install("mybench", env_type=EnvType.LOCAL, module_path=module_path)

        assert (env_dir / "data" / "setup_ran.txt").is_file()

    def test_installs_requirements_into_current_python(self, tmp_path: Path) -> None:
        """Local install must call uv pip install --python sys.executable."""
        module_path = _create_fake_package(tmp_path, with_requirements=True, with_setup=False)
        mgr = EnvironmentManager(base_dir=tmp_path / "envs")

        pip_calls: list[list[str]] = []

        def capture_run(cmd, **kwargs):
            if isinstance(cmd, list) and "pip" in cmd and "install" in cmd:
                pip_calls.append(list(cmd))
                return _docker_mock_result()
            return _real_subprocess_run(cmd, **kwargs)

        with mock.patch("subprocess.run", side_effect=capture_run):
            mgr.install("mybench", env_type=EnvType.LOCAL, module_path=module_path)

        assert len(pip_calls) == 1
        python_idx = pip_calls[0].index("--python") + 1
        assert pip_calls[0][python_idx] == sys.executable


# ---------------------------------------------------------------------------
# Docker install
# ---------------------------------------------------------------------------


class TestDockerInstall:
    """Docker install builds an image with deps baked in.

    The docker runner needs the image tag from the marker.
    """

    def test_builds_image_and_writes_marker(self, tmp_path: Path) -> None:
        module_path = _create_fake_package(tmp_path, with_requirements=True, with_setup=True)
        mgr = EnvironmentManager(base_dir=tmp_path / "envs")

        dockerfiles: list[str] = []

        def capture_run(cmd, **kwargs):
            if cmd[0] == "docker":
                if cmd[1] == "build":
                    df = Path(cmd[-1]) / "Dockerfile"
                    if df.exists():
                        dockerfiles.append(df.read_text())
                if cmd[1:3] == ["image", "inspect"]:
                    return _docker_mock_result(returncode=1)
                return _docker_mock_result()
            return _real_subprocess_run(cmd, **kwargs)

        with mock.patch("subprocess.run", side_effect=capture_run):
            env_dir = mgr.install("mybench", env_type=EnvType.DOCKER, module_path=module_path)

        assert len(dockerfiles) == 1
        assert "requirements.txt" in dockerfiles[0]
        assert "setup.sh" in dockerfiles[0]

        marker = json.loads((env_dir / ".installed").read_text())
        assert "docker" in marker
        assert "image" in marker["docker"]
        assert "installed_at" in marker["docker"]

    def test_reuses_existing_image(self, tmp_path: Path) -> None:
        module_path = _create_fake_package(tmp_path, with_requirements=True, with_setup=False)
        mgr = EnvironmentManager(base_dir=tmp_path / "envs")

        build_called = []

        def side_effect(cmd, **kwargs):
            if cmd[0] == "docker":
                if cmd[1] == "build":
                    build_called.append(True)
                if cmd[1:3] == ["image", "inspect"]:
                    return _docker_mock_result(returncode=0)
                return _docker_mock_result()
            return _real_subprocess_run(cmd, **kwargs)

        with mock.patch("subprocess.run", side_effect=side_effect):
            mgr.install("mybench", env_type=EnvType.DOCKER, module_path=module_path)

        assert len(build_called) == 0

    def test_force_rebuilds(self, tmp_path: Path) -> None:
        module_path = _create_fake_package(tmp_path, with_requirements=True, with_setup=False)
        mgr = EnvironmentManager(base_dir=tmp_path / "envs")

        build_calls: list[bool] = []

        def side_effect(cmd, **kwargs):
            if cmd[0] == "docker":
                if cmd[1] == "build":
                    build_calls.append(True)
                if cmd[1:3] == ["image", "inspect"]:
                    return _docker_mock_result(returncode=0)
                return _docker_mock_result()
            return _real_subprocess_run(cmd, **kwargs)

        with mock.patch("subprocess.run", side_effect=side_effect):
            mgr.install("mybench", env_type=EnvType.DOCKER, force=True, module_path=module_path)

        assert len(build_calls) == 1

    def test_includes_system_deps(self, tmp_path: Path) -> None:
        module_path = _create_fake_package(tmp_path, with_requirements=True, with_setup=False, with_system_deps=True)
        mgr = EnvironmentManager(base_dir=tmp_path / "envs")

        dockerfiles: list[str] = []

        def capture_run(cmd, **kwargs):
            if cmd[0] == "docker":
                if cmd[1] == "build":
                    df = Path(cmd[-1]) / "Dockerfile"
                    if df.exists():
                        dockerfiles.append(df.read_text())
                if cmd[1:3] == ["image", "inspect"]:
                    return _docker_mock_result(returncode=1)
                return _docker_mock_result()
            return _real_subprocess_run(cmd, **kwargs)

        with mock.patch("subprocess.run", side_effect=capture_run):
            mgr.install("mybench", env_type=EnvType.DOCKER, module_path=module_path)

        assert "apt-get install -y curl wget" in dockerfiles[0]

    def test_content_hash_differs(self, tmp_path: Path) -> None:
        module_path_a = _create_fake_package(tmp_path, with_requirements=True, with_setup=False)
        module_path_b = _create_fake_package(tmp_path, with_requirements=True, with_setup=False)

        parts_b = module_path_b.split(".")
        pkg_dir_b = tmp_path
        for part in parts_b[:-1]:
            pkg_dir_b = pkg_dir_b / part
        (pkg_dir_b / "requirements.txt").write_text("numpy\npandas\n")
        importlib.invalidate_caches()

        tags: list[str] = []

        def capture_run(cmd, **kwargs):
            if cmd[0] == "docker":
                if cmd[1] == "build":
                    idx = list(cmd).index("-t")
                    tags.append(cmd[idx + 1])
                if cmd[1:3] == ["image", "inspect"]:
                    return _docker_mock_result(returncode=1)
                return _docker_mock_result()
            return _real_subprocess_run(cmd, **kwargs)

        mgr = EnvironmentManager(base_dir=tmp_path / "envs")
        with mock.patch("subprocess.run", side_effect=capture_run):
            mgr.install("a", env_type=EnvType.DOCKER, module_path=module_path_a)
            mgr.install("b", env_type=EnvType.DOCKER, module_path=module_path_b)

        assert tags[0].split(":")[-1] != tags[1].split(":")[-1]


# ---------------------------------------------------------------------------
# Coexistence
# ---------------------------------------------------------------------------


class TestCoexistence:
    """Multiple env types can coexist for the same name.

    Runners pick whichever env type they need.
    """

    def test_venv_and_local_coexist(self, tmp_path: Path) -> None:
        module_path = _create_fake_package(tmp_path, with_requirements=False, with_setup=False)
        mgr = EnvironmentManager(base_dir=tmp_path / "envs")

        mgr.install("mybench", env_type=EnvType.VENV, module_path=module_path)
        mgr.install("mybench", env_type=EnvType.LOCAL, module_path=module_path)

        assert mgr.is_installed("mybench", env_type=EnvType.VENV)
        assert mgr.is_installed("mybench", env_type=EnvType.LOCAL)
        assert (mgr.env_path("mybench") / "venv").is_dir()

        marker = json.loads((mgr.env_path("mybench") / ".installed").read_text())
        assert "venv" in marker
        assert "local" in marker

    def test_venv_and_docker_coexist(self, tmp_path: Path) -> None:
        module_path = _create_fake_package(tmp_path, with_requirements=False, with_setup=False)
        mgr = EnvironmentManager(base_dir=tmp_path / "envs")

        mgr.install("mybench", env_type=EnvType.VENV, module_path=module_path)

        def docker_side_effect(cmd, **kwargs):
            if cmd[0] == "docker":
                if cmd[1:3] == ["image", "inspect"]:
                    return _docker_mock_result(returncode=1)
                return _docker_mock_result()
            return _real_subprocess_run(cmd, **kwargs)

        with mock.patch("subprocess.run", side_effect=docker_side_effect):
            mgr.install("mybench", env_type=EnvType.DOCKER, module_path=module_path)

        assert mgr.is_installed("mybench", env_type=EnvType.VENV)
        assert mgr.is_installed("mybench", env_type=EnvType.DOCKER)

    def test_all_three_coexist(self, tmp_path: Path) -> None:
        module_path = _create_fake_package(tmp_path, with_requirements=False, with_setup=False)
        mgr = EnvironmentManager(base_dir=tmp_path / "envs")

        mgr.install("mybench", env_type=EnvType.VENV, module_path=module_path)
        mgr.install("mybench", env_type=EnvType.LOCAL, module_path=module_path)

        def docker_side_effect(cmd, **kwargs):
            if cmd[0] == "docker":
                if cmd[1:3] == ["image", "inspect"]:
                    return _docker_mock_result(returncode=1)
                return _docker_mock_result()
            return _real_subprocess_run(cmd, **kwargs)

        with mock.patch("subprocess.run", side_effect=docker_side_effect):
            mgr.install("mybench", env_type=EnvType.DOCKER, module_path=module_path)

        marker = json.loads((mgr.env_path("mybench") / ".installed").read_text())
        assert set(marker.keys()) == {"venv", "local", "docker"}

    def test_force_reinstall_one_preserves_others(self, tmp_path: Path) -> None:
        module_path = _create_fake_package(tmp_path, with_requirements=False, with_setup=False)
        mgr = EnvironmentManager(base_dir=tmp_path / "envs")

        mgr.install("mybench", env_type=EnvType.VENV, module_path=module_path)
        mgr.install("mybench", env_type=EnvType.LOCAL, module_path=module_path)

        old_marker = json.loads((mgr.env_path("mybench") / ".installed").read_text())
        old_local_at = old_marker["local"]["installed_at"]

        mgr.install("mybench", env_type=EnvType.VENV, force=True, module_path=module_path)

        new_marker = json.loads((mgr.env_path("mybench") / ".installed").read_text())
        assert "venv" in new_marker
        assert "local" in new_marker
        assert new_marker["local"]["installed_at"] == old_local_at


# ---------------------------------------------------------------------------
# Uninstall
# ---------------------------------------------------------------------------


class TestUninstall:
    """Uninstall removes the specified env type without affecting others.

    When the last env type is removed, the whole directory is cleaned up.
    """

    def test_uninstall_venv(self, tmp_path: Path) -> None:
        module_path = _create_fake_package(tmp_path, with_requirements=False, with_setup=False)
        mgr = EnvironmentManager(base_dir=tmp_path / "envs")

        mgr.install("mybench", module_path=module_path)
        mgr.uninstall("mybench", env_type=EnvType.VENV)

        assert not mgr.is_installed("mybench", env_type=EnvType.VENV)
        assert not mgr.env_path("mybench").exists()

    def test_uninstall_local(self, tmp_path: Path) -> None:
        module_path = _create_fake_package(tmp_path, with_requirements=False, with_setup=False)
        mgr = EnvironmentManager(base_dir=tmp_path / "envs")

        mgr.install("mybench", env_type=EnvType.LOCAL, module_path=module_path)
        mgr.uninstall("mybench", env_type=EnvType.LOCAL)

        assert not mgr.is_installed("mybench", env_type=EnvType.LOCAL)

    def test_uninstall_docker_removes_image(self, tmp_path: Path) -> None:
        mgr = EnvironmentManager(base_dir=tmp_path / "envs")
        env_dir = mgr.env_path("mybench")
        env_dir.mkdir(parents=True)

        image_tag = "mybench:abc123"
        (env_dir / ".installed").write_text(
            json.dumps({"docker": {"installed_at": "2026-01-01T00:00:00Z", "image": image_tag}})
        )

        rmi_calls: list[list[str]] = []

        def side_effect(cmd, **kwargs):
            if cmd[0] == "docker" and cmd[1] == "rmi":
                rmi_calls.append(list(cmd))
                return _docker_mock_result()
            return _real_subprocess_run(cmd, **kwargs)

        with mock.patch("subprocess.run", side_effect=side_effect):
            mgr.uninstall("mybench", env_type=EnvType.DOCKER)

        assert len(rmi_calls) == 1
        assert rmi_calls[0] == ["docker", "rmi", image_tag]

    def test_uninstall_all(self, tmp_path: Path) -> None:
        module_path = _create_fake_package(tmp_path, with_requirements=False, with_setup=False)
        mgr = EnvironmentManager(base_dir=tmp_path / "envs")

        mgr.install("mybench", env_type=EnvType.VENV, module_path=module_path)
        mgr.install("mybench", env_type=EnvType.LOCAL, module_path=module_path)

        mgr.uninstall("mybench")

        assert not mgr.env_path("mybench").exists()
        assert not mgr.is_installed("mybench")

    def test_uninstall_one_keeps_others(self, tmp_path: Path) -> None:
        module_path = _create_fake_package(tmp_path, with_requirements=False, with_setup=False)
        mgr = EnvironmentManager(base_dir=tmp_path / "envs")

        mgr.install("mybench", env_type=EnvType.VENV, module_path=module_path)
        mgr.install("mybench", env_type=EnvType.LOCAL, module_path=module_path)

        mgr.uninstall("mybench", env_type=EnvType.VENV)

        assert not mgr.is_installed("mybench", env_type=EnvType.VENV)
        assert mgr.is_installed("mybench", env_type=EnvType.LOCAL)
        assert not (mgr.env_path("mybench") / "venv").exists()
        assert mgr.env_path("mybench").exists()

    def test_uninstall_last_removes_dir(self, tmp_path: Path) -> None:
        module_path = _create_fake_package(tmp_path, with_requirements=False, with_setup=False)
        mgr = EnvironmentManager(base_dir=tmp_path / "envs")

        mgr.install("mybench", env_type=EnvType.VENV, module_path=module_path)
        mgr.install("mybench", env_type=EnvType.LOCAL, module_path=module_path)

        mgr.uninstall("mybench", env_type=EnvType.VENV)
        assert mgr.env_path("mybench").exists()

        mgr.uninstall("mybench", env_type=EnvType.LOCAL)
        assert not mgr.env_path("mybench").exists()

    def test_uninstall_nonexistent_is_noop(self, tmp_path: Path) -> None:
        mgr = EnvironmentManager(base_dir=tmp_path / "envs")
        mgr.uninstall("nonexistent")
        mgr.uninstall("nonexistent", env_type=EnvType.VENV)

    def test_uninstall_all_with_docker_removes_image(self, tmp_path: Path) -> None:
        mgr = EnvironmentManager(base_dir=tmp_path / "envs")
        env_dir = mgr.env_path("mybench")
        env_dir.mkdir(parents=True)

        image_tag = "mybench:abc123"
        (env_dir / ".installed").write_text(
            json.dumps(
                {
                    "venv": {"installed_at": "2026-01-01T00:00:00Z"},
                    "docker": {"installed_at": "2026-01-01T00:00:00Z", "image": image_tag},
                }
            )
        )
        (env_dir / "venv").mkdir()

        rmi_calls: list[list[str]] = []

        def side_effect(cmd, **kwargs):
            if cmd[0] == "docker" and cmd[1] == "rmi":
                rmi_calls.append(list(cmd))
                return _docker_mock_result()
            return _real_subprocess_run(cmd, **kwargs)

        with mock.patch("subprocess.run", side_effect=side_effect):
            mgr.uninstall("mybench")

        assert len(rmi_calls) == 1
        assert not env_dir.exists()


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------


class TestQueries:
    """The list commands and evaluate flow depend on querying install state."""

    def test_is_installed_any(self, tmp_path: Path) -> None:
        module_path = _create_fake_package(tmp_path, with_requirements=False, with_setup=False)
        mgr = EnvironmentManager(base_dir=tmp_path / "envs")

        assert not mgr.is_installed("mybench")

        mgr.install("mybench", env_type=EnvType.LOCAL, module_path=module_path)
        assert mgr.is_installed("mybench")
        assert not mgr.is_installed("mybench", env_type=EnvType.VENV)
        assert mgr.is_installed("mybench", env_type=EnvType.LOCAL)

    def test_get_info(self, tmp_path: Path) -> None:
        module_path = _create_fake_package(tmp_path, with_requirements=False, with_setup=False)
        mgr = EnvironmentManager(base_dir=tmp_path / "envs")

        assert mgr.get_info("mybench") is None

        mgr.install("mybench", env_type=EnvType.VENV, module_path=module_path)
        mgr.install("mybench", env_type=EnvType.LOCAL, module_path=module_path)

        info = mgr.get_info("mybench")
        assert info is not None
        assert info["name"] == "mybench"
        assert "venv" in info["environments"]
        assert "local" in info["environments"]
        assert "installed_at" in info["environments"]["venv"]
        assert "python" in info["environments"]["local"]

    def test_list_installed(self, tmp_path: Path) -> None:
        module_path = _create_fake_package(tmp_path, with_requirements=False, with_setup=False)
        mgr = EnvironmentManager(base_dir=tmp_path / "envs")

        assert mgr.list_installed() == []

        mgr.install("alpha", module_path=module_path)
        mgr.install("beta", module_path=module_path)

        result = mgr.list_installed()
        assert len(result) == 2
        names = [r["name"] for r in result]
        assert names == ["alpha", "beta"]
        assert all("venv" in r["environments"] for r in result)

    def test_list_installed_includes_env_details(self, tmp_path: Path) -> None:
        module_path = _create_fake_package(tmp_path, with_requirements=False, with_setup=False)
        mgr = EnvironmentManager(base_dir=tmp_path / "envs")

        mgr.install("mybench", env_type=EnvType.VENV, module_path=module_path)
        mgr.install("mybench", env_type=EnvType.LOCAL, module_path=module_path)

        result = mgr.list_installed()
        assert len(result) == 1
        envs = result[0]["environments"]
        assert "venv" in envs
        assert "local" in envs
        assert "installed_at" in envs["venv"]
        assert "python" in envs["local"]

    def test_list_installed_nested_names(self, tmp_path: Path) -> None:
        """Names like benchmarks/tau2 should work with list_installed."""
        module_path = _create_fake_package(tmp_path, with_requirements=False, with_setup=False)
        mgr = EnvironmentManager(base_dir=tmp_path / "envs")

        mgr.install("benchmarks/tau2", module_path=module_path)
        mgr.install("agents/tool_calling", module_path=module_path)

        result = mgr.list_installed()
        names = [r["name"] for r in result]
        assert "benchmarks/tau2" in names
        assert "agents/tool_calling" in names


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


class TestPaths:
    """Path helpers for locating data and venv directories."""

    def test_env_path(self, tmp_path: Path) -> None:
        mgr = EnvironmentManager(base_dir=tmp_path / "envs")

        assert mgr.env_path("tau2") == tmp_path / "envs" / "tau2"
        assert mgr.env_path("benchmarks/tau2") == tmp_path / "envs" / "benchmarks" / "tau2"

    def test_venv_python(self, tmp_path: Path) -> None:
        mgr = EnvironmentManager(base_dir=tmp_path / "envs")

        expected = str(tmp_path / "envs" / "tau2" / "venv" / "bin" / "python")
        assert mgr.venv_python("tau2") == expected

    def test_default_base_dir(self) -> None:
        mgr = EnvironmentManager()
        assert mgr.base_dir == Path.home() / ".exgentic"


# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------


class TestMarkers:
    """Marker file must be robust against corruption and old formats."""

    def test_corrupted_marker_treated_as_empty(self, tmp_path: Path) -> None:
        mgr = EnvironmentManager(base_dir=tmp_path / "envs")
        env_dir = mgr.env_path("mybench")
        env_dir.mkdir(parents=True)
        (env_dir / ".installed").write_text("not json")

        assert not mgr.is_installed("mybench")
        assert mgr.get_info("mybench") is None

    def test_non_dict_marker_treated_as_empty(self, tmp_path: Path) -> None:
        mgr = EnvironmentManager(base_dir=tmp_path / "envs")
        env_dir = mgr.env_path("mybench")
        env_dir.mkdir(parents=True)
        (env_dir / ".installed").write_text('"just a string"')

        assert not mgr.is_installed("mybench")

    def test_empty_dict_marker_means_not_installed(self, tmp_path: Path) -> None:
        mgr = EnvironmentManager(base_dir=tmp_path / "envs")
        env_dir = mgr.env_path("mybench")
        env_dir.mkdir(parents=True)
        (env_dir / ".installed").write_text("{}")

        assert not mgr.is_installed("mybench")
        assert mgr.get_info("mybench") is None


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------


class TestFailureModes:
    def test_missing_uv(self, tmp_path: Path) -> None:
        mgr = EnvironmentManager(base_dir=tmp_path / "envs")

        with mock.patch("shutil.which", return_value=None):
            with pytest.raises(RuntimeError, match="Could not find 'uv'"):
                mgr.install("mybench")

    def test_broken_setup_sh_no_marker(self, tmp_path: Path) -> None:
        module_path = _create_fake_package(tmp_path, with_requirements=False, with_setup=False)
        parts = module_path.split(".")
        pkg_dir = tmp_path
        for part in parts[:-1]:
            pkg_dir = pkg_dir / part
        broken = pkg_dir / "setup.sh"
        broken.write_text("#!/usr/bin/env bash\nexit 1\n")
        broken.chmod(broken.stat().st_mode | stat.S_IEXEC)
        importlib.invalidate_caches()

        mgr = EnvironmentManager(base_dir=tmp_path / "envs")

        with pytest.raises(subprocess.CalledProcessError):
            mgr.install("mybench", module_path=module_path)

        assert not mgr.is_installed("mybench", env_type=EnvType.VENV)

    def test_missing_system_dep(self, tmp_path: Path) -> None:
        module_path = _create_fake_package(tmp_path, with_requirements=False, with_setup=False, with_system_deps=True)
        parts = module_path.split(".")
        pkg_dir = tmp_path
        for part in parts[:-1]:
            pkg_dir = pkg_dir / part
        (pkg_dir / "system-deps.txt").write_text("nonexistent_tool_xyz\n")
        importlib.invalidate_caches()

        mgr = EnvironmentManager(base_dir=tmp_path / "envs")

        original_which = shutil.which

        def which_no_fake(name):
            if name == "nonexistent_tool_xyz":
                return None
            return original_which(name)

        with mock.patch("exgentic.environment.helpers.shutil.which", side_effect=which_no_fake):
            with mock.patch("exgentic.environment.helpers._dpkg_installed", return_value=False):
                with pytest.raises(RuntimeError, match="nonexistent_tool_xyz"):
                    mgr.install("mybench", module_path=module_path)

    def test_venv_failure_preserves_coexisting_envs(self, tmp_path: Path) -> None:
        """If venv install fails, local install must be unaffected."""
        module_path = _create_fake_package(tmp_path, with_requirements=False, with_setup=False)
        mgr = EnvironmentManager(base_dir=tmp_path / "envs")

        mgr.install("mybench", env_type=EnvType.LOCAL, module_path=module_path)

        def fail_venv(cmd, **kwargs):
            if isinstance(cmd, list) and "venv" in cmd:
                raise subprocess.CalledProcessError(1, cmd)
            return _real_subprocess_run(cmd, **kwargs)

        with mock.patch("subprocess.run", side_effect=fail_venv):
            with pytest.raises(subprocess.CalledProcessError):
                mgr.install("mybench", env_type=EnvType.VENV, module_path=module_path)

        assert mgr.is_installed("mybench", env_type=EnvType.LOCAL)
        assert not mgr.is_installed("mybench", env_type=EnvType.VENV)

    def test_invalid_env_type(self, tmp_path: Path) -> None:
        mgr = EnvironmentManager(base_dir=tmp_path / "envs")
        with pytest.raises(ValueError):
            mgr.install("mybench", env_type="invalid")


# ---------------------------------------------------------------------------
# State transitions
# ---------------------------------------------------------------------------


class TestStateMachine:
    def test_install_uninstall_install_cycle(self, tmp_path: Path) -> None:
        module_path = _create_fake_package(tmp_path, with_requirements=False, with_setup=False)
        mgr = EnvironmentManager(base_dir=tmp_path / "envs")

        env1 = mgr.install("mybench", module_path=module_path)
        assert mgr.is_installed("mybench")

        mgr.uninstall("mybench")
        assert not mgr.is_installed("mybench")
        assert not env1.exists()

        env2 = mgr.install("mybench", module_path=module_path)
        assert mgr.is_installed("mybench")
        assert env2 == env1

    def test_double_uninstall(self, tmp_path: Path) -> None:
        module_path = _create_fake_package(tmp_path, with_requirements=False, with_setup=False)
        mgr = EnvironmentManager(base_dir=tmp_path / "envs")

        mgr.install("mybench", module_path=module_path)
        mgr.uninstall("mybench")
        mgr.uninstall("mybench")

        assert not mgr.is_installed("mybench")

    def test_install_without_module_path(self, tmp_path: Path) -> None:
        mgr = EnvironmentManager(base_dir=tmp_path / "envs")

        env_dir = mgr.install("mybench")

        assert mgr.is_installed("mybench", env_type=EnvType.VENV)
        assert (env_dir / "venv" / "bin" / "python").exists()

    def test_local_install_without_module_path(self, tmp_path: Path) -> None:
        mgr = EnvironmentManager(base_dir=tmp_path / "envs")

        env_dir = mgr.install("mybench", env_type=EnvType.LOCAL)

        assert mgr.is_installed("mybench", env_type=EnvType.LOCAL)
        assert env_dir.is_dir()


# ---------------------------------------------------------------------------
# Convenience accessors
# ---------------------------------------------------------------------------


class TestConvenienceAccessors:
    """Runners need quick access to docker image tags and local Python paths."""

    def test_docker_image_returns_tag(self, tmp_path: Path) -> None:
        mgr = EnvironmentManager(base_dir=tmp_path / "envs")
        env_dir = mgr.env_path("mybench")
        env_dir.mkdir(parents=True)
        (env_dir / ".installed").write_text(
            json.dumps({"docker": {"installed_at": "2026-01-01T00:00:00Z", "image": "mybench:abc123"}})
        )

        assert mgr.docker_image("mybench") == "mybench:abc123"

    def test_docker_image_returns_none_when_not_installed(self, tmp_path: Path) -> None:
        mgr = EnvironmentManager(base_dir=tmp_path / "envs")
        assert mgr.docker_image("mybench") is None

    def test_local_python_returns_path(self, tmp_path: Path) -> None:
        module_path = _create_fake_package(tmp_path, with_requirements=False, with_setup=False)
        mgr = EnvironmentManager(base_dir=tmp_path / "envs")

        mgr.install("mybench", env_type=EnvType.LOCAL, module_path=module_path)

        assert mgr.local_python("mybench") == sys.executable

    def test_local_python_returns_none_when_not_installed(self, tmp_path: Path) -> None:
        mgr = EnvironmentManager(base_dir=tmp_path / "envs")
        assert mgr.local_python("mybench") is None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestFindPackageFile:
    def test_finds_file(self, tmp_path: Path) -> None:
        module_path = _create_fake_package(tmp_path, with_requirements=True)
        result = find_package_file(module_path, "requirements.txt")
        assert result is not None
        assert result.name == "requirements.txt"

    def test_returns_none_for_missing(self, tmp_path: Path) -> None:
        module_path = _create_fake_package(tmp_path, with_requirements=False, with_setup=False)
        assert find_package_file(module_path, "nonexistent.txt") is None


class TestRequireUv:
    def test_returns_path(self) -> None:
        path = require_uv()
        assert "uv" in Path(path).name

    def test_raises_when_missing(self) -> None:
        with mock.patch("shutil.which", return_value=None):
            with pytest.raises(RuntimeError, match="Could not find 'uv'"):
                require_uv()


class TestBuildSubprocessEnv:
    """build_subprocess_env strips vars that could interfere with installs."""

    def test_strips_virtual_env(self) -> None:
        with mock.patch.dict(os.environ, {"VIRTUAL_ENV": "/some/venv"}, clear=False):
            env = build_subprocess_env()
        assert "VIRTUAL_ENV" not in env

    def test_strips_conda_vars(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"CONDA_DEFAULT_ENV": "base", "CONDA_PREFIX": "/opt/conda"},
            clear=False,
        ):
            env = build_subprocess_env()
        assert "CONDA_DEFAULT_ENV" not in env
        assert "CONDA_PREFIX" not in env

    def test_strips_uv_prefix_vars(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"UV_INDEX_URL": "https://evil.example.com", "UV_PYTHON": "3.8"},
            clear=False,
        ):
            env = build_subprocess_env()
        assert "UV_INDEX_URL" not in env
        assert "UV_PYTHON" not in env

    def test_strips_pip_prefix_vars(self) -> None:
        with mock.patch.dict(os.environ, {"PIP_INDEX_URL": "https://evil.example.com"}, clear=False):
            env = build_subprocess_env()
        assert "PIP_INDEX_URL" not in env

    def test_preserves_path_and_home(self) -> None:
        env = build_subprocess_env()
        assert "PATH" in env
        assert "HOME" in env

    def test_sets_git_lfs_skip_smudge(self) -> None:
        env = build_subprocess_env()
        assert env["GIT_LFS_SKIP_SMUDGE"] == "1"


# ---------------------------------------------------------------------------
# Project root & packages
# ---------------------------------------------------------------------------


class TestVenvProjectRoot:
    """Venv backend installs a Python project from project_root."""

    def test_project_root_installs_project(self, tmp_path: Path) -> None:
        project = _create_fake_project(tmp_path)
        mgr = EnvironmentManager(base_dir=tmp_path / "envs")

        pip_calls: list[list[str]] = []

        def capture_run(cmd, **kwargs):
            if isinstance(cmd, list) and "pip" in cmd and "install" in cmd:
                pip_calls.append(list(cmd))
                return _docker_mock_result()
            return _real_subprocess_run(cmd, **kwargs)

        with mock.patch("subprocess.run", side_effect=capture_run):
            mgr.install("mybench", project_root=project)

        # First pip call should install the project root.
        assert len(pip_calls) >= 1
        assert str(project) in pip_calls[0]

    def test_packages_installed_into_venv(self, tmp_path: Path) -> None:
        mgr = EnvironmentManager(base_dir=tmp_path / "envs")

        pip_calls: list[list[str]] = []

        def capture_run(cmd, **kwargs):
            if isinstance(cmd, list) and "pip" in cmd and "install" in cmd:
                pip_calls.append(list(cmd))
                return _docker_mock_result()
            return _real_subprocess_run(cmd, **kwargs)

        with mock.patch("subprocess.run", side_effect=capture_run):
            mgr.install("mybench", packages=["numpy", "pandas"])

        assert len(pip_calls) == 1
        assert "numpy" in pip_calls[0]
        assert "pandas" in pip_calls[0]

    def test_project_root_and_packages_combined(self, tmp_path: Path) -> None:
        project = _create_fake_project(tmp_path)
        mgr = EnvironmentManager(base_dir=tmp_path / "envs")

        pip_calls: list[list[str]] = []

        def capture_run(cmd, **kwargs):
            if isinstance(cmd, list) and "pip" in cmd and "install" in cmd:
                pip_calls.append(list(cmd))
                return _docker_mock_result()
            return _real_subprocess_run(cmd, **kwargs)

        with mock.patch("subprocess.run", side_effect=capture_run):
            mgr.install("mybench", project_root=project, packages=["numpy"])

        # Two separate pip calls: project root, then packages.
        assert len(pip_calls) == 2
        assert str(project) in pip_calls[0]
        assert "numpy" in pip_calls[1]

    def test_without_project_root_still_works(self, tmp_path: Path) -> None:
        module_path = _create_fake_package(tmp_path, with_requirements=False, with_setup=False)
        mgr = EnvironmentManager(base_dir=tmp_path / "envs")

        env_dir = mgr.install("mybench", module_path=module_path)

        assert (env_dir / "venv" / "bin" / "python").exists()
        assert mgr.is_installed("mybench", env_type=EnvType.VENV)


class TestLocalProjectRoot:
    """Local backend installs project_root and packages into host Python."""

    def test_project_root_installs_into_host_python(self, tmp_path: Path) -> None:
        project = _create_fake_project(tmp_path)
        mgr = EnvironmentManager(base_dir=tmp_path / "envs")

        pip_calls: list[list[str]] = []

        def capture_run(cmd, **kwargs):
            if isinstance(cmd, list) and "pip" in cmd and "install" in cmd:
                pip_calls.append(list(cmd))
                return _docker_mock_result()
            return _real_subprocess_run(cmd, **kwargs)

        with mock.patch("subprocess.run", side_effect=capture_run):
            mgr.install("mybench", env_type=EnvType.LOCAL, project_root=project)

        assert len(pip_calls) == 1
        assert str(project) in pip_calls[0]
        python_idx = pip_calls[0].index("--python") + 1
        assert pip_calls[0][python_idx] == sys.executable

    def test_packages_installs_into_host_python(self, tmp_path: Path) -> None:
        mgr = EnvironmentManager(base_dir=tmp_path / "envs")

        pip_calls: list[list[str]] = []

        def capture_run(cmd, **kwargs):
            if isinstance(cmd, list) and "pip" in cmd and "install" in cmd:
                pip_calls.append(list(cmd))
                return _docker_mock_result()
            return _real_subprocess_run(cmd, **kwargs)

        with mock.patch("subprocess.run", side_effect=capture_run):
            mgr.install("mybench", env_type=EnvType.LOCAL, packages=["numpy"])

        assert len(pip_calls) == 1
        assert "numpy" in pip_calls[0]


class TestDockerProjectRoot:
    """Docker backend uses two-layer build when project_root is provided."""

    def test_project_root_triggers_two_layer_build(self, tmp_path: Path) -> None:
        project = _create_fake_project(tmp_path)
        mgr = EnvironmentManager(base_dir=tmp_path / "envs")

        dockerfiles: list[str] = []

        def capture_run(cmd, **kwargs):
            if isinstance(cmd, list) and cmd[0] == "docker":
                if cmd[1] == "build":
                    # Read dockerfile from -f argument.
                    if "-f" in cmd:
                        df_idx = list(cmd).index("-f") + 1
                        df_path = Path(cmd[df_idx])
                        if df_path.exists():
                            dockerfiles.append(df_path.read_text())
                if cmd[1:3] == ["image", "inspect"]:
                    return _docker_mock_result(returncode=1)
                return _docker_mock_result()
            return _real_subprocess_run(cmd, **kwargs)

        with mock.patch("subprocess.run", side_effect=capture_run):
            mgr.install("mybench", env_type=EnvType.DOCKER, project_root=project)

        assert len(dockerfiles) == 1
        df = dockerfiles[0]
        assert "COPY pyproject.toml" in df
        assert "COPY src/ src/" in df
        assert "uv pip install --no-cache ." in df
        assert "uv pip install --no-cache --no-deps ." in df

    def test_docker_build_context_is_project_root(self, tmp_path: Path) -> None:
        project = _create_fake_project(tmp_path)
        mgr = EnvironmentManager(base_dir=tmp_path / "envs")

        build_contexts: list[str] = []

        def capture_run(cmd, **kwargs):
            if isinstance(cmd, list) and cmd[0] == "docker":
                if cmd[1] == "build":
                    build_contexts.append(cmd[-1])
                if cmd[1:3] == ["image", "inspect"]:
                    return _docker_mock_result(returncode=1)
                return _docker_mock_result()
            return _real_subprocess_run(cmd, **kwargs)

        with mock.patch("subprocess.run", side_effect=capture_run):
            mgr.install("mybench", env_type=EnvType.DOCKER, project_root=project)

        # Two builds: base image uses project_root as context, bench uses a temp dir.
        assert len(build_contexts) == 2
        assert build_contexts[0] == str(project)
        assert "exgentic-bench-" in build_contexts[1]

    def test_docker_without_project_root_uses_tmp_dir(self, tmp_path: Path) -> None:
        module_path = _create_fake_package(tmp_path, with_requirements=True, with_setup=False)
        mgr = EnvironmentManager(base_dir=tmp_path / "envs")

        build_contexts: list[str] = []

        def capture_run(cmd, **kwargs):
            if isinstance(cmd, list) and cmd[0] == "docker":
                if cmd[1] == "build":
                    build_contexts.append(cmd[-1])
                if cmd[1:3] == ["image", "inspect"]:
                    return _docker_mock_result(returncode=1)
                return _docker_mock_result()
            return _real_subprocess_run(cmd, **kwargs)

        with mock.patch("subprocess.run", side_effect=capture_run):
            mgr.install("mybench", env_type=EnvType.DOCKER, module_path=module_path)

        assert len(build_contexts) == 1
        assert "exgentic-docker-" in build_contexts[0]

    def test_docker_packages_in_dockerfile(self, tmp_path: Path) -> None:
        mgr = EnvironmentManager(base_dir=tmp_path / "envs")

        dockerfiles: list[str] = []

        def capture_run(cmd, **kwargs):
            if isinstance(cmd, list) and cmd[0] == "docker":
                if cmd[1] == "build":
                    df = Path(cmd[-1]) / "Dockerfile"
                    if df.exists():
                        dockerfiles.append(df.read_text())
                if cmd[1:3] == ["image", "inspect"]:
                    return _docker_mock_result(returncode=1)
                return _docker_mock_result()
            return _real_subprocess_run(cmd, **kwargs)

        with mock.patch("subprocess.run", side_effect=capture_run):
            mgr.install("mybench", env_type=EnvType.DOCKER, packages=["numpy", "pandas"])

        assert len(dockerfiles) == 1
        assert "uv pip install --no-cache numpy pandas" in dockerfiles[0]

    def test_content_hash_includes_project_root(self, tmp_path: Path) -> None:
        project = _create_fake_project(tmp_path, name="proj-a")
        mgr = EnvironmentManager(base_dir=tmp_path / "envs")

        tags: list[str] = []

        def capture_run(cmd, **kwargs):
            if isinstance(cmd, list) and cmd[0] == "docker":
                if cmd[1] == "build":
                    idx = list(cmd).index("-t")
                    tags.append(cmd[idx + 1])
                if cmd[1:3] == ["image", "inspect"]:
                    return _docker_mock_result(returncode=1)
                return _docker_mock_result()
            return _real_subprocess_run(cmd, **kwargs)

        with mock.patch("subprocess.run", side_effect=capture_run):
            mgr.install("a", env_type=EnvType.DOCKER, project_root=project)
            mgr.install("b", env_type=EnvType.DOCKER)

        # project_root install: 2 builds (base + bench); no-project_root install: 1 build.
        assert len(tags) == 3
        bench_tag_a = tags[1]  # bench tag for "a"
        single_tag_b = tags[2]  # single-image tag for "b"
        assert bench_tag_a.split(":")[-1] != single_tag_b.split(":")[-1]

    def test_project_root_with_force_includes(self, tmp_path: Path) -> None:
        project = _create_fake_project(tmp_path, name="withfi")
        # Add force-include to pyproject.toml.
        pyproject = project / "pyproject.toml"
        pyproject.write_text(
            pyproject.read_text()
            + textwrap.dedent(
                """\

            [tool.hatch.build.targets.wheel.force-include]
            "configs/" = "withfi/configs/"
            """
            )
        )
        mgr = EnvironmentManager(base_dir=tmp_path / "envs")

        dockerfiles: list[str] = []

        def capture_run(cmd, **kwargs):
            if isinstance(cmd, list) and cmd[0] == "docker":
                if cmd[1] == "build":
                    if "-f" in cmd:
                        df_idx = list(cmd).index("-f") + 1
                        df_path = Path(cmd[df_idx])
                        if df_path.exists():
                            dockerfiles.append(df_path.read_text())
                if cmd[1:3] == ["image", "inspect"]:
                    return _docker_mock_result(returncode=1)
                return _docker_mock_result()
            return _real_subprocess_run(cmd, **kwargs)

        with mock.patch("subprocess.run", side_effect=capture_run):
            mgr.install("mybench", env_type=EnvType.DOCKER, project_root=project)

        assert len(dockerfiles) == 1
        assert "mkdir -p 'configs/'" in dockerfiles[0]

    def test_two_builds_when_project_root_provided(self, tmp_path: Path) -> None:
        """Two docker build calls: one base image, one bench image."""
        project = _create_fake_project(tmp_path)
        mgr = EnvironmentManager(base_dir=tmp_path / "envs")

        base_builds: list[list] = []
        bench_builds: list[list] = []

        def capture_run(cmd, **kwargs):
            if isinstance(cmd, list) and cmd[0] == "docker":
                if cmd[1] == "build":
                    if "-f" in cmd:
                        base_builds.append(list(cmd))
                    else:
                        bench_builds.append(list(cmd))
                if cmd[1:3] == ["image", "inspect"]:
                    return _docker_mock_result(returncode=1)
                return _docker_mock_result()
            return _real_subprocess_run(cmd, **kwargs)

        with mock.patch("subprocess.run", side_effect=capture_run):
            mgr.install("mybench", env_type=EnvType.DOCKER, project_root=project)

        assert len(base_builds) == 1
        assert len(bench_builds) == 1

    def test_base_image_tag_has_prefix(self, tmp_path: Path) -> None:
        """Base image tag must start with 'exgentic-base:'."""
        project = _create_fake_project(tmp_path)
        mgr = EnvironmentManager(base_dir=tmp_path / "envs")

        base_tags: list[str] = []

        def capture_run(cmd, **kwargs):
            if isinstance(cmd, list) and cmd[0] == "docker":
                if cmd[1] == "build" and "-f" in cmd:
                    idx = list(cmd).index("-t")
                    base_tags.append(cmd[idx + 1])
                if cmd[1:3] == ["image", "inspect"]:
                    return _docker_mock_result(returncode=1)
                return _docker_mock_result()
            return _real_subprocess_run(cmd, **kwargs)

        with mock.patch("subprocess.run", side_effect=capture_run):
            mgr.install("mybench", env_type=EnvType.DOCKER, project_root=project)

        assert len(base_tags) == 1
        assert base_tags[0].startswith("exgentic-base:")

    def test_bench_image_from_base(self, tmp_path: Path) -> None:
        """Bench Dockerfile must start with FROM exgentic-base:..."""
        project = _create_fake_project(tmp_path)
        mgr = EnvironmentManager(base_dir=tmp_path / "envs")

        bench_dockerfiles: list[str] = []

        def capture_run(cmd, **kwargs):
            if isinstance(cmd, list) and cmd[0] == "docker":
                if cmd[1] == "build" and "-f" not in cmd:
                    df = Path(cmd[-1]) / "Dockerfile"
                    if df.exists():
                        bench_dockerfiles.append(df.read_text())
                if cmd[1:3] == ["image", "inspect"]:
                    return _docker_mock_result(returncode=1)
                return _docker_mock_result()
            return _real_subprocess_run(cmd, **kwargs)

        with mock.patch("subprocess.run", side_effect=capture_run):
            mgr.install("mybench", env_type=EnvType.DOCKER, project_root=project)

        assert len(bench_dockerfiles) == 1
        assert bench_dockerfiles[0].startswith("FROM exgentic-base:")

    def test_base_image_tag_stored_in_marker(self, tmp_path: Path) -> None:
        """Marker must record base_image so uninstall can clean it up."""
        project = _create_fake_project(tmp_path)
        mgr = EnvironmentManager(base_dir=tmp_path / "envs")

        def capture_run(cmd, **kwargs):
            if isinstance(cmd, list) and cmd[0] == "docker":
                if cmd[1:3] == ["image", "inspect"]:
                    return _docker_mock_result(returncode=1)
                return _docker_mock_result()
            return _real_subprocess_run(cmd, **kwargs)

        with mock.patch("subprocess.run", side_effect=capture_run):
            mgr.install("mybench", env_type=EnvType.DOCKER, project_root=project)

        marker = json.loads((mgr.env_path("mybench") / ".installed").read_text())
        assert "base_image" in marker["docker"]
        assert marker["docker"]["base_image"].startswith("exgentic-base:")

    def test_uninstall_attempts_base_image_removal(self, tmp_path: Path) -> None:
        """uninstall() attempts to remove the base image (rmi silently fails if still in use)."""
        mgr = EnvironmentManager(base_dir=tmp_path / "envs")
        env_dir = mgr.env_path("mybench")
        env_dir.mkdir(parents=True)
        (env_dir / ".installed").write_text(
            json.dumps(
                {
                    "docker": {
                        "installed_at": "2026-01-01T00:00:00Z",
                        "image": "mybench:abc123",
                        "base_image": "exgentic-base:def456",
                    }
                }
            )
        )

        rmi_calls: list[str] = []

        def side_effect(cmd, **kwargs):
            if cmd[0] == "docker" and cmd[1] == "rmi":
                rmi_calls.append(cmd[2])
                return _docker_mock_result()
            return _real_subprocess_run(cmd, **kwargs)

        with mock.patch("subprocess.run", side_effect=side_effect):
            mgr.uninstall("mybench", env_type=EnvType.DOCKER)

        assert "mybench:abc123" in rmi_calls
        assert "exgentic-base:def456" in rmi_calls

    def test_base_image_reused_on_second_install(self, tmp_path: Path) -> None:
        """Base image is only built once; second bench reuses it."""
        project = _create_fake_project(tmp_path)
        mgr = EnvironmentManager(base_dir=tmp_path / "envs")

        built_images: set[str] = set()
        base_builds = 0
        bench_builds = 0

        def capture_run(cmd, **kwargs):
            nonlocal base_builds, bench_builds
            if isinstance(cmd, list) and cmd[0] == "docker":
                if cmd[1] == "build":
                    idx = list(cmd).index("-t")
                    built_images.add(cmd[idx + 1])
                    if "-f" in cmd:
                        base_builds += 1
                    else:
                        bench_builds += 1
                elif cmd[1:3] == ["image", "inspect"]:
                    tag = cmd[-1]
                    return _docker_mock_result(returncode=0 if tag in built_images else 1)
                return _docker_mock_result()
            return _real_subprocess_run(cmd, **kwargs)

        with mock.patch("subprocess.run", side_effect=capture_run):
            mgr.install("bench-a", env_type=EnvType.DOCKER, project_root=project)
            mgr.install("bench-b", env_type=EnvType.DOCKER, project_root=project)

        assert base_builds == 1
        assert bench_builds == 2

    def test_image_version_changes_base_tag(self, tmp_path: Path) -> None:
        """Bumping _IMAGE_VERSION produces a different base tag."""
        from exgentic.environment.docker import DockerBackend

        project = _create_fake_project(tmp_path)

        tag_v1 = DockerBackend._base_image_tag(project)
        with mock.patch.object(DockerBackend, "_IMAGE_VERSION", "v99"):
            tag_v99 = DockerBackend._base_image_tag(project)

        assert tag_v1.startswith("exgentic-base:")
        assert tag_v99.startswith("exgentic-base:")
        assert tag_v1 != tag_v99


# ---------------------------------------------------------------------------
# Docker socket
# ---------------------------------------------------------------------------


class TestDockerSocket:
    """docker_socket=True installs Docker CLI binary in the bench/single image."""

    def test_docker_socket_adds_cli_to_bench_image(self, tmp_path: Path) -> None:
        """docker_socket=True adds Docker CLI RUN to the bench Dockerfile."""
        project = _create_fake_project(tmp_path)
        mgr = EnvironmentManager(base_dir=tmp_path / "envs")

        bench_dockerfiles: list[str] = []

        def capture_run(cmd, **kwargs):
            if isinstance(cmd, list) and cmd[0] == "docker":
                if cmd[1] == "build" and "-f" not in cmd:
                    df = Path(cmd[-1]) / "Dockerfile"
                    if df.exists():
                        bench_dockerfiles.append(df.read_text())
                if cmd[1:3] == ["image", "inspect"]:
                    return _docker_mock_result(returncode=1)
                return _docker_mock_result()
            return _real_subprocess_run(cmd, **kwargs)

        with mock.patch("subprocess.run", side_effect=capture_run):
            mgr.install("mybench", env_type=EnvType.DOCKER, project_root=project, docker_socket=True)

        assert len(bench_dockerfiles) == 1
        assert "docker.com/linux/static" in bench_dockerfiles[0]

    def test_docker_socket_not_in_base_image(self, tmp_path: Path) -> None:
        """docker_socket does NOT affect the base image — base must stay lean."""
        project = _create_fake_project(tmp_path)
        mgr = EnvironmentManager(base_dir=tmp_path / "envs")

        base_dockerfiles: list[str] = []

        def capture_run(cmd, **kwargs):
            if isinstance(cmd, list) and cmd[0] == "docker":
                if cmd[1] == "build" and "-f" in cmd:
                    df_idx = list(cmd).index("-f") + 1
                    df_path = Path(cmd[df_idx])
                    if df_path.exists():
                        base_dockerfiles.append(df_path.read_text())
                if cmd[1:3] == ["image", "inspect"]:
                    return _docker_mock_result(returncode=1)
                return _docker_mock_result()
            return _real_subprocess_run(cmd, **kwargs)

        with mock.patch("subprocess.run", side_effect=capture_run):
            mgr.install("mybench", env_type=EnvType.DOCKER, project_root=project, docker_socket=True)

        assert len(base_dockerfiles) == 1
        assert "docker.com/linux/static" not in base_dockerfiles[0]

    def test_docker_socket_in_single_image_path(self, tmp_path: Path) -> None:
        """docker_socket=True also adds Docker CLI in single-image (no project_root) path."""
        mgr = EnvironmentManager(base_dir=tmp_path / "envs")

        dockerfiles: list[str] = []

        def capture_run(cmd, **kwargs):
            if isinstance(cmd, list) and cmd[0] == "docker":
                if cmd[1] == "build":
                    df = Path(cmd[-1]) / "Dockerfile"
                    if df.exists():
                        dockerfiles.append(df.read_text())
                if cmd[1:3] == ["image", "inspect"]:
                    return _docker_mock_result(returncode=1)
                return _docker_mock_result()
            return _real_subprocess_run(cmd, **kwargs)

        with mock.patch("subprocess.run", side_effect=capture_run):
            mgr.install("mybench", env_type=EnvType.DOCKER, docker_socket=True)

        assert len(dockerfiles) == 1
        assert "docker.com/linux/static" in dockerfiles[0]

    def test_docker_socket_changes_image_hash(self, tmp_path: Path) -> None:
        """Same config with and without docker_socket produces different image tags."""
        mgr = EnvironmentManager(base_dir=tmp_path / "envs")

        tags: list[str] = []

        def capture_run(cmd, **kwargs):
            if isinstance(cmd, list) and cmd[0] == "docker":
                if cmd[1] == "build":
                    idx = list(cmd).index("-t")
                    tags.append(cmd[idx + 1])
                if cmd[1:3] == ["image", "inspect"]:
                    return _docker_mock_result(returncode=1)
                return _docker_mock_result()
            return _real_subprocess_run(cmd, **kwargs)

        with mock.patch("subprocess.run", side_effect=capture_run):
            mgr.install("bench-no-socket", env_type=EnvType.DOCKER)
            mgr.install("bench-with-socket", env_type=EnvType.DOCKER, docker_socket=True)

        assert len(tags) == 2
        assert tags[0].split(":")[-1] != tags[1].split(":")[-1]


# ---------------------------------------------------------------------------
# Docker build environment variable
# ---------------------------------------------------------------------------


class TestDockerBuildEnv:
    """EXGENTIC_DOCKER_BUILD=1 is set when running setup.sh during image builds."""

    def test_exgentic_docker_build_in_bench_setup_sh(self, tmp_path: Path) -> None:
        """EXGENTIC_DOCKER_BUILD=1 is prepended to the setup.sh RUN in bench image."""
        module_path = _create_fake_package(tmp_path, with_requirements=False, with_setup=True)
        project = _create_fake_project(tmp_path)
        mgr = EnvironmentManager(base_dir=tmp_path / "envs")

        bench_dockerfiles: list[str] = []

        def capture_run(cmd, **kwargs):
            if isinstance(cmd, list) and cmd[0] == "docker":
                if cmd[1] == "build" and "-f" not in cmd:
                    df = Path(cmd[-1]) / "Dockerfile"
                    if df.exists():
                        bench_dockerfiles.append(df.read_text())
                if cmd[1:3] == ["image", "inspect"]:
                    return _docker_mock_result(returncode=1)
                return _docker_mock_result()
            return _real_subprocess_run(cmd, **kwargs)

        with mock.patch("subprocess.run", side_effect=capture_run):
            mgr.install(
                "mybench",
                env_type=EnvType.DOCKER,
                project_root=project,
                module_path=module_path,
            )

        assert len(bench_dockerfiles) == 1
        assert "EXGENTIC_DOCKER_BUILD=1 bash /tmp/setup.sh" in bench_dockerfiles[0]

    def test_exgentic_docker_build_in_single_image_setup_sh(self, tmp_path: Path) -> None:
        """EXGENTIC_DOCKER_BUILD=1 is also present in the single-image (no project_root) path."""
        module_path = _create_fake_package(tmp_path, with_requirements=False, with_setup=True)
        mgr = EnvironmentManager(base_dir=tmp_path / "envs")

        dockerfiles: list[str] = []

        def capture_run(cmd, **kwargs):
            if isinstance(cmd, list) and cmd[0] == "docker":
                if cmd[1] == "build":
                    df = Path(cmd[-1]) / "Dockerfile"
                    if df.exists():
                        dockerfiles.append(df.read_text())
                if cmd[1:3] == ["image", "inspect"]:
                    return _docker_mock_result(returncode=1)
                return _docker_mock_result()
            return _real_subprocess_run(cmd, **kwargs)

        with mock.patch("subprocess.run", side_effect=capture_run):
            mgr.install("mybench", env_type=EnvType.DOCKER, module_path=module_path)

        assert len(dockerfiles) == 1
        assert "EXGENTIC_DOCKER_BUILD=1 bash /tmp/setup.sh" in dockerfiles[0]


# ---------------------------------------------------------------------------
# DockerBackend Dockerfile generation
# ---------------------------------------------------------------------------


class TestDockerBackendDockerfile:
    """Verify the Dockerfile content DockerBackend generates.

    These tests mock subprocess.run to capture the Dockerfile instead of
    actually building.  They verify critical assumptions:
    - exgentic is installed from source when a source dir is provided
    - requirements.txt and setup.sh are included
    - docker socket installs Docker CLI
    - extra dependencies are installed
    """

    def _capture_dockerfiles(self, tmp_path, **install_kwargs):
        """Run DockerBackend.install() with mocked docker, return list of Dockerfile contents."""
        from exgentic.environment.docker import DockerBackend

        dockerfiles: list[str] = []

        def side_effect(cmd, **kwargs):
            cmd = list(cmd)
            if cmd[0] == "docker" and cmd[1:3] == ["image", "inspect"]:
                # Image doesn't exist yet.
                return _docker_mock_result(returncode=1)
            if cmd[0] == "docker" and cmd[1] == "build":
                # Find the Dockerfile.
                if "-f" in cmd:
                    df_path = cmd[cmd.index("-f") + 1]
                else:
                    df_path = str(Path(cmd[-1]) / "Dockerfile")
                dockerfiles.append(Path(df_path).read_text())
                return _docker_mock_result()
            return _real_subprocess_run(cmd, **kwargs)

        backend = DockerBackend()
        env_dir = tmp_path / "env"
        env_dir.mkdir(parents=True, exist_ok=True)

        with mock.patch("subprocess.run", side_effect=side_effect):
            backend.install(env_dir, **install_kwargs)

        return dockerfiles

    def _capture_dockerfile(self, tmp_path, **install_kwargs):
        """Run DockerBackend.install() with mocked docker, return single Dockerfile content."""
        dockerfiles = self._capture_dockerfiles(tmp_path, **install_kwargs)
        assert len(dockerfiles) == 1, f"Expected 1 docker build, got {len(dockerfiles)}"
        return dockerfiles[0]

    def test_source_install_copies_source(self, tmp_path: Path) -> None:
        """When project_root is given, build a base + bench image pair."""
        # Create a fake project root.
        proj = tmp_path / "project"
        proj.mkdir()
        (proj / "pyproject.toml").write_text('[project]\nname = "fakepkg"\nversion = "0.1"\n')
        (proj / "README.md").write_text("# Fake\n")
        src = proj / "src" / "exgentic"
        src.mkdir(parents=True)
        (src / "__init__.py").write_text("")

        dockerfiles = self._capture_dockerfiles(
            tmp_path,
            name="benchmarks/test",
            module_path=None,
            project_root=proj,
        )

        assert len(dockerfiles) == 2, f"Expected 2 docker builds (base + bench), got {len(dockerfiles)}"
        base_df, bench_df = dockerfiles

        # Base image installs exgentic from source.
        assert "COPY pyproject.toml" in base_df
        assert "COPY src/ src/" in base_df
        assert "uv pip install --no-cache ." in base_df
        assert "uv pip install --no-cache --no-deps ." in base_df

        # Bench image is layered on top.
        assert "FROM exgentic-base:" in bench_df

    def test_pypi_install(self, tmp_path: Path) -> None:
        """When packages are given, RUN uv pip install."""
        df = self._capture_dockerfile(
            tmp_path,
            name="benchmarks/test",
            module_path=None,
            packages=["exgentic==1.2.3"],
        )

        assert "uv pip install --no-cache exgentic==1.2.3" in df
        assert "COPY src/" not in df

    def test_requirements_included(self, tmp_path: Path) -> None:
        """requirements.txt from module_path is installed."""
        module_path = _create_fake_package(tmp_path, with_requirements=True, with_setup=False)

        df = self._capture_dockerfile(
            tmp_path,
            name="benchmarks/test",
            module_path=module_path,
            packages=["exgentic==1.0"],
        )

        assert "requirements.txt" in df
        assert "uv pip install --no-cache -r" in df

    def test_setup_sh_included(self, tmp_path: Path) -> None:
        """setup.sh from module_path is run during build."""
        module_path = _create_fake_package(tmp_path, with_requirements=False, with_setup=True)

        df = self._capture_dockerfile(
            tmp_path,
            name="benchmarks/test",
            module_path=module_path,
            packages=["exgentic==1.0"],
        )

        assert "setup.sh" in df
        assert "EXGENTIC_DOCKER_BUILD=1 bash /tmp/setup.sh" in df

    def test_docker_socket_installs_cli(self, tmp_path: Path) -> None:
        """docker_socket=True installs Docker CLI in the image."""
        df = self._capture_dockerfile(
            tmp_path,
            name="benchmarks/test",
            module_path=None,
            docker_socket=True,
        )

        assert "docker-27.5.1.tgz" in df
        assert "/usr/local/bin docker/docker" in df

    def test_extra_dependencies(self, tmp_path: Path) -> None:
        """Extra packages are pip-installed."""
        df = self._capture_dockerfile(
            tmp_path,
            name="benchmarks/test",
            module_path=None,
            packages=["numpy", "pandas"],
        )

        assert "uv pip install --no-cache numpy pandas" in df

    def test_no_exgentic_without_packages(self, tmp_path: Path) -> None:
        """Without packages or project_root, no exgentic install lines."""
        df = self._capture_dockerfile(
            tmp_path,
            name="benchmarks/test",
            module_path=None,
        )

        assert "COPY src/" not in df
        assert "exgentic" not in df.lower() or "exgentic-docker" in df.lower()

    def test_image_tag_deterministic(self, tmp_path: Path) -> None:
        """Same inputs produce the same image tag."""
        from exgentic.environment.docker import DockerBackend

        tag1 = DockerBackend._image_tag("benchmarks/test", None, docker_socket=True)
        tag2 = DockerBackend._image_tag("benchmarks/test", None, docker_socket=True)
        assert tag1 == tag2

    def test_image_tag_changes_with_packages(self, tmp_path: Path) -> None:
        """Different packages produce different tags."""
        from exgentic.environment.docker import DockerBackend

        tag1 = DockerBackend._image_tag("benchmarks/test", None, packages=["exgentic==1.0"])
        tag2 = DockerBackend._image_tag("benchmarks/test", None, packages=["exgentic==2.0"])
        assert tag1 != tag2


# ---------------------------------------------------------------------------
# Docker integration (requires Docker/Podman running)
# ---------------------------------------------------------------------------

_docker_available = shutil.which("docker") is not None
if _docker_available:
    try:
        subprocess.run(["docker", "info"], check=True, capture_output=True, timeout=5)
    except Exception:
        _docker_available = False


@pytest.mark.skipif(not _docker_available, reason="Docker not available")
class TestDockerIntegration:
    """Real Docker tests -- actually build and remove images."""

    def test_docker_build_and_marker(self, tmp_path: Path) -> None:
        module_path = _create_fake_package(tmp_path, with_requirements=False, with_setup=False)
        mgr = EnvironmentManager(base_dir=tmp_path / "envs")

        mgr.install("inttest", env_type=EnvType.DOCKER, module_path=module_path)

        assert mgr.is_installed("inttest", env_type=EnvType.DOCKER)
        image_tag = mgr.docker_image("inttest")
        assert image_tag is not None

        result = subprocess.run(["docker", "image", "inspect", image_tag], check=False, capture_output=True, text=True)
        assert result.returncode == 0

        mgr.uninstall("inttest", env_type=EnvType.DOCKER)
        result = subprocess.run(["docker", "image", "inspect", image_tag], check=False, capture_output=True, text=True)
        assert result.returncode != 0

    def test_docker_reuses_existing_image(self, tmp_path: Path) -> None:
        module_path = _create_fake_package(tmp_path, with_requirements=False, with_setup=False)
        mgr = EnvironmentManager(base_dir=tmp_path / "envs")

        mgr.install("inttest2", env_type=EnvType.DOCKER, module_path=module_path)
        tag1 = mgr.docker_image("inttest2")

        mgr.install("inttest2", env_type=EnvType.DOCKER, force=True, module_path=module_path)
        tag2 = mgr.docker_image("inttest2")

        assert tag1 == tag2
        mgr.uninstall("inttest2")

    def test_docker_with_requirements(self, tmp_path: Path) -> None:
        module_path = _create_fake_package(tmp_path, with_requirements=True, with_setup=False)
        mgr = EnvironmentManager(base_dir=tmp_path / "envs")

        mgr.install("inttest3", env_type=EnvType.DOCKER, module_path=module_path)

        image_tag = mgr.docker_image("inttest3")
        assert image_tag is not None

        result = subprocess.run(
            ["docker", "run", "--rm", image_tag, "python", "-c", "import requests; print(requests.__version__)"],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert result.stdout.strip()

        mgr.uninstall("inttest3")

    def test_docker_coexists_with_venv(self, tmp_path: Path) -> None:
        module_path = _create_fake_package(tmp_path, with_requirements=False, with_setup=False)
        mgr = EnvironmentManager(base_dir=tmp_path / "envs")

        mgr.install("inttest4", env_type=EnvType.VENV, module_path=module_path)
        mgr.install("inttest4", env_type=EnvType.DOCKER, module_path=module_path)

        assert mgr.is_installed("inttest4", env_type=EnvType.VENV)
        assert mgr.is_installed("inttest4", env_type=EnvType.DOCKER)

        mgr.uninstall("inttest4", env_type=EnvType.DOCKER)
        assert mgr.is_installed("inttest4", env_type=EnvType.VENV)
        assert not mgr.is_installed("inttest4", env_type=EnvType.DOCKER)

        mgr.uninstall("inttest4")

    def test_docker_project_root_installs_package(self, tmp_path: Path) -> None:
        """Build Docker image with project_root and verify the package is importable inside."""
        project = _create_fake_project(tmp_path, name="mypkg")
        mgr = EnvironmentManager(base_dir=tmp_path / "envs")

        mgr.install("inttest5", env_type=EnvType.DOCKER, project_root=project)

        image_tag = mgr.docker_image("inttest5")
        assert image_tag is not None

        result = subprocess.run(
            ["docker", "run", "--rm", image_tag, "python", "-c", "import mypkg; print(mypkg.__version__)"],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "0.1.0"

        mgr.uninstall("inttest5")

    def test_venv_project_root_installs_package(self, tmp_path: Path) -> None:
        """Create venv with project_root and verify the package is importable inside."""
        project = _create_fake_project(tmp_path, name="mypkg2")
        mgr = EnvironmentManager(base_dir=tmp_path / "envs")

        mgr.install("inttest6", env_type=EnvType.VENV, project_root=project)

        venv_py = mgr.venv_python("inttest6")
        result = subprocess.run(
            [venv_py, "-c", "import mypkg2; print(mypkg2.__version__)"],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "0.1.0"

        mgr.uninstall("inttest6")

    def test_docker_socket_creates_working_docker_cli(self, tmp_path: Path) -> None:
        """docker_socket=True installs a functional Docker CLI binary inside the image."""
        mgr = EnvironmentManager(base_dir=tmp_path / "envs")

        mgr.install("inttest-ds", env_type=EnvType.DOCKER, docker_socket=True)

        image_tag = mgr.docker_image("inttest-ds")
        assert image_tag is not None

        result = subprocess.run(
            ["docker", "run", "--rm", image_tag, "docker", "--version"],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert "Docker version" in result.stdout

        mgr.uninstall("inttest-ds")

    def test_base_image_shared_between_benchmarks(self, tmp_path: Path) -> None:
        """Two benchmarks with the same project_root share the base image (1 base build, 2 bench builds)."""
        project = _create_fake_project(tmp_path, name="shared")
        mgr = EnvironmentManager(base_dir=tmp_path / "envs")

        # Use a unique name prefix to avoid collisions with other test runs.
        mgr.install("inttest-shared-a", env_type=EnvType.DOCKER, project_root=project)
        mgr.install("inttest-shared-b", env_type=EnvType.DOCKER, project_root=project)

        tag_a = mgr.docker_image("inttest-shared-a")
        tag_b = mgr.docker_image("inttest-shared-b")
        assert tag_a is not None
        assert tag_b is not None
        # Same project_root → same base tag prefix, different bench tags (different names).
        assert tag_a != tag_b
        assert tag_a.startswith("inttest-shared-a:")
        assert tag_b.startswith("inttest-shared-b:")

        # Both images must be runnable.
        for tag in (tag_a, tag_b):
            result = subprocess.run(
                ["docker", "run", "--rm", tag, "python", "-c", "import shared; print(shared.__version__)"],
                check=False,
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0, result.stderr
            assert result.stdout.strip() == "0.1.0"

        mgr.uninstall("inttest-shared-a")
        mgr.uninstall("inttest-shared-b")
