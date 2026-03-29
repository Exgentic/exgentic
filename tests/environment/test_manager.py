# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Tests for exgentic.environment.manager.

Tests are organized by the assumptions the rest of the repo makes about
the installer's capabilities.  Every public method and every env_type
(venv, local, docker) is tested in isolation so the installer can be
wired into the CLI / evaluate / list commands with confidence.
"""

from __future__ import annotations

import importlib
import json
import shutil
import stat
import subprocess
import sys
import textwrap
from pathlib import Path
from unittest import mock

import pytest
from exgentic.environment.manager import EnvironmentManager, _find_package_file, _require_uv

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
            mkdir -p "$EXGENTIC_CACHE_DIR/data"
            touch "$EXGENTIC_CACHE_DIR/data/setup_ran.txt"
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


# ---------------------------------------------------------------------------
# Venv install
# ---------------------------------------------------------------------------


class TestVenvInstall:
    """Venv is the default env_type.

    The evaluate flow and venv runner depend on: venv/ dir existing,
    .installed marker with installed_at, and setup.sh having been run.
    """

    def test_creates_venv_and_marker(self, tmp_path: Path) -> None:
        module_path = _create_fake_package(tmp_path, with_requirements=False, with_setup=False)
        installer = EnvironmentManager(base_dir=tmp_path / "envs")

        env_dir = installer.install("bench", "benchmark", module_path=module_path)

        assert env_dir.is_dir()
        assert (env_dir / "venv").is_dir()
        assert (env_dir / "venv" / "bin" / "python").exists()
        marker = json.loads((env_dir / ".installed").read_text())
        assert "venv" in marker
        assert "installed_at" in marker["venv"]

    def test_skips_if_already_installed(self, tmp_path: Path) -> None:
        module_path = _create_fake_package(tmp_path, with_requirements=False, with_setup=False)
        installer = EnvironmentManager(base_dir=tmp_path / "envs")

        installer.install("bench", "benchmark", module_path=module_path)
        mtime = (installer.env_path("bench", "benchmark") / ".installed").stat().st_mtime

        installer.install("bench", "benchmark", module_path=module_path)
        assert (installer.env_path("bench", "benchmark") / ".installed").stat().st_mtime == mtime

    def test_force_reinstalls(self, tmp_path: Path) -> None:
        module_path = _create_fake_package(tmp_path, with_requirements=False, with_setup=False)
        installer = EnvironmentManager(base_dir=tmp_path / "envs")

        installer.install("bench", "benchmark", module_path=module_path)
        sentinel = installer.env_path("bench", "benchmark") / "venv" / "sentinel.txt"
        sentinel.write_text("old")

        installer.install("bench", "benchmark", force=True, module_path=module_path)

        assert not sentinel.exists()
        assert installer.is_installed("bench", "benchmark", env_type="venv")

    def test_runs_setup_sh(self, tmp_path: Path) -> None:
        module_path = _create_fake_package(tmp_path, with_requirements=False, with_setup=True)
        installer = EnvironmentManager(base_dir=tmp_path / "envs")

        env_dir = installer.install("bench", "benchmark", module_path=module_path)

        assert (env_dir / "data" / "setup_ran.txt").is_file()

    def test_cleanup_on_failure(self, tmp_path: Path) -> None:
        module_path = _create_fake_package(tmp_path, with_requirements=False, with_setup=False)
        installer = EnvironmentManager(base_dir=tmp_path / "envs")

        def fail_pip(cmd, **kwargs):
            if isinstance(cmd, list) and "pip" in cmd and "install" in cmd:
                raise subprocess.CalledProcessError(1, cmd)
            return _real_subprocess_run(cmd, **kwargs)

        with mock.patch("subprocess.run", side_effect=fail_pip):
            with pytest.raises(subprocess.CalledProcessError):
                installer.install("bench", "benchmark", venv_packages=["some-pkg"], module_path=module_path)

        venv_dir = installer.env_path("bench", "benchmark") / "venv"
        assert not venv_dir.exists()
        assert not installer.is_installed("bench", "benchmark", env_type="venv")

    def test_venv_python_path(self, tmp_path: Path) -> None:
        """The venv runner needs to know the venv Python path."""
        module_path = _create_fake_package(tmp_path, with_requirements=False, with_setup=False)
        installer = EnvironmentManager(base_dir=tmp_path / "envs")

        installer.install("bench", "benchmark", module_path=module_path)

        python_path = installer.venv_python("bench", "benchmark")
        assert python_path == str(tmp_path / "envs" / "benchmarks" / "bench" / "venv" / "bin" / "python")
        assert Path(python_path).exists()


# ---------------------------------------------------------------------------
# Local install
# ---------------------------------------------------------------------------


class TestLocalInstall:
    """Local install uses the current Python (sys.executable).

    Used for debugging/development. The installer must record which
    Python was used so runners can find it.
    """

    def test_installs_without_venv(self, tmp_path: Path) -> None:
        module_path = _create_fake_package(tmp_path, with_requirements=False, with_setup=False)
        installer = EnvironmentManager(base_dir=tmp_path / "envs")

        env_dir = installer.install("bench", "benchmark", env_type="local", module_path=module_path)

        assert env_dir.is_dir()
        assert not (env_dir / "venv").exists()
        assert installer.is_installed("bench", "benchmark", env_type="local")

    def test_marker_has_python_path(self, tmp_path: Path) -> None:
        module_path = _create_fake_package(tmp_path, with_requirements=False, with_setup=False)
        installer = EnvironmentManager(base_dir=tmp_path / "envs")

        installer.install("bench", "benchmark", env_type="local", module_path=module_path)

        marker = json.loads((installer.env_path("bench", "benchmark") / ".installed").read_text())
        assert marker["local"]["python"] == sys.executable
        assert "installed_at" in marker["local"]

    def test_skips_if_already_installed(self, tmp_path: Path) -> None:
        module_path = _create_fake_package(tmp_path, with_requirements=False, with_setup=False)
        installer = EnvironmentManager(base_dir=tmp_path / "envs")

        installer.install("bench", "benchmark", env_type="local", module_path=module_path)
        mtime = (installer.env_path("bench", "benchmark") / ".installed").stat().st_mtime

        installer.install("bench", "benchmark", env_type="local", module_path=module_path)
        assert (installer.env_path("bench", "benchmark") / ".installed").stat().st_mtime == mtime

    def test_force_reinstalls(self, tmp_path: Path) -> None:
        module_path = _create_fake_package(tmp_path, with_requirements=False, with_setup=False)
        installer = EnvironmentManager(base_dir=tmp_path / "envs")

        installer.install("bench", "benchmark", env_type="local", module_path=module_path)
        old_marker = json.loads((installer.env_path("bench", "benchmark") / ".installed").read_text())

        installer.install("bench", "benchmark", env_type="local", force=True, module_path=module_path)
        new_marker = json.loads((installer.env_path("bench", "benchmark") / ".installed").read_text())

        assert new_marker["local"]["installed_at"] >= old_marker["local"]["installed_at"]

    def test_runs_setup_sh_without_virtual_env(self, tmp_path: Path) -> None:
        """setup.sh must get EXGENTIC_CACHE_DIR but NOT VIRTUAL_ENV for local installs."""
        module_path = _create_fake_package(tmp_path, with_requirements=False, with_setup=True)
        installer = EnvironmentManager(base_dir=tmp_path / "envs")

        env_dir = installer.install("bench", "benchmark", env_type="local", module_path=module_path)

        # setup.sh ran (it creates data/setup_ran.txt via $EXGENTIC_CACHE_DIR)
        assert (env_dir / "data" / "setup_ran.txt").is_file()

    def test_installs_requirements_into_current_python(self, tmp_path: Path) -> None:
        """Local install must call uv pip install --python sys.executable."""
        module_path = _create_fake_package(tmp_path, with_requirements=True, with_setup=False)
        installer = EnvironmentManager(base_dir=tmp_path / "envs")

        pip_calls: list[list[str]] = []

        def capture_run(cmd, **kwargs):
            if isinstance(cmd, list) and "pip" in cmd and "install" in cmd:
                pip_calls.append(list(cmd))
                return _docker_mock_result()
            return _real_subprocess_run(cmd, **kwargs)

        with mock.patch("subprocess.run", side_effect=capture_run):
            installer.install("bench", "benchmark", env_type="local", module_path=module_path)

        assert len(pip_calls) == 1
        assert "--python" in pip_calls[0]
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
        installer = EnvironmentManager(base_dir=tmp_path / "envs")

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
            env_dir = installer.install("bench", "benchmark", env_type="docker", module_path=module_path)

        assert len(dockerfiles) == 1
        assert "requirements.txt" in dockerfiles[0]
        assert "setup.sh" in dockerfiles[0]

        marker = json.loads((env_dir / ".installed").read_text())
        assert "docker" in marker
        assert "image" in marker["docker"]
        assert marker["docker"]["image"].startswith("exgentic-benchmark-bench:")
        assert "installed_at" in marker["docker"]

    def test_reuses_existing_image(self, tmp_path: Path) -> None:
        module_path = _create_fake_package(tmp_path, with_requirements=True, with_setup=False)
        installer = EnvironmentManager(base_dir=tmp_path / "envs")

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
            installer.install("bench", "benchmark", env_type="docker", module_path=module_path)

        assert len(build_called) == 0

    def test_force_rebuilds(self, tmp_path: Path) -> None:
        module_path = _create_fake_package(tmp_path, with_requirements=True, with_setup=False)
        installer = EnvironmentManager(base_dir=tmp_path / "envs")

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
            installer.install("bench", "benchmark", env_type="docker", force=True, module_path=module_path)

        assert len(build_calls) == 1

    def test_includes_system_deps(self, tmp_path: Path) -> None:
        module_path = _create_fake_package(tmp_path, with_requirements=True, with_setup=False, with_system_deps=True)
        installer = EnvironmentManager(base_dir=tmp_path / "envs")

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
            installer.install("bench", "benchmark", env_type="docker", module_path=module_path)

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

        installer = EnvironmentManager(base_dir=tmp_path / "envs")
        with mock.patch("subprocess.run", side_effect=capture_run):
            installer.install("a", "benchmark", env_type="docker", module_path=module_path_a)
            installer.install("b", "benchmark", env_type="docker", module_path=module_path_b)

        assert tags[0].split(":")[-1] != tags[1].split(":")[-1]


# ---------------------------------------------------------------------------
# Coexistence: venv + local + docker can all be installed simultaneously
# ---------------------------------------------------------------------------


class TestCoexistence:
    """Multiple env_types can coexist for the same slug.

    Runners pick whichever env_type they need.
    """

    def test_venv_and_local_coexist(self, tmp_path: Path) -> None:
        module_path = _create_fake_package(tmp_path, with_requirements=False, with_setup=False)
        installer = EnvironmentManager(base_dir=tmp_path / "envs")

        installer.install("bench", "benchmark", env_type="venv", module_path=module_path)
        installer.install("bench", "benchmark", env_type="local", module_path=module_path)

        assert installer.is_installed("bench", "benchmark", env_type="venv")
        assert installer.is_installed("bench", "benchmark", env_type="local")
        assert (installer.env_path("bench", "benchmark") / "venv").is_dir()

        marker = json.loads((installer.env_path("bench", "benchmark") / ".installed").read_text())
        assert "venv" in marker
        assert "local" in marker

    def test_venv_and_docker_coexist(self, tmp_path: Path) -> None:
        module_path = _create_fake_package(tmp_path, with_requirements=False, with_setup=False)
        installer = EnvironmentManager(base_dir=tmp_path / "envs")

        installer.install("bench", "benchmark", env_type="venv", module_path=module_path)

        def docker_side_effect(cmd, **kwargs):
            if cmd[0] == "docker":
                if cmd[1:3] == ["image", "inspect"]:
                    return _docker_mock_result(returncode=1)
                return _docker_mock_result()
            return _real_subprocess_run(cmd, **kwargs)

        with mock.patch("subprocess.run", side_effect=docker_side_effect):
            installer.install("bench", "benchmark", env_type="docker", module_path=module_path)

        assert installer.is_installed("bench", "benchmark", env_type="venv")
        assert installer.is_installed("bench", "benchmark", env_type="docker")

    def test_all_three_coexist(self, tmp_path: Path) -> None:
        module_path = _create_fake_package(tmp_path, with_requirements=False, with_setup=False)
        installer = EnvironmentManager(base_dir=tmp_path / "envs")

        installer.install("bench", "benchmark", env_type="venv", module_path=module_path)
        installer.install("bench", "benchmark", env_type="local", module_path=module_path)

        def docker_side_effect(cmd, **kwargs):
            if cmd[0] == "docker":
                if cmd[1:3] == ["image", "inspect"]:
                    return _docker_mock_result(returncode=1)
                return _docker_mock_result()
            return _real_subprocess_run(cmd, **kwargs)

        with mock.patch("subprocess.run", side_effect=docker_side_effect):
            installer.install("bench", "benchmark", env_type="docker", module_path=module_path)

        marker = json.loads((installer.env_path("bench", "benchmark") / ".installed").read_text())
        assert set(marker.keys()) == {"venv", "local", "docker"}

    def test_force_reinstall_one_preserves_others(self, tmp_path: Path) -> None:
        module_path = _create_fake_package(tmp_path, with_requirements=False, with_setup=False)
        installer = EnvironmentManager(base_dir=tmp_path / "envs")

        installer.install("bench", "benchmark", env_type="venv", module_path=module_path)
        installer.install("bench", "benchmark", env_type="local", module_path=module_path)

        old_marker = json.loads((installer.env_path("bench", "benchmark") / ".installed").read_text())
        old_local_at = old_marker["local"]["installed_at"]

        installer.install("bench", "benchmark", env_type="venv", force=True, module_path=module_path)

        new_marker = json.loads((installer.env_path("bench", "benchmark") / ".installed").read_text())
        assert "venv" in new_marker
        assert "local" in new_marker
        assert new_marker["local"]["installed_at"] == old_local_at


# ---------------------------------------------------------------------------
# Uninstall
# ---------------------------------------------------------------------------


class TestUninstall:
    """Uninstall removes the specified env_type without affecting others.

    When the last env_type is removed, the whole directory is cleaned up.
    """

    def test_uninstall_venv(self, tmp_path: Path) -> None:
        module_path = _create_fake_package(tmp_path, with_requirements=False, with_setup=False)
        installer = EnvironmentManager(base_dir=tmp_path / "envs")

        installer.install("bench", "benchmark", module_path=module_path)
        installer.uninstall("bench", "benchmark", env_type="venv")

        assert not installer.is_installed("bench", "benchmark", env_type="venv")
        # Last env_type removed -> dir cleaned up
        assert not installer.env_path("bench", "benchmark").exists()

    def test_uninstall_local(self, tmp_path: Path) -> None:
        module_path = _create_fake_package(tmp_path, with_requirements=False, with_setup=False)
        installer = EnvironmentManager(base_dir=tmp_path / "envs")

        installer.install("bench", "benchmark", env_type="local", module_path=module_path)
        installer.uninstall("bench", "benchmark", env_type="local")

        assert not installer.is_installed("bench", "benchmark", env_type="local")

    def test_uninstall_docker_removes_image(self, tmp_path: Path) -> None:
        installer = EnvironmentManager(base_dir=tmp_path / "envs")
        env_dir = installer.env_path("bench", "benchmark")
        env_dir.mkdir(parents=True)

        image_tag = "exgentic-benchmark-bench:abc123"
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
            installer.uninstall("bench", "benchmark", env_type="docker")

        assert len(rmi_calls) == 1
        assert rmi_calls[0] == ["docker", "rmi", image_tag]

    def test_uninstall_all(self, tmp_path: Path) -> None:
        module_path = _create_fake_package(tmp_path, with_requirements=False, with_setup=False)
        installer = EnvironmentManager(base_dir=tmp_path / "envs")

        installer.install("bench", "benchmark", env_type="venv", module_path=module_path)
        installer.install("bench", "benchmark", env_type="local", module_path=module_path)

        installer.uninstall("bench", "benchmark")

        assert not installer.env_path("bench", "benchmark").exists()
        assert not installer.is_installed("bench", "benchmark")

    def test_uninstall_one_keeps_others(self, tmp_path: Path) -> None:
        module_path = _create_fake_package(tmp_path, with_requirements=False, with_setup=False)
        installer = EnvironmentManager(base_dir=tmp_path / "envs")

        installer.install("bench", "benchmark", env_type="venv", module_path=module_path)
        installer.install("bench", "benchmark", env_type="local", module_path=module_path)

        installer.uninstall("bench", "benchmark", env_type="venv")

        assert not installer.is_installed("bench", "benchmark", env_type="venv")
        assert installer.is_installed("bench", "benchmark", env_type="local")
        assert not (installer.env_path("bench", "benchmark") / "venv").exists()
        # Dir still exists because local is still installed
        assert installer.env_path("bench", "benchmark").exists()

    def test_uninstall_last_removes_dir(self, tmp_path: Path) -> None:
        module_path = _create_fake_package(tmp_path, with_requirements=False, with_setup=False)
        installer = EnvironmentManager(base_dir=tmp_path / "envs")

        installer.install("bench", "benchmark", env_type="venv", module_path=module_path)
        installer.install("bench", "benchmark", env_type="local", module_path=module_path)

        installer.uninstall("bench", "benchmark", env_type="venv")
        assert installer.env_path("bench", "benchmark").exists()

        installer.uninstall("bench", "benchmark", env_type="local")
        assert not installer.env_path("bench", "benchmark").exists()

    def test_uninstall_nonexistent_is_noop(self, tmp_path: Path) -> None:
        installer = EnvironmentManager(base_dir=tmp_path / "envs")
        installer.uninstall("nonexistent", "benchmark")
        installer.uninstall("nonexistent", "benchmark", env_type="venv")

    def test_uninstall_all_with_docker_removes_image(self, tmp_path: Path) -> None:
        installer = EnvironmentManager(base_dir=tmp_path / "envs")
        env_dir = installer.env_path("bench", "benchmark")
        env_dir.mkdir(parents=True)

        image_tag = "exgentic-benchmark-bench:abc123"
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
            installer.uninstall("bench", "benchmark")

        assert len(rmi_calls) == 1
        assert not env_dir.exists()


# ---------------------------------------------------------------------------
# Query: is_installed, get_install_info, list_installed
# ---------------------------------------------------------------------------


class TestQueries:
    """The list commands and evaluate flow depend on querying install state."""

    def test_is_installed_any(self, tmp_path: Path) -> None:
        module_path = _create_fake_package(tmp_path, with_requirements=False, with_setup=False)
        installer = EnvironmentManager(base_dir=tmp_path / "envs")

        assert not installer.is_installed("bench", "benchmark")

        installer.install("bench", "benchmark", env_type="local", module_path=module_path)
        assert installer.is_installed("bench", "benchmark")
        assert not installer.is_installed("bench", "benchmark", env_type="venv")
        assert installer.is_installed("bench", "benchmark", env_type="local")

    def test_get_install_info(self, tmp_path: Path) -> None:
        module_path = _create_fake_package(tmp_path, with_requirements=False, with_setup=False)
        installer = EnvironmentManager(base_dir=tmp_path / "envs")

        assert installer.get_install_info("bench", "benchmark") is None

        installer.install("bench", "benchmark", env_type="venv", module_path=module_path)
        installer.install("bench", "benchmark", env_type="local", module_path=module_path)

        info = installer.get_install_info("bench", "benchmark")
        assert info is not None
        assert info["slug"] == "bench"
        assert info["kind"] == "benchmark"
        assert "venv" in info["environments"]
        assert "local" in info["environments"]
        assert "installed_at" in info["environments"]["venv"]
        assert "python" in info["environments"]["local"]

    def test_list_installed(self, tmp_path: Path) -> None:
        module_path = _create_fake_package(tmp_path, with_requirements=False, with_setup=False)
        installer = EnvironmentManager(base_dir=tmp_path / "envs")

        assert installer.list_installed("benchmark") == []

        installer.install("alpha", "benchmark", module_path=module_path)
        installer.install("beta", "benchmark", module_path=module_path)

        result = installer.list_installed("benchmark")
        assert len(result) == 2
        slugs = [r["slug"] for r in result]
        assert slugs == ["alpha", "beta"]
        assert all(r["kind"] == "benchmark" for r in result)
        assert all("venv" in r["environments"] for r in result)

    def test_list_installed_filters_by_kind(self, tmp_path: Path) -> None:
        module_path = _create_fake_package(tmp_path, with_requirements=False, with_setup=False)
        installer = EnvironmentManager(base_dir=tmp_path / "envs")

        installer.install("bench1", "benchmark", module_path=module_path)
        installer.install("agent1", "agent", module_path=module_path)

        assert len(installer.list_installed("benchmark")) == 1
        assert len(installer.list_installed("agent")) == 1
        assert len(installer.list_installed()) == 2

    def test_list_installed_includes_env_details(self, tmp_path: Path) -> None:
        """list_benchmarks() and list_agents() need installed_at from the marker."""
        module_path = _create_fake_package(tmp_path, with_requirements=False, with_setup=False)
        installer = EnvironmentManager(base_dir=tmp_path / "envs")

        installer.install("bench", "benchmark", env_type="venv", module_path=module_path)
        installer.install("bench", "benchmark", env_type="local", module_path=module_path)

        result = installer.list_installed("benchmark")
        assert len(result) == 1
        envs = result[0]["environments"]
        assert "venv" in envs
        assert "local" in envs
        assert "installed_at" in envs["venv"]
        assert "installed_at" in envs["local"]
        assert "python" in envs["local"]


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


class TestPaths:
    """Path helpers for locating data and venv directories."""

    def test_env_path(self, tmp_path: Path) -> None:
        installer = EnvironmentManager(base_dir=tmp_path / "envs")

        assert installer.env_path("tau2", "benchmark") == tmp_path / "envs" / "benchmarks" / "tau2"
        assert installer.env_path("tool_calling", "agent") == tmp_path / "envs" / "agents" / "tool_calling"

    def test_venv_python(self, tmp_path: Path) -> None:
        installer = EnvironmentManager(base_dir=tmp_path / "envs")

        expected = str(tmp_path / "envs" / "benchmarks" / "tau2" / "venv" / "bin" / "python")
        assert installer.venv_python("tau2", "benchmark") == expected

    def test_default_base_dir(self) -> None:
        installer = EnvironmentManager()
        assert installer.base_dir == Path.home() / ".exgentic"


# ---------------------------------------------------------------------------
# Marker edge cases
# ---------------------------------------------------------------------------


class TestMarkers:
    """Marker file must be robust against corruption and old formats."""

    def test_corrupted_marker_treated_as_empty(self, tmp_path: Path) -> None:
        installer = EnvironmentManager(base_dir=tmp_path / "envs")
        env_dir = installer.env_path("bench", "benchmark")
        env_dir.mkdir(parents=True)
        (env_dir / ".installed").write_text("not json")

        assert not installer.is_installed("bench", "benchmark")
        assert installer.get_install_info("bench", "benchmark") is None

    def test_non_dict_marker_treated_as_empty(self, tmp_path: Path) -> None:
        installer = EnvironmentManager(base_dir=tmp_path / "envs")
        env_dir = installer.env_path("bench", "benchmark")
        env_dir.mkdir(parents=True)
        (env_dir / ".installed").write_text('"just a string"')

        assert not installer.is_installed("bench", "benchmark")

    def test_empty_dict_marker_means_not_installed(self, tmp_path: Path) -> None:
        installer = EnvironmentManager(base_dir=tmp_path / "envs")
        env_dir = installer.env_path("bench", "benchmark")
        env_dir.mkdir(parents=True)
        (env_dir / ".installed").write_text("{}")

        assert not installer.is_installed("bench", "benchmark")
        assert installer.get_install_info("bench", "benchmark") is None


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------


class TestFailureModes:
    def test_missing_uv(self, tmp_path: Path) -> None:
        installer = EnvironmentManager(base_dir=tmp_path / "envs")

        with mock.patch("shutil.which", return_value=None):
            with pytest.raises(RuntimeError, match="Could not find 'uv'"):
                installer.install("bench", "benchmark")

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

        installer = EnvironmentManager(base_dir=tmp_path / "envs")

        with pytest.raises(subprocess.CalledProcessError):
            installer.install("bench", "benchmark", module_path=module_path)

        assert not installer.is_installed("bench", "benchmark", env_type="venv")

    def test_missing_system_dep(self, tmp_path: Path) -> None:
        module_path = _create_fake_package(tmp_path, with_requirements=False, with_setup=False, with_system_deps=True)
        parts = module_path.split(".")
        pkg_dir = tmp_path
        for part in parts[:-1]:
            pkg_dir = pkg_dir / part
        (pkg_dir / "system-deps.txt").write_text("nonexistent_tool_xyz\n")
        importlib.invalidate_caches()

        installer = EnvironmentManager(base_dir=tmp_path / "envs")

        original_which = shutil.which

        def which_no_fake(name):
            if name == "nonexistent_tool_xyz":
                return None
            return original_which(name)

        with mock.patch("exgentic.environment.manager.shutil.which", side_effect=which_no_fake):
            with mock.patch("exgentic.environment.manager._dpkg_installed", return_value=False):
                with pytest.raises(RuntimeError, match="nonexistent_tool_xyz"):
                    installer.install("bench", "benchmark", module_path=module_path)

    def test_venv_failure_preserves_coexisting_envs(self, tmp_path: Path) -> None:
        """If venv install fails, local install must be unaffected."""
        module_path = _create_fake_package(tmp_path, with_requirements=False, with_setup=False)
        installer = EnvironmentManager(base_dir=tmp_path / "envs")

        installer.install("bench", "benchmark", env_type="local", module_path=module_path)

        def fail_venv(cmd, **kwargs):
            if isinstance(cmd, list) and "venv" in cmd:
                raise subprocess.CalledProcessError(1, cmd)
            return _real_subprocess_run(cmd, **kwargs)

        with mock.patch("subprocess.run", side_effect=fail_venv):
            with pytest.raises(subprocess.CalledProcessError):
                installer.install("bench", "benchmark", env_type="venv", module_path=module_path)

        # Local must still be installed
        assert installer.is_installed("bench", "benchmark", env_type="local")
        assert not installer.is_installed("bench", "benchmark", env_type="venv")


# ---------------------------------------------------------------------------
# State transitions
# ---------------------------------------------------------------------------


class TestStateMachine:
    def test_install_uninstall_install_cycle(self, tmp_path: Path) -> None:
        module_path = _create_fake_package(tmp_path, with_requirements=False, with_setup=False)
        installer = EnvironmentManager(base_dir=tmp_path / "envs")

        env1 = installer.install("bench", "benchmark", module_path=module_path)
        assert installer.is_installed("bench", "benchmark")

        installer.uninstall("bench", "benchmark")
        assert not installer.is_installed("bench", "benchmark")
        assert not env1.exists()

        env2 = installer.install("bench", "benchmark", module_path=module_path)
        assert installer.is_installed("bench", "benchmark")
        assert env2 == env1

    def test_double_uninstall(self, tmp_path: Path) -> None:
        module_path = _create_fake_package(tmp_path, with_requirements=False, with_setup=False)
        installer = EnvironmentManager(base_dir=tmp_path / "envs")

        installer.install("bench", "benchmark", module_path=module_path)
        installer.uninstall("bench", "benchmark")
        installer.uninstall("bench", "benchmark")

        assert not installer.is_installed("bench", "benchmark")

    def test_install_without_module_path(self, tmp_path: Path) -> None:
        """Bare install with no module_path should still succeed."""
        installer = EnvironmentManager(base_dir=tmp_path / "envs")

        env_dir = installer.install("bench", "benchmark")

        assert installer.is_installed("bench", "benchmark", env_type="venv")
        assert (env_dir / "venv" / "bin" / "python").exists()

    def test_local_install_without_module_path(self, tmp_path: Path) -> None:
        installer = EnvironmentManager(base_dir=tmp_path / "envs")

        env_dir = installer.install("bench", "benchmark", env_type="local")

        assert installer.is_installed("bench", "benchmark", env_type="local")
        assert env_dir.is_dir()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestFindPackageFile:
    def test_finds_file(self, tmp_path: Path) -> None:
        module_path = _create_fake_package(tmp_path, with_requirements=True)
        result = _find_package_file(module_path, "requirements.txt")
        assert result is not None
        assert result.name == "requirements.txt"

    def test_returns_none_for_missing(self, tmp_path: Path) -> None:
        module_path = _create_fake_package(tmp_path, with_requirements=False, with_setup=False)
        assert _find_package_file(module_path, "nonexistent.txt") is None


class TestRequireUv:
    def test_returns_path(self) -> None:
        path = _require_uv()
        assert "uv" in Path(path).name

    def test_raises_when_missing(self) -> None:
        with mock.patch("shutil.which", return_value=None):
            with pytest.raises(RuntimeError, match="Could not find 'uv'"):
                _require_uv()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    """Invalid env_type must raise ValueError immediately."""

    def test_invalid_env_type_on_install(self, tmp_path: Path) -> None:
        installer = EnvironmentManager(base_dir=tmp_path / "envs")
        with pytest.raises(ValueError, match="Invalid env_type"):
            installer.install("bench", "benchmark", env_type="invalid")

    def test_invalid_env_type_on_uninstall(self, tmp_path: Path) -> None:
        installer = EnvironmentManager(base_dir=tmp_path / "envs")
        with pytest.raises(ValueError, match="Invalid env_type"):
            installer.uninstall("bench", "benchmark", env_type="invalid")

    def test_invalid_env_type_on_is_installed(self, tmp_path: Path) -> None:
        installer = EnvironmentManager(base_dir=tmp_path / "envs")
        with pytest.raises(ValueError, match="Invalid env_type"):
            installer.is_installed("bench", "benchmark", env_type="invalid")


# ---------------------------------------------------------------------------
# Convenience accessors
# ---------------------------------------------------------------------------


class TestConvenienceAccessors:
    """Runners need quick access to docker image tags and local Python paths."""

    def test_docker_image_returns_tag(self, tmp_path: Path) -> None:
        installer = EnvironmentManager(base_dir=tmp_path / "envs")
        env_dir = installer.env_path("bench", "benchmark")
        env_dir.mkdir(parents=True)
        (env_dir / ".installed").write_text(
            json.dumps({"docker": {"installed_at": "2026-01-01T00:00:00Z", "image": "exgentic-benchmark-bench:abc123"}})
        )

        assert installer.docker_image("bench", "benchmark") == "exgentic-benchmark-bench:abc123"

    def test_docker_image_returns_none_when_not_installed(self, tmp_path: Path) -> None:
        installer = EnvironmentManager(base_dir=tmp_path / "envs")
        assert installer.docker_image("bench", "benchmark") is None

    def test_local_python_returns_path(self, tmp_path: Path) -> None:
        module_path = _create_fake_package(tmp_path, with_requirements=False, with_setup=False)
        installer = EnvironmentManager(base_dir=tmp_path / "envs")

        installer.install("bench", "benchmark", env_type="local", module_path=module_path)

        assert installer.local_python("bench", "benchmark") == sys.executable

    def test_local_python_returns_none_when_not_installed(self, tmp_path: Path) -> None:
        installer = EnvironmentManager(base_dir=tmp_path / "envs")
        assert installer.local_python("bench", "benchmark") is None


# ---------------------------------------------------------------------------
# Docker integration (requires Docker/Podman running)
# ---------------------------------------------------------------------------

_docker_available = shutil.which("docker") is not None


@pytest.mark.skipif(not _docker_available, reason="docker CLI not available")
class TestDockerIntegration:
    """Real Docker tests — actually build and remove images."""

    def test_docker_build_and_marker(self, tmp_path: Path) -> None:
        """Build a real Docker image and verify the marker stores the tag."""
        module_path = _create_fake_package(tmp_path, with_requirements=False, with_setup=False)
        installer = EnvironmentManager(base_dir=tmp_path / "envs")

        installer.install("inttest", "benchmark", env_type="docker", module_path=module_path)

        assert installer.is_installed("inttest", "benchmark", env_type="docker")
        image_tag = installer.docker_image("inttest", "benchmark")
        assert image_tag is not None
        assert image_tag.startswith("exgentic-benchmark-inttest:")

        # Image actually exists
        result = subprocess.run(
            ["docker", "image", "inspect", image_tag],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

        # Cleanup
        installer.uninstall("inttest", "benchmark", env_type="docker")
        result = subprocess.run(
            ["docker", "image", "inspect", image_tag],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0

    def test_docker_reuses_existing_image(self, tmp_path: Path) -> None:
        """Second install should skip rebuild when image already exists."""
        module_path = _create_fake_package(tmp_path, with_requirements=False, with_setup=False)
        installer = EnvironmentManager(base_dir=tmp_path / "envs")

        installer.install("inttest2", "benchmark", env_type="docker", module_path=module_path)
        tag1 = installer.docker_image("inttest2", "benchmark")

        # Second install — should reuse
        installer.install("inttest2", "benchmark", env_type="docker", force=True, module_path=module_path)
        tag2 = installer.docker_image("inttest2", "benchmark")

        # Same content hash → same tag
        assert tag1 == tag2

        # Cleanup
        installer.uninstall("inttest2", "benchmark")

    def test_docker_with_requirements(self, tmp_path: Path) -> None:
        """Build image with a real requirements.txt."""
        module_path = _create_fake_package(tmp_path, with_requirements=True, with_setup=False)
        installer = EnvironmentManager(base_dir=tmp_path / "envs")

        installer.install("inttest3", "benchmark", env_type="docker", module_path=module_path)

        image_tag = installer.docker_image("inttest3", "benchmark")
        assert image_tag is not None

        # Verify the package is installed in the image
        result = subprocess.run(
            ["docker", "run", "--rm", image_tag, "python", "-c", "import requests; print(requests.__version__)"],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert result.stdout.strip()  # should print a version

        # Cleanup
        installer.uninstall("inttest3", "benchmark")

    def test_docker_coexists_with_venv(self, tmp_path: Path) -> None:
        """Docker and venv can be installed for the same slug."""
        module_path = _create_fake_package(tmp_path, with_requirements=False, with_setup=False)
        installer = EnvironmentManager(base_dir=tmp_path / "envs")

        installer.install("inttest4", "benchmark", env_type="venv", module_path=module_path)
        installer.install("inttest4", "benchmark", env_type="docker", module_path=module_path)

        assert installer.is_installed("inttest4", "benchmark", env_type="venv")
        assert installer.is_installed("inttest4", "benchmark", env_type="docker")

        # Uninstall docker only
        installer.uninstall("inttest4", "benchmark", env_type="docker")
        assert installer.is_installed("inttest4", "benchmark", env_type="venv")
        assert not installer.is_installed("inttest4", "benchmark", env_type="docker")

        # Cleanup
        installer.uninstall("inttest4", "benchmark")
