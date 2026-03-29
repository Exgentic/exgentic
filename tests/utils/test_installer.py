# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Tests for exgentic.utils.installer."""

from __future__ import annotations

import importlib
import json
import stat
import subprocess
import sys
import textwrap
from pathlib import Path
from unittest import mock

import pytest
from exgentic.utils.installer import EnvironmentInstaller, _find_package_file, _require_uv

_pkg_counter = 0


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
    """Create a minimal importable package with optional resource files.

    Returns the dotted module path for the package.  Each call produces a
    unique package name so that ``importlib`` caching does not interfere
    across tests.
    """
    global _pkg_counter
    _pkg_counter += 1
    tag = f"p{_pkg_counter}"

    # Use a unique top-level name per call so ``importlib.resources``
    # caching across tests (which use different ``tmp_path`` roots)
    # never interferes.
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

    # Make sure Python can import the package
    if str(tmp_path) not in sys.path:
        sys.path.insert(0, str(tmp_path))
    importlib.invalidate_caches()

    return f"{top}.{mid}.{leaf}.main"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEnvironmentInstaller:
    """Tests for EnvironmentInstaller."""

    def test_install_creates_environment(self, tmp_path: Path) -> None:
        module_path = _create_fake_package(tmp_path, with_requirements=False, with_setup=False)
        installer = EnvironmentInstaller(base_dir=tmp_path / "envs")

        env_dir = installer.install("my-bench", "benchmark", module_path=module_path)

        assert env_dir.is_dir()
        assert (env_dir / "venv").is_dir()
        assert (env_dir / "venv" / "bin" / "python").exists()
        assert (env_dir / ".installed").is_file()

    def test_install_skips_if_installed(self, tmp_path: Path) -> None:
        module_path = _create_fake_package(tmp_path, with_requirements=False, with_setup=False)
        installer = EnvironmentInstaller(base_dir=tmp_path / "envs")

        env_dir_1 = installer.install("my-bench", "benchmark", module_path=module_path)
        marker_mtime = (env_dir_1 / ".installed").stat().st_mtime

        # Second install should be a no-op -- marker mtime must not change.
        env_dir_2 = installer.install("my-bench", "benchmark", module_path=module_path)

        assert env_dir_2 == env_dir_1
        assert (env_dir_2 / ".installed").stat().st_mtime == marker_mtime

    def test_install_force_reinstalls(self, tmp_path: Path) -> None:
        module_path = _create_fake_package(tmp_path, with_requirements=False, with_setup=False)
        installer = EnvironmentInstaller(base_dir=tmp_path / "envs")

        installer.install("my-bench", "benchmark", module_path=module_path)
        sentinel = installer.env_path("my-bench", "benchmark") / "sentinel.txt"
        sentinel.write_text("should be removed")

        installer.install("my-bench", "benchmark", force=True, module_path=module_path)

        # The old sentinel must be gone because force wipes the directory.
        assert not sentinel.exists()
        assert installer.is_installed("my-bench", "benchmark")

    def test_uninstall_removes_environment(self, tmp_path: Path) -> None:
        module_path = _create_fake_package(tmp_path, with_requirements=False, with_setup=False)
        installer = EnvironmentInstaller(base_dir=tmp_path / "envs")

        installer.install("my-bench", "benchmark", module_path=module_path)
        assert installer.is_installed("my-bench", "benchmark")

        installer.uninstall("my-bench", "benchmark")
        assert not installer.env_path("my-bench", "benchmark").exists()

    def test_is_installed(self, tmp_path: Path) -> None:
        module_path = _create_fake_package(tmp_path, with_requirements=False, with_setup=False)
        installer = EnvironmentInstaller(base_dir=tmp_path / "envs")

        assert not installer.is_installed("my-bench", "benchmark")

        installer.install("my-bench", "benchmark", module_path=module_path)
        assert installer.is_installed("my-bench", "benchmark")

        installer.uninstall("my-bench", "benchmark")
        assert not installer.is_installed("my-bench", "benchmark")

    def test_list_installed(self, tmp_path: Path) -> None:
        module_path = _create_fake_package(tmp_path, with_requirements=False, with_setup=False)
        installer = EnvironmentInstaller(base_dir=tmp_path / "envs")

        assert installer.list_installed("benchmark") == []

        installer.install("alpha", "benchmark", module_path=module_path)
        installer.install("beta", "benchmark", module_path=module_path)

        assert installer.list_installed("benchmark") == ["alpha", "beta"]
        assert installer.list_installed("agent") == []
        assert installer.list_installed() == ["alpha", "beta"]

    def test_install_runs_setup_sh(self, tmp_path: Path) -> None:
        module_path = _create_fake_package(tmp_path, with_requirements=False, with_setup=True)
        installer = EnvironmentInstaller(base_dir=tmp_path / "envs")

        env_dir = installer.install("my-bench", "benchmark", module_path=module_path)

        # setup.sh writes setup_ran.txt into $EXGENTIC_CACHE_DIR/data/
        assert (env_dir / "data" / "setup_ran.txt").is_file()

    def test_install_fails_on_missing_uv(self, tmp_path: Path) -> None:
        installer = EnvironmentInstaller(base_dir=tmp_path / "envs")

        with mock.patch("shutil.which", return_value=None):
            with pytest.raises(RuntimeError, match="Could not find 'uv'"):
                installer.install("my-bench", "benchmark")


class TestFindPackageFile:
    """Tests for the _find_package_file helper."""

    def test_finds_file_in_package(self, tmp_path: Path) -> None:
        module_path = _create_fake_package(tmp_path, with_requirements=True)
        result = _find_package_file(module_path, "requirements.txt")
        assert result is not None
        assert result.name == "requirements.txt"

    def test_returns_none_for_missing_file(self, tmp_path: Path) -> None:
        module_path = _create_fake_package(tmp_path, with_requirements=False, with_setup=False)
        result = _find_package_file(module_path, "nonexistent.txt")
        assert result is None


_real_subprocess_run = subprocess.run


def _docker_mock_result(**overrides):
    """Create a mock subprocess result for docker commands."""
    result = mock.MagicMock()
    result.returncode = overrides.get("returncode", 0)
    result.stdout = overrides.get("stdout", "")
    result.stderr = overrides.get("stderr", "")
    return result


class TestDockerInstall:
    """Tests for docker-based install/uninstall."""

    def test_install_docker_generates_dockerfile(self, tmp_path: Path) -> None:
        module_path = _create_fake_package(tmp_path, with_requirements=True, with_setup=True)
        installer = EnvironmentInstaller(base_dir=tmp_path / "envs")

        dockerfiles: list[str] = []

        def capture_run(cmd, **kwargs):
            if cmd[0] == "docker":
                if cmd[1] == "build":
                    build_dir = Path(cmd[-1])
                    df = build_dir / "Dockerfile"
                    if df.exists():
                        dockerfiles.append(df.read_text())
                if cmd[1:3] == ["image", "inspect"]:
                    return _docker_mock_result(returncode=1)
                return _docker_mock_result()
            return _real_subprocess_run(cmd, **kwargs)

        with mock.patch("subprocess.run", side_effect=capture_run):
            env_dir = installer.install("my-bench", "benchmark", runner="docker", module_path=module_path)

        assert env_dir.is_dir()
        assert (env_dir / ".installed").is_file()
        assert len(dockerfiles) == 1
        df = dockerfiles[0]
        assert "FROM python:3.12-slim" in df
        assert "uv pip install" in df
        assert "requirements.txt" in df
        assert "setup.sh" in df

    def test_install_docker_reuses_existing_image(self, tmp_path: Path) -> None:
        module_path = _create_fake_package(tmp_path, with_requirements=True, with_setup=False)
        installer = EnvironmentInstaller(base_dir=tmp_path / "envs")

        build_called = []

        def side_effect(cmd, **kwargs):
            if cmd[0] == "docker":
                if cmd[1] == "build":
                    build_called.append(True)
                if cmd[1:3] == ["image", "inspect"]:
                    return _docker_mock_result(returncode=0)  # image exists
                return _docker_mock_result()
            return _real_subprocess_run(cmd, **kwargs)

        with mock.patch("subprocess.run", side_effect=side_effect):
            env_dir = installer.install("my-bench", "benchmark", runner="docker", module_path=module_path)

        assert env_dir.is_dir()
        assert len(build_called) == 0, "docker build should not be called when image exists"

    def test_install_docker_includes_system_deps(self, tmp_path: Path) -> None:
        module_path = _create_fake_package(tmp_path, with_requirements=True, with_setup=False, with_system_deps=True)
        installer = EnvironmentInstaller(base_dir=tmp_path / "envs")

        dockerfiles: list[str] = []

        def capture_run(cmd, **kwargs):
            if cmd[0] == "docker":
                if cmd[1] == "build":
                    build_dir = Path(cmd[-1])
                    df = build_dir / "Dockerfile"
                    if df.exists():
                        dockerfiles.append(df.read_text())
                if cmd[1:3] == ["image", "inspect"]:
                    return _docker_mock_result(returncode=1)
                return _docker_mock_result()
            return _real_subprocess_run(cmd, **kwargs)

        with mock.patch("subprocess.run", side_effect=capture_run):
            installer.install("my-bench", "benchmark", runner="docker", module_path=module_path)

        assert len(dockerfiles) == 1
        df = dockerfiles[0]
        assert "apt-get install -y curl wget" in df

    def test_uninstall_docker_removes_image(self, tmp_path: Path) -> None:
        installer = EnvironmentInstaller(base_dir=tmp_path / "envs")
        env_dir = installer.env_path("my-bench", "benchmark")
        env_dir.mkdir(parents=True)

        image_tag = "exgentic-benchmark-my-bench:abc123"
        (env_dir / ".installed").write_text(json.dumps({"runner": "docker", "image": image_tag}))

        rmi_calls: list[list[str]] = []

        def side_effect(cmd, **kwargs):
            if cmd[0] == "docker" and cmd[1] == "rmi":
                rmi_calls.append(list(cmd))
                return _docker_mock_result()
            return _real_subprocess_run(cmd, **kwargs)

        with mock.patch("subprocess.run", side_effect=side_effect):
            installer.uninstall("my-bench", "benchmark")

        assert not env_dir.exists()
        assert len(rmi_calls) == 1
        assert rmi_calls[0] == ["docker", "rmi", image_tag]

    def test_install_docker_content_hash_changes(self, tmp_path: Path) -> None:
        """Different requirements produce different image tags."""
        module_path_a = _create_fake_package(tmp_path, with_requirements=True, with_setup=False)
        module_path_b = _create_fake_package(tmp_path, with_requirements=True, with_setup=False)

        # Overwrite requirements for package b with different content
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

        installer = EnvironmentInstaller(base_dir=tmp_path / "envs")
        with mock.patch("subprocess.run", side_effect=capture_run):
            installer.install("bench-a", "benchmark", runner="docker", module_path=module_path_a)
            installer.install("bench-b", "benchmark", runner="docker", module_path=module_path_b)

        assert len(tags) == 2
        # Tags have different hashes because requirements differ
        tag_a_hash = tags[0].split(":")[-1]
        tag_b_hash = tags[1].split(":")[-1]
        assert tag_a_hash != tag_b_hash, f"Expected different hashes but got {tag_a_hash} for both"


class TestRequireUv:
    """Tests for the _require_uv helper."""

    def test_returns_path_when_available(self) -> None:
        path = _require_uv()
        assert path is not None
        assert "uv" in Path(path).name

    def test_raises_when_missing(self) -> None:
        with mock.patch("shutil.which", return_value=None):
            with pytest.raises(RuntimeError, match="Could not find 'uv'"):
                _require_uv()
