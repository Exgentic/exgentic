# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Tests for exgentic.utils.installer."""

from __future__ import annotations

import importlib
import json
import shutil
import stat
import subprocess
import sys
import textwrap
from datetime import datetime
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


class TestBuildRunnerDocker:
    """Tests for build_runner with runner='docker'."""

    def test_build_runner_docker_generates_dockerfile(self, tmp_path: Path) -> None:
        module_path = _create_fake_package(tmp_path, with_requirements=True, with_setup=True)
        installer = EnvironmentInstaller(base_dir=tmp_path / "envs")

        # First install (runner-agnostic)
        installer.install("my-bench", "benchmark", module_path=module_path)

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
            env_dir = installer.build_runner("my-bench", "benchmark", runner="docker", module_path=module_path)

        assert env_dir.is_dir()
        assert (env_dir / "docker" / "image_tag").is_file()
        assert len(dockerfiles) == 1
        df = dockerfiles[0]
        assert "FROM python:3.12-slim" in df
        assert "uv pip install" in df
        assert "requirements.txt" in df
        assert "setup.sh" in df

    def test_build_runner_docker_reuses_existing_image(self, tmp_path: Path) -> None:
        module_path = _create_fake_package(tmp_path, with_requirements=True, with_setup=False)
        installer = EnvironmentInstaller(base_dir=tmp_path / "envs")

        installer.install("my-bench", "benchmark", module_path=module_path)

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
            env_dir = installer.build_runner("my-bench", "benchmark", runner="docker", module_path=module_path)

        assert env_dir.is_dir()
        assert len(build_called) == 0, "docker build should not be called when image exists"

    def test_build_runner_docker_includes_system_deps(self, tmp_path: Path) -> None:
        module_path = _create_fake_package(tmp_path, with_requirements=True, with_setup=False, with_system_deps=True)
        installer = EnvironmentInstaller(base_dir=tmp_path / "envs")

        # Skip system-dep validation during install (wget may not be on host)
        with mock.patch("exgentic.utils.installer._validate_system_deps"):
            installer.install("my-bench", "benchmark", module_path=module_path)

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
            installer.build_runner("my-bench", "benchmark", runner="docker", module_path=module_path)

        assert len(dockerfiles) == 1
        df = dockerfiles[0]
        assert "apt-get install -y curl wget" in df

    def test_build_runner_docker_content_hash_changes(self, tmp_path: Path) -> None:
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
            installer.install("bench-a", "benchmark", module_path=module_path_a)
            installer.build_runner("bench-a", "benchmark", runner="docker", module_path=module_path_a)
            installer.install("bench-b", "benchmark", module_path=module_path_b)
            installer.build_runner("bench-b", "benchmark", runner="docker", module_path=module_path_b)

        assert len(tags) == 2
        # Tags have different hashes because requirements differ
        tag_a_hash = tags[0].split(":")[-1]
        tag_b_hash = tags[1].split(":")[-1]
        assert tag_a_hash != tag_b_hash, f"Expected different hashes but got {tag_a_hash} for both"

    def test_uninstall_docker_removes_image(self, tmp_path: Path) -> None:
        installer = EnvironmentInstaller(base_dir=tmp_path / "envs")
        env_dir = installer.env_path("my-bench", "benchmark")
        env_dir.mkdir(parents=True)

        image_tag = "exgentic-benchmark-my-bench:abc123"
        # Write .installed marker
        (env_dir / ".installed").write_text(json.dumps({"installed_at": "2026-01-01T00:00:00+00:00"}))
        # Write docker runner info
        docker_dir = env_dir / "docker"
        docker_dir.mkdir()
        (docker_dir / "image_tag").write_text(image_tag)

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


class TestDockerForceReinstall:
    """Tests for docker force-reinstall behaviour."""

    def test_force_reinstall_docker(self, tmp_path: Path) -> None:
        """force=True rebuilds the docker image even when it already exists."""
        module_path = _create_fake_package(tmp_path, with_requirements=True, with_setup=False)
        installer = EnvironmentInstaller(base_dir=tmp_path / "envs")

        installer.install("my-bench", "benchmark", module_path=module_path)

        build_calls: list[list[str]] = []

        def side_effect(cmd, **kwargs):
            if cmd[0] == "docker":
                if cmd[1] == "build":
                    build_calls.append(list(cmd))
                if cmd[1:3] == ["image", "inspect"]:
                    # Pretend the image already exists
                    return _docker_mock_result(returncode=0)
                return _docker_mock_result()
            return _real_subprocess_run(cmd, **kwargs)

        with mock.patch("subprocess.run", side_effect=side_effect):
            installer.build_runner("my-bench", "benchmark", runner="docker", force=True, module_path=module_path)

        assert len(build_calls) == 1, "docker build should be called when force=True even if image exists"


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


# ---------------------------------------------------------------------------
# Edge-case & failure-mode tests
# ---------------------------------------------------------------------------


class TestStateMachine:
    """Tests for install/uninstall state transitions."""

    def test_install_twice_without_force_skips(self, tmp_path: Path) -> None:
        """Second install without force is a no-op and returns the same path."""
        module_path = _create_fake_package(tmp_path, with_requirements=False, with_setup=False)
        installer = EnvironmentInstaller(base_dir=tmp_path / "envs")

        path1 = installer.install("bench", "benchmark", module_path=module_path)
        mtime1 = (path1 / ".installed").stat().st_mtime

        path2 = installer.install("bench", "benchmark", module_path=module_path)

        assert path1 == path2
        assert (path2 / ".installed").stat().st_mtime == mtime1

    def test_install_uninstall_install_cycle(self, tmp_path: Path) -> None:
        """Full install -> uninstall -> install cycle works cleanly."""
        module_path = _create_fake_package(tmp_path, with_requirements=False, with_setup=False)
        installer = EnvironmentInstaller(base_dir=tmp_path / "envs")

        env1 = installer.install("bench", "benchmark", module_path=module_path)
        assert installer.is_installed("bench", "benchmark")
        assert (env1 / "venv" / "bin" / "python").exists()

        installer.uninstall("bench", "benchmark")
        assert not installer.is_installed("bench", "benchmark")
        assert not env1.exists()

        env2 = installer.install("bench", "benchmark", module_path=module_path)
        assert installer.is_installed("bench", "benchmark")
        assert (env2 / "venv" / "bin" / "python").exists()
        assert env2 == env1  # same canonical path

    def test_uninstall_when_not_installed(self, tmp_path: Path) -> None:
        """Uninstalling something never installed is a no-op, no crash."""
        installer = EnvironmentInstaller(base_dir=tmp_path / "envs")

        # Should not raise
        installer.uninstall("nonexistent", "benchmark")
        assert not installer.is_installed("nonexistent", "benchmark")

    def test_uninstall_twice(self, tmp_path: Path) -> None:
        """Double uninstall is a no-op on the second call."""
        module_path = _create_fake_package(tmp_path, with_requirements=False, with_setup=False)
        installer = EnvironmentInstaller(base_dir=tmp_path / "envs")

        installer.install("bench", "benchmark", module_path=module_path)
        installer.uninstall("bench", "benchmark")
        assert not installer.is_installed("bench", "benchmark")

        # Second uninstall should be a safe no-op
        installer.uninstall("bench", "benchmark")
        assert not installer.is_installed("bench", "benchmark")


class TestFailureModes:
    """Tests for failure scenarios and error clarity."""

    def test_install_broken_setup_sh(self, tmp_path: Path) -> None:
        """setup.sh exits non-zero -> install fails, no .installed marker."""
        module_path = _create_fake_package(tmp_path, with_requirements=False, with_setup=False)
        # Write a broken setup.sh
        parts = module_path.split(".")
        pkg_dir = tmp_path
        for part in parts[:-1]:
            pkg_dir = pkg_dir / part
        broken_setup = pkg_dir / "setup.sh"
        broken_setup.write_text("#!/usr/bin/env bash\nexit 1\n")
        broken_setup.chmod(broken_setup.stat().st_mode | stat.S_IEXEC)
        importlib.invalidate_caches()

        installer = EnvironmentInstaller(base_dir=tmp_path / "envs")

        with pytest.raises(subprocess.CalledProcessError):
            installer.install("bench", "benchmark", module_path=module_path)

        env_dir = installer.env_path("bench", "benchmark")
        # .installed marker must NOT exist after a failed install
        assert not (env_dir / ".installed").exists()

    def test_install_missing_uv_clear_error(self, tmp_path: Path) -> None:
        """Uv not on PATH -> RuntimeError with helpful install message."""
        installer = EnvironmentInstaller(base_dir=tmp_path / "envs")

        with mock.patch("shutil.which", return_value=None):
            with pytest.raises(RuntimeError, match="Could not find 'uv'") as exc_info:
                installer.install("bench", "benchmark")
            # Error message should contain install instructions
            assert "install" in str(exc_info.value).lower()

    def test_build_runner_docker_missing_docker(self, tmp_path: Path) -> None:
        """Docker not available -> clear error on docker build_runner."""
        module_path = _create_fake_package(tmp_path, with_requirements=False, with_setup=False)
        installer = EnvironmentInstaller(base_dir=tmp_path / "envs")

        installer.install("bench", "benchmark", module_path=module_path)

        def fail_docker(cmd, **kwargs):
            if cmd[0] == "docker":
                raise FileNotFoundError("docker not found")
            return _real_subprocess_run(cmd, **kwargs)

        with mock.patch("subprocess.run", side_effect=fail_docker):
            with pytest.raises(FileNotFoundError):
                installer.build_runner("bench", "benchmark", runner="docker", module_path=module_path)

    def test_install_venv_missing_system_dep(self, tmp_path: Path) -> None:
        """system-deps.txt lists a tool not installed -> RuntimeError."""
        module_path = _create_fake_package(tmp_path, with_requirements=False, with_setup=False, with_system_deps=True)
        # Override system-deps.txt with a definitely-missing tool
        parts = module_path.split(".")
        pkg_dir = tmp_path
        for part in parts[:-1]:
            pkg_dir = pkg_dir / part
        (pkg_dir / "system-deps.txt").write_text("nonexistent_tool_xyz_12345\n")
        importlib.invalidate_caches()

        installer = EnvironmentInstaller(base_dir=tmp_path / "envs")

        # Mock both which and dpkg to ensure the tool appears missing
        original_which = shutil.which

        def which_no_fake(name):
            if name == "nonexistent_tool_xyz_12345":
                return None
            return original_which(name)

        with mock.patch("exgentic.utils.installer.shutil.which", side_effect=which_no_fake):
            with mock.patch("exgentic.utils.installer._dpkg_installed", return_value=False):
                with pytest.raises(RuntimeError, match="nonexistent_tool_xyz_12345"):
                    installer.install("bench", "benchmark", module_path=module_path)


class TestCleanupOnFailedInstall:
    """Tests that partial state is cleaned up when venv install fails."""

    def test_cleanup_on_failed_install(self, tmp_path: Path) -> None:
        """Verify env dir is removed when install fails partway through."""
        module_path = _create_fake_package(tmp_path, with_requirements=False, with_setup=False)
        installer = EnvironmentInstaller(base_dir=tmp_path / "envs")
        env_dir = installer.env_path("bench", "benchmark")

        # Make _find_project_root return a path so the exgentic install
        # step runs, then have that subprocess.run call fail.
        original_run = subprocess.run

        def fail_exgentic_install(cmd, **kwargs):
            # Let venv creation succeed but fail the pip install step
            if isinstance(cmd, list) and "pip" in cmd and "install" in cmd:
                raise subprocess.CalledProcessError(1, cmd)
            return original_run(cmd, **kwargs)

        with mock.patch("subprocess.run", side_effect=fail_exgentic_install):
            with mock.patch("exgentic.utils.installer._find_project_root", return_value=Path("/fake/root")):
                with pytest.raises(subprocess.CalledProcessError):
                    installer.install("bench", "benchmark", module_path=module_path)

        # The env directory must be cleaned up after failure
        assert not env_dir.exists(), "env_dir should be removed after a failed install"


class TestContentCorrectness:
    """Tests for marker content and directory structure."""

    def test_installed_marker_is_runner_agnostic(self, tmp_path: Path) -> None:
        """The .installed marker does not contain a runner field."""
        module_path = _create_fake_package(tmp_path, with_requirements=False, with_setup=False)
        installer = EnvironmentInstaller(base_dir=tmp_path / "envs")

        env_dir = installer.install("bench", "benchmark", module_path=module_path)

        marker = env_dir / ".installed"
        assert marker.is_file()
        info = json.loads(marker.read_text())
        assert "runner" not in info
        assert "installed_at" in info
        # Verify installed_at is a valid ISO timestamp
        ts = datetime.fromisoformat(info["installed_at"])
        assert ts.tzinfo is not None  # must be timezone-aware

    def test_build_runner_docker_creates_docker_dir(self, tmp_path: Path) -> None:
        """build_runner(runner='docker') creates docker/ dir with image_tag."""
        module_path = _create_fake_package(tmp_path, with_requirements=False, with_setup=False)
        installer = EnvironmentInstaller(base_dir=tmp_path / "envs")

        installer.install("bench", "benchmark", module_path=module_path)

        def side_effect(cmd, **kwargs):
            if cmd[0] == "docker":
                if cmd[1:3] == ["image", "inspect"]:
                    return _docker_mock_result(returncode=1)
                return _docker_mock_result()
            return _real_subprocess_run(cmd, **kwargs)

        with mock.patch("subprocess.run", side_effect=side_effect):
            env_dir = installer.build_runner("bench", "benchmark", runner="docker", module_path=module_path)

        assert (env_dir / "docker" / "image_tag").is_file()
        tag = (env_dir / "docker" / "image_tag").read_text().strip()
        assert tag.startswith("exgentic-benchmark-bench:")

    def test_get_install_info_returns_metadata(self, tmp_path: Path) -> None:
        """get_install_info returns marker contents as a dict."""
        module_path = _create_fake_package(tmp_path, with_requirements=False, with_setup=False)
        installer = EnvironmentInstaller(base_dir=tmp_path / "envs")

        assert installer.get_install_info("bench", "benchmark") is None

        installer.install("bench", "benchmark", module_path=module_path)
        info = installer.get_install_info("bench", "benchmark")

        assert info is not None
        assert "installed_at" in info

    def test_env_path_correct_structure(self, tmp_path: Path) -> None:
        """Verify the directory layout after install: venv/, .installed present."""
        module_path = _create_fake_package(tmp_path, with_requirements=False, with_setup=True)
        installer = EnvironmentInstaller(base_dir=tmp_path / "envs")

        env_dir = installer.install("bench", "benchmark", module_path=module_path)

        assert (env_dir / "venv").is_dir()
        assert (env_dir / "venv" / "bin" / "python").exists()
        assert (env_dir / ".installed").is_file()
        # setup.sh creates data/ directory
        assert (env_dir / "data").is_dir()
        assert (env_dir / "data" / "setup_ran.txt").is_file()

    def test_install_without_requirements(self, tmp_path: Path) -> None:
        """Benchmark with no requirements.txt still installs successfully."""
        module_path = _create_fake_package(tmp_path, with_requirements=False, with_setup=False)
        installer = EnvironmentInstaller(base_dir=tmp_path / "envs")

        env_dir = installer.install("bench", "benchmark", module_path=module_path)

        assert installer.is_installed("bench", "benchmark")
        assert (env_dir / "venv" / "bin" / "python").exists()

    def test_install_without_setup_sh(self, tmp_path: Path) -> None:
        """Benchmark with no setup.sh still installs successfully."""
        module_path = _create_fake_package(tmp_path, with_requirements=True, with_setup=False)
        installer = EnvironmentInstaller(base_dir=tmp_path / "envs")

        env_dir = installer.install("bench", "benchmark", module_path=module_path)

        assert installer.is_installed("bench", "benchmark")
        assert (env_dir / "venv" / "bin" / "python").exists()

    def test_install_without_both(self, tmp_path: Path) -> None:
        """Bare benchmark (no requirements, no setup) installs."""
        module_path = _create_fake_package(tmp_path, with_requirements=False, with_setup=False)
        installer = EnvironmentInstaller(base_dir=tmp_path / "envs")

        env_dir = installer.install("bench", "benchmark", module_path=module_path)

        assert installer.is_installed("bench", "benchmark")
        assert (env_dir / "venv").is_dir()
        assert (env_dir / ".installed").is_file()


# ---------------------------------------------------------------------------
# New tests for build_runner / has_runner
# ---------------------------------------------------------------------------


class TestBuildRunnerVenv:
    """Tests for build_runner with runner='venv'."""

    def test_build_runner_venv_is_noop(self, tmp_path: Path) -> None:
        """build_runner(runner='venv') is a no-op since install created venv."""
        module_path = _create_fake_package(tmp_path, with_requirements=False, with_setup=False)
        installer = EnvironmentInstaller(base_dir=tmp_path / "envs")

        env_dir = installer.install("bench", "benchmark", module_path=module_path)
        venv_mtime = (env_dir / "venv" / "bin" / "python").stat().st_mtime

        result = installer.build_runner("bench", "benchmark", runner="venv", module_path=module_path)

        assert result == env_dir
        # venv must not have been recreated
        assert (env_dir / "venv" / "bin" / "python").stat().st_mtime == venv_mtime


class TestHasRunner:
    """Tests for has_runner."""

    def test_has_runner_venv_false_before_install(self, tmp_path: Path) -> None:
        installer = EnvironmentInstaller(base_dir=tmp_path / "envs")
        assert not installer.has_runner("bench", "benchmark", runner="venv")

    def test_has_runner_venv_true_after_install(self, tmp_path: Path) -> None:
        module_path = _create_fake_package(tmp_path, with_requirements=False, with_setup=False)
        installer = EnvironmentInstaller(base_dir=tmp_path / "envs")

        installer.install("bench", "benchmark", module_path=module_path)
        assert installer.has_runner("bench", "benchmark", runner="venv")

    def test_has_runner_docker_false_before_build(self, tmp_path: Path) -> None:
        module_path = _create_fake_package(tmp_path, with_requirements=False, with_setup=False)
        installer = EnvironmentInstaller(base_dir=tmp_path / "envs")

        installer.install("bench", "benchmark", module_path=module_path)
        assert not installer.has_runner("bench", "benchmark", runner="docker")

    def test_has_runner_docker_true_after_build(self, tmp_path: Path) -> None:
        module_path = _create_fake_package(tmp_path, with_requirements=False, with_setup=False)
        installer = EnvironmentInstaller(base_dir=tmp_path / "envs")

        installer.install("bench", "benchmark", module_path=module_path)

        def side_effect(cmd, **kwargs):
            if cmd[0] == "docker":
                if cmd[1:3] == ["image", "inspect"]:
                    return _docker_mock_result(returncode=1)
                return _docker_mock_result()
            return _real_subprocess_run(cmd, **kwargs)

        with mock.patch("subprocess.run", side_effect=side_effect):
            installer.build_runner("bench", "benchmark", runner="docker", module_path=module_path)

        assert installer.has_runner("bench", "benchmark", runner="docker")


class TestBuildRunnerAutoInstalls:
    """Tests that build_runner auto-calls install() if not installed."""

    def test_build_runner_auto_installs(self, tmp_path: Path) -> None:
        """build_runner calls install() if not already installed."""
        module_path = _create_fake_package(tmp_path, with_requirements=False, with_setup=False)
        installer = EnvironmentInstaller(base_dir=tmp_path / "envs")

        assert not installer.is_installed("bench", "benchmark")

        env_dir = installer.build_runner("bench", "benchmark", runner="venv", module_path=module_path)

        assert installer.is_installed("bench", "benchmark")
        assert (env_dir / "venv" / "bin" / "python").exists()

    def test_build_runner_docker_auto_installs(self, tmp_path: Path) -> None:
        """build_runner(runner='docker') auto-installs then builds docker."""
        module_path = _create_fake_package(tmp_path, with_requirements=False, with_setup=False)
        installer = EnvironmentInstaller(base_dir=tmp_path / "envs")

        assert not installer.is_installed("bench", "benchmark")

        def side_effect(cmd, **kwargs):
            if cmd[0] == "docker":
                if cmd[1:3] == ["image", "inspect"]:
                    return _docker_mock_result(returncode=1)
                return _docker_mock_result()
            return _real_subprocess_run(cmd, **kwargs)

        with mock.patch("subprocess.run", side_effect=side_effect):
            env_dir = installer.build_runner("bench", "benchmark", runner="docker", module_path=module_path)

        assert installer.is_installed("bench", "benchmark")
        assert (env_dir / "venv" / "bin" / "python").exists()
        assert installer.has_runner("bench", "benchmark", runner="docker")


class TestUninstallRemovesAllRunners:
    """Tests that uninstall removes data + all runner environments."""

    def test_uninstall_removes_all_runners(self, tmp_path: Path) -> None:
        """Uninstall removes venv + docker image + all data."""
        module_path = _create_fake_package(tmp_path, with_requirements=False, with_setup=False)
        installer = EnvironmentInstaller(base_dir=tmp_path / "envs")

        installer.install("bench", "benchmark", module_path=module_path)

        # Simulate docker runner having been built
        env_dir = installer.env_path("bench", "benchmark")
        docker_dir = env_dir / "docker"
        docker_dir.mkdir()
        image_tag = "exgentic-benchmark-bench:abc123"
        (docker_dir / "image_tag").write_text(image_tag)

        assert installer.has_runner("bench", "benchmark", runner="venv")
        assert installer.has_runner("bench", "benchmark", runner="docker")

        rmi_calls: list[list[str]] = []

        def side_effect(cmd, **kwargs):
            if cmd[0] == "docker" and cmd[1] == "rmi":
                rmi_calls.append(list(cmd))
                return _docker_mock_result()
            return _real_subprocess_run(cmd, **kwargs)

        with mock.patch("subprocess.run", side_effect=side_effect):
            installer.uninstall("bench", "benchmark")

        assert not env_dir.exists()
        assert not installer.is_installed("bench", "benchmark")
        assert not installer.has_runner("bench", "benchmark", runner="venv")
        assert not installer.has_runner("bench", "benchmark", runner="docker")
        assert len(rmi_calls) == 1
        assert rmi_calls[0] == ["docker", "rmi", image_tag]
