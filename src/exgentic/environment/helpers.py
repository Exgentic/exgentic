# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Shared helpers for environment management."""

from __future__ import annotations

import os
import shutil
import subprocess
from importlib import resources
from pathlib import Path


def require_uv() -> str:
    """Return the path to ``uv``, raising a clear error if not found."""
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError(
            "Could not find 'uv' on PATH. " "Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh"
        )
    return uv


def build_subprocess_env() -> dict:
    """Build an env dict for subprocess calls."""
    env = os.environ.copy()
    env.pop("VIRTUAL_ENV", None)
    env["GIT_LFS_SKIP_SMUDGE"] = "1"
    return env


def install_project(uv: str, python_target: str, project_root: Path, env: dict) -> None:
    """Install a Python project from *project_root* into the target Python."""
    subprocess.run(
        [uv, "pip", "install", "--python", python_target, "--no-cache", str(project_root)],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )


def install_packages(uv: str, python_target: str, packages: list[str], env: dict) -> None:
    """Install packages into the target Python environment."""
    subprocess.run(
        [uv, "pip", "install", "--python", python_target, "--no-cache", *packages],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )


def install_requirements(uv: str, python_target: str, module_path: str, env: dict) -> None:
    """Find and install requirements.txt into the target Python."""
    req_path = find_package_file(module_path, "requirements.txt")
    if req_path is None:
        return
    lines = [
        line.strip() for line in req_path.read_text().splitlines() if line.strip() and not line.strip().startswith("#")
    ]
    if not lines:
        return
    subprocess.run(
        [uv, "pip", "install", "--python", python_target, "-r", str(req_path)],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )


def run_setup_sh(module_path: str, env_dir: Path, *, venv_dir: Path | None = None) -> None:
    """Run setup.sh in the environment directory.

    Args:
        module_path: Dotted module path for locating setup.sh.
        env_dir: Working directory for setup.sh execution.
        venv_dir: If set, activates the venv in the subprocess.
    """
    setup_path = find_package_file(module_path, "setup.sh")
    if setup_path is None:
        return
    env = os.environ.copy()
    if venv_dir is not None:
        env["VIRTUAL_ENV"] = str(venv_dir)
        env["PATH"] = str(venv_dir / "bin") + os.pathsep + env.get("PATH", "")
    subprocess.run(["bash", str(setup_path)], check=True, cwd=str(env_dir), env=env)


def find_package_file(module_path: str, filename: str) -> Path | None:
    """Locate *filename* in the package directory for *module_path*."""
    parts = module_path.split(".")
    for depth in range(len(parts) - 1, 1, -1):
        package = ".".join(parts[:depth])
        try:
            candidate = resources.files(package) / filename
        except Exception:
            continue
        if candidate.is_file():
            return Path(str(candidate))
    return None


def read_lines(path: Path) -> list[str]:
    """Read non-empty, non-comment lines from *path*."""
    return [line.strip() for line in path.read_text().splitlines() if line.strip() and not line.strip().startswith("#")]


def validate_system_deps(module_path: str) -> None:
    """Check that system packages from ``system-deps.txt`` are installed."""
    sysdeps_path = find_package_file(module_path, "system-deps.txt")
    if sysdeps_path is None:
        return
    pkgs = read_lines(sysdeps_path)
    if not pkgs:
        return
    missing = [p for p in pkgs if shutil.which(p) is None and not _dpkg_installed(p)]
    if missing:
        raise RuntimeError(
            f"Missing system packages required for install: {', '.join(missing)}. "
            "Install them with: sudo apt-get install -y " + " ".join(missing)
        )


def _dpkg_installed(package: str) -> bool:
    """Return *True* if *package* is installed via dpkg."""
    if shutil.which("dpkg") is None:
        return False
    result = subprocess.run(
        ["dpkg", "-s", package],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0
