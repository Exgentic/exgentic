# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Manages isolated environments for benchmarks and agents."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from importlib import resources
from pathlib import Path


class EnvironmentInstaller:
    """Manages isolated environments for benchmarks and agents.

    Each benchmark/agent gets its own self-contained environment at:
      ``{base_dir}/{kind}s/{slug}/``

    The environment contains:
      - ``venv/`` -- isolated Python venv with all dependencies
      - ``data/`` -- benchmark data files (populated by setup.sh)
      - ``.installed`` -- marker file
    """

    def __init__(self, base_dir: Path | None = None) -> None:
        self.base_dir = base_dir or Path.home() / ".exgentic"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def install(
        self,
        slug: str,
        kind: str = "benchmark",
        *,
        force: bool = False,
        module_path: str | None = None,
    ) -> Path:
        """Create a complete ready-to-run environment.

        Steps:
            1. Create dir at ``base_dir/{kind}s/{slug}/``
            2. Create a Python venv inside it
            3. Install *exgentic* into the venv
            4. Find and install ``requirements.txt`` into the venv
            5. Find and run ``setup.sh`` (with ``EXGENTIC_CACHE_DIR`` set)
            6. Write ``.installed`` marker

        Args:
            slug: Short identifier for the benchmark / agent.
            kind: ``"benchmark"`` or ``"agent"``.
            force: Re-create even if already installed.
            module_path: Dotted module path used to locate package resources
                (``requirements.txt``, ``setup.sh``).  When *None*, the
                resource-lookup steps are skipped.

        Returns:
            The environment directory path.
        """
        if not force and self.is_installed(slug, kind):
            return self.env_path(slug, kind)

        env_dir = self.env_path(slug, kind)
        if env_dir.exists():
            shutil.rmtree(env_dir)
        env_dir.mkdir(parents=True, exist_ok=True)

        uv_bin = _require_uv()

        # 1 & 2  -- create venv
        venv_dir = env_dir / "venv"
        subprocess.run(
            [
                uv_bin,
                "venv",
                str(venv_dir),
                "--python",
                f"{sys.version_info.major}.{sys.version_info.minor}",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        venv_python = str(venv_dir / "bin" / "python")

        # 3  -- install exgentic into the venv
        project_root = _find_project_root()
        if project_root is not None:
            subprocess.run(
                [uv_bin, "pip", "install", "--python", venv_python, "--no-cache", str(project_root)],
                check=True,
                capture_output=True,
                text=True,
            )

        # 4  -- install requirements.txt (if found)
        if module_path is not None:
            req_path = _find_package_file(module_path, "requirements.txt")
            if req_path is not None:
                lines = [
                    line.strip()
                    for line in req_path.read_text().splitlines()
                    if line.strip() and not line.strip().startswith("#")
                ]
                if lines:
                    env = os.environ.copy()
                    env.pop("VIRTUAL_ENV", None)
                    env["GIT_LFS_SKIP_SMUDGE"] = "1"
                    subprocess.run(
                        [uv_bin, "pip", "install", "--python", venv_python, "-r", str(req_path)],
                        check=True,
                        env=env,
                    )

        # 5  -- run setup.sh (if found)
        if module_path is not None:
            setup_path = _find_package_file(module_path, "setup.sh")
            if setup_path is not None:
                env = os.environ.copy()
                env["EXGENTIC_CACHE_DIR"] = str(env_dir)
                env["VIRTUAL_ENV"] = str(venv_dir)
                venv_bin = str(venv_dir / "bin")
                env["PATH"] = venv_bin + os.pathsep + env.get("PATH", "")
                subprocess.run(["bash", str(setup_path)], check=True, env=env)

        # 6  -- write .installed marker
        (env_dir / ".installed").write_text("")

        return env_dir

    def uninstall(self, slug: str, kind: str = "benchmark") -> None:
        """Remove the environment directory entirely."""
        env_dir = self.env_path(slug, kind)
        if env_dir.exists():
            shutil.rmtree(env_dir)

    def is_installed(self, slug: str, kind: str = "benchmark") -> bool:
        """Check if the ``.installed`` marker exists."""
        return (self.env_path(slug, kind) / ".installed").is_file()

    def env_path(self, slug: str, kind: str = "benchmark") -> Path:
        """Return the environment directory path."""
        return self.base_dir / f"{kind}s" / slug

    def list_installed(self, kind: str | None = None) -> list[str]:
        """List all installed environment slugs.

        Args:
            kind: ``"benchmark"``, ``"agent"``, or *None* for both.
        """
        kinds = [kind] if kind else ["benchmark", "agent"]
        slugs: list[str] = []
        for k in kinds:
            kind_dir = self.base_dir / f"{k}s"
            if not kind_dir.is_dir():
                continue
            for child in sorted(kind_dir.iterdir()):
                if child.is_dir() and (child / ".installed").is_file():
                    slugs.append(child.name)
        return slugs


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _require_uv() -> str:
    """Return the path to ``uv``, raising a clear error if not found."""
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError(
            "Could not find 'uv' on PATH. " "Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh"
        )
    return uv


def _find_project_root() -> Path | None:
    """Walk up from CWD looking for ``pyproject.toml``."""
    current = Path.cwd()
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    return None


def _find_package_file(module_path: str, filename: str) -> Path | None:
    """Locate *filename* in the package directory for *module_path*.

    Walks up from the deepest package toward the root until it finds the
    file.  This mirrors the logic in ``api.py`` but is self-contained.
    """
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
