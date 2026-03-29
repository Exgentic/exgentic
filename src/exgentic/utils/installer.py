# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Manages isolated environments for benchmarks and agents."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
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
        runner: str = "venv",
        force: bool = False,
        module_path: str | None = None,
    ) -> Path:
        """Create a complete ready-to-run environment.

        Args:
            slug: Short identifier for the benchmark / agent.
            kind: ``"benchmark"`` or ``"agent"``.
            runner: ``"venv"`` (default) or ``"docker"``.
            force: Re-create even if already installed.
            module_path: Dotted module path used to locate package resources
                (``requirements.txt``, ``setup.sh``, ``system-deps.txt``).
                When *None*, the resource-lookup steps are skipped.

        Returns:
            The environment directory path.
        """
        if runner == "docker":
            return self._install_docker(slug, kind, force=force, module_path=module_path)
        return self._install_venv(slug, kind, force=force, module_path=module_path)

    # ------------------------------------------------------------------
    # Venv-based install
    # ------------------------------------------------------------------

    def _install_venv(
        self,
        slug: str,
        kind: str,
        *,
        force: bool = False,
        module_path: str | None = None,
    ) -> Path:
        """Create a venv-based environment.

        Steps:
            1. Create dir at ``base_dir/{kind}s/{slug}/``
            2. Create a Python venv inside it
            3. Install *exgentic* into the venv
            4. Find and install ``requirements.txt`` into the venv
            5. Validate ``system-deps.txt`` packages are installed on host
            6. Find and run ``setup.sh`` (with ``EXGENTIC_CACHE_DIR`` set)
            7. Write ``.installed`` marker
        """
        if not force and self.is_installed(slug, kind):
            return self.env_path(slug, kind)

        env_dir = self.env_path(slug, kind)
        if env_dir.exists():
            shutil.rmtree(env_dir)
        env_dir.mkdir(parents=True, exist_ok=True)

        try:
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

            # Build a consistent env dict for subprocess calls:
            # - remove VIRTUAL_ENV so uv targets the right venv
            # - skip LFS smudge to speed up git-based installs
            env = os.environ.copy()
            env.pop("VIRTUAL_ENV", None)
            env["GIT_LFS_SKIP_SMUDGE"] = "1"

            # 3  -- install exgentic into the venv
            project_root = _find_project_root()
            if project_root is not None:
                subprocess.run(
                    [uv_bin, "pip", "install", "--python", venv_python, "--no-cache", str(project_root)],
                    check=True,
                    capture_output=True,
                    text=True,
                    env=env,
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
                        subprocess.run(
                            [uv_bin, "pip", "install", "--python", venv_python, "-r", str(req_path)],
                            check=True,
                            capture_output=True,
                            text=True,
                            env=env,
                        )

            # 5  -- validate system-deps.txt (if found)
            if module_path is not None:
                _validate_system_deps(module_path)

            # 6  -- run setup.sh (if found)
            # setup.sh intentionally prints to stdout for progress visibility
            if module_path is not None:
                setup_path = _find_package_file(module_path, "setup.sh")
                if setup_path is not None:
                    setup_env = env.copy()
                    setup_env["EXGENTIC_CACHE_DIR"] = str(env_dir)
                    setup_env["VIRTUAL_ENV"] = str(venv_dir)
                    venv_bin = str(venv_dir / "bin")
                    setup_env["PATH"] = venv_bin + os.pathsep + setup_env.get("PATH", "")
                    subprocess.run(["bash", str(setup_path)], check=True, env=setup_env)

            # 7  -- write .installed marker
            (env_dir / ".installed").write_text("")
        except BaseException:
            shutil.rmtree(env_dir, ignore_errors=True)
            raise

        return env_dir

    # ------------------------------------------------------------------
    # Docker-based install
    # ------------------------------------------------------------------

    def _install_docker(
        self,
        slug: str,
        kind: str,
        *,
        force: bool = False,
        module_path: str | None = None,
    ) -> Path:
        """Build a Docker image for the environment.

        The image tag is derived from a content hash of the package files
        (``requirements.txt``, ``setup.sh``, ``system-deps.txt``) so that
        identical inputs produce the same tag and rebuilds are skipped.
        """
        env_dir = self.env_path(slug, kind)

        # Gather package files ------------------------------------------
        req_path = _find_package_file(module_path, "requirements.txt") if module_path else None
        setup_path = _find_package_file(module_path, "setup.sh") if module_path else None
        sysdeps_path = _find_package_file(module_path, "system-deps.txt") if module_path else None

        # Content hash for image tag ------------------------------------
        hash_parts: list[str] = []
        if req_path is not None:
            hash_parts.append(req_path.read_text())
        if setup_path is not None:
            hash_parts.append(setup_path.read_text())
        if sysdeps_path is not None:
            hash_parts.append(sysdeps_path.read_text())
        h = hashlib.sha256()
        for part in hash_parts:
            h.update(part.encode())
            h.update(b"\x00")
        content_hash = h.hexdigest()[:12]
        image_tag = f"exgentic-{kind}-{slug}:{content_hash}"

        # Reuse existing image unless force -----------------------------
        if not force:
            result = subprocess.run(
                ["docker", "image", "inspect", image_tag],
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                # Ensure cache dir and marker exist
                env_dir.mkdir(parents=True, exist_ok=True)
                (env_dir / ".installed").write_text(json.dumps({"runner": "docker", "image": image_tag}))
                return env_dir

        # Build Dockerfile ----------------------------------------------
        tmp_dir = Path(tempfile.mkdtemp(prefix="exgentic-docker-"))
        try:
            lines = [
                "FROM python:3.12-slim",
            ]

            # System dependencies
            if sysdeps_path is not None:
                pkgs = _read_lines(sysdeps_path)
                if pkgs:
                    lines.append(
                        "RUN apt-get update && apt-get install -y " + " ".join(pkgs) + " && rm -rf /var/lib/apt/lists/*"
                    )

            lines.extend(
                [
                    "RUN pip install --no-cache-dir uv",
                    "ENV UV_SYSTEM_PYTHON=true",
                ]
            )

            # Requirements
            if req_path is not None:
                shutil.copy2(req_path, tmp_dir / "requirements.txt")
                lines.append("COPY requirements.txt /tmp/")
                lines.append("RUN GIT_LFS_SKIP_SMUDGE=1 uv pip install --no-cache -r /tmp/requirements.txt")

            # Setup script
            if setup_path is not None:
                shutil.copy2(setup_path, tmp_dir / "setup.sh")
                lines.append("COPY setup.sh /tmp/")
                lines.append("RUN bash /tmp/setup.sh")

            dockerfile_content = "\n".join(lines) + "\n"
            (tmp_dir / "Dockerfile").write_text(dockerfile_content)

            # Build
            subprocess.run(
                ["docker", "build", "-t", image_tag, str(tmp_dir)],
                check=True,
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        # Write marker --------------------------------------------------
        env_dir.mkdir(parents=True, exist_ok=True)
        (env_dir / ".installed").write_text(json.dumps({"runner": "docker", "image": image_tag}))
        return env_dir

    def uninstall(self, slug: str, kind: str = "benchmark") -> None:
        """Remove the environment directory entirely.

        For docker-based environments, also removes the Docker image.
        """
        env_dir = self.env_path(slug, kind)
        if not env_dir.exists():
            return

        marker = env_dir / ".installed"
        if marker.is_file():
            try:
                info = json.loads(marker.read_text())
                if info.get("runner") == "docker" and info.get("image"):
                    subprocess.run(
                        ["docker", "rmi", info["image"]],
                        check=False,
                        capture_output=True,
                        text=True,
                    )
            except (json.JSONDecodeError, KeyError):
                pass

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


def _read_lines(path: Path) -> list[str]:
    """Read non-empty, non-comment lines from *path*."""
    return [line.strip() for line in path.read_text().splitlines() if line.strip() and not line.strip().startswith("#")]


def _validate_system_deps(module_path: str) -> None:
    """Check that system packages from ``system-deps.txt`` are installed.

    Raises :class:`RuntimeError` with a clear message listing missing
    packages when any are not found on the host.
    """
    sysdeps_path = _find_package_file(module_path, "system-deps.txt")
    if sysdeps_path is None:
        return
    pkgs = _read_lines(sysdeps_path)
    if not pkgs:
        return
    missing = [p for p in pkgs if shutil.which(p) is None and not _dpkg_installed(p)]
    if missing:
        raise RuntimeError(
            f"Missing system packages required for venv install: {', '.join(missing)}. "
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
