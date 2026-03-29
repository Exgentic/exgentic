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
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path


class EnvironmentManager:
    """Manages isolated environments for benchmarks and agents.

    Each benchmark/agent gets its own directory at:
      ``{base_dir}/{kind}s/{slug}/``

    Multiple environment types can coexist for the same benchmark/agent:
      - ``venv`` -- isolated Python venv with dependencies
      - ``local`` -- dependencies installed into the current Python
      - ``docker`` -- Docker image with dependencies baked in

    The ``.installed`` marker file tracks which environment types are
    installed as a JSON dict keyed by environment type.
    """

    MARKER_FILE = ".installed"
    VALID_ENV_TYPES = ("venv", "local", "docker")

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
        env_type: str = "venv",
        force: bool = False,
        module_path: str | None = None,
        venv_packages: list[str] | None = None,
    ) -> Path:
        """Install a benchmark or agent environment.

        Args:
            slug: Short identifier (e.g. ``"tau2"``).
            kind: ``"benchmark"`` or ``"agent"``.
            env_type: ``"venv"`` (default), ``"local"``, or ``"docker"``.
            force: Re-create even if already installed.
            module_path: Dotted module path for locating package resources.
            venv_packages: Extra packages to install into the venv
                (e.g. ``["exgentic"]``).  Only used for ``env_type="venv"``.

        Returns:
            The environment directory path.
        """
        self._validate_env_type(env_type)
        if env_type == "docker":
            return self._install_docker(slug, kind, force=force, module_path=module_path)
        if env_type == "local":
            return self._install_local(slug, kind, force=force, module_path=module_path)
        return self._install_venv(slug, kind, force=force, module_path=module_path, venv_packages=venv_packages)

    def uninstall(
        self,
        slug: str,
        kind: str = "benchmark",
        *,
        env_type: str | None = None,
    ) -> None:
        """Remove an installed environment.

        Args:
            slug: Short identifier.
            kind: ``"benchmark"`` or ``"agent"``.
            env_type: Specific type to remove, or *None* to remove all.
        """
        if env_type is not None:
            self._validate_env_type(env_type)

        env_dir = self.env_path(slug, kind)
        if not env_dir.exists():
            return

        if env_type is None:
            marker = self._read_marker(slug, kind)
            docker_info = marker.get("docker", {})
            if docker_info.get("image"):
                subprocess.run(
                    ["docker", "rmi", docker_info["image"]],
                    check=False,
                    capture_output=True,
                    text=True,
                )
            shutil.rmtree(env_dir)
            return

        if env_type == "venv":
            venv_dir = env_dir / "venv"
            if venv_dir.exists():
                shutil.rmtree(venv_dir)

        elif env_type == "docker":
            marker = self._read_marker(slug, kind)
            docker_info = marker.get("docker", {})
            if docker_info.get("image"):
                subprocess.run(
                    ["docker", "rmi", docker_info["image"]],
                    check=False,
                    capture_output=True,
                    text=True,
                )

        # For "local", we can't remove deps from the global env.
        self._remove_marker_entry(slug, kind, env_type)

        # If no env types remain, clean up the directory.
        if not self._read_marker(slug, kind) and env_dir.exists():
            shutil.rmtree(env_dir)

    def is_installed(
        self,
        slug: str,
        kind: str = "benchmark",
        *,
        env_type: str | None = None,
    ) -> bool:
        """Check if an environment is installed.

        Args:
            slug: Short identifier.
            kind: ``"benchmark"`` or ``"agent"``.
            env_type: Check a specific type, or *None* for any type.
        """
        if env_type is not None:
            self._validate_env_type(env_type)
        marker = self._read_marker(slug, kind)
        if env_type is None:
            return bool(marker)
        return env_type in marker

    def get_install_info(self, slug: str, kind: str = "benchmark") -> dict | None:
        """Return installation info or *None* if not installed.

        Returns a dict with ``slug``, ``kind``, and ``environments``
        (a dict keyed by env_type with install details).
        """
        marker = self._read_marker(slug, kind)
        if not marker:
            return None
        return {"slug": slug, "kind": kind, "environments": marker}

    def list_installed(self, kind: str | None = None) -> list[dict]:
        """List all installed environments with details.

        Args:
            kind: ``"benchmark"``, ``"agent"``, or *None* for both.

        Returns:
            List of dicts with ``slug``, ``kind``, and ``environments``.
        """
        kinds = [kind] if kind else ["benchmark", "agent"]
        result: list[dict] = []
        for k in kinds:
            kind_dir = self.base_dir / f"{k}s"
            if not kind_dir.is_dir():
                continue
            for child in sorted(kind_dir.iterdir()):
                if not child.is_dir():
                    continue
                marker_path = child / self.MARKER_FILE
                if not marker_path.is_file():
                    continue
                try:
                    marker = json.loads(marker_path.read_text())
                except (json.JSONDecodeError, ValueError):
                    continue
                if marker:
                    result.append({"slug": child.name, "kind": k, "environments": marker})
        return result

    def env_path(self, slug: str, kind: str = "benchmark") -> Path:
        """Return the environment directory path."""
        return self.base_dir / f"{kind}s" / slug

    def venv_python(self, slug: str, kind: str = "benchmark") -> str:
        """Return the path to the venv Python binary."""
        return str(self.env_path(slug, kind) / "venv" / "bin" / "python")

    def docker_image(self, slug: str, kind: str = "benchmark") -> str | None:
        """Return the Docker image tag, or *None* if not installed."""
        marker = self._read_marker(slug, kind)
        docker_info = marker.get("docker", {})
        return docker_info.get("image")

    def local_python(self, slug: str, kind: str = "benchmark") -> str | None:
        """Return the Python path used for local install, or *None*."""
        marker = self._read_marker(slug, kind)
        local_info = marker.get("local", {})
        return local_info.get("python")

    def _validate_env_type(self, env_type: str) -> None:
        if env_type not in self.VALID_ENV_TYPES:
            raise ValueError(f"Invalid env_type {env_type!r}. Must be one of: {', '.join(self.VALID_ENV_TYPES)}")

    # ------------------------------------------------------------------
    # Marker management
    # ------------------------------------------------------------------

    def _read_marker(self, slug: str, kind: str) -> dict:
        marker = self.env_path(slug, kind) / self.MARKER_FILE
        if not marker.is_file():
            return {}
        try:
            data = json.loads(marker.read_text())
            return data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, ValueError):
            return {}

    def _write_marker(self, slug: str, kind: str, data: dict) -> None:
        env_dir = self.env_path(slug, kind)
        env_dir.mkdir(parents=True, exist_ok=True)
        (env_dir / self.MARKER_FILE).write_text(json.dumps(data, indent=2))

    def _add_marker_entry(self, slug: str, kind: str, env_type: str, info: dict) -> None:
        data = self._read_marker(slug, kind)
        data[env_type] = info
        self._write_marker(slug, kind, data)

    def _remove_marker_entry(self, slug: str, kind: str, env_type: str) -> None:
        data = self._read_marker(slug, kind)
        data.pop(env_type, None)
        if data:
            self._write_marker(slug, kind, data)
        else:
            marker = self.env_path(slug, kind) / self.MARKER_FILE
            if marker.exists():
                marker.unlink()

    # ------------------------------------------------------------------
    # Install: venv
    # ------------------------------------------------------------------

    def _install_venv(
        self,
        slug: str,
        kind: str,
        *,
        force: bool,
        module_path: str | None,
        venv_packages: list[str] | None = None,
    ) -> Path:
        if not force and self.is_installed(slug, kind, env_type="venv"):
            return self.env_path(slug, kind)

        env_dir = self.env_path(slug, kind)
        env_dir.mkdir(parents=True, exist_ok=True)
        venv_dir = env_dir / "venv"

        if venv_dir.exists():
            shutil.rmtree(venv_dir)
        self._remove_marker_entry(slug, kind, "venv")

        try:
            uv = _require_uv()

            subprocess.run(
                [uv, "venv", str(venv_dir), "--python", f"{sys.version_info.major}.{sys.version_info.minor}"],
                check=True,
                capture_output=True,
                text=True,
            )

            venv_py = str(venv_dir / "bin" / "python")
            env = _build_subprocess_env()

            if venv_packages:
                _install_packages(uv, venv_py, venv_packages, env)

            if module_path is not None:
                _install_requirements(uv, venv_py, module_path, env)
                _validate_system_deps(module_path)
                _run_setup_sh(module_path, env_dir, venv_dir=venv_dir)

            self._add_marker_entry(slug, kind, "venv", {"installed_at": _now_iso()})
        except BaseException:
            if venv_dir.exists():
                shutil.rmtree(venv_dir, ignore_errors=True)
            raise

        return env_dir

    # ------------------------------------------------------------------
    # Install: local
    # ------------------------------------------------------------------

    def _install_local(
        self,
        slug: str,
        kind: str,
        *,
        force: bool,
        module_path: str | None,
    ) -> Path:
        if not force and self.is_installed(slug, kind, env_type="local"):
            return self.env_path(slug, kind)

        env_dir = self.env_path(slug, kind)
        env_dir.mkdir(parents=True, exist_ok=True)
        self._remove_marker_entry(slug, kind, "local")

        if module_path is not None:
            uv = _require_uv()
            env = _build_subprocess_env()
            _install_requirements(uv, sys.executable, module_path, env)
            _validate_system_deps(module_path)
            _run_setup_sh(module_path, env_dir)

        self._add_marker_entry(
            slug,
            kind,
            "local",
            {
                "installed_at": _now_iso(),
                "python": sys.executable,
            },
        )

        return env_dir

    # ------------------------------------------------------------------
    # Install: docker
    # ------------------------------------------------------------------

    def _install_docker(
        self,
        slug: str,
        kind: str,
        *,
        force: bool,
        module_path: str | None,
    ) -> Path:
        if not force and self.is_installed(slug, kind, env_type="docker"):
            return self.env_path(slug, kind)

        env_dir = self.env_path(slug, kind)
        self._remove_marker_entry(slug, kind, "docker")

        # Gather package files
        req_path = _find_package_file(module_path, "requirements.txt") if module_path else None
        setup_path = _find_package_file(module_path, "setup.sh") if module_path else None
        sysdeps_path = _find_package_file(module_path, "system-deps.txt") if module_path else None

        # Content hash for image tag
        h = hashlib.sha256()
        for path in (req_path, setup_path, sysdeps_path):
            if path is not None:
                h.update(path.read_text().encode())
            h.update(b"\x00")
        image_tag = f"exgentic-{kind}-{slug}:{h.hexdigest()[:12]}"

        # Reuse existing image unless force
        if not force:
            result = subprocess.run(
                ["docker", "image", "inspect", image_tag],
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                env_dir.mkdir(parents=True, exist_ok=True)
                self._add_marker_entry(
                    slug,
                    kind,
                    "docker",
                    {
                        "installed_at": _now_iso(),
                        "image": image_tag,
                    },
                )
                return env_dir

        # Build Dockerfile
        tmp_dir = Path(tempfile.mkdtemp(prefix="exgentic-docker-"))
        try:
            lines = ["FROM python:3.12-slim"]

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

            if req_path is not None:
                shutil.copy2(req_path, tmp_dir / "requirements.txt")
                lines.append("COPY requirements.txt /tmp/")
                lines.append("RUN GIT_LFS_SKIP_SMUDGE=1 uv pip install --no-cache -r /tmp/requirements.txt")

            if setup_path is not None:
                shutil.copy2(setup_path, tmp_dir / "setup.sh")
                lines.append("COPY setup.sh /tmp/")
                lines.append("RUN bash /tmp/setup.sh")

            (tmp_dir / "Dockerfile").write_text("\n".join(lines) + "\n")

            subprocess.run(
                ["docker", "build", "-t", image_tag, str(tmp_dir)],
                check=True,
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        env_dir.mkdir(parents=True, exist_ok=True)
        self._add_marker_entry(
            slug,
            kind,
            "docker",
            {
                "installed_at": _now_iso(),
                "image": image_tag,
            },
        )
        return env_dir


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------


def _now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _require_uv() -> str:
    """Return the path to ``uv``, raising a clear error if not found."""
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError(
            "Could not find 'uv' on PATH. " "Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh"
        )
    return uv


def _build_subprocess_env() -> dict:
    """Build an env dict for subprocess calls."""
    env = os.environ.copy()
    env.pop("VIRTUAL_ENV", None)
    env["GIT_LFS_SKIP_SMUDGE"] = "1"
    return env


def _install_packages(uv: str, python_target: str, packages: list[str], env: dict) -> None:
    """Install packages into the target Python environment."""
    subprocess.run(
        [uv, "pip", "install", "--python", python_target, "--no-cache", *packages],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )


def _install_requirements(uv: str, python_target: str, module_path: str, env: dict) -> None:
    """Find and install requirements.txt into the target Python."""
    req_path = _find_package_file(module_path, "requirements.txt")
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


def _run_setup_sh(module_path: str, env_dir: Path, *, venv_dir: Path | None = None) -> None:
    """Run setup.sh with the appropriate environment variables."""
    setup_path = _find_package_file(module_path, "setup.sh")
    if setup_path is None:
        return
    env = os.environ.copy()
    env["EXGENTIC_CACHE_DIR"] = str(env_dir)
    if venv_dir is not None:
        env["VIRTUAL_ENV"] = str(venv_dir)
        env["PATH"] = str(venv_dir / "bin") + os.pathsep + env.get("PATH", "")
    subprocess.run(["bash", str(setup_path)], check=True, env=env)


def _find_package_file(module_path: str, filename: str) -> Path | None:
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


def _read_lines(path: Path) -> list[str]:
    """Read non-empty, non-comment lines from *path*."""
    return [line.strip() for line in path.read_text().splitlines() if line.strip() and not line.strip().startswith("#")]


def _validate_system_deps(module_path: str) -> None:
    """Check that system packages from ``system-deps.txt`` are installed."""
    sysdeps_path = _find_package_file(module_path, "system-deps.txt")
    if sysdeps_path is None:
        return
    pkgs = _read_lines(sysdeps_path)
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
