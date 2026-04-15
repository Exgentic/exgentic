# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Docker environment: builds a Docker image with dependencies baked in."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import tomllib

from .helpers import find_package_file, get_exgentic_version, read_lines

_DOCKER_CLI_VERSION = "27.5.1"
_DOCKER_CLI_RUN = (
    "RUN apt-get update && apt-get install -y --no-install-recommends curl"
    " && rm -rf /var/lib/apt/lists/*"
    f" && ARCH=$(uname -m)"
    f" && curl -fsSL https://download.docker.com/linux/static/stable/${{ARCH}}/docker-{_DOCKER_CLI_VERSION}.tgz"
    " | tar xz --strip-components=1 -C /usr/local/bin docker/docker"
)


class DockerBackend:
    """Backend that builds a Docker image with dependencies baked in.

    When *project_root* is provided the build is split into two images:

    1. **Base image** (``exgentic-base:{hash}``) — Python + git + uv + the
       project installed via a two-layer cache-friendly build.  This image is
       shared across all environments that use the same project root and is
       only rebuilt when the project's ``pyproject.toml`` or
       ``_IMAGE_VERSION`` changes.

    2. **Bench image** (``{name}:{hash}``) — ``FROM exgentic-base`` with
       benchmark-specific system deps, requirements, extra packages, an
       optional Docker CLI (for sibling-container benchmarks), and the
       setup script.

    When *project_root* is ``None`` the legacy single-image build is used.
    """

    _IMAGE_VERSION = "v1"

    def install(
        self,
        env_dir: Path,
        *,
        module_path: str | None = None,
        **kwargs: object,
    ) -> dict:
        """Build a Docker image for the environment.

        Args:
            env_dir: Root directory for this environment.
            module_path: Dotted module path for locating package resources.
            **kwargs: Accepts ``name``, ``force``, ``project_root`` (Path),
                ``packages`` (list[str]), and ``docker_socket`` (bool).

        Returns:
            Marker data with ``image`` tag.
        """
        name: str = kwargs.get("name", env_dir.name)  # type: ignore[assignment]
        force: bool = kwargs.get("force", False)  # type: ignore[assignment]
        project_root: Path | None = kwargs.get("project_root")  # type: ignore[assignment]
        packages: list[str] | None = kwargs.get("packages")  # type: ignore[assignment]
        docker_socket: bool = bool(kwargs.get("docker_socket", False))

        req_path = find_package_file(module_path, "requirements.txt") if module_path else None
        setup_path = find_package_file(module_path, "setup.sh") if module_path else None
        sysdeps_path = find_package_file(module_path, "system-deps.txt") if module_path else None

        if project_root is not None:
            return self._install_with_base(
                name,
                project_root,
                req_path=req_path,
                setup_path=setup_path,
                sysdeps_path=sysdeps_path,
                packages=packages,
                docker_socket=docker_socket,
                force=force,
            )

        image_tag = self._image_tag(
            name,
            req_path,
            setup_path,
            sysdeps_path,
            packages=packages,
            docker_socket=docker_socket,
        )
        if not force and self._image_exists(image_tag):
            return {"image": image_tag, "exgentic_version": get_exgentic_version()}
        self._build_image(
            image_tag,
            req_path,
            setup_path,
            sysdeps_path,
            packages=packages,
            docker_socket=docker_socket,
        )
        return {"image": image_tag, "exgentic_version": get_exgentic_version()}

    def exists(self, env_dir: Path, marker_data: dict) -> bool:
        """Docker images are managed externally; trust the marker."""
        return bool(marker_data.get("image"))

    def uninstall(self, env_dir: Path, marker_data: dict) -> None:
        """Remove the Docker images referenced in the marker data.

        The bench image is always removed.  The base image (if present) is
        attempted too — ``docker rmi`` will silently fail if another bench
        image still depends on it, so the last environment using a base tag
        will clean it up automatically.
        """
        for key in ("image", "base_image"):
            tag = marker_data.get(key)
            if tag:
                subprocess.run(
                    ["docker", "rmi", tag],
                    check=False,
                    capture_output=True,
                    text=True,
                )

    # ------------------------------------------------------------------
    # Two-image path (project_root provided)
    # ------------------------------------------------------------------

    def _install_with_base(
        self,
        name: str,
        project_root: Path,
        *,
        req_path: Path | None,
        setup_path: Path | None,
        sysdeps_path: Path | None,
        packages: list[str] | None,
        docker_socket: bool,
        force: bool,
    ) -> dict:
        base_tag = self._base_image_tag(project_root)
        bench_tag = self._bench_image_tag(
            name,
            base_tag,
            req_path,
            setup_path,
            sysdeps_path,
            packages=packages,
            docker_socket=docker_socket,
        )

        if not force and self._image_exists(bench_tag):
            return {"image": bench_tag, "exgentic_version": get_exgentic_version()}

        if force or not self._image_exists(base_tag):
            self._build_base_image(base_tag, project_root)

        self._build_bench_image(
            bench_tag,
            base_tag,
            req_path=req_path,
            setup_path=setup_path,
            sysdeps_path=sysdeps_path,
            packages=packages,
            docker_socket=docker_socket,
        )
        return {"image": bench_tag, "base_image": base_tag, "exgentic_version": get_exgentic_version()}

    @classmethod
    def _base_image_tag(cls, project_root: Path) -> str:
        """Deterministic tag for the shared base image."""
        h = hashlib.sha256()
        h.update(cls._IMAGE_VERSION.encode())
        h.update(b"\x00")
        pyproject = project_root / "pyproject.toml"
        if pyproject.is_file():
            h.update(pyproject.read_text().encode())
        h.update(b"\x00")
        return f"exgentic-base:{h.hexdigest()[:12]}"

    @staticmethod
    def _bench_image_tag(
        name: str,
        base_tag: str,
        *file_paths: Path | None,
        packages: list[str] | None = None,
        docker_socket: bool = False,
    ) -> str:
        """Deterministic tag for the benchmark-specific image."""
        h = hashlib.sha256()
        h.update(base_tag.encode())
        h.update(b"\x00")
        for path in file_paths:
            if path is not None:
                h.update(path.read_text().encode())
            h.update(b"\x00")
        if packages:
            h.update("\n".join(sorted(packages)).encode())
            h.update(b"\x00")
        if docker_socket:
            h.update(b"docker-socket\x00")
        safe_name = name.replace("/", "-")
        return f"{safe_name}:{h.hexdigest()[:12]}"

    @staticmethod
    def _build_base_image(tag: str, project_root: Path) -> None:
        """Build the shared base image: Python + git + uv + project installed."""
        py_version = f"{sys.version_info.major}.{sys.version_info.minor}"

        lines = [
            f"FROM python:{py_version}-slim",
            "RUN apt-get update && apt-get install -y --no-install-recommends git git-lfs"
            " && rm -rf /var/lib/apt/lists/* && git lfs install",
            "RUN pip install --no-cache-dir uv",
            "ENV UV_SYSTEM_PYTHON=true",
            "WORKDIR /app",
        ]

        # Layer 1 — install dependencies (cached unless pyproject.toml changes).
        copy_parts = ["COPY pyproject.toml ./"]
        if (project_root / "README.md").is_file():
            copy_parts.append("COPY README.md ./")
        lines.extend(copy_parts)

        pyproject_data = tomllib.loads((project_root / "pyproject.toml").read_text())
        pkg_name = pyproject_data.get("project", {}).get("name", "").replace("-", "_")
        if pkg_name:
            lines.append(f"RUN mkdir -p src/{pkg_name} && touch src/{pkg_name}/__init__.py")

        force_includes = (
            pyproject_data.get("tool", {})
            .get("hatch", {})
            .get("build", {})
            .get("targets", {})
            .get("wheel", {})
            .get("force-include", {})
        )
        for src_path in force_includes:
            lines.append(f"RUN mkdir -p '{src_path}'")

        lines.append("RUN uv pip install --no-cache .")

        # Layer 2 — install source code only (fast, deps already cached).
        lines.extend(
            [
                "COPY src/ src/",
                "RUN uv pip install --no-cache --no-deps .",
            ]
        )

        with tempfile.NamedTemporaryFile(
            mode="w",
            prefix="exgentic-base-",
            suffix=".Dockerfile",
            delete=False,
        ) as fh:
            fh.write("\n".join(lines) + "\n")
            dockerfile = Path(fh.name)
        try:
            subprocess.run(
                ["docker", "build", "-f", str(dockerfile), "-t", tag, str(project_root)],
                check=True,
            )
        finally:
            dockerfile.unlink(missing_ok=True)

    @staticmethod
    def _build_bench_image(
        tag: str,
        base_tag: str,
        *,
        req_path: Path | None = None,
        setup_path: Path | None = None,
        sysdeps_path: Path | None = None,
        packages: list[str] | None = None,
        docker_socket: bool = False,
    ) -> None:
        """Build the benchmark-specific image on top of the base image."""
        tmp_dir = Path(tempfile.mkdtemp(prefix="exgentic-bench-"))
        try:
            lines = [f"FROM {base_tag}"]

            if sysdeps_path is not None:
                pkgs = read_lines(sysdeps_path)
                if pkgs:
                    lines.append(
                        "RUN apt-get update && apt-get install -y " + " ".join(pkgs) + " && rm -rf /var/lib/apt/lists/*"
                    )

            if req_path is not None:
                shutil.copy2(req_path, tmp_dir / "requirements.txt")
                lines.append("COPY requirements.txt /tmp/")
                lines.append("RUN GIT_LFS_SKIP_SMUDGE=1 uv pip install --no-cache -r /tmp/requirements.txt")

            if packages:
                lines.append(f"RUN uv pip install --no-cache {' '.join(packages)}")

            if docker_socket:
                lines.append(_DOCKER_CLI_RUN)

            if setup_path is not None:
                shutil.copy2(setup_path, tmp_dir / "setup.sh")
                lines.append("COPY setup.sh /tmp/")
                lines.append("RUN EXGENTIC_DOCKER_BUILD=1 bash /tmp/setup.sh")

            (tmp_dir / "Dockerfile").write_text("\n".join(lines) + "\n")
            subprocess.run(
                ["docker", "build", "-t", tag, str(tmp_dir)],
                check=True,
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    # ------------------------------------------------------------------
    # Single-image path (no project_root)
    # ------------------------------------------------------------------

    @staticmethod
    def _image_tag(
        name: str,
        *file_paths: Path | None,
        packages: list[str] | None = None,
        docker_socket: bool = False,
    ) -> str:
        """Compute a deterministic image tag from content hashes."""
        h = hashlib.sha256()
        for path in file_paths:
            if path is not None:
                h.update(path.read_text().encode())
            h.update(b"\x00")
        if packages:
            h.update("\n".join(sorted(packages)).encode())
            h.update(b"\x00")
        if docker_socket:
            h.update(b"docker-socket\x00")
        safe_name = name.replace("/", "-")
        return f"{safe_name}:{h.hexdigest()[:12]}"

    @staticmethod
    def _image_exists(tag: str) -> bool:
        result = subprocess.run(
            ["docker", "image", "inspect", tag],
            check=False,
            capture_output=True,
            text=True,
        )
        return result.returncode == 0

    @staticmethod
    def _build_image(
        tag: str,
        req_path: Path | None,
        setup_path: Path | None,
        sysdeps_path: Path | None,
        *,
        packages: list[str] | None = None,
        docker_socket: bool = False,
    ) -> None:
        """Build a single image without a project root."""
        py_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        tmp_dir = Path(tempfile.mkdtemp(prefix="exgentic-docker-"))
        try:
            lines = [f"FROM python:{py_version}-slim"]

            if sysdeps_path is not None:
                pkgs = read_lines(sysdeps_path)
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

            if packages:
                lines.append(f"RUN uv pip install --no-cache {' '.join(packages)}")

            if docker_socket:
                lines.append(_DOCKER_CLI_RUN)

            if setup_path is not None:
                shutil.copy2(setup_path, tmp_dir / "setup.sh")
                lines.append("COPY setup.sh /tmp/")
                lines.append("RUN EXGENTIC_DOCKER_BUILD=1 bash /tmp/setup.sh")

            (tmp_dir / "Dockerfile").write_text("\n".join(lines) + "\n")

            subprocess.run(
                ["docker", "build", "-t", tag, str(tmp_dir)],
                check=True,
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
