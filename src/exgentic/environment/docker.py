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

from .helpers import find_package_file, read_lines


class DockerBackend:
    """Backend that builds a Docker image with dependencies baked in."""

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
                and ``packages`` (list[str]).

        Returns:
            Marker data with ``image`` tag.
        """
        name: str = kwargs.get("name", env_dir.name)  # type: ignore[assignment]
        force: bool = kwargs.get("force", False)  # type: ignore[assignment]
        project_root: Path | None = kwargs.get("project_root")  # type: ignore[assignment]
        packages: list[str] | None = kwargs.get("packages")  # type: ignore[assignment]

        req_path = find_package_file(module_path, "requirements.txt") if module_path else None
        setup_path = find_package_file(module_path, "setup.sh") if module_path else None
        sysdeps_path = find_package_file(module_path, "system-deps.txt") if module_path else None

        image_tag = self._image_tag(
            name,
            req_path,
            setup_path,
            sysdeps_path,
            project_root=project_root,
            packages=packages,
        )

        if not force and self._image_exists(image_tag):
            return {"image": image_tag}

        if project_root is not None:
            self._build_project_image(
                image_tag,
                project_root,
                req_path=req_path,
                setup_path=setup_path,
                sysdeps_path=sysdeps_path,
                packages=packages,
            )
        else:
            self._build_image(
                image_tag,
                req_path,
                setup_path,
                sysdeps_path,
                packages=packages,
            )
        return {"image": image_tag}

    def uninstall(self, env_dir: Path, marker_data: dict) -> None:
        """Remove the Docker image referenced in the marker data."""
        image = marker_data.get("image")
        if image:
            subprocess.run(
                ["docker", "rmi", image],
                check=False,
                capture_output=True,
                text=True,
            )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _image_tag(
        name: str,
        *file_paths: Path | None,
        project_root: Path | None = None,
        packages: list[str] | None = None,
    ) -> str:
        """Compute a deterministic image tag from content hashes."""
        h = hashlib.sha256()
        for path in file_paths:
            if path is not None:
                h.update(path.read_text().encode())
            h.update(b"\x00")
        if project_root is not None:
            pyproject = project_root / "pyproject.toml"
            if pyproject.is_file():
                h.update(pyproject.read_text().encode())
            h.update(b"\x00")
        if packages:
            h.update("\n".join(sorted(packages)).encode())
            h.update(b"\x00")
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
    ) -> None:
        """Build an image without a project root (original behavior)."""
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

            if setup_path is not None:
                shutil.copy2(setup_path, tmp_dir / "setup.sh")
                lines.append("COPY setup.sh /tmp/")
                lines.append("RUN bash /tmp/setup.sh")

            (tmp_dir / "Dockerfile").write_text("\n".join(lines) + "\n")

            subprocess.run(
                ["docker", "build", "-t", tag, str(tmp_dir)],
                check=True,
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    @staticmethod
    def _build_project_image(
        tag: str,
        project_root: Path,
        *,
        req_path: Path | None = None,
        setup_path: Path | None = None,
        sysdeps_path: Path | None = None,
        packages: list[str] | None = None,
    ) -> None:
        """Build an image with two-layer project install for caching."""
        py_version = f"{sys.version_info.major}.{sys.version_info.minor}"

        lines = [
            f"FROM python:{py_version}-slim",
            "RUN apt-get update && apt-get install -y --no-install-recommends git git-lfs"
            " && rm -rf /var/lib/apt/lists/* && git lfs install",
            "RUN pip install --no-cache-dir uv",
            "ENV UV_SYSTEM_PYTHON=true",
            "WORKDIR /app",
        ]

        if sysdeps_path is not None:
            pkgs = read_lines(sysdeps_path)
            if pkgs:
                lines.append(
                    "RUN apt-get update && apt-get install -y " + " ".join(pkgs) + " && rm -rf /var/lib/apt/lists/*"
                )

        # Layer 1 — install dependencies (cached unless pyproject.toml changes).
        copy_parts = ["COPY pyproject.toml ./"]
        if (project_root / "README.md").is_file():
            copy_parts.append("COPY README.md ./")
        lines.append(" ".join(copy_parts) if len(copy_parts) == 1 else "\n".join(copy_parts))

        # Determine the package name for the stub __init__.py.
        pyproject_data = tomllib.loads((project_root / "pyproject.toml").read_text())
        pkg_name = pyproject_data.get("project", {}).get("name", "").replace("-", "_")
        if pkg_name:
            lines.append(f"RUN mkdir -p src/{pkg_name} && touch src/{pkg_name}/__init__.py")

        # Create stub directories for hatch force-include paths.
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

        # Module requirements.
        if req_path is not None:
            try:
                rel = req_path.resolve().relative_to(project_root.resolve())
                lines.append(f"COPY {rel} /tmp/requirements.txt")
            except ValueError:
                # req_path is outside project_root — copy into build context.
                shutil.copy2(req_path, project_root / ".tmp_requirements.txt")
                lines.append("COPY .tmp_requirements.txt /tmp/requirements.txt")
            lines.append("RUN GIT_LFS_SKIP_SMUDGE=1 uv pip install --no-cache -r /tmp/requirements.txt")

        # Extra packages.
        if packages:
            lines.append(f"RUN uv pip install --no-cache {' '.join(packages)}")

        # Setup script.
        if setup_path is not None:
            try:
                rel = setup_path.resolve().relative_to(project_root.resolve())
                lines.append(f"COPY {rel} /tmp/setup.sh")
            except ValueError:
                shutil.copy2(setup_path, project_root / ".tmp_setup.sh")
                lines.append("COPY .tmp_setup.sh /tmp/setup.sh")
            lines.append("RUN bash /tmp/setup.sh")

        # Write Dockerfile to a temp file and build with project_root as context.
        dockerfile = Path(tempfile.mktemp(prefix="exgentic-dockerfile-", suffix=".Dockerfile"))
        try:
            dockerfile.write_text("\n".join(lines) + "\n")
            subprocess.run(
                ["docker", "build", "-f", str(dockerfile), "-t", tag, str(project_root)],
                check=True,
            )
        finally:
            dockerfile.unlink(missing_ok=True)
