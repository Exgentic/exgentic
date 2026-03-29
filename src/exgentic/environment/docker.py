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
            **kwargs: Accepts ``name`` (environment name) and ``force``.

        Returns:
            Marker data with ``image`` tag.
        """
        name: str = kwargs.get("name", env_dir.name)  # type: ignore[assignment]
        force: bool = kwargs.get("force", False)  # type: ignore[assignment]

        req_path = find_package_file(module_path, "requirements.txt") if module_path else None
        setup_path = find_package_file(module_path, "setup.sh") if module_path else None
        sysdeps_path = find_package_file(module_path, "system-deps.txt") if module_path else None

        image_tag = self._image_tag(name, req_path, setup_path, sysdeps_path)

        if not force and self._image_exists(image_tag):
            return {"image": image_tag}

        self._build_image(image_tag, req_path, setup_path, sysdeps_path)
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
    def _image_tag(name: str, *file_paths: Path | None) -> str:
        """Compute a deterministic image tag from content hashes."""
        h = hashlib.sha256()
        for path in file_paths:
            if path is not None:
                h.update(path.read_text().encode())
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
    ) -> None:
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
