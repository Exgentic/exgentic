# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""DockerRunner — runs the HTTP service inside a Docker container.

Uses the same HTTPTransport as ServiceRunner, but the uvicorn server
runs inside a container instead of a local thread.
"""

from __future__ import annotations

import atexit
import hashlib
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from ._utils import (
    find_free_port,
    find_project_root,
    inject_exgentic_env,
    make_close,
    prepare_subprocess_env,
    serialize_kwargs,
)
from .service import HTTPTransport, _wait_for_health
from .transport import ObjectProxy


def _docker(*args: str, check: bool = True, **kwargs: Any) -> subprocess.CompletedProcess:
    docker_bin = shutil.which("docker")
    if docker_bin is None:
        raise RuntimeError("docker CLI not found on PATH")
    return subprocess.run([docker_bin, *args], check=check, **kwargs)


class DockerRunner:
    """Start a containerised HTTP service and return an ObjectProxy.

    Parameters
    ----------
    target_cls:    Class to instantiate inside the container.
    image:         Pre-built image name (skips building).
    dockerfile:    Path to a Dockerfile to build from.
    port:          Host port to bind (auto-selected if None).
    docker_args:   Extra arguments forwarded to ``docker run``.
    dependencies:  Pip packages to install in the image.
    setup_script:  Path to a shell script to run during image build.
                   This is the primary way benchmarks declare their
                   environment — the same script users run locally.
    docker_socket: Mount the host Docker socket into the container.
                   Needed for benchmarks like SWE-bench that create
                   sibling containers via the Docker API.
    volumes:       Host-to-container volume mappings (``{host: container}``).
    requirements_txt: Path to a requirements.txt to install in the image.
    """

    _BASE_IMAGE = "python:3.12-slim"
    _IMAGE_VERSION = "v24"  # bump to invalidate cached images

    def __init__(
        self,
        target_cls: type | str,
        *args: Any,
        image: str | None = None,
        dockerfile: str | None = None,
        port: int | None = None,
        docker_args: list[str] | None = None,
        dependencies: list[str] | None = None,
        setup_script: str | None = None,
        docker_socket: bool = False,
        volumes: dict[str, str] | None = None,
        requirements_txt: str | None = None,
        **kwargs: Any,
    ) -> None:
        if args:
            raise ValueError(
                "DockerRunner requires keyword-only constructor arguments. "
                "Pass all arguments as kwargs instead of positional args."
            )
        self._target_cls = target_cls
        self._kwargs = kwargs
        self._image = image
        self._dockerfile = dockerfile
        self._port = port or find_free_port()
        self._docker_args = docker_args or []
        self._dependencies = dependencies or []
        self._setup_script = setup_script
        self._docker_socket = docker_socket
        self._volumes = volumes or {}
        self._requirements_txt = requirements_txt
        self._container_id: str | None = None

    # ── image handling ───────────────────────────────────────────────

    def _ensure_image(self) -> str:
        if self._image:
            return self._image

        if self._dockerfile:
            tag = f"exgentic-runner-custom:{hash(self._dockerfile) & 0xFFFFFFFF:08x}"
            path = Path(self._dockerfile)
            _docker("build", "-t", tag, "-f", str(path), str(path.parent), capture_output=True)
            return tag

        return self._build_default_image()

    def _image_tag(self) -> str:
        """Compute a deterministic image tag from all build inputs."""
        parts: list[str] = []
        if self._requirements_txt:
            req_path = Path(self._requirements_txt)
            if req_path.exists():
                parts.append("reqs:" + req_path.read_text())
        if self._dependencies:
            parts.append("deps:" + " ".join(sorted(self._dependencies)))
        if self._setup_script:
            script_path = Path(self._setup_script)
            if script_path.exists():
                parts.append("setup:" + script_path.read_text())
        if self._docker_socket:
            parts.append("docker-cli")
        if not parts:
            return f"exgentic-runner:{self._IMAGE_VERSION}"
        parts.insert(0, self._IMAGE_VERSION)
        content_hash = hashlib.sha256("\n".join(parts).encode()).hexdigest()[:12]
        return f"exgentic-runner:{content_hash}"

    def _build_default_image(self) -> str:
        tag = self._image_tag()

        # Reuse if already built.
        if _docker("image", "inspect", tag, check=False, capture_output=True).returncode == 0:
            return tag

        root = find_project_root()
        tmp = Path(tempfile.mkdtemp(prefix="exgentic-docker-"))

        # Build Dockerfile lines.  Build context is the project root.
        # Dependencies are installed first with a stub package so the
        # heavy layer is cached across source-code changes.
        lines = [
            f"FROM {self._BASE_IMAGE}",
            "RUN apt-get update && apt-get install -y --no-install-recommends git git-lfs"
            " && rm -rf /var/lib/apt/lists/* && git lfs install",
            "RUN pip install --no-cache-dir uv",
            "ENV UV_SYSTEM_PYTHON=true",
            "WORKDIR /app",
            # Layer 1 — install dependencies (cached unless pyproject.toml changes).
            "COPY pyproject.toml README.md ./",
            "RUN mkdir -p src/exgentic && touch src/exgentic/__init__.py",
        ]

        # Create stub directories for force-include paths so hatch can
        # build the wheel during the deps-only install (the real files
        # arrive in the later COPY src/ layer).
        import tomllib

        pyproject = tomllib.loads((root / "pyproject.toml").read_text())
        force_includes = (
            pyproject.get("tool", {})
            .get("hatch", {})
            .get("build", {})
            .get("targets", {})
            .get("wheel", {})
            .get("force-include", {})
        )
        for src_path in force_includes:
            lines.append(f"RUN mkdir -p '{src_path}'")

        lines.extend(
            [
                "RUN uv pip install --no-cache .",
                # Layer 2 — install source code only (fast, deps already cached).
                "COPY src/ src/",
                "RUN uv pip install --no-cache --no-deps .",
            ]
        )
        if self._requirements_txt:
            req_path = Path(self._requirements_txt)
            if req_path.exists():
                # Use absolute path in COPY via the relative path from root.
                rel = req_path.resolve().relative_to(root.resolve())
                lines.append(f"COPY {rel} /tmp/requirements.txt")
                # Skip Git LFS smudge filters — pip only needs Python source.
                # Benchmarks that need LFS data fetch it in setup.sh instead.
                lines.append("RUN GIT_LFS_SKIP_SMUDGE=1 uv pip install --no-cache -r /tmp/requirements.txt")

        if self._dependencies:
            lines.append(f"RUN uv pip install --no-cache {' '.join(self._dependencies)}")

        if self._docker_socket:
            # Install the Docker CLI (static binary) so the container can
            # manage sibling containers via the mounted Docker socket.
            # Using a static download avoids Debian-version-specific apt repos.
            lines.append(
                "RUN apt-get update && apt-get install -y --no-install-recommends curl && "
                "rm -rf /var/lib/apt/lists/* && "
                "ARCH=$(uname -m) && "
                "curl -fsSL https://download.docker.com/linux/static/stable/${ARCH}/docker-27.5.1.tgz "
                "| tar xz --strip-components=1 -C /usr/local/bin docker/docker"
            )

        if self._setup_script:
            script_path = Path(self._setup_script)
            if not script_path.exists():
                raise FileNotFoundError(f"Setup script not found: {self._setup_script}")
            rel = script_path.resolve().relative_to(root.resolve())
            lines.append(f"COPY {rel} /tmp/setup.sh")
            lines.append("RUN EXGENTIC_DOCKER_BUILD=1 bash /tmp/setup.sh")

        (tmp / "Dockerfile").write_text("\n".join(lines) + "\n")
        result = _docker(
            "build",
            "-t",
            tag,
            "-f",
            str(tmp / "Dockerfile"),
            str(root),
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Docker build failed:\n{result.stderr}")
        return tag

    # ── container lifecycle ──────────────────────────────────────────

    def start(self) -> ObjectProxy:
        image = self._ensure_image()

        if isinstance(self._target_cls, str):
            cls_ref = self._target_cls
        else:
            cls_ref = f"{self._target_cls.__module__}:{self._target_cls.__qualname__}"
        kwargs_flag, kwargs_value = serialize_kwargs(self._kwargs)

        run_args: list[str] = ["run", "-d", "-p", f"{self._port}:8080"]

        # Forward host environment into the container (API tokens, user
        # config) while excluding system-level and IDE vars.
        env = prepare_subprocess_env()
        inject_exgentic_env(env)
        cache_dir = env.get("EXGENTIC_CACHE_DIR", "")

        for k, v in env.items():
            run_args.extend(["-e", f"{k}={v}"])

        # Mount Docker socket for sibling container access.
        if self._docker_socket:
            run_args.extend(["-v", "/var/run/docker.sock:/var/run/docker.sock"])

        # Always mount the cache dir so benchmarks that skip data downloads
        # during Docker build (e.g. browsecompplus) can access host-side data,
        # and benchmarks that bake data into the image (e.g. appworld) can
        # also work since the volume mount overlays the image path.
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        run_args.extend(["-v", f"{cache_dir}:{cache_dir}"])

        # Mount volumes.  Resolve to absolute paths (Docker requires them)
        # and ensure source directories exist — Docker Desktop on macOS
        # cannot create mount sources in some protected paths.
        for host_path, container_path in self._volumes.items():
            host_path = str(Path(host_path).resolve())
            container_path = str(Path(container_path)) if Path(container_path).is_absolute() else container_path
            Path(host_path).mkdir(parents=True, exist_ok=True)
            run_args.extend(["-v", f"{host_path}:{container_path}"])

        run_args.extend(self._docker_args)
        run_args.extend(
            [
                image,
                "exgentic",
                "serve",
                "--cls",
                cls_ref,
                kwargs_flag,
                kwargs_value,
                "--host",
                "0.0.0.0",
                "--port",
                "8080",
            ]
        )

        result = _docker(*run_args, capture_output=True, text=True)
        self._container_id = result.stdout.strip()
        atexit.register(self._stop_container)

        url = f"http://127.0.0.1:{self._port}"
        try:
            _wait_for_health(url, timeout=60.0)
        except TimeoutError:
            cid = self._container_id or ""
            logs = _docker("logs", cid, check=False, capture_output=True, text=True)
            status = _docker(
                "inspect", "--format", "{{.State.Status}}", cid, check=False, capture_output=True, text=True
            )
            self._stop_container()
            raise TimeoutError(
                f"Container did not become healthy within 60s.\n"
                f"Status: {status.stdout.strip()}\n"
                f"Logs:\n{logs.stdout}\n{logs.stderr}"
            ) from None

        transport = HTTPTransport(url, timeout=600.0)
        proxy = ObjectProxy(transport)
        object.__setattr__(proxy, "close", make_close(transport, self._stop_container))
        return proxy

    def _stop_container(self) -> None:
        if self._container_id is None:
            return
        cid = self._container_id
        self._container_id = None
        try:
            _docker("stop", "-t", "2", cid, check=False, capture_output=True)
            _docker("rm", "-f", cid, check=False, capture_output=True)
        except Exception:
            pass
