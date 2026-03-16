# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""DockerRunner — runs the HTTP service inside a Docker container.

Uses the same HTTPTransport as ServiceRunner, but the uvicorn server
runs inside a container instead of a local thread.
"""

from __future__ import annotations

import atexit
import base64
import hashlib
import shutil
import socket
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import cloudpickle as cp

from .service import HTTPTransport, _wait_for_health
from .transport import ObjectProxy


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


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

    def __init__(
        self,
        target_cls: type,
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
        self._target_cls = target_cls
        self._args = args
        self._kwargs = kwargs
        self._image = image
        self._dockerfile = dockerfile
        self._port = port or _find_free_port()
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
            _docker("build", "-t", tag, "-f", str(path), str(path.parent),
                    capture_output=True)
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
            return "exgentic-runner:latest"
        content_hash = hashlib.sha256("\n".join(parts).encode()).hexdigest()[:12]
        return f"exgentic-runner:{content_hash}"

    def _build_default_image(self) -> str:
        tag = self._image_tag()

        # Reuse if already built.
        if _docker("image", "inspect", tag, check=False, capture_output=True).returncode == 0:
            return tag

        root = self._find_project_root()
        tmp = Path(tempfile.mkdtemp(prefix="exgentic-docker-"))

        # Build Dockerfile lines.
        lines = [
            f"FROM {self._BASE_IMAGE}",
            "RUN pip install --no-cache-dir uv",
            "ENV UV_SYSTEM_PYTHON=true",
            "WORKDIR /app",
            "COPY pyproject.toml README.md ./",
            "COPY src/ src/",
        ]
        lines.append("RUN uv pip install --no-cache .")
        if self._requirements_txt:
            req_path = Path(self._requirements_txt)
            if req_path.exists():
                shutil.copy2(req_path, tmp / "requirements.txt")
                lines.append("COPY requirements.txt /tmp/requirements.txt")
                lines.append("RUN uv pip install --no-cache -r /tmp/requirements.txt")

        if self._dependencies:
            lines.append(f"RUN uv pip install --no-cache {' '.join(self._dependencies)}")

        # Install Docker CLI if docker_socket is requested.
        if self._docker_socket:
            lines.append(
                "RUN apt-get update && apt-get install -y --no-install-recommends "
                "docker.io && rm -rf /var/lib/apt/lists/*"
            )

        # Copy and run the setup script.
        if self._setup_script:
            script_path = Path(self._setup_script)
            if not script_path.exists():
                raise FileNotFoundError(f"Setup script not found: {self._setup_script}")
            # Copy the script into the build context.
            shutil.copy2(script_path, tmp / "setup.sh")
            lines.append("COPY setup.sh /tmp/setup.sh")
            lines.append("RUN EXGENTIC_DOCKER_BUILD=1 bash /tmp/setup.sh")

        lines.append('ENTRYPOINT ["python", "-c"]')

        (tmp / "Dockerfile").write_text("\n".join(lines) + "\n")
        result = _docker(
            "build", "-t", tag, "-f", str(tmp / "Dockerfile"), str(root),
            check=False, capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Docker build failed:\n{result.stderr}")
        return tag

    @staticmethod
    def _find_project_root() -> Path:
        for parent in Path(__file__).resolve().parents:
            if (parent / "pyproject.toml").exists():
                return parent
        raise FileNotFoundError("Cannot find project root (pyproject.toml)")

    # ── container lifecycle ──────────────────────────────────────────

    def start(self) -> ObjectProxy:
        image = self._ensure_image()

        # Pickle the object, registering the module for by-value pickling
        # so cloudpickle embeds the class definition.
        obj = self._target_cls(*self._args, **self._kwargs)
        cls_module_name = getattr(self._target_cls, "__module__", None)
        cls_module = sys.modules.get(cls_module_name) if cls_module_name else None

        registered = False
        if cls_module is not None:
            try:
                cp.register_pickle_by_value(cls_module)
                registered = True
            except Exception:
                pass
        try:
            payload_b64 = base64.b64encode(cp.dumps(obj)).decode("ascii")
        finally:
            if registered and cls_module is not None:
                try:
                    cp.unregister_pickle_by_value(cls_module)
                except Exception:
                    pass

        serve_script = (
            "from exgentic.core.context import init_context_from_env\n"
            "try:\n    init_context_from_env()\nexcept RuntimeError:\n    pass\n"
            "import base64, cloudpickle as cp\n"
            "from exgentic.adapters.runners.service import serve\n"
            f"obj = cp.loads(base64.b64decode('{payload_b64}'))\n"
            "serve(obj, host='0.0.0.0', port=8080)"
        )

        # Build docker run arguments.
        run_args: list[str] = ["run", "-d", "-p", f"{self._port}:8080"]

        # Forward context env vars into the container.
        from ...core.context import context_env
        for k, v in context_env().items():
            run_args.extend(["-e", f"{k}={v}"])

        # Mount Docker socket for sibling container access.
        if self._docker_socket:
            run_args.extend(["-v", "/var/run/docker.sock:/var/run/docker.sock"])

        # Mount volumes.
        for host_path, container_path in self._volumes.items():
            run_args.extend(["-v", f"{host_path}:{container_path}"])

        run_args.extend(self._docker_args)
        run_args.extend([image, serve_script])

        result = _docker(*run_args, capture_output=True, text=True)
        self._container_id = result.stdout.strip()
        atexit.register(self._stop_container)

        url = f"http://127.0.0.1:{self._port}"
        try:
            _wait_for_health(url, timeout=60.0)
        except TimeoutError:
            cid = self._container_id or ""
            logs = _docker("logs", cid, check=False, capture_output=True, text=True)
            status = _docker("inspect", "--format", "{{.State.Status}}", cid,
                             check=False, capture_output=True, text=True)
            self._stop_container()
            raise TimeoutError(
                f"Container did not become healthy within 60s.\n"
                f"Status: {status.stdout.strip()}\n"
                f"Logs:\n{logs.stdout}\n{logs.stderr}"
            )

        transport = HTTPTransport(url)
        proxy = ObjectProxy(transport)
        runner_ref = self

        def _close() -> None:
            transport.close()
            runner_ref._stop_container()

        object.__setattr__(proxy, "close", _close)
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
