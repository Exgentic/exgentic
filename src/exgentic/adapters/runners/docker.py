# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""DockerRunner — runs the HTTP service inside a Docker container.

Uses the same HTTPTransport as ServiceRunner, but the uvicorn server
runs inside a container instead of a local thread.

The Docker image is managed by the EnvironmentManager — DockerRunner
only starts the container and wires up volumes, ports and env vars.
"""

from __future__ import annotations

import atexit
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ...core.context import Role

from ._utils import (
    find_free_port,
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
    env_name:      Environment name for EnvironmentManager (e.g. "benchmarks/bfcl").
    module_path:   Dotted module path for locating package resources.
    image:         Pre-built image name (skips EM lookup).
    dockerfile:    Path to a Dockerfile to build from.
    port:          Host port to bind (auto-selected if None).
    docker_args:   Extra arguments forwarded to ``docker run``.
    dependencies:  Pip packages to install in the image.
    docker_socket: Mount the host Docker socket into the container.
    volumes:       Host-to-container volume mappings (``{host: container}``).
    """

    def __init__(
        self,
        target_cls: type | str,
        *args: Any,
        env_name: str = "",
        module_path: str = "",
        image: str | None = None,
        dockerfile: str | None = None,
        port: int | None = None,
        docker_args: list[str] | None = None,
        dependencies: list[str] | None = None,
        docker_socket: bool = False,
        volumes: dict[str, str] | None = None,
        role: Role | None = None,
        **kwargs: Any,
    ) -> None:
        if args:
            raise ValueError(
                "DockerRunner requires keyword-only constructor arguments. "
                "Pass all arguments as kwargs instead of positional args."
            )
        self._target_cls = target_cls
        self._kwargs = kwargs
        self._env_name = env_name
        self._module_path = module_path
        self._image = image
        self._dockerfile = dockerfile
        self._port = port or find_free_port()
        self._docker_args = docker_args or []
        self._dependencies = dependencies or []
        self._docker_socket = docker_socket
        self._volumes = volumes or {}
        self._role = role
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

        if not self._env_name:
            raise RuntimeError(
                "DockerRunner requires 'env_name' (and usually 'module_path') "
                "when no 'image' or 'dockerfile' is provided."
            )

        # Use EM's pre-built image.
        from ...environment.instance import get_manager

        mgr = get_manager()
        image = mgr.docker_image(self._env_name)
        if image:
            return image

        # Not pre-installed — install via EM now.
        from ...environment import EnvType
        from ...environment.helpers import get_exgentic_install_target

        project_root, packages = get_exgentic_install_target()
        all_packages = (packages or []) + list(self._dependencies)
        mgr.install(
            self._env_name,
            env_type=EnvType.DOCKER,
            module_path=self._module_path,
            docker_socket=self._docker_socket,
            project_root=project_root,
            packages=all_packages or None,
        )
        image = mgr.docker_image(self._env_name)
        if not image:
            raise RuntimeError(f"EM install succeeded but no Docker image found for {self._env_name}")
        return image

    # ── container lifecycle ──────────────────────────────────────────

    def start(self) -> ObjectProxy:
        image = self._ensure_image()

        if isinstance(self._target_cls, str):
            cls_ref = self._target_cls
        else:
            cls_ref = f"{self._target_cls.__module__}:{self._target_cls.__qualname__}"
        kwargs_flag, kwargs_value = serialize_kwargs(self._kwargs)

        run_args: list[str] = ["run", "-d", "-p", f"{self._port}:8080"]

        # Forward only model-provider credentials + point at runtime.json.
        # Exgentic context and settings travel via runtime.json, not env.
        env = prepare_subprocess_env()
        inject_exgentic_env(env, role=self._role)

        for k, v in env.items():
            run_args.extend(["-e", f"{k}={v}"])

        # Mount Docker socket for sibling container access.
        if self._docker_socket:
            run_args.extend(["-v", "/var/run/docker.sock:/var/run/docker.sock"])

        # Always mount the EnvironmentManager cache dir so benchmarks that
        # skip data downloads during Docker build (e.g. browsecompplus) can
        # access host-side data, and benchmarks that bake data into the
        # image (e.g. appworld) can also work since the volume mount
        # overlays the image path.
        from ...environment.instance import get_manager

        cache_dir = str(get_manager().base_dir)
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        run_args.extend(["-v", f"{cache_dir}:{cache_dir}"])

        # Mount the runtime.json file (read-only) so the container can
        # bootstrap context, settings, and OTEL propagation.  Mounting
        # the file (not its parent dir) avoids shadowing the outer
        # outputs/ volume: the benchmark/agent needs to WRITE results
        # into its runtime dir, so a read-only directory mount would
        # block those writes.
        runtime_file = env.get("EXGENTIC_RUNTIME_FILE")
        if runtime_file:
            Path(runtime_file).parent.mkdir(parents=True, exist_ok=True)
            run_args.extend(["-v", f"{runtime_file}:{runtime_file}:ro"])

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
