# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel


class ExecutionBackend(str, Enum):
    """Execution backend for CLI runs."""

    PROCESS = "process"
    PODMAN = "podman"
    DOCKER = "docker"
    AUTO = "auto"


def resolve_container_backend() -> ExecutionBackend:
    """Auto-detect container runtime: prefer podman, fallback to docker."""
    if shutil.which("podman"):
        return ExecutionBackend.PODMAN
    if shutil.which("docker"):
        return ExecutionBackend.DOCKER
    raise RuntimeError("Neither podman nor docker found")


@dataclass
class CLIResult:
    stdout: str
    stderr: str
    code: int


class CLIStartError(RuntimeError):
    """Raised when a CLI fails to spawn."""


class CLIExecutionError(RuntimeError):
    """Raised when a CLI exits with a non-zero status."""

    def __init__(
        self,
        message: str,
        *,
        code: int,
        stdout: str,
        stderr: str,
        cmd: list[str],
    ) -> None:
        super().__init__(message)
        self.code = code
        self.stdout = stdout
        self.stderr = stderr
        self.cmd = cmd

    def __str__(self) -> str:
        parts = [super().__str__()]
        if self.stderr:
            parts.append(f"STDERR:\n{self.stderr.rstrip()}")
        if self.stdout:
            parts.append(f"STDOUT:\n{self.stdout.rstrip()}")
        return "\n".join(parts)


class BaseCLIConfig(BaseModel):
    """Common config fields shared by CLI wrappers."""

    mcp_host: str
    mcp_port: int
    provider_url: str
    image: str
    image_workdir: str = "/work"
    env: Optional[dict[str, str]] = None


class ProcessRunner:
    """Shared subprocess execution logic (spawn + communicate + timeout + kill).

    Concrete runners implement how cmd/env/cfg_root are transformed.
    """

    def __init__(self, log_path, logger):
        super().__init__()
        self.log_path = log_path
        self._logger = logger
        self._last_cmd: list[str] = []
        self._proc: Optional[subprocess.Popen[str]] = None

    def _write_log(
        self,
        stdout: str,
        stderr: str,
        *,
        returncode: int,
        config: BaseCLIConfig,
    ) -> None:
        if not self.log_path:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            config_json = config.model_dump_json(indent=2)
        except Exception:
            config_json = str(config)
        with open(self.log_path, "w", encoding="utf-8") as fh:
            fh.write(f"Command: {shlex.join(self._last_cmd)}\n")
            fh.write("Config:\n")
            fh.write(f"{config_json}\n")
            fh.write(f"Exit code: {returncode}\n\n")
            if stdout:
                fh.write("STDOUT:\n")
                fh.write(stdout)
                if not stdout.endswith("\n"):
                    fh.write("\n")
            if stderr:
                if stdout:
                    fh.write("\n")
                fh.write("STDERR:\n")
                fh.write(stderr)
                if not stderr.endswith("\n"):
                    fh.write("\n")

    def run(
        self,
        *,
        cmd: list[str],
        env: dict[str, str],
        cfg_root: Path,
        config: BaseCLIConfig,
        spawn_error_message: str,
        stdin_devnull: bool = False,
    ) -> CLIResult:
        self._last_cmd = cmd
        stdout: str = ""
        stderr: str = ""
        code: int = -1

        popen_stdin = subprocess.DEVNULL if stdin_devnull else None
        timeout_s: Optional[float] = None

        try:
            self._proc = subprocess.Popen(
                cmd,
                stdin=popen_stdin,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
            )
        except Exception as exc:
            stderr = f"{spawn_error_message}: {exc}"
            self._write_log(stdout, stderr, returncode=code, config=config)
            raise CLIStartError(f"{spawn_error_message}: {exc}") from exc

        try:
            try:
                stdout, stderr = self._proc.communicate(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                self._logger.warning("CLI timed out; terminating: %s", shlex.join(cmd))
                self._proc.terminate()
                try:
                    stdout, stderr = self._proc.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    self._logger.warning("CLI did not terminate; killing it: %s", shlex.join(cmd))
                    self._proc.kill()
                    stdout, stderr = self._proc.communicate()
            code = self._proc.returncode or 0
        finally:
            self._write_log(stdout or "", stderr or "", returncode=code, config=config)
            if code != 0:
                self._logger.warning("CLI exited non-zero (%s): %s", code, shlex.join(cmd))
        if code != 0:
            raise CLIExecutionError(
                f"CLI exited non-zero ({code}): {shlex.join(cmd)}",
                code=code,
                stdout=stdout or "",
                stderr=stderr or "",
                cmd=cmd,
            )
        return CLIResult(stdout=stdout or "", stderr=stderr or "", code=code)

    def close(self) -> None:
        proc = self._proc
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._logger.warning("CLI process did not terminate; killing it.")
                self._proc.kill()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._logger.warning("CLI process did not exit after kill.")


class ContainerRunner(ProcessRunner):
    """Shared helpers for container-based runners."""

    host_gateway: str = "host.docker.internal"

    def _rewrite_mcp_config_path(self, inner_cmd: list[str], workdir: str) -> list[str]:
        if "--mcp-config" in inner_cmd:
            i = inner_cmd.index("--mcp-config")
            if i + 1 < len(inner_cmd):
                inner_cmd[i + 1] = f"{workdir}/mcp.json"
        return inner_cmd

    def _container_env_from(
        self,
        env: dict[str, str],
        host_gateway: str,
        config: BaseCLIConfig,
    ) -> dict[str, str]:
        forwarded: dict[str, str] = {}

        for k in (
            # Anthropic (Claude Code)
            "ANTHROPIC_BASE_URL",
            "ANTHROPIC_API_KEY",
            "ANTHROPIC_AUTH_TOKEN",
            # OpenAI (Codex)
            "OPENAI_API_KEY",
            "OPENAI_API_BASE",
            # Google (Gemini)
            "GEMINI_API_KEY",
            "GOOGLE_GEMINI_BASE_URL",
        ):
            if k in env:
                forwarded[k] = env[k]
        for k, v in env.items():
            if k.startswith("EXGENTIC_CTX_"):
                forwarded[k] = v
        if config.env:
            for k, v in config.env.items():
                forwarded[k] = v

        for k in (
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "NO_PROXY",
            "http_proxy",
            "https_proxy",
            "no_proxy",
        ):
            if k in env:
                forwarded[k] = env[k]

        if "PYTHONIOENCODING" in env:
            forwarded["PYTHONIOENCODING"] = env["PYTHONIOENCODING"]

        for k, v in forwarded.items():
            if "://127.0.0.1:" in v:
                forwarded[k] = v.replace("://127.0.0.1:", f"://{host_gateway}:")
            if "://localhost:" in v:
                forwarded[k] = v.replace("://localhost:", f"://{host_gateway}:")

        return forwarded

    def _patch_mcp_json(self, *, cfg_root: Path, host_gateway: str) -> None:
        mcp_path = cfg_root / "mcp.json"
        if not mcp_path.exists():
            return

        try:
            data = json.loads(mcp_path.read_text(encoding="utf-8"))
            env_cfg = data.get("mcpServers", {}).get("environment", {})
            url = env_cfg.get("url", "")
            if not isinstance(url, str) or not url:
                return

            if "://127.0.0.1:" in url:
                data["mcpServers"]["environment"]["url"] = url.replace("://127.0.0.1:", f"://{host_gateway}:")
                mcp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
                self._logger.debug("Patched mcp.json for container host gateway: %s", mcp_path)
            elif "://localhost:" in url:
                data["mcpServers"]["environment"]["url"] = url.replace("://localhost:", f"://{host_gateway}:")
                mcp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
                self._logger.debug("Patched mcp.json for container host gateway: %s", mcp_path)

        except Exception as exc:
            self._logger.warning("Failed to patch mcp.json (%s): %s", mcp_path, exc)


class PodmanRunner(ContainerRunner):
    """Run the inner command inside a container via Podman."""

    host_gateway = "host.containers.internal"

    def __init__(self, log_path, logger):
        super().__init__(log_path, logger)

    def run(
        self,
        *,
        cmd: list[str],
        env: dict[str, str],
        cfg_root: Path,
        config: BaseCLIConfig,
        spawn_error_message: str,
    ) -> CLIResult:
        runtime = "podman"
        host_gateway = "host.containers.internal"
        host_cfg_root = str(cfg_root.resolve())

        # Patch mcp.json so container uses host gateway (not 127.0.0.1/localhost)
        self._patch_mcp_json(cfg_root=cfg_root, host_gateway=host_gateway)

        inner_cmd = self._rewrite_mcp_config_path(list(cmd), workdir=config.image_workdir)

        # Minimal env forwarding into container (encoded via podman -e flags)
        container_env = self._container_env_from(env, host_gateway=host_gateway, config=config)
        container_env["HOME"] = config.image_workdir
        connection_args = self._resolve_podman_connection_args()
        user_args: list[str] = []
        uid = getattr(os, "getuid", None)
        gid = getattr(os, "getgid", None)
        if callable(uid) and callable(gid):
            try:
                user_args = ["--user", f"{uid()}:{gid()}"]
            except Exception:
                user_args = []
        wrapped_cmd: list[str] = [
            runtime,
            *connection_args,
            "run",
            "--rm",
            *user_args,
            "-v",
            f"{host_cfg_root}:{config.image_workdir}:Z",
            "-w",
            config.image_workdir,
        ]
        for k, v in container_env.items():
            wrapped_cmd.extend(["-e", f"{k}={v}"])
        wrapped_cmd.append(str(config.image))
        wrapped_cmd.extend(inner_cmd)

        # Important: do NOT keep stdin open (avoids "podman run never ends")
        return super().run(
            cmd=wrapped_cmd,
            env=env,
            cfg_root=cfg_root,
            config=config,
            spawn_error_message=spawn_error_message,
            stdin_devnull=True,
        )

    def _resolve_podman_connection_args(self) -> list[str]:
        """Return extra args for the `podman` CLI to select a connection.

        Priority:
        1) PODMAN_CONNECTION env var
        2) auto-detect default connection from `podman system connection list --format json`
        3) fallback: no args (let Podman decide; works on native Linux / preconfigured env)
        """
        # 1) environment override (nice for CI/users)
        env_name = os.environ.get("PODMAN_CONNECTION")
        if env_name:
            return ["--connection", env_name]

        # 2) auto-detect default connection
        try:
            proc = subprocess.run(
                ["podman", "system", "connection", "list", "--format", "json"],
                check=False,
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0:
                self._logger.info(
                    "podman system connection list failed (rc=%s): %s",
                    proc.returncode,
                    (proc.stderr or "").strip(),
                )
                return []

            data = json.loads(proc.stdout or "[]")
            # entries look like: {"Name": "...", "URI": "...", "Identity": "...", "Default": true, ...}
            default = next((x for x in data if x.get("Default") is True), None)
            if default and default.get("Name"):
                return [
                    "--url",
                    str(default["URI"]),
                    "--identity",
                    str(default["Identity"]),
                ]
        except Exception as exc:
            self._logger.info("Failed to auto-detect Podman connection: %r", exc)
            return []

        return []


class DockerRunner(ContainerRunner):
    """Run the inner command inside a container via Docker."""

    def __init__(self, log_path, logger):
        super().__init__(log_path, logger)

    def run(
        self,
        *,
        cmd: list[str],
        env: dict[str, str],
        cfg_root: Path,
        config: BaseCLIConfig,
        spawn_error_message: str,
    ) -> CLIResult:
        runtime = "docker"
        host_gateway = "host.docker.internal"
        host_cfg_root = str(cfg_root.resolve())

        # Patch mcp.json so container uses host gateway (not 127.0.0.1/localhost)
        self._patch_mcp_json(cfg_root=cfg_root, host_gateway=host_gateway)

        inner_cmd = self._rewrite_mcp_config_path(list(cmd), workdir=config.image_workdir)

        # Minimal env forwarding into container (encoded via docker -e flags)
        container_env = self._container_env_from(env, host_gateway=host_gateway, config=config)
        container_env["HOME"] = config.image_workdir
        user_args: list[str] = []
        uid = getattr(os, "getuid", None)
        gid = getattr(os, "getgid", None)
        if callable(uid) and callable(gid):
            try:
                user_args = ["--user", f"{uid()}:{gid()}"]
            except Exception:
                user_args = []
        wrapped_cmd: list[str] = [
            runtime,
            "run",
            "--rm",
            "--add-host",
            f"{host_gateway}:host-gateway",
            *user_args,
            "-v",
            f"{host_cfg_root}:{config.image_workdir}",
            "-w",
            config.image_workdir,
        ]
        for k, v in container_env.items():
            wrapped_cmd.extend(["-e", f"{k}={v}"])
        wrapped_cmd.append(str(config.image))
        wrapped_cmd.extend(inner_cmd)

        # Important: do NOT keep stdin open (avoids "docker run never ends")
        return super().run(
            cmd=wrapped_cmd,
            env=env,
            cfg_root=cfg_root,
            config=config,
            spawn_error_message=spawn_error_message,
            stdin_devnull=True,
        )
