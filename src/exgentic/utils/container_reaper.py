# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Container reaper — prevent orphaned Docker containers leaking on unclean exit.

When ``exgentic`` spawns Docker containers (directly via ``DockerRunner`` or
indirectly through benchmark/agent subprocesses such as ``minisweagent``),
the container may outlive its owning Python process if that process dies
abnormally (SIGKILL, uncaught exception, crashed grandchild, …).  Because
containers live inside the Docker daemon — outside the process group of
their spawner — process-group kills do not touch them.

The reaper addresses this with two complementary mechanisms:

1. **Labeled-container reaping.**  Every container we spawn is tagged with
   ``exgentic.owner_pid=<pid>``.  The :func:`reap_orphaned_containers`
   helper lists all containers bearing that label whose owner PID is no
   longer a running process and removes them.  Call it at batch / run
   startup to sweep leftovers from prior crashed runs.

2. **Graceful process-exit cleanup.**  :func:`install_cleanup_handlers`
   registers an ``atexit`` hook plus ``SIGTERM`` / ``SIGINT`` handlers that
   reap containers labeled with the *current* PID, catching the common
   case where the parent receives a termination signal and must shut down
   its children before the process group dies.

Callers instrument ``docker run`` invocations by prepending the flags
returned by :func:`docker_run_label_args` to their command lines.  All
exgentic-owned containers therefore share a single identifying label
scheme and can be swept in a single pass.
"""

from __future__ import annotations

import atexit
import logging
import os
import shutil
import signal
import subprocess
from collections.abc import Iterable

# All exgentic-owned containers carry this label.  The value is the PID of
# the Python process that created the container.
LABEL_OWNER_PID = "exgentic.owner_pid"

_handlers_installed = False
_log = logging.getLogger(__name__)


def docker_run_label_args(pid: int | None = None) -> list[str]:
    """Return ``["--label", "exgentic.owner_pid=<pid>"]`` for a ``docker run``.

    Caller splices the result into its ``docker run`` argument list so the
    spawned container is tagged with the creator's PID.  When ``pid`` is
    ``None`` the current process PID is used.
    """
    if pid is None:
        pid = os.getpid()
    return ["--label", f"{LABEL_OWNER_PID}={pid}"]


def _docker_bin() -> str | None:
    return shutil.which("docker")


def _pid_alive(pid: int) -> bool:
    """Return True if ``pid`` is a running process on this host.

    ``os.kill(pid, 0)`` raises ``ProcessLookupError`` for dead PIDs and
    ``PermissionError`` for processes owned by another user (which is
    still alive, so we treat it as True).
    """
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _list_labeled_containers(
    *,
    runner: object | None = None,
) -> list[tuple[str, int]]:
    """Return ``[(container_id, owner_pid), ...]`` for all labeled containers.

    Uses ``docker ps -a --filter label=exgentic.owner_pid`` so stopped
    containers are included as well — a crashed process may have left a
    container behind in ``Created`` or ``Exited`` state.
    """
    binary = _docker_bin()
    if binary is None:
        return []
    run = runner or subprocess.run
    try:
        result = run(
            [
                binary,
                "ps",
                "-a",
                "--filter",
                f"label={LABEL_OWNER_PID}",
                "--format",
                '{{.ID}}\t{{.Label "' + LABEL_OWNER_PID + '"}}',
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        return []
    if result.returncode != 0:
        return []
    out: list[tuple[str, int]] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        cid, pid_str = parts[0], parts[1]
        try:
            pid = int(pid_str)
        except ValueError:
            continue
        out.append((cid, pid))
    return out


def _remove_containers(
    ids: Iterable[str],
    *,
    runner: object | None = None,
) -> int:
    binary = _docker_bin()
    if binary is None:
        return 0
    run = runner or subprocess.run
    removed = 0
    for cid in ids:
        try:
            result = run(
                [binary, "rm", "-f", cid],
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except Exception:
            continue
        if result.returncode == 0:
            removed += 1
    return removed


def reap_orphaned_containers(
    *,
    logger: logging.Logger | None = None,
    runner: object | None = None,
) -> int:
    """Remove labeled containers whose owner PID is no longer running.

    Safe to call concurrently with active sibling exgentic processes: only
    containers whose recorded owner PID is *dead* are removed.  Returns
    the number of containers removed.
    """
    log = logger or _log
    stale: list[str] = []
    for cid, owner_pid in _list_labeled_containers(runner=runner):
        if not _pid_alive(owner_pid):
            stale.append(cid)
    if not stale:
        return 0
    log.warning(
        "Reaping %d orphaned exgentic container(s) from dead owners.",
        len(stale),
    )
    return _remove_containers(stale, runner=runner)


def reap_own_containers(
    *,
    pid: int | None = None,
    logger: logging.Logger | None = None,
    runner: object | None = None,
) -> int:
    """Remove all containers labeled with ``pid`` (default: current PID).

    Used at graceful process exit to clean up containers owned by *this*
    process without touching containers belonging to sibling processes.
    """
    log = logger or _log
    target_pid = os.getpid() if pid is None else pid
    own: list[str] = []
    for cid, owner_pid in _list_labeled_containers(runner=runner):
        if owner_pid == target_pid:
            own.append(cid)
    if not own:
        return 0
    log.info("Cleaning up %d exgentic container(s) owned by PID %d.", len(own), target_pid)
    return _remove_containers(own, runner=runner)


def install_cleanup_handlers(
    *,
    logger: logging.Logger | None = None,
) -> None:
    """Install atexit + SIGTERM / SIGINT handlers that reap own containers.

    Idempotent: repeat calls are no-ops.  Signal handlers chain to the
    previously-installed handler so existing behaviour (e.g. ``KeyboardInterrupt``
    on SIGINT) is preserved.
    """
    global _handlers_installed
    if _handlers_installed:
        return
    _handlers_installed = True

    def _cleanup_once() -> None:
        try:
            reap_own_containers(logger=logger)
        except Exception:
            # Cleanup is best-effort; never raise from a shutdown path.
            pass

    atexit.register(_cleanup_once)

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            prev = signal.getsignal(sig)
        except (ValueError, OSError):
            continue

        def _make_handler(previous, signum=sig):
            def _handler(signum_arg, frame):
                _cleanup_once()
                if callable(previous):
                    previous(signum_arg, frame)
                elif previous == signal.SIG_DFL:
                    # Restore and re-raise so default disposition still applies.
                    signal.signal(signum_arg, signal.SIG_DFL)
                    os.kill(os.getpid(), signum_arg)

            return _handler

        try:
            signal.signal(sig, _make_handler(prev))
        except (ValueError, OSError):
            # Signals may not be installable off the main thread or under
            # some test runners; cleanup still runs via atexit.
            continue
