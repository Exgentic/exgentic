# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Container reaper: clean up orphaned exgentic-owned Docker containers.

Each container spawned by exgentic is tagged with two labels:

* ``exgentic.owner_pid`` — creator PID
* ``exgentic.owner_token`` — per-process UUID4 generated at import time

The PID alone is not sufficient: on abnormal termination a PID can be
recycled to an unrelated process, and a co-tenant could deliberately
spawn a container with our PID as its label (label-collision DoS). The
token closes both holes — ``reap_own_containers`` requires both labels
to match, and ``reap_orphaned_containers`` only removes containers whose
PID is dead on this host.
"""

from __future__ import annotations

import atexit
import contextlib
import logging
import os
import shutil
import signal
import subprocess
import uuid
from collections.abc import Iterable, Iterator

LABEL_OWNER_PID = "exgentic.owner_pid"
LABEL_OWNER_TOKEN = "exgentic.owner_token"

OWN_TOKEN = uuid.uuid4().hex

_handlers_installed = False
_log = logging.getLogger(__name__)


def docker_run_label_args(pid: int | None = None, token: str | None = None) -> list[str]:
    """Return ``--label`` flags for a ``docker run`` to tag our containers."""
    if pid is None:
        pid = os.getpid()
    if token is None:
        token = OWN_TOKEN
    return [
        "--label",
        f"{LABEL_OWNER_PID}={pid}",
        "--label",
        f"{LABEL_OWNER_TOKEN}={token}",
    ]


def _docker_bin() -> str | None:
    return shutil.which("docker")


def _pid_alive(pid: int) -> bool:
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
) -> list[tuple[str, int, str]]:
    """Return ``[(container_id, owner_pid, owner_token), ...]`` for our containers."""
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
                "--filter",
                f"label={LABEL_OWNER_TOKEN}",
                "--format",
                "{{.ID}}\t" + '{{.Label "' + LABEL_OWNER_PID + '"}}\t' + '{{.Label "' + LABEL_OWNER_TOKEN + '"}}',
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
    out: list[tuple[str, int, str]] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        cid, pid_str, token = parts[0], parts[1], parts[2]
        if not token:
            continue
        try:
            pid = int(pid_str)
        except ValueError:
            continue
        out.append((cid, pid, token))
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

    A container is reaped only when ``_pid_alive(owner_pid)`` is False.
    Containers with a live PID are left alone regardless of token —
    another exgentic process (or an unrelated process at a recycled PID)
    owns them.
    """
    log = logger or _log
    stale: list[str] = []
    for cid, owner_pid, _owner_token in _list_labeled_containers(runner=runner):
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
    token: str | None = None,
    logger: logging.Logger | None = None,
    runner: object | None = None,
) -> int:
    """Remove containers tagged with both our PID and our token."""
    log = logger or _log
    target_pid = os.getpid() if pid is None else pid
    target_token = OWN_TOKEN if token is None else token
    own: list[str] = []
    for cid, owner_pid, owner_token in _list_labeled_containers(runner=runner):
        if owner_pid == target_pid and owner_token == target_token:
            own.append(cid)
    if not own:
        return 0
    log.info("Cleaning up %d exgentic container(s) owned by PID %d.", len(own), target_pid)
    return _remove_containers(own, runner=runner)


@contextlib.contextmanager
def docker_sdk_label_injection(
    pid: int | None = None,
    token: str | None = None,
) -> Iterator[None]:
    """Patch ``docker`` SDK so containers created via it carry our labels.

    SWE-bench harness spawns ``sweb.eval.*`` containers via the Python
    Docker SDK (``ContainerCollection.create``), not the docker CLI, so
    ``--label`` flags never reach them. This wraps ``create`` while the
    block runs to splice in the owner labels. If ``docker`` isn't
    importable (host process has no Docker SDK), the block is a no-op.
    """
    owner_pid = os.getpid() if pid is None else pid
    owner_token = OWN_TOKEN if token is None else token
    labels = {LABEL_OWNER_PID: str(owner_pid), LABEL_OWNER_TOKEN: owner_token}

    try:
        from docker.models.containers import ContainerCollection
    except Exception:
        yield
        return

    original = ContainerCollection.create

    def patched(self, image, command=None, **kwargs):
        existing = kwargs.get("labels") or {}
        if isinstance(existing, list):
            merged = list(existing)
            merged.extend(f"{k}={v}" for k, v in labels.items())
            kwargs["labels"] = merged
        else:
            merged_map = dict(existing)
            merged_map.update(labels)
            kwargs["labels"] = merged_map
        return original(self, image, command=command, **kwargs)

    ContainerCollection.create = patched
    try:
        yield
    finally:
        ContainerCollection.create = original


def install_cleanup_handlers(
    *,
    logger: logging.Logger | None = None,
) -> None:
    """Register atexit + SIGTERM / SIGINT handlers that reap own containers.

    Idempotent. Signal handlers reap, then invoke the previous handler if
    it was a callable; for ``SIG_DFL`` they re-raise the signal under the
    default disposition; for ``SIG_IGN`` they return silently.
    """
    global _handlers_installed
    if _handlers_installed:
        return
    _handlers_installed = True

    def _cleanup_once() -> None:
        try:
            reap_own_containers(logger=logger)
        except Exception:
            pass

    atexit.register(_cleanup_once)

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            prev = signal.getsignal(sig)
        except (ValueError, OSError):
            continue

        def _handler(signum, frame, previous=prev):
            _cleanup_once()
            if callable(previous):
                previous(signum, frame)
            elif previous == signal.SIG_DFL:
                signal.signal(signum, signal.SIG_DFL)
                os.kill(os.getpid(), signum)

        try:
            signal.signal(sig, _handler)
        except (ValueError, OSError):
            continue
