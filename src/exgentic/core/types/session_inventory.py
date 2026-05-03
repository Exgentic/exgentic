# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Single source of truth for filesystem scans of a session tree.

Multiple callers (``batch status``, ``_load_results_summary``, future
aggregation paths) need to walk ``sessions/*/results.json`` for the same
underlying data: the parsed payload and the file's mtime. Each used to
glob and stat independently; for a 4600-session tree on NFS that meant
two-to-three full walks per ``batch status`` invocation.

``scan_sessions`` does the walk once, in parallel, and returns typed
:class:`SessionRecord` snapshots. Consumers compose pure functions over
the records — no further I/O required.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_MAX_SCAN_WORKERS = 32
"""Upper bound on threads used to fan out per-session results reads."""


@dataclass(frozen=True)
class SessionRecord:
    """Filesystem snapshot of a single session's results artifact.

    Holds everything any consumer needs from a session directory after a
    single read: the parsed ``results.json`` payload and its mtime. Not
    a status enum — that's :class:`SessionStatus`'s job, which factors in
    lock files and the orchestrator's state machine.
    """

    session_dir: Path
    results_path: Path
    results: dict[str, Any] | None
    """Parsed ``results.json``; ``None`` if missing or unparseable."""
    mtime: float | None
    """``results.json`` mtime; ``None`` if the file is missing."""


def scan_sessions(
    sessions_dir: Path,
    *,
    max_workers: int = _MAX_SCAN_WORKERS,
) -> list[SessionRecord]:
    """Walk ``sessions_dir/*/results.json`` once, in parallel.

    Returns one :class:`SessionRecord` per existing ``results.json``.
    Sessions without a ``results.json`` are not included — callers that
    care about missing sessions should consult :class:`SessionStatus`.

    The walk is order-unstable: callers that care about input/output
    correspondence (the orchestrator) should use
    ``RunStatus.from_session_configs`` instead, which keeps task-id
    ordering. Callers here (status display, live aggregation) reduce
    the records and don't depend on order.
    """
    paths = list(sessions_dir.glob("*/results.json"))
    if not paths:
        return []
    workers = max(1, min(max_workers, len(paths)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_read_session_record, paths))


def _read_session_record(results_path: Path) -> SessionRecord:
    try:
        results: dict[str, Any] | None = json.loads(results_path.read_text(encoding="utf-8"))
        if not isinstance(results, dict):
            results = None
    except Exception:
        results = None
    try:
        mtime: float | None = results_path.stat().st_mtime
    except OSError:
        mtime = None
    return SessionRecord(
        session_dir=results_path.parent,
        results_path=results_path,
        results=results,
        mtime=mtime,
    )
