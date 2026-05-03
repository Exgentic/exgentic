# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Single-walk parallel scan of a session tree's results.json files."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_MAX_SCAN_WORKERS = 32


@dataclass(frozen=True)
class SessionRecord:
    """Parsed ``results.json`` and its mtime for one session.

    Distinct from :class:`SessionStatus`, which is the orchestrator's
    state-machine view (RUNNING/COMPLETED/INCOMPLETE/MISSING).
    """

    session_dir: Path
    results_path: Path
    results: dict[str, Any] | None
    mtime: float | None


def scan_sessions(
    sessions_dir: Path,
    *,
    max_workers: int = _MAX_SCAN_WORKERS,
) -> list[SessionRecord]:
    """Parallel walk of ``sessions_dir/*/results.json``.

    Order is unspecified. Sessions without a ``results.json`` are skipped.
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
