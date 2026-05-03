# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

import json
from pathlib import Path

from exgentic.core.types.session_inventory import scan_sessions


def _write_session(sessions_dir: Path, session_id: str, payload: dict | None) -> Path:
    session_dir = sessions_dir / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    if payload is not None:
        (session_dir / "results.json").write_text(json.dumps(payload), encoding="utf-8")
    return session_dir


def test_empty_sessions_dir(tmp_path: Path):
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    assert scan_sessions(sessions_dir) == []


def test_missing_sessions_dir(tmp_path: Path):
    assert scan_sessions(tmp_path / "nonexistent") == []


def test_skips_sessions_without_results_json(tmp_path: Path):
    sessions_dir = tmp_path / "sessions"
    _write_session(sessions_dir, "with-results", {"score": 1.0})
    _write_session(sessions_dir, "no-results", None)
    records = scan_sessions(sessions_dir)
    ids = {r.session_dir.name for r in records}
    assert ids == {"with-results"}


def test_parses_results_payload(tmp_path: Path):
    sessions_dir = tmp_path / "sessions"
    _write_session(sessions_dir, "s1", {"score": 0.42, "agent_cost": 1.5})
    [r] = scan_sessions(sessions_dir)
    assert r.results == {"score": 0.42, "agent_cost": 1.5}
    assert r.results_path == sessions_dir / "s1" / "results.json"
    assert r.session_dir == sessions_dir / "s1"
    assert r.mtime is not None


def test_corrupt_json_yields_record_with_none_results(tmp_path: Path):
    sessions_dir = tmp_path / "sessions"
    session_dir = sessions_dir / "broken"
    session_dir.mkdir(parents=True)
    (session_dir / "results.json").write_text("not json", encoding="utf-8")
    [r] = scan_sessions(sessions_dir)
    assert r.results is None
    assert r.mtime is not None


def test_non_dict_json_yields_none_results(tmp_path: Path):
    sessions_dir = tmp_path / "sessions"
    session_dir = sessions_dir / "list-payload"
    session_dir.mkdir(parents=True)
    (session_dir / "results.json").write_text("[1, 2, 3]", encoding="utf-8")
    [r] = scan_sessions(sessions_dir)
    assert r.results is None


def test_parallel_scan_returns_all_records(tmp_path: Path):
    sessions_dir = tmp_path / "sessions"
    for i in range(64):  # > pool size of 32 forces real parallelism
        _write_session(sessions_dir, f"s{i:02d}", {"score": i / 100})
    records = scan_sessions(sessions_dir)
    assert len(records) == 64
    scores = sorted(r.results["score"] for r in records)
    assert scores == [i / 100 for i in range(64)]
