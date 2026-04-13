# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Tests for SessionStatus._extract_result_status."""

from __future__ import annotations

import json
from pathlib import Path

from exgentic.core.types.session import SessionOutcomeStatus, SessionStatus


def _write(path: Path, data: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_success(tmp_path: Path):
    p = _write(tmp_path / "results.json", {"status": "success"})
    assert SessionStatus._extract_result_status(p) == SessionOutcomeStatus.SUCCESS


def test_error(tmp_path: Path):
    p = _write(tmp_path / "results.json", {"status": "error"})
    assert SessionStatus._extract_result_status(p) == SessionOutcomeStatus.ERROR


def test_cancelled(tmp_path: Path):
    p = _write(
        tmp_path / "results.json",
        {"details": {"session_metadata": {"error_source": "cancelled"}}},
    )
    assert SessionStatus._extract_result_status(p) == SessionOutcomeStatus.CANCELLED


def test_agent_error(tmp_path: Path):
    p = _write(
        tmp_path / "results.json",
        {"details": {"session_metadata": {"error_source": "agent", "error": "crash"}}},
    )
    assert SessionStatus._extract_result_status(p) == SessionOutcomeStatus.ERROR


def test_finished_successful(tmp_path: Path):
    p = _write(tmp_path / "results.json", {"success": True, "is_finished": True})
    assert SessionStatus._extract_result_status(p) == SessionOutcomeStatus.SUCCESS


def test_finished_unsuccessful(tmp_path: Path):
    p = _write(tmp_path / "results.json", {"success": False, "is_finished": True})
    assert SessionStatus._extract_result_status(p) == SessionOutcomeStatus.UNSUCCESSFUL


def test_unfinished(tmp_path: Path):
    p = _write(tmp_path / "results.json", {"is_finished": False})
    assert SessionStatus._extract_result_status(p) == SessionOutcomeStatus.UNFINISHED


def test_empty_json_returns_unknown(tmp_path: Path):
    p = _write(tmp_path / "results.json", {})
    assert SessionStatus._extract_result_status(p) == SessionOutcomeStatus.UNKNOWN


def test_all_none_returns_unknown(tmp_path: Path):
    p = _write(tmp_path / "results.json", {"score": None, "success": None, "status": None})
    assert SessionStatus._extract_result_status(p) == SessionOutcomeStatus.UNKNOWN


def test_corrupt_json_returns_unknown(tmp_path: Path):
    p = tmp_path / "results.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("not json", encoding="utf-8")
    assert SessionStatus._extract_result_status(p) == SessionOutcomeStatus.UNKNOWN


def test_missing_file_returns_unknown(tmp_path: Path):
    p = tmp_path / "results.json"
    assert SessionStatus._extract_result_status(p) == SessionOutcomeStatus.UNKNOWN
