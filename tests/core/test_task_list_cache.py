# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

from exgentic.core.types.run import (
    _load_cached_task_ids,
    _save_cached_task_ids,
)


def _fake_manager(tmp_path):
    from exgentic.environment.manager import EnvironmentManager

    return EnvironmentManager(base_dir=tmp_path)


def test_save_and_load_round_trip(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "exgentic.environment.instance.get_manager",
        lambda: _fake_manager(tmp_path),
    )
    ids = ["0", "1", "2", "42"]
    _save_cached_task_ids("gsm8k", "main", ids)

    result = _load_cached_task_ids("gsm8k", "main")
    assert result == ids


def test_load_returns_none_when_no_cache(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "exgentic.environment.instance.get_manager",
        lambda: _fake_manager(tmp_path),
    )
    assert _load_cached_task_ids("gsm8k", "main") is None


def test_load_returns_none_on_corrupt_file(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "exgentic.environment.instance.get_manager",
        lambda: _fake_manager(tmp_path),
    )
    cache_dir = tmp_path / "benchmarks" / "gsm8k"
    cache_dir.mkdir(parents=True)
    (cache_dir / "task_ids_main.json").write_text("not json{{{")

    assert _load_cached_task_ids("gsm8k", "main") is None


def test_different_subsets_cached_separately(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "exgentic.environment.instance.get_manager",
        lambda: _fake_manager(tmp_path),
    )
    _save_cached_task_ids("appworld", "train", ["a", "b"])
    _save_cached_task_ids("appworld", "dev", ["x", "y", "z"])

    assert _load_cached_task_ids("appworld", "train") == ["a", "b"]
    assert _load_cached_task_ids("appworld", "dev") == ["x", "y", "z"]
