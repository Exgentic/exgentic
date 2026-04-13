# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

from exgentic.core.types.run import (
    _load_cached_task_ids,
    _save_cached_task_ids,
    _task_ids_cache_key,
)


def _fake_manager(tmp_path):
    from exgentic.environment.manager import EnvironmentManager

    return EnvironmentManager(base_dir=tmp_path)


def test_save_and_load_round_trip(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "exgentic.environment.instance.get_manager",
        lambda: _fake_manager(tmp_path),
    )
    key = _task_ids_cache_key("gsm8k", "main")
    ids = ["0", "1", "2", "42"]
    _save_cached_task_ids("gsm8k", key, ids)

    result = _load_cached_task_ids("gsm8k", key)
    assert result == ids


def test_load_returns_none_when_no_cache(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "exgentic.environment.instance.get_manager",
        lambda: _fake_manager(tmp_path),
    )
    key = _task_ids_cache_key("gsm8k", "main")
    assert _load_cached_task_ids("gsm8k", key) is None


def test_load_returns_none_on_corrupt_file(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "exgentic.environment.instance.get_manager",
        lambda: _fake_manager(tmp_path),
    )
    cache_dir = tmp_path / "benchmarks" / "gsm8k"
    cache_dir.mkdir(parents=True)
    (cache_dir / "task_ids_main.json").write_text("not json{{{")

    key = _task_ids_cache_key("gsm8k", "main")
    assert _load_cached_task_ids("gsm8k", key) is None


def test_different_subsets_cached_separately(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "exgentic.environment.instance.get_manager",
        lambda: _fake_manager(tmp_path),
    )
    key_train = _task_ids_cache_key("appworld", "train")
    key_dev = _task_ids_cache_key("appworld", "dev")
    _save_cached_task_ids("appworld", key_train, ["a", "b"])
    _save_cached_task_ids("appworld", key_dev, ["x", "y", "z"])

    assert _load_cached_task_ids("appworld", key_train) == ["a", "b"]
    assert _load_cached_task_ids("appworld", key_dev) == ["x", "y", "z"]


def test_different_kwargs_cached_separately(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "exgentic.environment.instance.get_manager",
        lambda: _fake_manager(tmp_path),
    )
    key1 = _task_ids_cache_key("test", "main", {"tasks": ["a"]})
    key2 = _task_ids_cache_key("test", "main", {"tasks": ["a", "b"]})
    assert key1 != key2

    _save_cached_task_ids("test", key1, ["a"])
    _save_cached_task_ids("test", key2, ["a", "b"])

    assert _load_cached_task_ids("test", key1) == ["a"]
    assert _load_cached_task_ids("test", key2) == ["a", "b"]


def test_no_kwargs_same_as_none(tmp_path, monkeypatch) -> None:
    assert _task_ids_cache_key("x", "main") == _task_ids_cache_key("x", "main", None)
    assert _task_ids_cache_key("x", "main") == _task_ids_cache_key("x", "main", {})
