# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

from exgentic.core.context import run_scope, try_get_context
from exgentic.core.types import RunConfig
from exgentic.integrations.litellm.cache_utils import build_litellm_cache
from exgentic.utils.settings import ExgenticSettings, resolve_cache_path


def test_resolve_cache_path_uses_base_dir_for_relative_paths() -> None:
    assert resolve_cache_path(".exgentic_cache", ".litellm_cache") == ".exgentic_cache/.litellm_cache"


def test_resolve_cache_path_keeps_absolute_paths() -> None:
    assert resolve_cache_path(".exgentic_cache", "/tmp/litellm") == "/tmp/litellm"


def test_build_litellm_cache_resolves_relative_path_under_cache_dir(tmp_path) -> None:
    base_dir = tmp_path / "cache-root"
    settings = ExgenticSettings(
        cache_dir=str(base_dir),
        litellm_cache_dir=".litellm_cache",
    )
    cache = build_litellm_cache(settings)
    assert cache.cache.disk_cache.directory == str(base_dir / ".litellm_cache")


def test_run_config_to_session_config_preserves_cache_dir() -> None:
    run_config = RunConfig(
        benchmark="tau2",
        agent="tool_calling",
        cache_dir="/tmp/exgentic-cache",
    )
    session_config = run_config.to_session_config("task-1")
    assert session_config.cache_dir == "/tmp/exgentic-cache"


def test_run_context_sets_and_restores_cache_env(monkeypatch) -> None:
    monkeypatch.delenv("EXGENTIC_CTX_CACHE_DIR", raising=False)
    with run_scope(
        run_id="cache-test",
        output_dir="./outputs",
        cache_dir="./cache",
    ):
        ctx = try_get_context()
        assert ctx is not None
        assert ctx.cache_dir == "./cache"
