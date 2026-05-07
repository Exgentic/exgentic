# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Anonymous Authors.

from __future__ import annotations

from framework.core.context import run_scope, try_get_context
from framework.core.types import RunConfig
from framework.integrations.litellm.cache_utils import build_litellm_cache
from framework.utils.settings import FrameworkSettings, resolve_cache_path


def test_resolve_cache_path_uses_base_dir_for_relative_paths() -> None:
    assert resolve_cache_path(".framework", ".litellm_cache") == ".framework/.litellm_cache"


def test_resolve_cache_path_keeps_absolute_paths() -> None:
    assert resolve_cache_path(".framework", "/tmp/litellm") == "/tmp/litellm"


def test_build_litellm_cache_resolves_relative_path_under_cache_dir(tmp_path) -> None:
    base_dir = tmp_path / "cache-root"
    settings = FrameworkSettings(
        cache_dir=str(base_dir),
        litellm_cache_dir=".litellm_cache",
    )
    cache = build_litellm_cache(settings)
    assert cache.cache.disk_cache.directory == str(base_dir / ".litellm_cache")


def test_run_config_to_session_config_preserves_cache_dir() -> None:
    run_config = RunConfig(
        benchmark="gsm8k",
        agent="tool_calling",
        cache_dir="/tmp/framework-cache",
    )
    session_config = run_config.to_session_config("task-1")
    assert session_config.cache_dir == "/tmp/framework-cache"


def test_run_context_sets_and_restores_cache_env(monkeypatch) -> None:
    monkeypatch.delenv("FRAMEWORK_CTX_CACHE_DIR", raising=False)
    with run_scope(
        run_id="cache-test",
        output_dir="./outputs",
        cache_dir="./cache",
    ):
        ctx = try_get_context()
        assert ctx is not None
        # Context resolves relative paths to absolute.
        assert ctx.cache_dir.endswith("/cache")
        assert not ctx.cache_dir.startswith(".")
