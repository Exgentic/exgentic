# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

from exgentic.core.context import Context, Role, set_context
from exgentic.integrations.litellm.cache.log import CacheLogger


def test_cache_logger_uses_context_session_path(tmp_path) -> None:
    ctx = Context(
        run_id="run-cache",
        output_dir=str(tmp_path),
        cache_dir=str(tmp_path / "cache"),
        session_id="sess-1",
        role=Role.AGENT,
    )
    set_context(ctx)

    logger = CacheLogger(disk_cache_dir=str(tmp_path / "cache"), strip_time=False)
    logger.hit()

    expected = tmp_path / "run-cache" / "sessions" / "sess-1" / "agent" / "litellm" / "cache.log"
    assert expected.exists()
