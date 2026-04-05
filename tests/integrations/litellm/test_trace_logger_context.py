# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

import os
from pathlib import Path

from exgentic.core.context import (
    Context,
    Role,
    init_context,
    save_service_runtime,
    set_context,
)
from exgentic.integrations.litellm.trace_logger import TraceLogger


def test_trace_logger_initializes_context_from_env(tmp_path: Path):
    ctx = Context(
        run_id="run-env",
        output_dir=str(tmp_path),
        cache_dir=str(tmp_path),
        session_id="sess-1",
    )
    set_context(ctx)

    # Write a per-service runtime.json for the benchmark role.
    path = save_service_runtime(Role.BENCHMARK)
    assert path.exists()

    old_val = os.environ.get("EXGENTIC_RUNTIME_FILE")
    os.environ["EXGENTIC_RUNTIME_FILE"] = str(path)
    try:
        init_context()
        logger = TraceLogger()
        log_path = logger._resolve_log_path({})
        assert "run-env" in log_path
    finally:
        if old_val is None:
            os.environ.pop("EXGENTIC_RUNTIME_FILE", None)
        else:
            os.environ["EXGENTIC_RUNTIME_FILE"] = old_val
