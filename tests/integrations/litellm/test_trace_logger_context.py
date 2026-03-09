# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

import os
from pathlib import Path

from exgentic.core.context import (
    ENV_OTEL_SPAN_ID,
    ENV_OTEL_TRACE_ID,
    Context,
    init_context_from_env,
)
from exgentic.integrations.litellm.trace_logger import TraceLogger


def test_trace_logger_initializes_context_from_env(tmp_path: Path):
    ctx = Context(run_id="run-env", output_dir=str(tmp_path), cache_dir=str(tmp_path))
    env = ctx.to_env()
    os.environ.update(env)
    os.environ.pop(ENV_OTEL_TRACE_ID, None)
    os.environ.pop(ENV_OTEL_SPAN_ID, None)

    # Force env init.
    init_context_from_env()
    logger = TraceLogger()
    path = logger._resolve_log_path({})

    assert "run-env" in path

    for k in env:
        os.environ.pop(k, None)
