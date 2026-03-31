# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

import os
from pathlib import Path

from exgentic.core.context import (
    Context,
    init_context,
    save_runtime,
    set_context,
)
from exgentic.integrations.litellm.trace_logger import TraceLogger


def test_trace_logger_initializes_context_from_env(tmp_path: Path):
    ctx = Context(run_id="run-env", output_dir=str(tmp_path), cache_dir=str(tmp_path))
    set_context(ctx)

    # Write runtime.json to the run dir (tmp_path/run-env is the run root).
    runtime_dir = tmp_path / "run-env"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    save_runtime(runtime_dir)

    old_val = os.environ.get("EXGENTIC_RUNTIME_DIR")
    os.environ["EXGENTIC_RUNTIME_DIR"] = str(runtime_dir)
    try:
        init_context()
        logger = TraceLogger()
        path = logger._resolve_log_path({})
        assert "run-env" in path
    finally:
        if old_val is None:
            os.environ.pop("EXGENTIC_RUNTIME_DIR", None)
        else:
            os.environ["EXGENTIC_RUNTIME_DIR"] = old_val
