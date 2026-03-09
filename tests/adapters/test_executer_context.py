# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

import pytest
from exgentic.adapters.executors.executer import RemoteProcessExecuter
from exgentic.core.context import Context, set_context


class _ContextEcho:
    def __init__(self):
        pass

    def run_id(self) -> str:
        from exgentic.core.context import try_get_context

        ctx = try_get_context()
        return ctx.run_id if ctx is not None else ""


def test_remote_process_executer_passes_context():
    ctx = Context(run_id="run-ctx", output_dir="/tmp/out", cache_dir="/tmp/cache")
    set_context(ctx)

    exe = RemoteProcessExecuter(_ContextEcho)
    try:
        exe.start()
    except PermissionError:
        pytest.skip("multiprocessing semaphores not available in this environment")
    try:
        assert exe.call("run_id") == "run-ctx"
    finally:
        exe.shutdown()
