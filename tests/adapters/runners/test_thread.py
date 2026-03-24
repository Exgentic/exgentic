# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Thread-specific tests that verify actual thread isolation."""

from __future__ import annotations

import threading

from exgentic.adapters.runners import with_runner

from .conftest import Calculator


def test_runs_in_different_thread():
    calc = with_runner(Calculator, runner="thread", value=0)
    try:
        remote_tid = calc.thread_id()
        local_tid = threading.get_ident()
        assert remote_tid != local_tid
    finally:
        calc.close()


def test_runs_in_same_pid():
    import os

    calc = with_runner(Calculator, runner="thread", value=0)
    try:
        assert calc.pid() == os.getpid()
    finally:
        calc.close()


def test_close_joins_thread():
    calc = with_runner(Calculator, runner="thread", value=0)
    transport = object.__getattribute__(calc, "_transport")
    thread = transport._thread
    assert thread is not None and thread.is_alive()
    calc.close()
    assert not thread.is_alive()
