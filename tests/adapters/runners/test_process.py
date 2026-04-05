# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Process-specific tests that verify actual process isolation."""

from __future__ import annotations

import gc
import os

import pytest
from exgentic.adapters.runners import with_runner

from .conftest import Calculator


def _make_process_calc(**kwargs):
    try:
        return with_runner(Calculator, runner="process", **kwargs)
    except PermissionError:
        pytest.skip("multiprocessing semaphores not available")


def test_runs_in_different_pid():
    calc = _make_process_calc(value=0)
    try:
        assert calc.pid() != os.getpid()
    finally:
        calc.close()


def test_crash_isolation():
    """Errors in the remote process don't crash the proxy."""
    calc = _make_process_calc(value=0)
    try:
        with pytest.raises(ZeroDivisionError):
            calc.divide(1, 0)
        assert calc.add(1, 2) == 3
    finally:
        calc.close()
        # Force cleanup of multiprocessing resources to avoid interference
        # with subsequent thread tests (CPython 3.11 bug workaround).
        gc.collect()
