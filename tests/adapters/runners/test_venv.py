# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Venv runner tests (skipped when uv is unavailable)."""

from __future__ import annotations

import os
import shutil

import pytest
from exgentic.adapters.runners import with_runner

from .conftest import Calculator

# Skip entire module if uv is not available.
_uv_available = shutil.which("uv") is not None

pytestmark = [
    pytest.mark.skipif(not _uv_available, reason="uv not available"),
]


@pytest.fixture(scope="module")
def venv_calc():
    """Shared venv-backed Calculator (venv creation is slow)."""
    proxy = with_runner(Calculator, runner="venv", value=10)
    yield proxy
    try:
        proxy.close()
    except Exception:
        pass


def test_call_method(venv_calc):
    assert venv_calc.add(2, 3) == 5


def test_accumulate(venv_calc):
    assert venv_calc.accumulate(5) == 15


def test_get_attribute(venv_calc):
    assert venv_calc.value == 15


def test_set_attribute(venv_calc):
    venv_calc.value = 42
    assert venv_calc.value == 42


def test_error_propagation(venv_calc):
    with pytest.raises(ZeroDivisionError):
        venv_calc.divide(1, 0)


def test_echo(venv_calc):
    assert venv_calc.echo({"key": [1, 2, 3]}) == {"key": [1, 2, 3]}


def test_different_pid(venv_calc):
    """Venv runner should run in a separate process."""
    assert venv_calc.pid() != os.getpid()
