# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Venv runner tests (skipped when uv is unavailable)."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import time

import pytest
from exgentic.adapters.runners import with_runner
from exgentic.adapters.runners.venv import VenvRunner

from .conftest import Calculator

# Skip entire module if uv is not available.
_uv_available = shutil.which("uv") is not None

pytestmark = [
    pytest.mark.skipif(not _uv_available, reason="uv not available"),
]


@pytest.fixture(scope="module")
def venv_calc():
    """Shared venv-backed Calculator (venv creation is slow)."""
    proxy = with_runner(
        Calculator,
        runner="venv",
        env_name="tests/calculator",
        module_path="exgentic.testing.calculator",
        value=10,
    )
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


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@pytest.mark.skipif(platform.system() != "Linux", reason="killpg reaping semantics differ on macOS /proc")
def test_stop_process_reaps_grandchildren_via_killpg(tmp_path):
    pid_file = tmp_path / "grandchild.pid"
    script = (
        "import os, sys\n"
        f"pid_file = {str(pid_file)!r}\n"
        "gpid = os.fork()\n"
        "if gpid == 0:\n"
        "    with open(pid_file, 'w') as f:\n"
        "        f.write(str(os.getpid()))\n"
        "        f.flush()\n"
        "    os.execvp('sleep', ['sleep', '120'])\n"
        "os.execvp('sleep', ['sleep', '120'])\n"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and not pid_file.exists():
        time.sleep(0.05)
    assert pid_file.exists(), "grandchild did not record its PID"
    grandchild_pid = int(pid_file.read_text().strip())
    assert _pid_alive(grandchild_pid)

    runner = VenvRunner.__new__(VenvRunner)
    runner._process = proc
    runner._stop_process()

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and _pid_alive(grandchild_pid):
        time.sleep(0.05)
    assert not _pid_alive(grandchild_pid)
