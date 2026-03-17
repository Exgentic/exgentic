# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Docker runner tests (skipped when Docker is unavailable)."""

from __future__ import annotations

import shutil
import subprocess
import sys

import pytest

from exgentic.adapters.runners import with_runner
from .conftest import Calculator

# Skip entire module if docker is not available.
_docker_available = shutil.which("docker") is not None
if _docker_available:
    try:
        subprocess.run(["docker", "info"], check=True, capture_output=True, timeout=5)
    except Exception:
        _docker_available = False

# The Docker image uses Python 3.12; cloudpickle payloads from other
# minor versions can fail to deserialize, so skip unless the host matches.
_python_matches = sys.version_info[:2] == (3, 12)

pytestmark = [
    pytest.mark.skipif(not _docker_available, reason="Docker not available"),
    pytest.mark.skipif(not _python_matches, reason="Docker image uses Python 3.12"),
]


@pytest.fixture(scope="module")
def docker_calc():
    """Shared docker-backed Calculator (building is slow)."""
    proxy = with_runner(Calculator, runner="docker", value=10)
    yield proxy
    try:
        proxy.close()
    except Exception:
        pass


def test_call_method(docker_calc):
    assert docker_calc.add(2, 3) == 5


def test_accumulate(docker_calc):
    assert docker_calc.accumulate(5) == 15


def test_get_attribute(docker_calc):
    assert docker_calc.value == 15


def test_set_attribute(docker_calc):
    docker_calc.value = 42
    assert docker_calc.value == 42


def test_error_propagation(docker_calc):
    with pytest.raises(ZeroDivisionError):
        docker_calc.divide(1, 0)


def test_echo(docker_calc):
    assert docker_calc.echo({"key": [1, 2, 3]}) == {"key": [1, 2, 3]}
