# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Shared fixtures for runner/transport tests.

The ``Calculator`` class and the parametrized ``calc`` fixture are used
across all transport test modules so that every transport is verified
against the exact same behavioural contract.
"""

from __future__ import annotations

import pytest
from exgentic.adapters.runners import with_runner
from exgentic.testing.calculator import Calculator, CalculatorError

# Re-export so existing test imports keep working.
__all__ = ["Calculator", "CalculatorError"]

# Runners available for the current milestone.
_AVAILABLE_RUNNERS = ["direct", "thread", "process", "service"]


@pytest.fixture(params=_AVAILABLE_RUNNERS)
def runner_name(request):
    return request.param


@pytest.fixture
def calc(runner_name):
    proxy = with_runner(Calculator, runner=runner_name, value=10)
    yield proxy
    try:
        proxy.close()
    except Exception:
        pass
