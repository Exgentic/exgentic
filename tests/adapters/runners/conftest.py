# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Shared fixtures for runner/transport tests.

The ``Calculator`` class and the parametrized ``calc`` fixture are used
across all transport test modules so that every transport is verified
against the exact same behavioural contract.
"""

from __future__ import annotations

import os
import threading

import pytest

from exgentic.adapters.runners import with_runner


class CalculatorError(Exception):
    """Custom exception for testing cross-transport error propagation."""

    def __init__(self, message: str, code: int = 0) -> None:
        super().__init__(message)
        self.code = code


class Calculator:
    """Dummy target for transport tests."""

    def __init__(self, value: int = 0) -> None:
        self.value = value

    def add(self, a: int, b: int) -> int:
        return a + b

    def accumulate(self, n: int) -> int:
        self.value += n
        return self.value

    def divide(self, a: int, b: int) -> float:
        return a / b

    def fail_custom(self) -> None:
        raise CalculatorError("something went wrong", code=42)

    def thread_id(self) -> int:
        return threading.get_ident()

    def pid(self) -> int:
        return os.getpid()

    def echo(self, obj: object) -> object:
        return obj


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
