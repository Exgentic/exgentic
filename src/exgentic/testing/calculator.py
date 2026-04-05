# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Dummy Calculator class for runner/transport tests.

Lives inside the installed package so it is importable inside Docker
containers (where ``tests.*`` is not installed).
"""

from __future__ import annotations

import os
import threading


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
