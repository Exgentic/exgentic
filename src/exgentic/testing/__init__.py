# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Lightweight test fixtures for integration testing.

These classes are intentionally kept inside the installed package so they
are available inside Docker containers (where ``tests.*`` is not installed).
They are *not* registered in the benchmark/agent registry by default.
"""

from .agent import (
    BAD_ACTION_TYPE,
    FINISH_ACTION_TYPE,
    GOOD_ACTION_TYPE,
    BadAction,
    EmptyArgs,
    FinishAction,
    GoodAction,
    TestAgent,
    TestAgentInstance,
)
from .benchmark import TestBenchmark, TestEvaluator, TestSession
from .calculator import Calculator, CalculatorError
from .docker_session import DockerSession

__all__ = [
    "BAD_ACTION_TYPE",
    "BadAction",
    "Calculator",
    "CalculatorError",
    "DockerSession",
    "EmptyArgs",
    "FINISH_ACTION_TYPE",
    "FinishAction",
    "GOOD_ACTION_TYPE",
    "GoodAction",
    "TestAgent",
    "TestAgentInstance",
    "TestBenchmark",
    "TestEvaluator",
    "TestSession",
]
