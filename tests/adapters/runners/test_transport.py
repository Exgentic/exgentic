# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Behavioural contract tests — run against every available transport.

When a new transport is added, append it to ``_AVAILABLE_RUNNERS``
in ``conftest.py`` and all tests here will automatically cover it.
"""

from __future__ import annotations

import pytest

# ── method calls ─────────────────────────────────────────────────────


def test_call_method(calc):
    assert calc.add(2, 3) == 5


def test_call_method_kwargs(calc):
    assert calc.add(a=2, b=3) == 5


def test_call_method_mixed_args(calc):
    assert calc.add(2, b=3) == 5


# ── stateful operations ─────────────────────────────────────────────


def test_accumulate(calc):
    assert calc.accumulate(5) == 15  # started at 10
    assert calc.accumulate(5) == 20


# ── attribute access ─────────────────────────────────────────────────


def test_get_attribute(calc):
    assert calc.value == 10


def test_set_attribute(calc):
    calc.value = 42
    assert calc.value == 42


# ── error propagation ────────────────────────────────────────────────


def test_builtin_error(calc):
    with pytest.raises(ZeroDivisionError):
        calc.divide(1, 0)


def test_attribute_error(calc):
    with pytest.raises(AttributeError):
        _ = calc.nonexistent_attribute


def test_custom_exception(calc):
    """Custom (non-builtin) exceptions preserve type and attributes."""
    from .conftest import CalculatorError

    with pytest.raises(CalculatorError) as exc_info:
        calc.fail_custom()
    assert "something went wrong" in str(exc_info.value)
    assert exc_info.value.code == 42


def test_remote_traceback(calc):
    """Non-direct transports attach __remote_traceback__."""
    try:
        calc.divide(1, 0)
    except ZeroDivisionError as exc:
        if hasattr(exc, "__remote_traceback__"):
            assert isinstance(exc.__remote_traceback__, str)


# ── echo / serialization ────────────────────────────────────────────


def test_echo_int(calc):
    assert calc.echo(42) == 42


def test_echo_string(calc):
    assert calc.echo("hello") == "hello"


def test_echo_list(calc):
    assert calc.echo([1, 2, 3]) == [1, 2, 3]


def test_echo_dict(calc):
    assert calc.echo({"a": 1}) == {"a": 1}


def test_echo_none(calc):
    assert calc.echo(None) is None
