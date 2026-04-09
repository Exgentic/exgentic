# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from types import SimpleNamespace

import pytest
from exgentic.interfaces.dashboard.views.forms import _collect_values


def _ctrl(value):
    """Minimal mock for a NiceGUI control with a .value attribute."""
    return SimpleNamespace(value=value)


class TestCollectValuesSelect:
    def test_basic(self):
        controls = {"mode": ("select", _ctrl("fast"), False)}
        assert _collect_values(controls) == {"mode": "fast"}

    def test_optional_none(self):
        controls = {"mode": ("select", _ctrl(None), True)}
        assert _collect_values(controls) == {"mode": None}

    def test_optional_empty_string(self):
        controls = {"mode": ("select", _ctrl(""), True)}
        assert _collect_values(controls) == {"mode": None}

    def test_required_empty_string_kept(self):
        controls = {"mode": ("select", _ctrl(""), False)}
        assert _collect_values(controls) == {"mode": ""}


class TestCollectValuesNumber:
    def test_int(self):
        controls = {"count": ("number", _ctrl(5), False, int)}
        assert _collect_values(controls) == {"count": 5}
        assert isinstance(_collect_values(controls)["count"], int)

    def test_float(self):
        controls = {"temp": ("number", _ctrl(0.7), False, float)}
        assert _collect_values(controls) == {"temp": pytest.approx(0.7)}
        assert isinstance(_collect_values(controls)["temp"], float)

    def test_optional_none(self):
        controls = {"count": ("number", _ctrl(None), True, int)}
        assert _collect_values(controls) == {"count": None}

    def test_required_none_defaults_to_zero(self):
        controls = {"count": ("number", _ctrl(None), False, int)}
        assert _collect_values(controls) == {"count": 0}

    def test_string_coerced_to_int(self):
        controls = {"count": ("number", _ctrl("10"), False, int)}
        assert _collect_values(controls) == {"count": 10}

    def test_float_truncated_to_int(self):
        controls = {"count": ("number", _ctrl(3.9), False, int)}
        assert _collect_values(controls) == {"count": 3}


class TestCollectValuesCheckbox:
    def test_true(self):
        controls = {"verbose": ("checkbox", _ctrl(True), False)}
        assert _collect_values(controls) == {"verbose": True}

    def test_false(self):
        controls = {"verbose": ("checkbox", _ctrl(False), False)}
        assert _collect_values(controls) == {"verbose": False}

    def test_none_is_falsy(self):
        controls = {"verbose": ("checkbox", _ctrl(None), False)}
        assert _collect_values(controls) == {"verbose": False}


class TestCollectValuesJson:
    def test_dict(self):
        controls = {"extra": ("json", _ctrl('{"a": 1}'), False, "dict")}
        assert _collect_values(controls) == {"extra": {"a": 1}}

    def test_list(self):
        controls = {"tags": ("json", _ctrl("[1, 2]"), False, "list")}
        assert _collect_values(controls) == {"tags": [1, 2]}

    def test_empty_string_dict(self):
        controls = {"extra": ("json", _ctrl(""), False, "dict")}
        assert _collect_values(controls) == {"extra": {}}

    def test_empty_string_list(self):
        controls = {"tags": ("json", _ctrl(""), False, "list")}
        assert _collect_values(controls) == {"tags": []}

    def test_none_dict(self):
        controls = {"extra": ("json", _ctrl(None), False, "dict")}
        assert _collect_values(controls) == {"extra": {}}

    def test_invalid_json_dict_fallback(self):
        controls = {"extra": ("json", _ctrl("not json"), False, "dict")}
        assert _collect_values(controls) == {"extra": {}}

    def test_invalid_json_list_fallback(self):
        controls = {"tags": ("json", _ctrl("{bad"), False, "list")}
        assert _collect_values(controls) == {"tags": []}


class TestCollectValuesText:
    def test_basic(self):
        controls = {"name": ("text", _ctrl("hello"), False)}
        assert _collect_values(controls) == {"name": "hello"}

    def test_optional_none(self):
        controls = {"name": ("text", _ctrl(None), True)}
        assert _collect_values(controls) == {"name": None}

    def test_optional_empty(self):
        controls = {"name": ("text", _ctrl(""), True)}
        assert _collect_values(controls) == {"name": None}

    def test_required_empty_kept(self):
        controls = {"name": ("text", _ctrl(""), False)}
        assert _collect_values(controls) == {"name": ""}


class TestCollectValuesMultipleFields:
    def test_mixed(self):
        controls = {
            "benchmark": ("select", _ctrl("swe-bench"), False),
            "num_tasks": ("number", _ctrl(10), False, int),
            "verbose": ("checkbox", _ctrl(True), False),
            "extra": ("json", _ctrl('{"k": "v"}'), False, "dict"),
            "model": ("text", _ctrl("gpt-4"), False),
        }
        result = _collect_values(controls)
        assert result == {
            "benchmark": "swe-bench",
            "num_tasks": 10,
            "verbose": True,
            "extra": {"k": "v"},
            "model": "gpt-4",
        }

    def test_empty_controls(self):
        assert _collect_values({}) == {}
