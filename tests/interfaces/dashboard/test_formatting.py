# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.


from exgentic.interfaces.dashboard.views.formatting import (
    _format_error_message,
    _format_payload,
    _format_value,
    _normalize_action_payload,
    _normalize_observation_item,
    _normalize_observation_payload,
)


class TestFormatValue:
    def test_none(self):
        assert _format_value(None) == "-"

    def test_nan(self):
        assert _format_value(float("nan")) == "NaN"

    def test_integer_float(self):
        assert _format_value(100.0) == "100"

    def test_large_float(self):
        result = _format_value(123.456)
        assert result == "123.46"

    def test_medium_float(self):
        result = _format_value(1.5)
        assert result == "1.5"

    def test_small_float(self):
        result = _format_value(0.001234)
        assert result == "0.001234"

    def test_zero(self):
        assert _format_value(0.0) == "0"

    def test_string(self):
        assert _format_value("hello") == "hello"

    def test_negative_large(self):
        result = _format_value(-200.5)
        assert result == "-200.5"

    def test_negative_small(self):
        result = _format_value(-0.0005)
        assert result == "-0.0005"


class TestFormatPayload:
    def test_dict(self):
        result = _format_payload({"key": "value"})
        assert '"key"' in result
        assert '"value"' in result

    def test_truncation(self):
        result = _format_payload({"data": "x" * 5000}, limit=100)
        assert result.endswith("...")
        assert len(result) == 103  # 100 + "..."

    def test_non_serializable(self):
        result = _format_payload(object())
        assert isinstance(result, str)

    def test_list(self):
        result = _format_payload([1, 2, 3])
        assert "1" in result


class TestFormatErrorMessage:
    def test_string(self):
        assert _format_error_message("some error") == "some error"

    def test_string_truncation(self):
        long_msg = "x" * 5000
        result = _format_error_message(long_msg, limit=100)
        assert len(result) == 103
        assert result.endswith("...")

    def test_non_string_delegates_to_format_payload(self):
        result = _format_error_message({"error": "detail"})
        assert "error" in result


class TestNormalizeActionPayload:
    def test_none(self):
        assert _normalize_action_payload(None) == {"name": None, "arguments": None}

    def test_single_action(self):
        payload = {"name": "click", "arguments": {"x": 10}}
        result = _normalize_action_payload(payload)
        assert result == {"name": "click", "arguments": {"x": 10}}

    def test_parallel_actions(self):
        payload = {
            "type": "parallel",
            "actions": [
                {"name": "a1", "arguments": {"k": "v"}},
                {"name": "a2", "arguments": None},
            ],
        }
        result = _normalize_action_payload(payload)
        assert result["mode"] == "parallel"
        assert len(result["actions"]) == 2
        assert result["actions"][0]["name"] == "a1"

    def test_sequential_actions(self):
        payload = {"type": "sequential", "actions": [{"name": "step1", "arguments": {}}]}
        result = _normalize_action_payload(payload)
        assert result["mode"] == "sequential"

    def test_non_dict_action_in_list(self):
        payload = {"type": "parallel", "actions": ["raw_string"]}
        result = _normalize_action_payload(payload)
        assert result["actions"][0]["name"] == "raw_string"

    def test_raw_payload(self):
        result = _normalize_action_payload("just a string")
        assert result == {"raw": "just a string"}

    def test_empty_actions_list(self):
        payload = {"type": "parallel", "actions": []}
        result = _normalize_action_payload(payload)
        assert result == {"mode": "parallel", "actions": []}


class TestNormalizeObservationItem:
    def test_none(self):
        assert _normalize_observation_item(None) is None

    def test_dict_with_result_dict_message(self):
        item = {"result": {"sender": "bot", "message": "hi"}}
        result = _normalize_observation_item(item)
        assert result == {"sender": "bot", "message": "hi"}

    def test_dict_with_result_plain(self):
        item = {"result": 42}
        assert _normalize_observation_item(item) == 42

    def test_dict_with_sender_message(self):
        item = {"sender": "user", "message": "hello"}
        result = _normalize_observation_item(item)
        assert result == {"sender": "user", "message": "hello"}

    def test_dict_passthrough(self):
        item = {"data": [1, 2]}
        assert _normalize_observation_item(item) == {"data": [1, 2]}

    def test_non_dict(self):
        assert _normalize_observation_item("text") == "text"


class TestNormalizeObservationPayload:
    def test_none(self):
        assert _normalize_observation_payload(None) is None

    def test_dict_with_observations_list(self):
        payload = {"observations": [{"result": 1}, {"result": 2}]}
        result = _normalize_observation_payload(payload)
        assert result == [1, 2]

    def test_dict_without_observations(self):
        payload = {"sender": "bot", "message": "hi"}
        result = _normalize_observation_payload(payload)
        assert result == {"sender": "bot", "message": "hi"}

    def test_list(self):
        payload = [{"result": "a"}, None]
        result = _normalize_observation_payload(payload)
        assert result == ["a", None]

    def test_scalar(self):
        assert _normalize_observation_payload("raw") == "raw"
