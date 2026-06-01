# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Unit tests for the spec-shape normalizers in trace_logger.

The OTel GenAI semantic conventions define canonical shapes for
gen_ai.tool.definitions and gen_ai.input.messages. LiteLLM hands the
logger whatever the underlying provider used (OpenAI native, Anthropic
native, ...) so the logger must normalize before serializing — otherwise
the spans are non-conformant and downstream consumers (e.g. the
Exgentic agent-llm-traces-v2 dataset) inherit shape bugs.

These tests exist to catch regressions: if the normalizer drifts, the
dataset rebuilders we patched (v2.10, v2.11) would have to keep fixing
the same bugs forever.
"""

from __future__ import annotations

import json

from exgentic.integrations.litellm.trace_logger import (
    _convert_input_messages_to_parts,
    _normalize_tool_definitions,
)

# ---------------------------------------------------------------------------
# _normalize_tool_definitions
# ---------------------------------------------------------------------------


def test_normalize_openai_native_unwraps_function_envelope():
    """OpenAI's envelope must unwrap to the flat spec shape.

    OpenAI ships `{type:function, function:{name, description, parameters}}`.
    OTel spec wants flat `{type:function, name, description, parameters}`.
    """
    out = _normalize_tool_definitions(
        [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "search the web",
                    "parameters": {"type": "object"},
                },
            }
        ]
    )
    assert out == [
        {
            "type": "function",
            "name": "search",
            "description": "search the web",
            "parameters": {"type": "object"},
        }
    ]


def test_normalize_anthropic_native_adds_type_and_renames_input_schema():
    """Anthropic ships `{name, description, input_schema}` with no `type`."""
    out = _normalize_tool_definitions(
        [
            {
                "name": "search",
                "description": "search the web",
                "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}},
            }
        ]
    )
    assert out == [
        {
            "type": "function",
            "name": "search",
            "description": "search the web",
            "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
        }
    ]


def test_normalize_already_spec_shape_is_idempotent():
    spec = [{"type": "function", "name": "search", "parameters": {}}]
    assert _normalize_tool_definitions(spec) == spec


def test_normalize_preserves_provider_extension_type():
    """Provider-extension types pass through unchanged.

    e.g. Anthropic's `computer_20241022` — only spec-required fields are
    normalized, extension types pass through.
    """
    out = _normalize_tool_definitions([{"type": "computer_20241022", "name": "computer"}])
    assert out == [{"type": "computer_20241022", "name": "computer"}]


def test_normalize_non_list_passes_through():
    assert _normalize_tool_definitions(None) is None
    assert _normalize_tool_definitions("not a list") == "not a list"


# ---------------------------------------------------------------------------
# _convert_input_messages_to_parts
# ---------------------------------------------------------------------------


def test_convert_openai_plain_string_content():
    out = _convert_input_messages_to_parts(
        [
            {"role": "system", "content": "be helpful"},
            {"role": "user", "content": "hi"},
        ]
    )
    assert out == [
        {"role": "system", "parts": [{"type": "text", "content": "be helpful"}]},
        {"role": "user", "parts": [{"type": "text", "content": "hi"}]},
    ]


def test_convert_anthropic_tool_use_block_becomes_tool_call_part():
    """Anthropic `tool_use` blocks must become OTel `tool_call` parts.

    Not text parts (the Issue #14 anti-pattern).
    """
    out = _convert_input_messages_to_parts(
        [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "let me search"},
                    {"type": "tool_use", "id": "tu_1", "name": "search", "input": {"q": "x"}},
                ],
            }
        ]
    )
    assert out == [
        {
            "role": "assistant",
            "parts": [
                {"type": "text", "content": "let me search"},
                {"type": "tool_call", "id": "tu_1", "name": "search", "arguments": {"q": "x"}},
            ],
        }
    ]


def test_convert_anthropic_tool_result_block_becomes_tool_call_response():
    """Anthropic `tool_result` block becomes a `tool_call_response` part.

    The `result` field is always a JSON-stringified blocks list — the v2.10
    contract — so `json.loads(result)` always returns `[{type, ...}]`.
    """
    out = _convert_input_messages_to_parts(
        [
            {
                "role": "user",
                "content": [
                    {
                        "tool_use_id": "tu_1",
                        "type": "tool_result",
                        "content": [{"type": "text", "text": "result text"}],
                    }
                ],
            }
        ]
    )
    [m] = out
    assert m["role"] == "user"
    [p] = m["parts"]
    assert p["type"] == "tool_call_response"
    assert p["id"] == "tu_1"
    assert json.loads(p["result"]) == [{"type": "text", "text": "result text"}]


def test_convert_anthropic_multiblock_tool_result_preserves_structure():
    """Multi-block content must be preserved one block per element.

    So the wire structure the model saw is recoverable.
    """
    out = _convert_input_messages_to_parts(
        [
            {
                "role": "user",
                "content": [
                    {
                        "tool_use_id": "tu_1",
                        "type": "tool_result",
                        "content": [
                            {"type": "text", "text": "part A"},
                            {"type": "text", "text": "part B"},
                        ],
                    }
                ],
            }
        ]
    )
    [p] = out[0]["parts"]
    blocks = json.loads(p["result"])
    assert blocks == [
        {"type": "text", "text": "part A"},
        {"type": "text", "text": "part B"},
    ]


def test_convert_anthropic_string_tool_result_wraps_as_single_text_block():
    """Plain-string `tool_result.content` wraps as a single text block.

    Rare but allowed by Anthropic. Wrap so `result` shape stays uniform.
    """
    out = _convert_input_messages_to_parts(
        [
            {
                "role": "user",
                "content": [
                    {
                        "tool_use_id": "tu_1",
                        "type": "tool_result",
                        "content": "plain text result",
                    }
                ],
            }
        ]
    )
    [p] = out[0]["parts"]
    assert json.loads(p["result"]) == [{"type": "text", "text": "plain text result"}]


def test_convert_openai_tool_message_becomes_tool_call_response():
    out = _convert_input_messages_to_parts(
        [
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "tool output",
            }
        ]
    )
    [m] = out
    [p] = m["parts"]
    assert p["type"] == "tool_call_response"
    assert p["id"] == "call_1"
    assert json.loads(p["result"]) == [{"type": "text", "text": "tool output"}]


def test_convert_openai_assistant_tool_calls_become_tool_call_parts():
    out = _convert_input_messages_to_parts(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "call_1", "function": {"name": "search", "arguments": '{"q": "x"}'}},
                ],
            }
        ]
    )
    [m] = out
    [p] = m["parts"]
    assert p == {"type": "tool_call", "id": "call_1", "name": "search", "arguments": {"q": "x"}}


def test_convert_anthropic_thinking_block_preserves_signature():
    out = _convert_input_messages_to_parts(
        [
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "let me think", "signature": "sig123"},
                ],
            }
        ]
    )
    [p] = out[0]["parts"]
    assert p == {"type": "thinking", "thinking": "let me think", "signature": "sig123"}


def test_convert_non_list_passes_through():
    assert _convert_input_messages_to_parts(None) is None
    assert _convert_input_messages_to_parts("not a list") == "not a list"
