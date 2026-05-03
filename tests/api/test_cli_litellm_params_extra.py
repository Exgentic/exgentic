# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""End-to-end test that the CLI `--set ...litellm_params_extra=...` form works.

Specifically: the JSON value is parsed, validates against the agent /
benchmark factory schema, and lands in the right kwargs dict.
"""

from __future__ import annotations

import pytest
from exgentic.interfaces.cli.options import (
    _parse_set_list,
    _set_nested,
    _validate_set_keys_for_agent,
    _validate_set_keys_for_benchmark,
)

EXTRAS_JSON = '{"api_base":"https://example.invalid/v1","extra_headers":{"X-Backend-Auth":"k"}}'
EXPECTED = {
    "api_base": "https://example.invalid/v1",
    "extra_headers": {"X-Backend-Auth": "k"},
}


@pytest.mark.parametrize(
    "agent_slug",
    [
        "tool_calling",
        "openai_solo",
        "smolagents_code",
        "smolagents_tool",
        "codex_cli",
        "gemini_cli",
        "claude_code",
    ],
)
def test_cli_set_litellm_params_extra_validates_and_lands_in_agent_kwargs(agent_slug):
    items = _parse_set_list((f"agent.litellm_params_extra={EXTRAS_JSON}",))
    _validate_set_keys_for_agent(agent_slug, items)

    agent_kwargs: dict = {}
    for group, path, value in items:
        if group == "agent":
            _set_nested(agent_kwargs, path, value)

    assert agent_kwargs == {"litellm_params_extra": EXPECTED}


def test_cli_set_user_simulator_litellm_params_extra_for_tau2():
    items = _parse_set_list((f"benchmark.user_simulator_litellm_params_extra={EXTRAS_JSON}",))
    _validate_set_keys_for_benchmark("tau2", items)

    bench_kwargs: dict = {}
    for group, path, value in items:
        if group == "benchmark":
            _set_nested(bench_kwargs, path, value)

    assert bench_kwargs == {"user_simulator_litellm_params_extra": EXPECTED}


def test_cli_set_litellm_params_extra_unknown_field_rejected():
    """Sanity check that the validator actually rejects unknown agent overrides.

    Without this, the test above wouldn't be proving anything.
    """
    import click

    items = _parse_set_list(("agent.not_a_real_field=42",))
    with pytest.raises(click.ClickException, match="Unknown agent override"):
        _validate_set_keys_for_agent("tool_calling", items)
