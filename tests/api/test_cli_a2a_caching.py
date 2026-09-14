# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""The a2a command defaults LiteLLM response caching off.

An A2A agent process is long-lived and serves many runs, so a cached response
from an earlier run would be returned for a repeated task — silently
invalidating re-measurement. The default is scoped to this command: other entry
points keep the global default, and an explicit ``EXGENTIC_LITELLM_CACHING``
still wins.
"""

from __future__ import annotations

import pytest
from exgentic.utils.settings import ExgenticSettings


def _resolve(env_value: str | None) -> bool:
    """Return the caching value the a2a command would end up using."""
    settings = ExgenticSettings() if env_value is None else ExgenticSettings(litellm_caching=env_value == "true")
    if env_value is not None:
        # Mirror pydantic-settings: a value sourced from the environment is
        # recorded in model_fields_set, unlike a class default.
        settings.model_fields_set.add("litellm_caching")

    # The logic under test, as it appears in commands/a2a.py.
    if "litellm_caching" not in settings.model_fields_set:
        settings.litellm_caching = False
    return settings.litellm_caching


def test_global_default_is_unchanged():
    """Other entry points must still get caching on by default."""
    assert ExgenticSettings().litellm_caching is True


def test_a2a_defaults_caching_off():
    """With no env var, a2a turns caching off."""
    assert _resolve(None) is False


@pytest.mark.parametrize("env_value,expected", [("true", True), ("false", False)])
def test_explicit_env_var_wins(env_value, expected):
    """An explicit EXGENTIC_LITELLM_CACHING overrides the a2a default."""
    assert _resolve(env_value) is expected


def test_env_var_is_read_from_environment(monkeypatch):
    """EXGENTIC_LITELLM_CACHING is the documented knob and is honoured."""
    monkeypatch.setenv("EXGENTIC_LITELLM_CACHING", "false")
    settings = ExgenticSettings()
    assert settings.litellm_caching is False
    assert "litellm_caching" in settings.model_fields_set

    monkeypatch.setenv("EXGENTIC_LITELLM_CACHING", "true")
    settings = ExgenticSettings()
    assert settings.litellm_caching is True
    assert "litellm_caching" in settings.model_fields_set


def test_a2a_command_disables_caching_by_default(monkeypatch):
    """End-to-end: invoking a2a_cmd flips the live setting.

    The caching decision happens before the command connects to MCP, so the
    unreachable --mcp address below aborts the command *after* the code under
    test has run. get_settings() is lru_cached, so the object the command
    mutates is the same one this test observes.
    """
    monkeypatch.delenv("EXGENTIC_LITELLM_CACHING", raising=False)

    from click.testing import CliRunner
    from exgentic.interfaces.cli.main import cli
    from exgentic.utils.settings import get_settings

    settings = get_settings()
    # Assigning the attribute re-adds it to model_fields_set, which the command
    # reads as an explicit override — so set the value first, then clear the
    # marker to model "came from the class default".
    settings.litellm_caching = True
    settings.model_fields_set.discard("litellm_caching")
    assert settings.litellm_caching is True

    try:
        CliRunner().invoke(cli, ["a2a", "--agent", "tool_calling", "--mcp", "http://127.0.0.1:1/mcp"])
        assert get_settings() is settings, "settings object should be the lru_cached singleton"
        assert settings.litellm_caching is False
    finally:
        # Restore the global default so later tests in this process are unaffected.
        settings.litellm_caching = True
