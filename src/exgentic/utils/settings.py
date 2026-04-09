# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

DOTENV_PATH = Path(os.environ.get("EXGENTIC_DOTENV_PATH", ".env"))

load_dotenv(DOTENV_PATH)

if TYPE_CHECKING:
    from ..integrations.litellm.config import LitellmSettings

RunnerName = Literal["direct", "thread", "process", "service", "docker", "venv"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


def resolve_cache_path(cache_dir: str, cache_path: str) -> str:
    base = Path(cache_dir).expanduser()
    target = Path(cache_path).expanduser()
    if target.is_absolute():
        return str(target)
    return str(base / target)


class ExgenticSettings(BaseSettings):
    """Global settings for Exgentic."""

    default_runner: RunnerName = "venv"
    output_dir: str = "./outputs"
    # Logging level (env: EXGENTIC_LOG_LEVEL)
    log_level: LogLevel = "INFO"
    debug: bool = False
    litellm_log_level: LogLevel = "WARNING"
    litellm_delete_time_from_cache_key: bool = False
    litellm_caching: bool = True
    cache_dir: str = "~/.exgentic"
    litellm_cache_dir: str = "~/.cache/exgentic/litellm"
    dotenv_path: str = str(DOTENV_PATH)
    otel_enabled: bool = False
    otel_record_content: bool = True

    _litellm_cache_configured: bool = False

    def resolved_litellm_cache_dir(self) -> str:
        return resolve_cache_path(self.cache_dir, self.litellm_cache_dir)

    def get_env(self) -> dict[str, str]:
        """Return env vars for all settings fields (EXGENTIC_*)."""
        env: dict[str, str] = {}
        prefix = self.model_config.get("env_prefix") or ""
        for name, _field in type(self).model_fields.items():
            if name.startswith("_"):
                continue
            value = getattr(self, name, None)
            if value is None:
                continue
            key = f"{prefix}{name}".upper()
            if isinstance(value, bool):
                env[key] = "true" if value else "false"
            else:
                env[key] = str(value)
        return env

    def get_overrides(self) -> dict[str, Any]:
        """Return settings values that differ from class defaults."""
        overrides: dict[str, Any] = {}
        for name, field in type(self).model_fields.items():
            if name.startswith("_"):
                continue
            default = field.default
            current = getattr(self, name, None)
            if current != default:
                overrides[name] = current
        return overrides

    def model_post_init(self, __context) -> None:
        if self.debug and self.log_level != "DEBUG":
            object.__setattr__(self, "log_level", "DEBUG")
        elif self.log_level == "DEBUG" and not self.debug:
            object.__setattr__(self, "debug", True)
        self.configure_litellm()
        object.__setattr__(self, "_litellm_cache_configured", True)

    def configure_litellm(self, *, cache_only: bool = False) -> None:
        from ..integrations.litellm.config import configure_litellm

        configure_litellm(config=self.to_litellm_config(), cache_only=cache_only)

    def to_litellm_config(self) -> LitellmSettings:
        from ..integrations.litellm.config import LitellmSettings

        return LitellmSettings(
            litellm_caching=self.litellm_caching,
            litellm_delete_time_from_cache_key=self.litellm_delete_time_from_cache_key,
            cache_dir=self.cache_dir,
            litellm_cache_dir=self.litellm_cache_dir,
            log_level=self.litellm_log_level,
        )

    def __setattr__(self, name: str, value: Any) -> None:
        prev_value = self.__dict__.get(name, None)
        super().__setattr__(name, value)
        if name == "debug" and value != prev_value and bool(value):
            if self.log_level != "DEBUG":
                object.__setattr__(self, "log_level", "DEBUG")
        if name == "log_level" and value != prev_value:
            is_debug = str(value).upper() == "DEBUG"
            if self.__dict__.get("debug") != is_debug:
                object.__setattr__(self, "debug", is_debug)
        if (
            name
            in {
                "litellm_delete_time_from_cache_key",
                "litellm_caching",
                "cache_dir",
                "litellm_cache_dir",
            }
            and self._litellm_cache_configured
            and value != prev_value
        ):
            self.configure_litellm(cache_only=True)
            object.__setattr__(self, "_litellm_cache_configured", True)
        if name == "litellm_log_level" and value != prev_value:
            self.configure_litellm(cache_only=False)

    model_config = SettingsConfigDict(
        env_prefix="EXGENTIC_",
        case_sensitive=False,
    )


@lru_cache(maxsize=1)
def get_settings() -> ExgenticSettings:
    return ExgenticSettings()  # type: ignore[arg-type]
