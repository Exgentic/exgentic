# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, field_validator


class RetryStrategy(str, Enum):
    EXPONENTIAL_BACKOFF = "exponential_backoff_retry"
    CONSTANT = "constant_retry"


class ModelSettings(BaseModel):
    temperature: float | None = 1.0
    top_p: float | None = None
    max_tokens: int | None = None
    reasoning_effort: str | None = None
    num_retries: int | None = 5
    retry_after: float = 0.5
    retry_strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF

    @field_validator("temperature")
    @classmethod
    def _validate_temperature(cls, value: float | None) -> float | None:
        if value is not None and value < 0:
            raise ValueError("temperature must be >= 0")
        return value

    @field_validator("max_tokens")
    @classmethod
    def _validate_max_tokens(cls, value: int | None) -> int | None:
        if value is not None and value < 0:
            raise ValueError("max_tokens must be >= 0")
        return value

    @field_validator("top_p")
    @classmethod
    def _validate_top_p(cls, value: float | None) -> float | None:
        if value is None:
            return value
        if value < 0 or value > 1:
            raise ValueError("top_p must be between 0 and 1")
        return value

    @field_validator("num_retries")
    @classmethod
    def _validate_num_retries(cls, value: int | None) -> int | None:
        if value is None:
            return None
        if value < 0:
            raise ValueError("num_retries must be >= 0")
        return value

    @field_validator("retry_after")
    @classmethod
    def _validate_retry_after(cls, value: float) -> float:
        if value < 0:
            raise ValueError("retry_after must be >= 0")
        return value
