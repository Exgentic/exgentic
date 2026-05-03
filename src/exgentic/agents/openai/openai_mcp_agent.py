# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from ...core.agent import Agent
from ...core.types import ModelSettings


class MCPConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    client_session_timeout_seconds: float | None = None
    http_timeout_seconds: float | None = None
    sse_read_timeout_seconds: float | None = None
    http_connect_timeout_seconds: float | None = None
    headers: dict[str, str] | None = None
    terminate_on_close: bool = True
    max_retry_attempts: int = -1
    retry_backoff_seconds_base: float = 1.0
    cache_tools_list: bool = False
    use_structured_content: bool = False
    skip_health_check: bool = False
    name: str | None = None
    message_handler: Any | None = None


class OpenAIMCPAgent(Agent):
    display_name: ClassVar[str] = "OpenAI Solo"
    slug_name: ClassVar[str] = "openai_solo"

    model: str
    max_steps: int = 150
    model_settings: ModelSettings | None = None
    mcp_config: MCPConfig | dict | None = None
    litellm_params_extra: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def _get_instance_class(cls):
        from .instance import OpenAIMCPAgentInstance

        return OpenAIMCPAgentInstance

    @classmethod
    def _get_instance_class_ref(cls) -> str:
        return "exgentic.agents.openai.instance:OpenAIMCPAgentInstance"

    def _get_instance_kwargs(
        self,
        session_id: str,
    ) -> dict[str, Any]:
        return {
            "session_id": session_id,
            "model_id": self.model,
            "max_steps": self.max_steps,
            "model_settings": self.model_settings,
            "mcp_config": self.mcp_config,
            "litellm_params_extra": self.litellm_params_extra,
        }

    @property
    def model_name(self) -> str:  # type: ignore[override]
        return str(self.model).split("/")[-1]

    def get_models_names(self) -> list[str]:  # type: ignore[override]
        return [str(self.model)]
