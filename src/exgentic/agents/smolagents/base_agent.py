# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from typing import Any, ClassVar

from ...core.agent import Agent
from ...core.types import ModelSettings


class SmolagentBaseAgent(Agent):
    display_name: ClassVar[str] = "SmolAgents Base Agent"
    slug_name: ClassVar[str] = "smolagents_base"

    model: str = "watsonx/meta-llama/llama-3-3-70b-instruct"
    max_steps: int = 150
    model_settings: ModelSettings | None = None
    retry_on_all_errors: bool = True

    def get_instance_kwargs(
        self,
        session_id: str,
    ) -> dict[str, Any]:
        return {
            "session_id": session_id,
            "model_id": self.model,
            "max_steps": self.max_steps,
            "model_settings": self.model_settings,
            "retry_on_all_errors": self.retry_on_all_errors,
        }

    @property
    def model_name(self) -> str:  # type: ignore[override]
        return str(self.model).split("/")[-1]

    def get_models_names(self) -> list[str]:  # type: ignore[override]
        return [str(self.model)]
