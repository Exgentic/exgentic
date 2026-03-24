# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

import os

import yaml
from smolagents import CodeAgent as SmolagentBaseCodeAgent
from smolagents.tools import Tool
from smolagents.utils import AgentError

from .base_instance import SmolagentBaseAgentInstance


class SmolagentCodeAgentInstance(SmolagentBaseAgentInstance):
    """Smolagent implementation."""

    def run_smolagent(self, tools: list[Tool]):
        # Load custom structured prompt templates from YAML next to this module
        prompt_path = os.path.join(os.path.dirname(__file__), "structured_code_agent.yaml")
        try:
            with open(prompt_path, encoding="utf-8-sig") as f:
                prompt_templates = yaml.safe_load(f)
        except Exception:
            prompt_templates = None

        self._agent = SmolagentBaseCodeAgent(
            tools=tools,
            model=self.get_internal_model(),
            prompt_templates=prompt_templates,
            use_structured_outputs_internally=True,
            logger=self.get_smolagent_logger(),
        )
        # Remove built-in final_answer; termination should happen by interacting with the benchmark (finish action).
        self._agent.tools.pop("final_answer", None)

        prompt = f"Task: {self.task}\n\n"
        if self.context:
            prompt += f"Context: {self.context}\n\n"
        prompt += (
            "Complete this task using the available functions. "
            "Each function corresponds to an action you can take to solve the given task.\n"
            "Every action should be taken only by calling one of the functions. "
            "If one function fail, consider using another, at any given point one of the functions\n"
            "can be a valid next step. At any point you should executing actions by writing code. "
            "do not call tools with tool calling mechanism.\n\n"
            "Printing or any other code will be visible only by you alone.\n\n"
            # "Always provide parameter names when calling function. Do not rely on positional arguments.\n"
        )
        if self.initial_observation is not None and not self.initial_observation.is_empty():
            text = str(self.initial_observation).strip()
            if text:
                prompt += f"\nFirst Observation: {text}\n"
        try:
            self._agent.run(task=prompt, max_steps=self.max_steps)

        except AgentError as e:
            self.logger.info(f"AgentError: {e}")
            raise
