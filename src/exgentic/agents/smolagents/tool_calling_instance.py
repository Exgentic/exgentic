# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from smolagents import ToolCallingAgent
from smolagents.tools import Tool
from smolagents.utils import AgentError

from .base_instance import SmolagentBaseAgentInstance


class SmolagentToolCallingAgentInstance(SmolagentBaseAgentInstance):
    """Smolagent implementation."""

    def run_smolagent(self, tools: list[Tool]):
        self._agent = ToolCallingAgent(
            tools=tools,
            model=self.get_internal_model(),
            # use_structured_outputs_internally=True,
            logger=self.get_smolagent_logger(),
        )

        prompt = f"Task: {self.task}\n\n"
        if self.context:
            prompt += f"Context: {self.context}\n\n"
        prompt += (
            "Complete this task using the available tools. "
            "Each tool corresponds to an action you can take in the environment.\n"
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
