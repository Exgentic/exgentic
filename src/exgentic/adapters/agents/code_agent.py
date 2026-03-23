# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from abc import abstractmethod
from typing import Callable, List, Optional

from ...core import Observation
from ..actions.functions import action_type_to_function
from .coordinator import AgentCoordinator, CoordinatedAgent


class CodeAgentInstance(CoordinatedAgent, AgentCoordinator):
    """Base class for code-based agents that inherits both roles."""

    def __init__(self, session_id: str):
        self.initial_observation: Optional[Observation] = None
        # Initialize AgentCoordinator with self as the internal agent
        AgentCoordinator.__init__(self, session_id, self)

    def run(self, adapter) -> None:
        """Implementation of CoordinatedAgent.run that converts actions to functions.

        When the code agent calls one of the functions, what actually happens, is that the
        AgentCoordinator.execute() method is called with the action. This places the action in an
        action queue, creates a future for the result, and waits for it.

        The AgentCordinator, which is running in  a different thread, waits for an action in the
        queue, fetches it and passes it to the benchmark environment.  When the AgentCordinator receives
        the coressponding observation, it places it in the result future.

        This unblocks the CoordinateAgent, and cause the function to return the value.
        The code agent then continues its run.

        """
        functions = []

        for action_type in self.actions:
            function = action_type_to_function(action_type, self.execute)
            functions.append(function)

        # Block until the environment delivers the initial observation via adapter.react()
        self.initial_observation = adapter.get_observation()

        try:
            self.run_code_agent(functions)
        finally:
            self.execute(None)  # Mark execution as done, by returning no action to the benchmark.

    @abstractmethod
    def run_code_agent(self, functions: List[Callable]) -> None:
        """Subclasses implement their code agent logic here."""
        pass
