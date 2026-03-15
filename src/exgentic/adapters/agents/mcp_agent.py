# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

import abc
from abc import abstractmethod
from typing import Any, Callable, Dict, List, Optional

from ...core.types import ActionType
from ..actions.functions import action_type_to_function
from .code_agent import CodeAgentInstance
from .mcp_server import MCPServer


class MCPAgentInstance(CodeAgentInstance, abc.ABC):
    """Sync-first base class.

    - run_code_agent(): sync
    - run_mcp_agent(): ABSTRACT SYNC (subclass decides implementation strategy)
    """

    def __init__(
        self,
        session_id: str,
        task: str,
        context: Dict[str, Any],
        actions: List[ActionType],
    ):
        self._mcp_server: Optional[MCPServer] = None

        super().__init__(session_id, task, context, actions)

    def run(self, adapter) -> None:
        functions: List[Callable[..., Any]] = []
        for action_type in self.actions:
            if action_type.is_finish:
                function = action_type_to_function(action_type, self._submit_finish_action)
            else:
                function = action_type_to_function(action_type, self.execute)
            functions.append(function)

        self.initial_observation = adapter.get_observation()

        try:
            self.run_code_agent(functions)
        finally:
            self.execute(None)

    def run_code_agent(self, functions: List[Callable[..., Any]]) -> Any:
        """Fully synchronous entrypoint."""
        self.logger.info("Starting MCP server for agent tools")
        server = MCPServer(
            tools=functions,
            log_dir=self.paths.agent_dir,
            logger=self.logger,
            stringify_empty_output=self._stringify_empty_output(),
        )
        self._mcp_server = server
        self.mcp = server.mcp
        started = False
        try:
            with server:
                started = True
                self.logger.info(
                    "MCP server ready at http://%s:%s/mcp",
                    server.connect_host,
                    server.port,
                )
                return self.run_mcp_agent(server.connect_host, server.port)
        finally:
            if started:
                self.logger.info("MCP server stopped")

    def _stringify_empty_output(self) -> bool:
        return False

    def _submit_finish_action(self, action) -> None:
        with self._condition:
            self._raise_if_agent_failed()
            if self._closed:
                if action is not None:
                    raise RuntimeError("execute() called after close()")
                return

            self.logger.info("Received finish action (turn=%s): %s", self._turn, action)
            self._pending_actions.append(action)
            self._condition.notify_all()
            return

    def close_mcp_agent(self) -> None:
        self.close()

    def close(self) -> None:
        super().close()
        if self._mcp_server is not None:
            self.logger.info("Stopping MCP server")
            self._mcp_server.stop(raise_on_timeout=False)

    @abstractmethod
    def run_mcp_agent(self, mcp_host: str, mcp_port: int) -> Any:
        """ABSTRACT SYNC.

        Subclass may implement:
          - purely sync logic, OR
          - a sync wrapper over an async core (via run_sync, etc.)
        """
        ...


class MCPAgent(MCPAgentInstance, abc.ABC):
    """Backwards-compatible alias for MCPAgentInstance."""

    pass
