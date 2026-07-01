# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

import json
import logging
import uuid
from typing import Any
from urllib.request import Request, urlopen

from ...core.agent_instance import AgentInstance
from ...core.types import (
    Action,
    Message,
    MessageAction,
    Observation,
)

logger = logging.getLogger(__name__)


class A2AError(Exception):
    """Raised when the A2A agent returns an error or an unexpected response."""


class A2AAgentInstance(AgentInstance):
    """Agent instance that sends prompts to an A2A-compliant agent.

    Communication flow:
      1. On ``start()``, discover the remote agent via its agent card.
      2. On the first ``react()`` call, send the task prompt to
         ``POST {url}/message:send`` (JSON-RPC 2.0) and return the
         response text as a ``MessageAction``.
      3. On subsequent ``react()`` calls, return ``None`` (done).

    Uses only ``urllib`` from the standard library so there are no
    extra dependencies beyond what the exgentic core already provides.
    """

    def __init__(
        self,
        session_id: str,
        url: str,
        timeout: float = 120.0,
    ) -> None:
        super().__init__(session_id)
        self.url = url.rstrip("/")
        self.timeout = timeout
        self._agent_card: dict[str, Any] | None = None
        self._done = False

    # ------------------------------------------------------------------
    # Agent card discovery
    # ------------------------------------------------------------------

    def _discover_agent_card(self) -> dict[str, Any]:
        """Fetch ``/.well-known/agent-card.json`` from the remote agent."""
        if self._agent_card is not None:
            return self._agent_card

        card_url = f"{self.url}/.well-known/agent-card.json"
        self.logger.info("Discovering A2A agent card at %s", card_url)

        request = Request(card_url, method="GET")
        request.add_header("Accept", "application/json")

        with urlopen(request, timeout=self.timeout) as response:  # noqa: S310
            body = response.read().decode("utf-8")

        self._agent_card = json.loads(body)
        self.logger.info(
            "A2A agent card: name=%s, version=%s",
            self._agent_card.get("name"),
            self._agent_card.get("version"),
        )
        return self._agent_card

    # ------------------------------------------------------------------
    # Send message via JSON-RPC 2.0
    # ------------------------------------------------------------------

    def _send_message(self, text: str) -> str:
        """Send a task prompt and return the response text.

        Uses ``POST {url}/message:send`` with a JSON-RPC 2.0 envelope.
        """
        endpoint = f"{self.url}/message:send"
        request_id = str(uuid.uuid4())

        payload = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": text}],
                },
            },
        }

        self.logger.info("Sending A2A message to %s (id=%s)", endpoint, request_id)
        self.logger.debug("A2A request payload: %s", payload)

        data = json.dumps(payload).encode("utf-8")
        request = Request(endpoint, data=data, method="POST")
        request.add_header("Content-Type", "application/json")
        request.add_header("Accept", "application/json")

        with urlopen(request, timeout=self.timeout) as response:  # noqa: S310
            body = response.read().decode("utf-8")

        result = json.loads(body)
        self.logger.debug("A2A raw response: %s", result)

        # Handle JSON-RPC error
        if "error" in result:
            error = result["error"]
            raise A2AError(
                f"A2A JSON-RPC error: code={error.get('code')}, "
                f"message={error.get('message')}"
            )

        return self._extract_response_text(result)

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_response_text(rpc_response: dict[str, Any]) -> str:
        """Extract text from ``result.artifacts[].parts[].text``."""
        rpc_result = rpc_response.get("result", {})
        artifacts = rpc_result.get("artifacts", [])

        texts: list[str] = []
        for artifact in artifacts:
            for part in artifact.get("parts", []):
                if part.get("kind") == "text" and "text" in part:
                    texts.append(part["text"])

        if not texts:
            raise A2AError(
                f"No text parts found in A2A response artifacts: {rpc_response}"
            )

        return "\n".join(texts)

    # ------------------------------------------------------------------
    # AgentInstance interface
    # ------------------------------------------------------------------

    def _build_prompt(self) -> str:
        """Combine task and context into a single prompt string."""
        parts: list[str] = []

        if self.context:
            for key, value in self.context.items():
                parts.append(f"<{key}>\n{value}\n</{key}>")

        parts.append(self.task)
        return "\n\n".join(parts)

    def react(self, observation: Observation | None) -> Action | None:
        """Send the task to the A2A agent and return the response.

        Returns a ``MessageAction`` on the first call, ``None`` on
        subsequent calls to signal completion.
        """
        if self._done:
            return None

        self._done = True

        # Discover the remote agent (validates connectivity)
        try:
            self._discover_agent_card()
        except Exception:
            self.logger.warning(
                "Agent card discovery failed; proceeding with message send",
                exc_info=True,
            )

        prompt = self._build_prompt()
        self.logger.info("A2A prompt: %s", prompt[:200])

        response_text = self._send_message(prompt)
        self.logger.info("A2A response: %s", response_text[:200])

        return MessageAction(arguments=Message(content=response_text))

    def close(self) -> None:
        """No resources to clean up."""
        pass
