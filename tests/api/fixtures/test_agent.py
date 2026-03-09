# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

import hashlib
import random
from typing import Any, ClassVar, Literal, Optional

from exgentic.core.agent import Agent
from exgentic.core.agent_instance import AgentInstance
from exgentic.core.types import Action, ActionType, SingleAction, SingleObservation
from pydantic import BaseModel


class EmptyArgs(BaseModel):
    pass


class GoodAction(SingleAction):
    name: Literal["good"] = "good"
    arguments: EmptyArgs


class BadAction(SingleAction):
    name: Literal["bad"] = "bad"
    arguments: EmptyArgs


class FinishAction(SingleAction):
    name: Literal["finish"] = "finish"
    arguments: EmptyArgs


GOOD_ACTION_TYPE = ActionType(
    name="good",
    description="Good action",
    cls=GoodAction,
)
BAD_ACTION_TYPE = ActionType(
    name="bad",
    description="Bad action",
    cls=BadAction,
)
FINISH_ACTION_TYPE = ActionType(
    name="finish",
    description="Finish action",
    cls=FinishAction,
    is_finish=True,
)


class TestAgent(Agent):
    __test__ = False
    display_name: ClassVar[str] = "Test Agent"
    slug_name: ClassVar[str] = "test_agent"

    seed: int = 0
    policy: Literal[
        "random",
        "good_only",
        "bad_only",
        "good_then_finish",
        "finish_immediately",
        "return_none",
        "invalid_action",
        "raise_error",
    ] = "random"
    finish_after: int = 2
    max_steps: int | None = None

    def assign(
        self,
        task: str,
        context: dict[str, Any],
        actions: list[ActionType],
        session_id: str,
    ) -> TestAgentInstance:
        return TestAgentInstance(
            session_id=session_id,
            seed=self.seed,
            policy=self.policy,
            finish_after=self.finish_after,
            max_steps=self.max_steps,
        )


class TestAgentInstance(AgentInstance):
    __test__ = False

    def __init__(
        self,
        *,
        session_id: str,
        seed: int,
        policy: str,
        finish_after: int,
        max_steps: int | None,
    ) -> None:
        super().__init__(session_id=session_id)
        self._seed = seed
        self._policy = policy
        self._finish_after = finish_after
        self._rng = self._build_rng(session_id, seed)
        self._step = 0
        self.max_steps = max_steps

    @staticmethod
    def _build_rng(session_id: str, seed: int) -> random.Random:
        payload = f"{session_id}:{seed}".encode()
        digest = hashlib.sha256(payload).digest()
        value = int.from_bytes(digest[:4], "big")
        return random.Random(value)

    def react(self, observation: Optional[SingleObservation]) -> Optional[Action]:
        self._step += 1
        if self._policy == "return_none":
            return None
        if self._policy == "raise_error":
            raise RuntimeError("agent failure")
        if self._policy == "invalid_action":
            return "not-an-action"  # type: ignore[return-value]
        if self._policy == "finish_immediately":
            return FinishAction(arguments=EmptyArgs())
        if self._policy == "good_only":
            return GoodAction(arguments=EmptyArgs())
        if self._policy == "bad_only":
            return BadAction(arguments=EmptyArgs())
        if self._policy == "good_then_finish":
            if self._step >= self._finish_after:
                return FinishAction(arguments=EmptyArgs())
            return GoodAction(arguments=EmptyArgs())
        # random policy
        choice = self._rng.choice([GOOD_ACTION_TYPE, BAD_ACTION_TYPE, FINISH_ACTION_TYPE])
        if choice is GOOD_ACTION_TYPE:
            return GoodAction(arguments=EmptyArgs())
        if choice is BAD_ACTION_TYPE:
            return BadAction(arguments=EmptyArgs())
        return FinishAction(arguments=EmptyArgs())

    def close(self) -> None:
        return None
