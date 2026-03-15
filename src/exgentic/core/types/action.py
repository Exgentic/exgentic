# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

import inspect
import uuid
from abc import ABC, abstractmethod
from typing import Any, Literal, Optional, get_type_hints

from pydantic import BaseModel, Field


class Action(BaseModel, ABC):
    @abstractmethod
    def to_action_list(self):
        pass

    pass


class ValidationReport(BaseModel):
    valid: bool = True
    name_valid: bool = True
    args_valid: bool = True
    error: Optional[str] = None
    details: dict[str, Any] = Field(default_factory=dict)


class SingleAction(Action):
    name: str
    arguments: BaseModel
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), frozen=True)
    validation: ValidationReport = Field(default_factory=ValidationReport)

    def to_action_list(self):
        return [self]


class Message(BaseModel):
    content: str


class MessageAction(SingleAction):
    name: Literal["message"] = "message"
    arguments: Message


class FinishAction(SingleAction):
    name: str
    arguments: BaseModel


class ParallelAction(Action):
    actions: list[SingleAction]

    def to_action_list(self):
        return self.actions


class SequentialAction(Action):
    actions: list[SingleAction]

    def to_action_list(self):
        return self.actions


class ActionType(BaseModel):
    name: str
    description: str
    cls: type[SingleAction]
    # Hints for agent/adapter handling (optional)
    is_message: bool = False
    is_finish: bool = False
    is_hidden: bool = False

    @property
    def arguments(self) -> type[BaseModel]:
        """Return the resolved pydantic model type for the action's arguments.

        Handles forward-referenced annotations (e.g., from `from __future__ import annotations`).
        Falls back to the raw annotation if resolution fails.
        """
        module = inspect.getmodule(self.cls)
        globalns = vars(module) if module else {}
        try:
            hints = get_type_hints(self.cls, globalns=globalns, localns=globalns)
            arg_t = hints.get("arguments")
            if arg_t is not None:
                if isinstance(arg_t, str):
                    resolved = globalns.get(arg_t)
                    if isinstance(resolved, type):
                        return resolved  # type: ignore[return-value]
                return arg_t  # type: ignore[return-value]
        except Exception:
            pass
        arg_t = self.cls.__annotations__.get("arguments")
        if isinstance(arg_t, str):
            resolved = globalns.get(arg_t)
            if isinstance(resolved, type):
                return resolved  # type: ignore[return-value]
        return arg_t  # type: ignore[return-value]

    def build_action(self, arguments: Any, *, action_id: Optional[str] = None) -> SingleAction:
        """Convenience wrapper around core.actions.build_action for this action type."""
        from ..actions import build_action  # Local import to avoid circular dependency

        return build_action(self, arguments, action_id=action_id)
