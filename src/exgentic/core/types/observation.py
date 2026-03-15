# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from .action import SingleAction


class Observation(BaseModel, ABC):
    """Base class to encapsulate both single and multiple observations."""

    @abstractmethod
    def to_observation_list(self):
        pass

    @abstractmethod
    def is_empty(self) -> bool:
        """Returns True if the observation carries no meaningful content."""
        raise NotImplementedError


class SingleObservation(Observation):
    """Observation that includes a single arbitrary result and a list of actions that invoked it.

    (i.e. the actions that generated it the observation)

    To allow the observation to be used in place of the results in CodeAgents,
    we map all the magic methods of the observation to the result object.
    """

    result: Any
    invoking_actions: list[SingleAction] = Field(repr=False, default_factory=list)

    def to_observation_list(self):
        return [self]

    def is_empty(self) -> bool:
        if self.result is None:
            return True
        if isinstance(self.result, str) and self.result.strip() == "":
            return True
        if isinstance(self.result, (list, tuple, set, dict)) and len(self.result) == 0:
            return True
        return False

    def __str__(self, *args, **kwargs):
        return self.result.__str__(*args, **kwargs)

    def __repr__(self, *args, **kwargs):
        return self.result.__repr__(*args, **kwargs)

    def __len__(self, *args, **kwargs):
        return self.result.__len__(*args, **kwargs)

    def __getitem__(self, *args, **kwargs):
        return self.result.__getitem__(*args, **kwargs)

    def __setitem__(self, *args, **kwargs):
        return self.result.__setitem__(*args, **kwargs)

    def __delitem__(self, *args, **kwargs):
        return self.result.__delitem__(*args, **kwargs)

    def __iter__(self, *args, **kwargs):
        return self.result.__iter__(*args, **kwargs)

    def __contains__(self, *args, **kwargs):
        return self.result.__contains__(*args, **kwargs)

    def __eq__(self, *args, **kwargs):
        return self.result.__eq__(*args, **kwargs)

    def __ne__(self, *args, **kwargs):
        return self.result.__ne__(*args, **kwargs)

    def __lt__(self, *args, **kwargs):
        return self.result.__lt__(*args, **kwargs)

    def __le__(self, *args, **kwargs):
        return self.result.__le__(*args, **kwargs)

    def __gt__(self, *args, **kwargs):
        return self.result.__gt__(*args, **kwargs)

    def __ge__(self, *args, **kwargs):
        return self.result.__ge__(*args, **kwargs)

    def __add__(self, *args, **kwargs):
        return self.result.__add__(*args, **kwargs)

    def __sub__(self, *args, **kwargs):
        return self.result.__sub__(*args, **kwargs)

    def __mul__(self, *args, **kwargs):
        return self.result.__mul__(*args, **kwargs)

    def __truediv__(self, *args, **kwargs):
        return self.result.__truediv__(*args, **kwargs)

    def __floordiv__(self, *args, **kwargs):
        return self.result.__floordiv__(*args, **kwargs)

    def __mod__(self, *args, **kwargs):
        return self.result.__mod__(*args, **kwargs)

    def __pow__(self, *args, **kwargs):
        return self.result.__pow__(*args, **kwargs)

    def __and__(self, *args, **kwargs):
        return self.result.__and__(*args, **kwargs)

    def __or__(self, *args, **kwargs):
        return self.result.__or__(*args, **kwargs)

    def __xor__(self, *args, **kwargs):
        return self.result.__xor__(*args, **kwargs)

    def __lshift__(self, *args, **kwargs):
        return self.result.__lshift__(*args, **kwargs)

    def __rshift__(self, *args, **kwargs):
        return self.result.__rshift__(*args, **kwargs)

    def __neg__(self, *args, **kwargs):
        return self.result.__neg__(*args, **kwargs)

    def __pos__(self, *args, **kwargs):
        return self.result.__pos__(*args, **kwargs)

    def __abs__(self, *args, **kwargs):
        return self.result.__abs__(*args, **kwargs)

    def __invert__(self, *args, **kwargs):
        return self.result.__invert__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.result.__call__(*args, **kwargs)


class MessagePayload(BaseModel):
    sender: str
    message: str


class MessageObservation(SingleObservation):
    """Observation carrying a structured message payload (sender + message)."""

    result: MessagePayload


class EmptyObservation(SingleObservation):
    """Explicit empty observation to signal 'no initial content'."""

    result: Any = None
    invoking_actions: list[SingleAction] = Field(repr=False, default_factory=list)

    def is_empty(self) -> bool:
        return True


class MultiObservation(Observation):
    """An observation which is actually a collection of multiple observations."""

    observations: list[SingleObservation]

    def to_observation_list(self):
        return self.observations

    def is_empty(self) -> bool:
        return all(obs.is_empty() for obs in self.observations)
