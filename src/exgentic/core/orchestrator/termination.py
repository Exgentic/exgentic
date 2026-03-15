# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.


class SessionTerminationError(Exception):
    """Internal control-flow signal for clean session termination."""


class SessionCancelError(SessionTerminationError):
    pass


class RunCancelError(SessionTerminationError):
    pass


class AgentTerminationError(SessionTerminationError):
    pass


class BenchmarkTerminationError(SessionTerminationError):
    pass


class SessionLimitReachedError(SessionTerminationError):
    def __init__(
        self,
        *,
        reason: str,
        max_steps: int,
        max_actions: int,
        steps: int,
        actions: int,
    ) -> None:
        super().__init__()
        self.reason = reason
        self.max_steps = max_steps
        self.max_actions = max_actions
        self.steps = steps
        self.actions = actions

    def __str__(self) -> str:
        return (
            f"limit_reached ({self.reason}): "
            f"steps={self.steps}/{self.max_steps}, "
            f"actions={self.actions}/{self.max_actions}"
        )


class AgentError(SessionTerminationError):
    def __init__(self, error: Exception | None = None) -> None:
        super().__init__()
        self.error = error


class BenchmarkError(SessionTerminationError):
    def __init__(self, error: Exception | None = None) -> None:
        super().__init__()
        self.error = error


class InvalidActionError(ValueError):
    def __init__(self, action) -> None:
        self.action = action
        super().__init__(f"illegal action returned from agent: {action}")


class InvalidObservationError(ValueError):
    def __init__(self, observation) -> None:
        self.observation = observation
        super().__init__(f"illegal observation returned from session: {observation}")
