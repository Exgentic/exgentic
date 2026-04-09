# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from ...integrations.litellm.health import HealthCheckError
from ..types import SessionConfig
from .controller import Controller
from .observer import Observer
from .termination import (
    AgentError,
    AgentTerminationError,
    BenchmarkError,
    BenchmarkTerminationError,
    RunCancelError,
    SessionCancelError,
    SessionLimitReachedError,
)
from .tracker import Tracker


def _close_session_agent(session, agent_instance) -> None:
    session.close()
    agent_instance.close()


def run_session(
    session_config: SessionConfig,
    session,
    agent,
    observers: list[Observer] | None = None,
    controllers: list[Controller] | None = None,
    *,
    tracker: Tracker | None = None,
) -> None:
    """Process a single session.

    The *agent_instance* is created via ``agent.get_instance()`` for
    runner isolation.
    """
    if tracker is None:
        tracker = Tracker(observers=observers, controllers=controllers)

    # session_scope is entered by the caller (run_session_config); role
    # is baked into each service at spawn time via per-service runtime.json.
    tracker.on_session_creation(session)

    try:
        agent_instance = agent.get_instance(session_id=session.session_id)

        observation = session.start()

        agent_instance.start(
            task=session.task,
            context=session.context,
            actions=session.actions,
        )

        tracker.on_session_start(session, agent_instance, observation)
        while not session.done() and observation is not None:
            try:
                action = agent_instance.react(observation)
            except Exception as exc:
                tracker.on_react_error(session, exc)

            tracker.on_react_success(session, action)

            try:
                observation = session.step(action)
            except Exception as exc:
                tracker.on_step_error(session, exc)

            tracker.on_step_success(session, observation)

        if session.done():
            raise BenchmarkTerminationError()
        raise AgentTerminationError()

    except HealthCheckError as exc:
        tracker.on_session_error(session, AgentError(exc))
    except KeyboardInterrupt:
        tracker.on_session_error(session, RunCancelError())
        _close_session_agent(session, agent_instance)
        raise
    except AgentError as exc:
        tracker.on_session_error(session, exc)
        _close_session_agent(session, agent_instance)
    except SessionLimitReachedError as exc:
        tracker.on_session_scoring(session)
        score = session.score()
        score.is_finished = False
        score.session_metadata = {
            **(score.session_metadata or {}),
            "limit_reached": True,
            "limit_reason": exc.reason,
            "max_steps": exc.max_steps,
            "max_actions": exc.max_actions,
            "steps": exc.steps,
            "actions": exc.actions,
        }
        tracker.on_session_success(session, score, agent_instance)
        _close_session_agent(session, agent_instance)
    except (AgentTerminationError, BenchmarkTerminationError):
        tracker.on_session_scoring(session)
        score = session.score()
        if score.is_finished is None:
            score.is_finished = True
        tracker.on_session_success(session, score, agent_instance)
        _close_session_agent(session, agent_instance)
    except BenchmarkError as exc:
        tracker.on_session_error(session, exc)
        agent_instance.close()
    except SessionCancelError as exc:
        tracker.on_session_error(session, exc)
        _close_session_agent(session, agent_instance)
    except RunCancelError as exc:
        tracker.on_session_error(session, exc)
        _close_session_agent(session, agent_instance)
        raise
