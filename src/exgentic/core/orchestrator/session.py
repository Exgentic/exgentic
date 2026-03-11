# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from ..context import agent_scope, benchmark_scope, session_scope
from ..types import SessionConfig, SessionIndex
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
    benchmark,
    agent,
    observers: list[Observer] | None = None,
    controllers: list[Controller] | None = None,
    *,
    tracker: Tracker | None = None,
) -> None:
    """Process a single session."""
    if tracker is None:
        tracker = Tracker(observers=observers, controllers=controllers)

    session_id = session_config.get_session_id()
    session = benchmark.create_session(SessionIndex(task_id=str(session_config.task_id), session_id=session_id))
    with session_scope(session.session_id, task_id=session.task_id):
        tracker.on_session_creation(session)

        agent_instance = agent.assign(
            task=session.task,
            context=session.context,
            actions=session.actions,
            session_id=session.session_id,
        )

        with benchmark_scope():
            observation = session.start()

        with agent_scope():
            agent_instance.start()

        try:
            tracker.on_session_start(session, agent_instance, observation)
            while not session.done() and observation is not None:
                try:
                    with agent_scope():
                        action = agent_instance.react(observation)
                except Exception as exc:
                    tracker.on_react_error(session, exc)

                tracker.on_react_success(
                    session,
                    action,
                )

                try:
                    with benchmark_scope():
                        observation = session.step(action)
                except Exception as exc:
                    tracker.on_step_error(
                        session,
                        exc,
                    )

                tracker.on_step_success(
                    session,
                    observation,
                )

            if session.done():
                raise BenchmarkTerminationError()
            raise AgentTerminationError()

        except KeyboardInterrupt:
            _close_session_agent(session, agent_instance)
            tracker.on_session_error(session, RunCancelError())
            raise
        except AgentError as exc:
            _close_session_agent(session, agent_instance)
            tracker.on_session_error(session, exc)
        except SessionLimitReachedError as exc:
            with benchmark_scope():
                tracker.on_session_scoring(session)
            _close_session_agent(session, agent_instance)
            with benchmark_scope():
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
        except (AgentTerminationError, BenchmarkTerminationError):
            with benchmark_scope():
                tracker.on_session_scoring(session)
            _close_session_agent(session, agent_instance)
            with benchmark_scope():
                score = session.score()
            if score.is_finished is None:
                score.is_finished = True
            tracker.on_session_success(session, score, agent_instance)
        except BenchmarkError as exc:
            agent_instance.close()
            tracker.on_session_error(session, exc)
        except SessionCancelError as exc:
            _close_session_agent(session, agent_instance)
            tracker.on_session_error(session, exc)
        except RunCancelError as exc:
            _close_session_agent(session, agent_instance)
            tracker.on_session_error(session, exc)
            raise
