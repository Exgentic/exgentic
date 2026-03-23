# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from ...adapters.runners import with_runner
from ..context import agent_scope, benchmark_scope, session_scope
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

    The *session* and *agent_instance* are created via ``with_runner()``
    for isolation.
    """
    if tracker is None:
        tracker = Tracker(observers=observers, controllers=controllers)

    with session_scope(session.session_id, task_id=session.task_id):
        agent_instance = with_runner(
            agent.get_instance_class_ref(),
            runner=agent.resolve_runner(),
            **agent.get_instance_kwargs(session_id=session.session_id),
            **agent.runner_kwargs(),
        )

        with benchmark_scope():
            observation = session.start()

        with agent_scope():
            agent_instance.start(
                task=session.task,
                context=session.context,
                actions=session.actions,
            )

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
            tracker.on_session_error(session, RunCancelError())
            _close_session_agent(session, agent_instance)
            raise
        except AgentError as exc:
            tracker.on_session_error(session, exc)
            _close_session_agent(session, agent_instance)
        except SessionLimitReachedError as exc:
            with benchmark_scope():
                tracker.on_session_scoring(session)
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
            _close_session_agent(session, agent_instance)
        except (AgentTerminationError, BenchmarkTerminationError):
            with benchmark_scope():
                tracker.on_session_scoring(session)
            with benchmark_scope():
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
