# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

from abc import ABC, abstractmethod
from argparse import Action
from queue import Queue
from threading import Event, Lock, Semaphore
from typing import Any, Generic, Optional, TypeVar

from ...core.session import Session
from ...core.types import Observation

DONE = object()


class BaseProxySession(Session, ABC):
    """Generic queue-based proxy session.

    Implements a simple rendezvous between an external driver (agent/framework)
    and a foreign environment adapter via two blocking queues.
    """

    def __init__(self):
        super().__init__()
        self.completed = False
        self.step_count = 0
        self._to_agent: Queue = Queue()
        self._from_agent: Queue = Queue()
        self._last_observation: Optional[Any] = None

    def step(self, action: Action) -> Optional[Observation]:
        self.step_count += 1
        self._from_agent.put(action)
        next_obs = self._to_agent.get()
        return next_obs

    def start(self) -> Optional[Observation]:
        result = self._to_agent.get()
        return result

    def done(self) -> bool:
        result = self.completed
        return result

    def score(self) -> dict:
        return {
            "success": self.completed,
            "steps": self.step_count,
            "score": max(0.0, 1.0 - (self.step_count - 1) * 0.1),
        }

    def close(self):
        self.completed = True
        try:
            self._from_agent.put_nowait(DONE)
        except Exception:
            pass
        try:
            self._to_agent.put_nowait(DONE)
        except Exception:
            pass

    # --- Hooks for subclasses ---
    def put_observation(self, obs: Any) -> None:
        self._last_observation = obs
        self._to_agent.put(obs)

    def wait_for_action(self) -> Optional[Any]:
        item = self._from_agent.get()
        if item is DONE:
            self.completed = True
            return None
        return item


_PAIRING_SEMAPHORE: Semaphore = Semaphore(1)  # Only 1 session can be staged at a time
_PAIRING_LOCK: Lock = Lock()
_CURRENT_SESSION: Optional[PairableProxySession] = None


class PairableProxySession(BaseProxySession):
    """Proxy session that can be staged and paired with a proxy agent automatically."""

    def __init__(self):
        super().__init__()
        self._paired_event: Event = Event()

    # Pairing API
    def stage_for_pairing(self) -> None:
        _PAIRING_SEMAPHORE.acquire()  # Blocks until slot available
        with _PAIRING_LOCK:
            global _CURRENT_SESSION
            _CURRENT_SESSION = self

    def _mark_paired(self) -> None:
        self._paired_event.set()

    def waiting_for_pairing(self):
        return _CURRENT_SESSION == self and not self._paired_event.is_set()

    def unstage_for_pairing(self):
        if self.waiting_for_pairing():
            with _PAIRING_LOCK:
                global _CURRENT_SESSION
                assert _CURRENT_SESSION == self
                _CURRENT_SESSION = None
            _PAIRING_SEMAPHORE.release()

    def pair_to_agent(self, timeout: Optional[float] = 10.0) -> None:
        ok = self._paired_event.wait(timeout=timeout)
        if not ok:
            raise RuntimeError("Timed out waiting for proxy agent to pair with session")

    # Gate helpers
    @classmethod
    def block_pairing(cls) -> None:
        pass  # No longer needed with lock-based approach

    @classmethod
    def allow_pairing(cls) -> None:
        pass  # No longer needed with lock-based approach

    @classmethod
    def pairing_allowed(cls) -> bool:
        return True  # Always allowed with lock-based approach


SessionT = TypeVar("SessionT", bound=BaseProxySession)


class BaseProxyAgent(ABC, Generic[SessionT]):
    """Base mixin providing generic step handling between a proxy session and an external environment.

    Uses core terms (session, observation, action).

    Adapters should call `handle_observation(observation, state)` from their
    environment-specific entrypoint.
    """

    def _ensure_session(self, state: Optional[SessionT], observation: Any) -> SessionT:
        if state is None:
            return self.create_session(observation)
        self.update_session_observation(state, observation)
        return state

    def handle_observation(self, observation: Any, state: Optional[SessionT]):
        """Generic step handler: ensure session, wait for action, translate response.

        - observation: an environment-specific observation object
        - state: the proxy session instance (or None for a new session)
        Returns (environment-specific response, new_state).
        """
        session = self._ensure_session(state, observation)
        action = session.wait_for_action()
        response_obj, new_state = self.action_to_response(action, observation, session)
        return response_obj, new_state

    # --- Subclass hooks ---
    @abstractmethod
    def create_session(self, first_observation: Any) -> SessionT:
        pass

    @abstractmethod
    def update_session_observation(self, session: SessionT, observation: Any) -> None:
        pass

    @abstractmethod
    def action_to_response(self, action: Any, observation: Any, session: SessionT):
        pass


class PairableProxyAgent(BaseProxyAgent[SessionT]):
    """Proxy agent that adopts the currently staged PairableProxySession."""

    def adopt_staged_session(self) -> SessionT:
        with _PAIRING_LOCK:
            global _CURRENT_SESSION
            sess = _CURRENT_SESSION
            if sess is None:
                raise RuntimeError("No staged session available for pairing")
            _CURRENT_SESSION = None
        sess._mark_paired()  # type: ignore[attr-defined]
        _PAIRING_SEMAPHORE.release()  # Release slot for next session
        return sess  # type: ignore[return-value]

    # Gate helpers
    @classmethod
    def block_pairing(cls) -> None:
        PairableProxySession.block_pairing()

    @classmethod
    def allow_pairing(cls) -> None:
        PairableProxySession.allow_pairing()

    @classmethod
    def pairing_allowed(cls) -> bool:
        return PairableProxySession.pairing_allowed()
