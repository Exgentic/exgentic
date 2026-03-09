# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Cache logging utilities."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from ....core.context import try_get_context
from ....utils.settings import get_settings


def _resolve_session_id() -> Optional[str]:
    ctx = try_get_context()
    if ctx is not None:
        return ctx.session_id
    return None


def _cache_log_path(output_dir: str, run_id: str, session_id: Optional[str], role: str, filename: str) -> Path:
    base = Path(output_dir) / run_id
    if session_id:
        return base / "sessions" / session_id / role / "litellm" / filename
    return base / "run" / "litellm" / filename


class CacheLogger:
    """File-based cache logger scoped to run + session.

    INFO  - "hit" / "miss" for every lookup
    DEBUG - structured detail lines and JSON material dumps
    """

    def __init__(self, disk_cache_dir: Optional[str], strip_time: bool) -> None:
        self._disk_cache_dir = disk_cache_dir
        self._strip_time = strip_time
        self._logger: Optional[logging.Logger] = None
        self._logger_key: Optional[tuple[str, str, Optional[str]]] = None
        self._init_logged = False

    @property
    def raw(self) -> logging.Logger:
        """The underlying stdlib logger (for external consumers)."""
        return self._ensure()

    @property
    def is_debug(self) -> bool:
        return self._ensure().isEnabledFor(logging.DEBUG)

    def hit(self) -> None:
        self._ensure().info("hit")

    def miss(self) -> None:
        self._ensure().info("miss")

    def detail(self, status: str, **fields: Any) -> None:
        log = self._ensure()
        if not log.isEnabledFor(logging.DEBUG):
            return
        parts = [status] + [f"{k}={v}" for k, v in fields.items() if v is not None]
        log.debug(" ".join(parts))

    def material(self, payload: dict[str, Any]) -> None:
        log = self._ensure()
        if log.isEnabledFor(logging.DEBUG):
            log.debug(
                "cache_material %s",
                json.dumps(payload, ensure_ascii=True, separators=(",", ":")),
            )

    # -- private --

    def _ensure(self) -> logging.Logger:
        ctx = try_get_context()
        if ctx is None:
            return self._noop_logger()

        session_id = _resolve_session_id()
        role = ctx.role.value if hasattr(ctx, "role") else "framework"
        key = (ctx.run_id, ctx.output_dir, session_id, role)
        if self._logger is not None and self._logger_key == key:
            return self._logger

        name = f"exgentic.cache.{ctx.run_id}" + (f".{session_id}" if session_id else "")
        logger = logging.getLogger(name)

        if self._logger_key != key:
            for h in list(logger.handlers):
                logger.removeHandler(h)
                h.close()
            self._init_logged = False

        if not logger.handlers:
            path = _cache_log_path(ctx.output_dir, ctx.run_id, session_id, role, "cache.log")
            path.parent.mkdir(parents=True, exist_ok=True)
            level = logging.DEBUG if get_settings().log_level == "DEBUG" else logging.INFO
            handler = logging.FileHandler(path, encoding="utf-8")
            handler.setLevel(level)
            handler.setFormatter(logging.Formatter("%(message)s"))
            logger.addHandler(handler)
            logger.setLevel(level)
            logger.propagate = False

        if not self._init_logged and logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "cache_init disk_cache_dir=%s delete_time_from_messages=%s",
                self._disk_cache_dir,
                self._strip_time,
            )
            self._init_logged = True

        self._logger = logger
        self._logger_key = key
        return logger

    def _noop_logger(self) -> logging.Logger:
        logger = logging.getLogger("exgentic.cache.noop")
        if not any(isinstance(h, logging.NullHandler) for h in logger.handlers):
            logger.addHandler(logging.NullHandler())
        logger.propagate = False
        self._logger = logger
        self._logger_key = None
        return logger
