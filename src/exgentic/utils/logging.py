# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

# exgentic/benchmarks/swebench/util.py
import logging
import sys
from contextlib import contextmanager


class SessionLogHandler(logging.Handler):
    """Forward logs from other loggers into an Exgentic session logger, with source label."""

    def __init__(self, session_logger: logging.Logger):
        super().__init__()
        self.session_logger = session_logger

    def emit(self, record: logging.LogRecord) -> None:
        msg = self.format(record)
        source = record.name or "logger"
        self.session_logger.log(record.levelno, f"[{source}] {msg}")


def hook_loggers_into_session(
    session_logger: logging.Logger,
    logger_names: list[str],
    level: int = logging.INFO,
) -> None:
    """Redirect logs from the given logger names into session_logger.

    Each line will be prefixed with [logger_name].
    """
    handler = SessionLogHandler(session_logger)
    # You can add a formatter if you want timestamps inside msg
    # handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))

    for name in logger_names:
        lg = logging.getLogger(name)
        lg.handlers.clear()  # prevent duplicate stdout noise
        lg.addHandler(handler)
        lg.setLevel(level)
        lg.propagate = False


class StreamToLogger:
    """File-like object that sends writes to session_logger with a given source label."""

    def __init__(self, session_logger: logging.Logger, source: str, level: int):
        self.logger = session_logger
        self.source = source
        self.level = level
        self._buf = ""

    def write(self, message: str) -> None:
        if not message:
            return
        self._buf += message
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            line = line.rstrip("\r")
            if line.strip():
                self.logger.log(self.level, f"[{self.source}] {line}")

    def flush(self) -> None:
        if self._buf.strip():
            self.logger.log(self.level, f"[{self.source}] {self._buf.strip()}")
        self._buf = ""


@contextmanager
def capture_stdio_to_session(
    session_logger: logging.Logger,
    stdout_level: int = logging.INFO,
    stderr_level: int = logging.WARNING,
):
    """Redirect sys.stdout and sys.stderr into session_logger.

    Lines from stdout are prefixed with [stdout], from stderr with [stderr].
    """
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = StreamToLogger(session_logger, "stdout", stdout_level)
    sys.stderr = StreamToLogger(session_logger, "stderr", stderr_level)
    try:
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout, sys.stderr = old_stdout, old_stderr
