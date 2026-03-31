# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Basic logging setup for Exgentic components."""
import logging
import os
import sys
from collections.abc import Callable
from pathlib import Path

from uvicorn.logging import DefaultFormatter

from ...core.context import try_get_context
from ...utils.settings import get_settings


def _build_console_handler(*, log_level: int, formatter: logging.Formatter) -> logging.Handler:
    try:
        from rich.logging import RichHandler
    except Exception:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(log_level)
        handler.setFormatter(formatter)
        return handler

    console_formatter = logging.Formatter("%(message)s")
    handler = RichHandler(
        show_time=False,
        show_level=True,
        show_path=False,
        rich_tracebacks=True,
        markup=False,
    )
    handler.setLevel(log_level)
    handler.setFormatter(console_formatter)
    return handler


def get_logger(
    name: str,
    log_file_path: str | None = None,
    *,
    console: bool | None = None,
    propagate: bool | None = None,
) -> logging.Logger:
    """Get a logger with standard Exgentic configuration.

    - If `log_file_path` is provided: attach a file handler. By default, do not propagate
      to avoid console duplication; set `propagate=True` if you want bubbling.
    - If `log_file_path` is not provided: no console by default unless `console=True` or
      `EXGENTIC_CONSOLE_LOG` is truthy. Optional default file sink via context run dir.
    - Console handlers use Rich when available, falling back to a standard stream handler.

    Callers decide explicitly if they want console output by passing `console=True`.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    settings = get_settings()
    log_level = logging._nameToLevel.get(settings.log_level.upper(), logging.INFO)
    logger.setLevel(log_level)

    formatter = logging.Formatter("%(levelname)s | %(message)s")

    if log_file_path:
        # File-targeted logger
        log_file = Path(log_file_path)
        try:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_file, encoding="utf-8")
        except OSError:
            # Fall back to console logging when file logging isn't writable.
            ch = _build_console_handler(log_level=logging.INFO, formatter=formatter)
            logger.addHandler(ch)
            if propagate is None:
                propagate = False
            logger.propagate = propagate
            return logger
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        # Optional console mirroring if explicitly requested
        if bool(console):
            ch = _build_console_handler(log_level=logging.INFO, formatter=formatter)
            logger.addHandler(ch)

        # Default: avoid bubbling to parent to prevent duplicates
        if propagate is None:
            propagate = False
        logger.propagate = propagate
    else:
        # Non-file logger
        enable_console_env = os.environ.get("EXGENTIC_CONSOLE_LOG", "").lower() in (
            "1",
            "true",
            "yes",
        )
        if bool(console) or enable_console_env:
            ch = _build_console_handler(log_level=logging.INFO, formatter=formatter)
            logger.addHandler(ch)

        # Optional default file sink (single consolidated log) if requested
        ctx = try_get_context()
        if ctx is not None:
            lf = Path(ctx.output_dir) / "exgentic.log"  # global log, not run-scoped
            try:
                lf.parent.mkdir(parents=True, exist_ok=True)
                fh = logging.FileHandler(lf, encoding="utf-8")
            except OSError:
                fh = None
            if fh is not None:
                fh.setLevel(logging.DEBUG)
                fh.setFormatter(formatter)
                logger.addHandler(fh)

        # Default propagation for non-file logger: True unless explicitly set
        if propagate is not None:
            logger.propagate = propagate

    return logger


def get_disabled_logger(name: str | None = None) -> logging.Logger:
    """Return a logger that discards all messages (NullHandler, no propagation)."""
    lname = name or f"{__name__}.noop"
    logger = logging.getLogger(lname)
    if not any(isinstance(h, logging.NullHandler) for h in logger.handlers):
        logger.addHandler(logging.NullHandler())
    logger.propagate = False
    logger.setLevel(logging.CRITICAL)
    return logger


def close_logger(log: logging.Logger) -> None:
    """Close and detach all file handlers from the given logger."""
    handlers_to_close = list(log.handlers)

    for handler in handlers_to_close:
        if isinstance(handler, logging.FileHandler):
            handler.close()
            log.removeHandler(handler)


def configure_warnings_logging(
    run_dir_base: str | None = None,
    run_id: str | None = None,
    *,
    replace_existing_file_handlers: bool = True,
) -> str:
    """Set up logging for Python warnings to a `warnings.log` file.

    - Computes `<run_dir>/<run_id>/run/warnings.log` using provided args or env/settings.
    - Enables `logging.captureWarnings(True)`.
    - Optionally replaces existing FileHandlers on the `py.warnings` logger.

    Returns the path to the warnings log file as a string.
    """
    from ...utils.paths import RunPaths

    ctx = try_get_context()
    if ctx is not None:
        base = run_dir_base or ctx.output_dir
    else:
        settings = get_settings()
        base = run_dir_base or settings.output_dir
    if run_id is None:
        ctx = try_get_context()
        rid = ctx.run_id if ctx is not None else "default"
    else:
        rid = run_id
    warnings_path = RunPaths(run_id=rid, output_dir=base).warnings
    warnings_path.parent.mkdir(parents=True, exist_ok=True)

    logging.captureWarnings(True)
    wlogger = logging.getLogger("py.warnings")

    # Remove conflicting handlers per requested policy
    if replace_existing_file_handlers:
        for h in list(wlogger.handlers):
            if isinstance(h, logging.FileHandler):
                wlogger.removeHandler(h)
                try:
                    h.close()
                except Exception:
                    pass
    else:
        for h in list(wlogger.handlers):
            try:
                if isinstance(h, logging.FileHandler) and h.baseFilename == str(warnings_path):
                    wlogger.removeHandler(h)
                    try:
                        h.close()
                    except Exception:
                        pass
            except Exception:
                pass

    wfh = logging.FileHandler(str(warnings_path), encoding="utf-8")
    wfh.setLevel(logging.WARNING)
    wfh.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))
    wlogger.addHandler(wfh)
    wlogger.setLevel(logging.WARNING)
    wlogger.propagate = False
    return str(warnings_path)


def configure_uvicorn_file_logging(log_path: Path, *, thread_id: int) -> Callable[[], None]:
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(
        DefaultFormatter(
            "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    handler.addFilter(lambda record: record.thread == thread_id)

    uvicorn_loggers = [
        logging.getLogger("uvicorn"),
        logging.getLogger("uvicorn.error"),
        logging.getLogger("uvicorn.access"),
    ]
    prev_logger_state: dict[logging.Logger, tuple[int, bool, list[logging.Handler]]] = {}
    for lg in uvicorn_loggers:
        removed_handlers = list(lg.handlers)
        for h in removed_handlers:
            lg.removeHandler(h)
        prev_logger_state[lg] = (lg.level, lg.propagate, removed_handlers)
        lg.addHandler(handler)
        lg.setLevel(logging.INFO)
        lg.propagate = False

    def _cleanup() -> None:
        for lg in uvicorn_loggers:
            lg.removeHandler(handler)
            prev_state = prev_logger_state.get(lg)
            if prev_state is not None:
                lg.setLevel(prev_state[0])
                lg.propagate = prev_state[1]
                for h in prev_state[2]:
                    if h not in lg.handlers:
                        lg.addHandler(h)
        handler.close()

    return _cleanup


def configure_library_file_logging(
    log_path: Path, *, logger_names: list[str], thread_id: int | None = None
) -> Callable[[], None]:
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    if thread_id is not None:
        handler.addFilter(lambda record: record.thread == thread_id)

    prefixes = tuple(logger_names)
    names = set(logger_names)
    for name in logging.root.manager.loggerDict:
        if name.startswith(prefixes):
            names.add(name)

    prev_logger_state: dict[logging.Logger, tuple[int, bool, list[logging.Handler]]] = {}
    for name in names:
        lg = logging.getLogger(name)
        removed_handlers = list(lg.handlers)
        for h in removed_handlers:
            lg.removeHandler(h)
        prev_logger_state[lg] = (lg.level, lg.propagate, removed_handlers)
        lg.addHandler(handler)
        lg.setLevel(logging.INFO)
        lg.propagate = False

    def _cleanup() -> None:
        for lg, (level, propagate, removed) in prev_logger_state.items():
            lg.removeHandler(handler)
            lg.setLevel(level)
            lg.propagate = propagate
            for h in removed:
                if h not in lg.handlers:
                    lg.addHandler(h)
        handler.close()

    return _cleanup


def attach_library_logger_to_handler(
    library_logger_name: str,
    handler: logging.Handler,
    *,
    level: int = logging.DEBUG,
    propagate: bool = False,
) -> tuple[logging.Logger, list[logging.Handler], bool]:
    """Attach a library logger to the given handler.

    Returns a tuple of (logger, previous_handlers, previous_propagate) so callers
    can restore the original configuration later.
    """
    logger = logging.getLogger(library_logger_name)
    prev_handlers = list(logger.handlers)
    prev_propagate = logger.propagate

    handler.setLevel(level)
    logger.handlers = [handler]
    logger.propagate = propagate

    return logger, prev_handlers, prev_propagate


def restore_library_logger(
    logger: logging.Logger,
    handlers: list[logging.Handler],
    propagate: bool,
) -> None:
    """Restore a library logger's handlers and propagation flag."""
    logger.handlers = handlers
    logger.propagate = propagate


def add_loguru_file_sink(file_obj, level: str = "DEBUG", colorize: bool = False):
    """Add a Loguru sink for the given file-like object.

    Returns the sink id, or None if Loguru is not available.
    """
    try:
        from loguru import logger as _loguru  # type: ignore[import-not-found]
    except Exception:
        return None

    try:
        return _loguru.add(file_obj, level=level, colorize=colorize)
    except Exception:
        return None


def remove_loguru_sink(sink_id) -> None:
    """Remove a previously registered Loguru sink, ignoring errors."""
    if sink_id is None:
        return

    try:
        from loguru import logger as _loguru  # type: ignore[import-not-found]
    except Exception:
        return

    try:
        _loguru.remove(sink_id)
    except Exception:
        pass
