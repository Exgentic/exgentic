# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from ...core.context import get_context
from ...core.orchestrator.observer import Observer
from ..logging import configure_warnings_logging


class WarningsObserver(Observer):
    def __init__(self, *, replace_existing_file_handlers: bool = True) -> None:
        self._replace = replace_existing_file_handlers
        self._configured = False

    def on_run_start(self, run_config) -> None:
        if self._configured:
            return
        ctx = get_context()
        try:
            configure_warnings_logging(
                ctx.output_dir,
                ctx.run_id,
                replace_existing_file_handlers=self._replace,
            )
        except OSError:
            # Ignore failures in read-only runs.
            return
        self._configured = True
