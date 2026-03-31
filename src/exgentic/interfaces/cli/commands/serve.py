# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

import logging

import rich_click as click

from ..options import apply_debug_mode

logger = logging.getLogger(__name__)


@click.command("serve")
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", type=int, default=8080, help="Port to listen on")
@click.option("--cls", required=True, help="Class to serve (module.path:ClassName)")
@click.option("--kwargs", "kwargs_json", default=None, help="JSON constructor kwargs")
@click.option("--kwargs-b64", "kwargs_b64", default=None, help="Base64-encoded cloudpickle kwargs")
@click.option("--debug", is_flag=True, hidden=True)
def serve_cmd(host: str, port: int, cls: str, kwargs_json: str | None, kwargs_b64: str | None, debug: bool) -> None:
    """Serve a class instance over HTTP."""
    apply_debug_mode(debug)

    import importlib
    import json

    from ....core.context import try_init_context

    try_init_context()

    from ....adapters.runners.service import serve

    # Import the target module FIRST so that package __init__.py files
    # (which may set environment variables like TAU2_DATA_DIR) run before
    # cloudpickle deserialization triggers transitive library imports.
    module_path, attr_name = cls.rsplit(":", 1)
    mod = importlib.import_module(module_path)
    klass = getattr(mod, attr_name)

    # Deserialize kwargs: JSON (preferred) or cloudpickle fallback.
    if kwargs_b64 is not None:
        import base64

        import cloudpickle as cp

        kw = cp.loads(base64.b64decode(kwargs_b64))
    elif kwargs_json is not None:
        kw = json.loads(kwargs_json)
    else:
        kw = {}

    obj = klass(**kw)
    serve(obj, host=host, port=port)
