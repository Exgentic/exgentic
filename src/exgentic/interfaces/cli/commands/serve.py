# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

import rich_click as click

from ..options import apply_debug_mode


@click.command("serve")
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", type=int, default=8080, help="Port to listen on")
@click.option("--object-b64", required=True, help="Base64-encoded cloudpickle payload")
@click.option("--debug", is_flag=True, hidden=True)
def serve_cmd(host: str, port: int, object_b64: str, debug: bool) -> None:
    """Serve a pickled object over HTTP."""
    apply_debug_mode(debug)

    from ....core.context import init_context_from_env

    try:
        init_context_from_env()
    except RuntimeError:
        pass

    import base64

    import cloudpickle as cp

    from ....adapters.runners.service import serve

    obj = cp.loads(base64.b64decode(object_b64))
    serve(obj, host=host, port=port)
