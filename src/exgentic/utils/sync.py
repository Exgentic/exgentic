# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

import asyncio
import threading
from typing import Any, Coroutine, Optional

_loop: Optional[asyncio.AbstractEventLoop] = None
_thread: Optional[threading.Thread] = None
_ready = threading.Event()
_lock = threading.Lock()


def _loop_thread_main() -> None:
    global _loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    _loop = loop
    _ready.set()
    loop.run_forever()


def run_sync(coro: Coroutine[Any, Any, Any], timeout: float | None = None) -> Any:
    """Run an async coroutine from sync code using ONE long-lived event loop.

    Safe from any thread in this process.

    Do not call from an async context.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        raise RuntimeError("run_sync() called from a running event loop; use `await` instead")

    global _loop, _thread
    with _lock:
        if _loop is None or not _loop.is_running():
            _ready.clear()
            _thread = threading.Thread(target=_loop_thread_main, name="exgentic-async-loop", daemon=True)
            _thread.start()

    if not _ready.wait(timeout=5.0) or _loop is None:
        raise RuntimeError("Failed to start shared asyncio loop")

    fut = asyncio.run_coroutine_threadsafe(coro, _loop)
    return fut.result(timeout=timeout)
