# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

import asyncio
import gc


def close_aiohttp_sessions_silently() -> None:
    """Best-effort cleanup for stray aiohttp sessions to avoid resource warnings."""
    try:
        import aiohttp  # type: ignore
    except Exception:
        return

    sessions = [obj for obj in gc.get_objects() if isinstance(obj, aiohttp.ClientSession) and not obj.closed]
    if not sessions:
        return

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(asyncio.gather(*(s.close() for s in sessions), return_exceptions=True))
    asyncio.set_event_loop(None)
    loop.close()
