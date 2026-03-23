# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Shared retriever service for BrowseCompPlus search index.

The Retriever loads the heavy search index once and serves queries.
It can run via any runner (direct, service, docker, etc.), allowing
a single index copy to be shared across all sessions.
"""

import argparse
import json
import threading
from typing import Any


class Retriever:
    """Loads a search index and serves queries.

    Designed to run via ``with_runner()`` in any runner.
    """

    def __init__(self, searcher_type: str, **searcher_args: Any) -> None:
        # Import torch/safetensors before the searcher module to avoid a
        # native-library initialisation conflict with FAISS that causes
        # segfaults on Apple Silicon (faiss_searcher.py imports faiss
        # before torch at module level).
        import safetensors  # noqa: F401
        import torch  # noqa: F401
        from searcher.searchers import SearcherType

        searcher_class = SearcherType.get_searcher_class(searcher_type)
        parser = argparse.ArgumentParser()
        searcher_class.parse_args(parser)

        cli: list[str] = []
        for key, value in searcher_args.items():
            flag = f"--{key.replace('_', '-')}"
            if isinstance(value, bool):
                if value:
                    cli.append(flag)
            else:
                cli.extend([flag, str(value)])

        args = parser.parse_args(cli)
        self._searcher = searcher_class(args)

    def search(self, query: str, k: int) -> list:
        return self._searcher.search(query, k)

    def get_document(self, docid: str) -> dict | None:
        return self._searcher.get_document(docid)


class RetrieverClient:
    """Lazy HTTP client to a remote Retriever service.

    Picklable — stores only the URL. Connects on first use.
    This allows it to survive serialization into Docker containers.
    """

    def __init__(self, url: str) -> None:
        self._url = url
        self._proxy: Any = None

    def _connect(self) -> None:
        if self._proxy is None:
            from ...adapters.runners.service import HTTPTransport
            from ...adapters.runners.transport import ObjectProxy

            self._proxy = ObjectProxy(HTTPTransport(self._url))

    def search(self, query: str, k: int) -> list:
        self._connect()
        return self._proxy.search(query, k)

    def get_document(self, docid: str) -> dict | None:
        self._connect()
        return self._proxy.get_document(docid)

    def close(self) -> None:
        if self._proxy is not None:
            try:
                self._proxy.close()
            except Exception:
                pass
            self._proxy = None

    def __getstate__(self) -> dict:
        return {"url": self._url}

    def __setstate__(self, state: dict) -> None:
        self._url = state["url"]
        self._proxy = None


# ── Shared retriever cache ────────────────────────────────────────────

_cache_lock = threading.Lock()
_cache: dict[str, Any] = {}


def get_shared_retriever(
    runner: str,
    runner_kwargs: dict[str, Any] | None = None,
    **retriever_kwargs: Any,
) -> Any:
    """Get or create a shared Retriever running in the specified runner."""
    from ...adapters.runners import with_runner

    key = json.dumps(retriever_kwargs, sort_keys=True, default=str)
    if key not in _cache:
        with _cache_lock:
            if key not in _cache:
                _cache[key] = with_runner(
                    Retriever,
                    runner=runner,
                    **(runner_kwargs or {}),
                    **retriever_kwargs,
                )
    return _cache[key]


def get_retriever_url(proxy: Any) -> str:
    """Extract the HTTP URL from a retriever proxy."""
    from ...adapters.runners.service import HTTPTransport

    transport = object.__getattribute__(proxy, "_transport")
    if isinstance(transport, HTTPTransport):
        return transport._base_url
    raise ValueError(
        "Cannot extract URL from non-HTTP retriever. Use runner='service' or runner='docker' for the retriever."
    )
