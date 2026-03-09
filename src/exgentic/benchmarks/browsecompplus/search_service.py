# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Thread-safe search service with semaphore for concurrency control."""

import argparse
import atexit
import json
import logging
import os
import threading
import time
from typing import Optional

import psutil


class SearchService:
    """Thread-safe singleton service for managing searcher instances."""

    _instance = None
    _instance_lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._searchers = {}
                    cls._instance._cache_lock = threading.RLock()
                    cls._instance._search_semaphore = threading.Semaphore(5)
                    cls._instance._shutdown = False
                    # Register cleanup to run when program exits
                    atexit.register(cls._instance.shutdown)
        return cls._instance

    def shutdown(self):
        """Clean up all searcher instances and their resources."""
        with self._cache_lock:
            if self._shutdown:
                return
            self._shutdown = True

            logger = logging.getLogger(__name__)
            logger.info("Shutting down SearchService and cleaning up models...")

            for cache_key, searcher in self._searchers.items():
                try:
                    # Try to close/cleanup the searcher if it has such methods
                    if hasattr(searcher, "close"):
                        searcher.close()
                    elif hasattr(searcher, "shutdown"):
                        searcher.shutdown()

                    if hasattr(searcher, "searcher") and hasattr(searcher.searcher, "close"):
                        searcher.searcher.close()

                except Exception as e:
                    logger.warning(f"Error cleaning up searcher {cache_key}: {e}")

            self._searchers.clear()
            logger.info("SearchService shutdown complete")

    def get_or_create_searcher(
        self,
        searcher_type: str,
        searcher_class: type,
        logger: Optional[logging.Logger] = None,
        **searcher_args,
    ):
        """Get or create a searcher instance with thread-safe access."""
        if logger is None:
            logger = logging.getLogger(__name__)

        # Create cache key
        args_str = json.dumps(searcher_args, sort_keys=True, default=str)
        cache_key = f"{searcher_type}:{args_str}"

        # Check if already cached
        if cache_key in self._searchers:
            logger.info(
                f"Reusing cached searcher: {searcher_type} (PID: {os.getpid()}, Thread: {threading.get_ident()})"
            )
            return ThreadSafeSearcherWrapper(self._searchers[cache_key], self._search_semaphore)

        # Create new searcher (with lock)
        with self._cache_lock:
            if cache_key in self._searchers:
                return ThreadSafeSearcherWrapper(self._searchers[cache_key], self._search_semaphore)

            process = psutil.Process(os.getpid())
            mem_before = process.memory_info().rss / 1024 / 1024
            thread_id = threading.get_ident()
            logger.info(
                f"Loading searcher model (before: {mem_before:.1f} MB, PID: {os.getpid()}, Thread: {thread_id})"
            )

            # Instantiate searcher
            searcher = self._instantiate_with_overrides(searcher_class, **searcher_args)
            self._searchers[cache_key] = searcher

            mem_after = process.memory_info().rss / 1024 / 1024
            logger.info(
                f"Searcher model loaded: {searcher_type} "
                f"(after: {mem_after:.1f} MB, delta: {mem_after - mem_before:.1f} MB)"
            )

            return ThreadSafeSearcherWrapper(searcher, self._search_semaphore)

    @staticmethod
    def _instantiate_with_overrides(cls, **overrides):
        """Instantiate a class that uses argparse for configuration."""
        parser = argparse.ArgumentParser()
        cls.parse_args(parser)

        cli = []
        for key, value in overrides.items():
            flag = f"--{key.replace('_', '-')}"
            if isinstance(value, bool):
                if value:
                    cli.append(flag)
            else:
                cli.extend([flag, str(value)])

        args = parser.parse_args(cli)
        return cls(args)


class ThreadSafeSearcherWrapper:
    """Wrapper that limits concurrent access to the searcher."""

    def __init__(self, searcher, semaphore):
        self._searcher = searcher
        self._semaphore = semaphore
        self._search_count = 0

    def search(self, query: str, k: int):
        """Thread-safe search with semaphore and lock for async safety."""
        # Use both semaphore and lock to ensure serialization even in async contexts
        acquired = self._semaphore.acquire(blocking=True, timeout=300)
        if not acquired:
            raise TimeoutError("Failed to acquire semaphore for search")
        try:
            self._search_count += 1
            count = self._search_count

            logger = logging.getLogger(__name__)
            logger.debug(f"Search #{count} starting (thread={threading.get_ident()})")

            result = self._searcher.search(query, k)
            time.sleep(0.01)
            logger.debug(f"Search #{count} completed")
            return result
        finally:
            self._semaphore.release()

    def get_document(self, docid: str):
        """Thread-safe document retrieval with semaphore and lock."""
        acquired = self._semaphore.acquire(blocking=True, timeout=300)
        if not acquired:
            raise TimeoutError("Failed to acquire semaphore for get_document")
        try:
            return self._searcher.get_document(docid)
        finally:
            self._semaphore.release()

    def __getattr__(self, name):
        """Forward other attributes to the wrapped searcher."""
        return getattr(self._searcher, name)


def get_search_service() -> SearchService:
    """Get the singleton SearchService instance."""
    return SearchService()
