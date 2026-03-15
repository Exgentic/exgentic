# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

import builtins as _bl
import multiprocessing as mp
import traceback
import weakref
from typing import Any

from ...core.context import (
    context_env_scope,
    init_context_from_env,
    set_context,
    try_get_context,
)

try:
    import cloudpickle as _cp
except Exception:  # pragma: no cover
    import pickle as _cp  # type: ignore


def _dumps(obj: Any) -> bytes:
    return _cp.dumps(obj)


def _loads(data: Any) -> Any:
    # Allow backward-compatibility if data is not bytes
    if isinstance(data, (bytes, bytearray, memoryview)):
        return _cp.loads(data)
    return data


def _worker_process(queue_in, queue_out):
    """Generic worker that executes commands on an object (cloudpickle-based)."""
    # First message contains the class and init args (pickled)
    first = queue_in.get()
    init_msg = _loads(first)
    # Support both legacy (cls, args, kwargs) and tagged ('init', cls, args, kwargs)
    if isinstance(init_msg, tuple) and init_msg and init_msg[0] == "init":
        _, cls, args, kwargs, ctx = init_msg
    else:
        cls, args, kwargs = init_msg
        ctx = None

    if ctx is not None:
        set_context(ctx)
    else:
        try:
            init_context_from_env()
        except RuntimeError:
            pass

    try:
        # Create instance
        if hasattr(cls, "get_worker"):
            obj = cls.get_worker(*args, **kwargs)
        else:
            obj = cls(*args, **kwargs)
        queue_out.put(_dumps(("ready", None)))
    except Exception as e:
        queue_out.put(_dumps(("error", (type(e).__name__, str(e), traceback.format_exc()))))
        return

    # Process commands
    while True:
        try:
            raw = queue_in.get()
            if raw is None:
                break
            cmd = _loads(raw)

            op, name, *params = cmd

            try:
                if op == "get":
                    result = getattr(obj, name)
                elif op == "set":
                    setattr(obj, name, params[0])
                    result = None
                elif op == "call":
                    result = getattr(obj, name)(*params[0], **params[1])
                elif op == "del":
                    delattr(obj, name)
                    result = None
                else:
                    raise ValueError(f"Unknown op: {op}")

                queue_out.put(_dumps(("ok", result)))

            except Exception as e:
                queue_out.put(_dumps(("error", (type(e).__name__, str(e), traceback.format_exc()))))

        except (EOFError, BrokenPipeError):
            break


class RemoteProcess:
    """Proxy that forwards operations to a worker process."""

    def __init__(self, cls, *args, **kwargs):
        self._cls = cls

        # Create queues and process
        ctx = mp.get_context("spawn")
        self._queue_in = ctx.Queue()
        self._queue_out = ctx.Queue()
        self._process = ctx.Process(target=_worker_process, args=(self._queue_in, self._queue_out), daemon=True)
        with context_env_scope():
            self._process.start()

        # Send class and init args (pickled)
        ctx = try_get_context()
        self._queue_in.put(_dumps(("init", cls, args, kwargs, ctx)))

        # Wait for ready signal (block until child is ready)
        raw = self._queue_out.get()
        status, data = _loads(raw)
        if status == "error":
            self._cleanup()
            exc_type, exc_msg, exc_tb = data
            raise RuntimeError(f"Worker init failed: {exc_type}: {exc_msg}")

        # Setup cleanup
        self._finalizer = weakref.finalize(
            self,
            self._cleanup_resources,
            self._process,
            self._queue_in,
            self._queue_out,
        )

    def _cleanup(self):
        """Manual cleanup."""
        try:
            self._queue_in.put(None)
            self._process.join(timeout=1.0)
        except Exception:
            pass
        if self._process.is_alive():
            self._process.terminate()
        try:
            self._queue_in.close()
            self._queue_out.close()
        except Exception:
            pass

    @staticmethod
    def _cleanup_resources(process, queue_in, queue_out):
        """Cleanup for finalizer."""
        try:
            queue_in.put(None)
            process.join(timeout=1.0)
        except Exception:
            pass
        if process.is_alive():
            process.terminate()
        try:
            queue_in.close()
            queue_out.close()
        except Exception:
            pass

    def _send(self, cmd, timeout: float | None = None):
        """Send command and get response (optional timeout)."""
        if not self._process.is_alive():
            raise RuntimeError("Worker process died")

        self._queue_in.put(_dumps(cmd))
        raw = self._queue_out.get(timeout=timeout)
        status, data = _loads(raw)

        if status == "error":
            exc_type, exc_msg, exc_tb = data
            exc_class = getattr(_bl, exc_type, None)
            if isinstance(exc_class, type) and issubclass(exc_class, BaseException):
                raise exc_class(exc_msg)
            raise RuntimeError(f"{exc_type}: {exc_msg}")

        return data

    def __getattr__(self, name):
        # Check if it's a method (best effort; still forwarded if not resolvable here)
        if hasattr(self._cls, name) and callable(getattr(self._cls, name)):

            def method(*args, **kwargs):
                return self._send(("call", name, args, kwargs))

            method.__name__ = name
            return method
        # Get attribute
        return self._send(("get", name))

    def __setattr__(self, name, value):
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            self._send(("set", name, value))

    def __delattr__(self, name):
        if name.startswith("_"):
            object.__delattr__(self, name)
        else:
            self._send(("del", name))

    def __del__(self):
        pass  # Finalizer handles cleanup

    def shutdown(self) -> None:
        """Explicitly stop the worker process and detach the finalizer.

        Note: The proxied object's own `close()` method (if any) is forwarded
        to the remote process. Call `shutdown()` to terminate the RPC worker
        itself.
        """
        self._cleanup()
        try:
            self._finalizer.detach()
        except Exception:
            pass


def remote_process_class(cls):
    """Decorator that makes a class run in a separate process.

    IMPORTANT: The decorated class must be defined at module level
    (not inside if __name__ == "__main__") for pickling to work.

    Usage:
        @remote_process_class
        class MyClass:
            def __init__(self, value):
                self.value = value

            def work(self):
                return self.value * 2

        obj = MyClass(10)      # Starts remote process
        result = obj.work()    # Runs in separate process
        obj.value = 20         # Attribute access works
    """
    # CRITICAL: Don't replace the class!
    # Instead, monkey-patch __new__ to return a proxy
    original_new = cls.__new__
    original_init = cls.__init__

    # Store original class reference
    _original_class = cls

    def new_new(cls, *args, **kwargs):
        # When someone tries to create an instance, return a proxy instead
        if cls is _original_class:
            return RemoteProcess(_original_class, *args, **kwargs)
        # For subclasses, use original __new__
        if original_new is object.__new__:
            return original_new(cls)
        return original_new(cls, *args, **kwargs)

    # Replace __new__ method
    cls.__new__ = new_new

    # Make __init__ do nothing when called on proxy
    def new_init(self, *args, **kwargs):
        if isinstance(self, RemoteProcess):
            pass  # Proxy already initialized
        else:
            original_init(self, *args, **kwargs)

    cls.__init__ = new_init

    # Provide a factory that the worker process can use to build a real instance
    @classmethod
    def get_worker(c, *args, **kwargs):
        # Bypass the proxying __new__/__init__ to construct a real object
        inst = object.__new__(c)
        original_init(inst, *args, **kwargs)
        return inst

    cls.get_worker = get_worker  # type: ignore[attr-defined]

    # Return the SAME class (modified in place)
    return cls


# Define test class at module level for pickling
@remote_process_class
class Worker:
    def __init__(self, name, value=0):
        import os

        self.name = name
        self.value = value
        self.pid = os.getpid()
        print(f"Worker {name} initialized in process {self.pid}")

    def compute(self, n):
        """CPU-intensive task."""
        total = sum(i**2 for i in range(n))
        return total

    def get_info(self):
        import os

        return {
            "name": self.name,
            "value": self.value,
            "pid": self.pid,
            "main": os.getpid(),
        }

    def error(self):
        raise ValueError(f"Error from {self.name}")


@remote_process_class
class CustomWorker:
    def __init__(self, x):
        self.x = x

    @classmethod
    def get_worker(cls, x):
        import os

        print(f"Custom init in process {os.getpid()}")
        return cls(x * 2)

    def get_x(self):
        return self.x


# Example usage
if __name__ == "__main__":
    import os
    import time

    print(f"Main process: {os.getpid()}\n")

    # Create workers
    w1 = Worker("W1", 10)
    w2 = Worker("W2", 20)

    # Test basic operations
    print("Worker info:")
    print(f"  W1: {w1.get_info()}")
    print(f"  W2: {w2.get_info()}")

    # Test attribute access
    print(f"\nW1 value: {w1.value}")
    w1.value = 100
    print(f"W1 value after update: {w1.value}")

    # Test parallel computation
    print("\nParallel computation:")
    start = time.time()
    r1 = w1.compute(1000000)
    r2 = w2.compute(1000000)
    print(f"  Results: {r1}, {r2}")
    print(f"  Time: {time.time() - start:.2f}s")

    # Test exception
    print("\nException handling:")
    try:
        w1.error()
    except ValueError as e:
        print(f"  Caught: {e}")

    # Test custom worker
    print("\nCustom worker:")
    cw = CustomWorker(5)
    print(f"  Custom worker x: {cw.get_x()}")  # Should be 10

    print("\nDone!")
