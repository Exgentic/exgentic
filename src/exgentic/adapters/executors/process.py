# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Generic Process Manager.

Run code in isolated child processes with simple RPC communication.

Use cases:
- Isolate heavy dependencies
- Prevent memory leaks
- Sandbox unstable code
- Multiple isolated environments

Example:
    class DatabaseWorker(ProcessWorker):
        def initialize(self):
            self.db = connect_db()
            return {"status": "ready"}

        def handle_operation(self, op, params):
            if op == "query":
                return {"result": self.db.execute(params["sql"])}

        def cleanup(self):
            self.db.close()

    class DatabaseManager(ProcessManager):
        def _child_process_main(self, conn, db_url):
            DatabaseWorker(conn).run()

        def query(self, sql):
            return self.send_request("query", sql=sql)["result"]
"""

from abc import ABC, abstractmethod
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection
from typing import Any, Dict, Optional

from ...core.context import context_env_scope


class ProcessWorker(ABC):
    """Base class for workers that run in child processes."""

    def __init__(self, conn: Connection):
        self.conn = conn

    def initialize(self) -> Optional[Dict[str, Any]]:
        """Initialize and optionally return handshake data."""
        return None

    @abstractmethod
    def handle_operation(self, operation: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle an operation request."""
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Clean up resources before exit."""
        pass

    def run(self) -> None:
        """Main RPC loop."""
        # Send initialization handshake (optional)
        init_data = self.initialize()
        self.conn.send(init_data or {})

        # Handle requests
        while True:
            msg = self.conn.recv()
            op = msg.get("op")

            if op == "close":
                self.cleanup()
                self.conn.send({"ok": True})
                break

            try:
                response = self.handle_operation(op, msg)
                self.conn.send(response)
            except Exception as e:
                self.conn.send({"ok": False, "error": str(e)})


class ProcessManager(ABC):
    """Base class for managing child processes."""

    def __init__(self, *args, **kwargs):
        parent_conn, child_conn = Pipe()
        self._conn = parent_conn
        self._proc = Process(
            target=self._child_process_main,
            args=(child_conn, *args),
            kwargs=kwargs,
            daemon=True,
        )
        with context_env_scope():
            self._proc.start()
        self._init_data = self._conn.recv()  # Always receive ready signal

    @abstractmethod
    def _child_process_main(self, conn: Connection, *args, **kwargs) -> None:
        """Entry point for child process."""
        pass

    def send_request(self, operation: str, **params) -> Any:
        """Send request to child process."""
        self._conn.send({"op": operation, **params})
        return self._conn.recv()

    def close(self) -> None:
        """Shut down child process."""
        try:
            self._conn.send({"op": "close"})
            self._conn.recv()
        finally:
            self._proc.join(timeout=5)
