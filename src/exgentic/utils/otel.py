# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Utility functions for OpenTelemetry initialization and logging.

This module provides OTEL setup functions and structured logging for OTEL operations.
"""

import base64
import json
import os
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path, PurePath
from typing import Any, Dict, Mapping, Optional, Sequence, Union

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
    OTLPSpanExporter as GrpcOTLPSpanExporter,
)
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter as HttpOTLPSpanExporter,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
from opentelemetry.sdk.trace.id_generator import IdGenerator
from opentelemetry.trace import Tracer
from opentelemetry.util.types import AttributeValue

OTEL_SPAN_ATTRIBUTE_NAMESPACE = "exgentic"


class UrandomIdGenerator(IdGenerator):
    """ID generator using os.urandom for unique random IDs."""

    def generate_span_id(self) -> int:
        return int.from_bytes(os.urandom(8), "big")

    def generate_trace_id(self) -> int:
        return int.from_bytes(os.urandom(16), "big")


def init_tracing_from_env(
    service_name: Optional[str] = None,
    use_urandom_ids: bool = True,
    use_simple_processor: bool = True,
) -> Tracer:
    """Initialize OpenTelemetry tracing from environment variables.

    This function is idempotent - if a TracerProvider is already set, it will not
    reinitialize and will just return a tracer.

    Args:
        service_name: Optional service name override. Defaults to OTEL_SERVICE_NAME env var or "exgentic".
        use_urandom_ids: Whether to use UrandomIdGenerator for span/trace IDs. Default True.
        use_simple_processor: If True, use SimpleSpanProcessor (immediate export) instead of BatchSpanProcessor.
                             Useful for subprocesses that may exit before batch export completes.

    Returns:
        A Tracer instance.

    Env vars honored by the SDK / exporters include (non-exhaustive):
      - OTEL_SERVICE_NAME
      - OTEL_SERVICE_VERSION
      - OTEL_RESOURCE_ATTRIBUTES
      - OTEL_TRACES_SAMPLER, OTEL_TRACES_SAMPLER_ARG
      - OTEL_EXPORTER_OTLP_{ENDPOINT,HEADERS,PROTOCOL}
      - OTEL_EXPORTER_OTLP_TRACES_{ENDPOINT,HEADERS,PROTOCOL,COMPRESSION,TIMEOUT}
    """
    # Check if tracer provider is already set
    current_provider = trace.get_tracer_provider()
    # ProxyTracerProvider means no real provider is set
    if type(current_provider).__name__ != "ProxyTracerProvider":
        # Already initialized, just return a tracer
        return trace.get_tracer(__name__)

    # Initialize the tracer provider
    resource = Resource.create(
        {
            "service.name": service_name or os.getenv("OTEL_SERVICE_NAME", "exgentic"),
            "service.version": os.getenv("OTEL_SERVICE_VERSION", "1.0.0"),
            "service.namespace": os.getenv("OTEL_SERVICE_NAMESPACE", "exgentic"),
            "deployment.environment.name": os.getenv("DEPLOYMENT_ENVIRONMENT", "dev"),
        }
    )

    id_generator = UrandomIdGenerator() if use_urandom_ids else None
    provider = TracerProvider(resource=resource, id_generator=id_generator)
    trace.set_tracer_provider(provider)

    protocol = os.getenv("OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf").strip().lower()
    exporter = GrpcOTLPSpanExporter() if protocol == "grpc" else HttpOTLPSpanExporter()

    # Use SimpleSpanProcessor for immediate export (useful for subprocesses)
    # or BatchSpanProcessor for better performance (default)
    if use_simple_processor:
        provider.add_span_processor(SimpleSpanProcessor(exporter))
    else:
        provider.add_span_processor(BatchSpanProcessor(exporter))

    return trace.get_tracer(__name__)


def check_otel_collector_health(timeout: int = 5) -> tuple[bool, Optional[str]]:
    """Check if the OTEL collector endpoint is reachable and protocol matches.

    Verifies:
    1. Endpoint is reachable (socket connection)
    2. Protocol (HTTP/gRPC) matches what the server supports

    Args:
        timeout: Connection timeout in seconds (default: 5)

    Returns:
        Tuple of (is_healthy, error_message)
        - (True, None) if collector is reachable and protocol matches
        - (False, error_message) if collector is not reachable or protocol mismatch
    """
    import socket
    from urllib.parse import urlparse

    # Get endpoint and protocol from environment
    protocol = os.getenv("OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf").strip().lower()

    # Try traces-specific endpoint first, fall back to general endpoint
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")

    if not endpoint:
        return False, "OTEL_EXPORTER_OTLP_ENDPOINT not specified"

    host = "Unknown"
    port = 0
    try:
        parsed = urlparse(endpoint)
        host = parsed.hostname
        port = parsed.port

        if not host or not port:
            return False, f"Invalid endpoint URL: {endpoint}"

        # Step 1: Check if endpoint is reachable via socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()

        if result != 0:
            return False, f"Cannot connect to OTEL collector at {host}:{port}"

        # Step 2: Verify protocol matches by attempting a minimal request
        if protocol.startswith("http"):
            # For HTTP protocol, try a simple HTTP request to verify it's an HTTP server
            try:
                import http.client

                # Determine if we should use HTTPS
                use_https = parsed.scheme == "https"
                conn_class = http.client.HTTPSConnection if use_https else http.client.HTTPConnection

                conn = conn_class(host, port, timeout=timeout)
                # Try to access the OTLP traces endpoint
                conn.request("POST", "/v1/traces", headers={"Content-Type": "application/x-protobuf"})
                response = conn.getresponse()
                conn.close()

                # We expect either 200 (OK), 400 (bad request - empty body), or 405 (method not allowed)
                # What we DON'T want is connection refused or protocol errors
                if response.status in (200, 400, 405, 415):  # 415 = Unsupported Media Type
                    return True, None
                return False, f"HTTP endpoint responded with unexpected status: {response.status}"

            except http.client.HTTPException as e:
                return False, f"HTTP protocol error: {e!s}. Server {endpoint} may not support HTTP protocol."
            except Exception as e:
                # If we can connect via socket but HTTP fails, likely a protocol mismatch
                return False, f"Protocol mismatch: configured as HTTP but server {endpoint} may be gRPC. Error: {e!s}"

        elif protocol == "grpc":
            # For gRPC, attempt a basic protocol check
            # We'll try to send a minimal gRPC frame to verify it's a gRPC server
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(timeout)
                sock.connect((host, port))

                # Send HTTP/2 connection preface followed by SETTINGS frame
                # This is what a real gRPC client sends
                preface = b"PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n"
                # HTTP/2 SETTINGS frame: length=0, type=4, flags=0, stream_id=0
                settings_frame = b"\x00\x00\x00\x04\x00\x00\x00\x00\x00"
                sock.sendall(preface + settings_frame)

                # Try to receive a response
                # A gRPC/HTTP2 server MUST respond with a SETTINGS frame
                sock.settimeout(2)  # Short timeout for response
                response = sock.recv(1024)
                sock.close()

                # Validate we got a proper HTTP/2 response
                # HTTP/2 frames start with 3-byte length, 1-byte type, 1-byte flags, 4-byte stream ID
                if len(response) >= 9:
                    # Check if we got a SETTINGS frame (type=4) or other valid HTTP/2 frame
                    frame_type = response[3]
                    if frame_type in (0x04, 0x00, 0x01):  # SETTINGS, DATA, or HEADERS frame
                        return True, None

                # If we got a response but it's not HTTP/2, it's likely HTTP/1.1
                if response:
                    if response.startswith(b"HTTP/1"):
                        return (
                            False,
                            f"Server at {endpoint} is HTTP/1.1, not gRPC. Use 'http/protobuf' protocol instead.",
                        )
                    return (
                        False,
                        f"Server at {endpoint} responded but not with valid HTTP/2 frames. May not be a gRPC server.",
                    )

                return False, f"gRPC endpoint at {endpoint} did not respond to HTTP/2 preface"

            except socket.timeout:
                return (
                    False,
                    f"Timeout waiting for gRPC response from {endpoint}. Server may not support gRPC protocol.",
                )
            except Exception as e:
                return False, f"gRPC protocol check failed: {e!s}. Server {endpoint} may not support gRPC protocol."

        else:
            return False, f"Unknown protocol: {protocol}. Expected 'http/protobuf' or 'grpc'"

    except socket.gaierror:
        return False, f"Cannot resolve hostname: {host}"
    except socket.timeout:
        return False, f"Connection timeout to {host}:{port}"
    except Exception as e:
        return False, f"Error checking OTEL collector: {e!s}"


def flush_traces(timeout_millis: int = 30000) -> bool:
    """Flush all pending spans to the configured exporter.

    This ensures that all spans are exported before the process exits or
    when you want to guarantee delivery at a specific point (e.g., end of session).

    Args:
        timeout_millis: Maximum time to wait for flush to complete, in milliseconds.
                       Default is 30000 (30 seconds).

    Returns:
        True if flush succeeded, False otherwise.
    """
    try:
        provider = trace.get_tracer_provider()
        # Check if we have a real TracerProvider (not ProxyTracerProvider)
        if type(provider).__name__ == "ProxyTracerProvider":
            return True  # No real provider, nothing to flush
        return provider.force_flush(timeout_millis)
    except Exception:
        return False


_Primitive = (str, bool, int, float)


def _to_primitive_number(x: Any) -> Optional[Union[int, float]]:
    """Convert numeric-ish types to plain Python int/float."""
    if isinstance(x, bool):
        # bool is a subclass of int; do not coerce.
        return None
    if isinstance(x, (int, float)):
        return x
    if isinstance(x, Decimal):
        # Prefer float for AttributeValue; fall back to string elsewhere if needed.
        return float(x)
    return None


def _canonical_attr_type(x: Any) -> Optional[type]:
    """Return which primitive type x maps to, or None if not primitive."""
    if isinstance(x, bool):
        return bool
    if isinstance(x, int) and not isinstance(x, bool):
        return int
    if isinstance(x, float):
        return float
    if isinstance(x, str):
        return str
    return None


def _json_default(o: Any) -> Any:
    """Safe fallback for json.dumps(default=...)."""
    # Pydantic v2
    if hasattr(o, "model_dump"):
        try:
            return o.model_dump()
        except Exception:
            pass
    # Pydantic v1
    if hasattr(o, "dict"):
        try:
            return o.dict()
        except Exception:
            pass
    # dataclasses
    try:
        from dataclasses import asdict, is_dataclass

        if is_dataclass(o):
            return asdict(o)
    except Exception:
        pass
    # Datetime/Date
    if isinstance(o, (datetime, date)):
        return o.isoformat()
    # Paths
    if isinstance(o, (PurePath,)):
        return str(o)
    # Bytes-like → base64 wrapper so it round-trips
    if isinstance(o, (bytes, bytearray, memoryview)):
        return {"__bytes_b64__": base64.b64encode(bytes(o)).decode("ascii")}
    # Fallback: repr as string
    return str(o)


def _to_homogeneous_sequence(seq: Sequence[Any]) -> Optional[Sequence[Union[str, bool, int, float]]]:
    """Try to coerce a sequence into a homogeneous list of primitives allowed by AttributeValue.

    Returns list on success, None on failure.
    """
    arr = list(seq)

    primed = []
    for v in arr:
        if v is None:
            # None in arrays is not portably supported; bail to JSON outside.  [2](https://opentelemetry.io/docs/specs/otel/common/)
            return None
        if isinstance(v, _Primitive):
            primed.append(v)
            continue
        # Try numeric coercion (Decimal)
        num = _to_primitive_number(v)
        if num is not None:
            primed.append(num)
            continue
        # Datetimes/paths/bytes -> string
        if isinstance(v, (datetime, date, PurePath, bytes, bytearray, memoryview)):
            primed.append(str(_json_default(v)))
            continue
        # Not representable as primitive
        return None

    # Check homogeneity (bool must not mix with ints)
    types = {_canonical_attr_type(x) for x in primed}
    if None in types:
        return None
    if len(types) == 1:
        return primed
    # Allow implicit upcast to float when mixing int/float
    if types == {int, float}:
        return [float(x) for x in primed]
    # Mixed types like str+int or bool+int are not allowed for attribute arrays.  [2](https://opentelemetry.io/docs/specs/otel/common/)
    return None


def to_otel_attribute_value(value: Any, *, prefer_json: bool = True) -> Optional[AttributeValue]:
    """Convert an arbitrary value into an OpenTelemetry-Python AttributeValue for spans.

    Returns:
        - A valid AttributeValue (str|bool|int|float|homogeneous Sequence thereof) on success.
        - None if the attribute should be skipped (e.g., value is None).
    """
    # 1) None: skip (undefined/strongly discouraged).  [3](https://opentelemetry-python.readthedocs.io/en/latest/api/trace.span.html)
    if value is None:
        return None

    # 2) Accepted primitives
    if isinstance(value, _Primitive):
        return bool(value) if isinstance(value, bool) else value  # type: ignore[return-value]

    # 3) Numeric-like -> int/float
    num = _to_primitive_number(value)
    if num is not None:
        return num  # type: ignore[return-value]

    # 4) datetime/date -> ISO string; Path -> str; bytes -> base64 string
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, (PurePath,)):
        return str(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return base64.b64encode(bytes(value)).decode("ascii")

    # 5) Sequences -> attempt homogeneous primitive array; else JSON
    from collections.abc import Sequence as _Seq

    if isinstance(value, _Seq) and not isinstance(value, (str, bytes, bytearray, memoryview)):
        coerced = _to_homogeneous_sequence(value)
        if coerced is not None:
            return coerced  # type: ignore[return-value]
        if prefer_json:
            try:
                return json.dumps(value, default=_json_default, ensure_ascii=False, sort_keys=True)
            except Exception:
                return str(value)

    # 6) Mappings/objects -> JSON string
    if isinstance(value, Mapping) or hasattr(value, "__dict__") or hasattr(value, "model_dump"):
        try:
            return json.dumps(value, default=_json_default, ensure_ascii=False, sort_keys=True)
        except Exception:
            return str(value)

    # 7) Fallback: string
    return str(value)


def get_session_logger(session_root: Path, name: str) -> "OtelLogger":
    """Get a session-specific OTEL logger.

    Creates a logger that writes to <session_root>/otel.log with standardized
    formatting for OTEL operations across different processes.

    Args:
        session_root: Path to the session output directory (not run directory)
        name: Name of the logger (e.g., "session_span_manager", "otel_callback")

    Returns:
        OtelLogger instance configured for the session
    """
    # Import here to avoid circular dependency at module level
    from ..observers.logging import get_logger

    log_path = session_root / "otel.log"
    # Use session_id to make logger unique per session
    # This ensures each session gets its own file handler for otel.log
    logger = get_logger(f"{name}.{session_root.name}", str(log_path), console=False)

    return OtelLogger(logger, scope=name)


class OtelLogger:
    """Structured logger for OpenTelemetry operations.

    Provides standardized logging methods for common OTEL operations like
    span creation, attribute setting, context file I/O, etc.
    """

    def __init__(self, logger, scope: str):
        self._logger = logger
        self._scope = scope

    def _format_message(self, message: str) -> str:
        return f"[{self._scope}] {message}"

    def debug(self, message: str) -> None:
        self._logger.debug(self._format_message(message))

    def info(self, message: str) -> None:
        self._logger.info(self._format_message(message))

    def warning(self, message: str) -> None:
        self._logger.warning(self._format_message(message))

    def error(self, message: str) -> None:
        self._logger.error(self._format_message(message))

    # Standardized OTEL operation logging methods

    def log_tracer_init(self, tracer_params: Dict[str, Any]) -> None:
        params_str = ", ".join(f"{k}={v}" for k, v in tracer_params.items())
        self.info(f"TRACER_INIT | {params_str}")

    def log_span_start(
        self,
        span_name: str,
        span_id: str,
        trace_id: str,
        parent_span_id: Optional[str] = None,
        is_root: bool = False,
        depth: Optional[int] = None,
        start_time: Optional[datetime] = None,
    ) -> None:
        root_marker = " [ROOT]" if is_root else ""
        parent_info = f" parent={parent_span_id}" if parent_span_id else ""
        depth_info = f" depth={depth}" if depth is not None else ""
        timestamp = start_time.strftime("%Y-%m-%d %H:%M:%S.%f")
        time_info = f" start_time={timestamp}" if start_time else ""
        self.info(
            f"SPAN_START{root_marker} | name='{span_name}' id={span_id} trace={trace_id}"
            f"{parent_info}{depth_info}{time_info}"
        )

    def log_span_rename(self, old_name: str, new_name: str, span_id: str) -> None:
        self.info(f"SPAN_RENAME | '{old_name}' -> '{new_name}' id={span_id}")

    def log_span_end(
        self,
        span_name: str,
        span_id: str,
        is_root: bool = False,
        status: Optional[str] = None,
        depth: Optional[int] = None,
        end_time: Optional[datetime] = None,
    ) -> None:
        root_marker = " [ROOT]" if is_root else ""
        status_info = f" status={status}" if status else ""
        depth_info = f" depth={depth}" if depth is not None else ""
        time_info = f" end_time={end_time.strftime('%Y-%m-%d %H:%M:%S.%f')}" if end_time else ""
        self.info(f"SPAN_END{root_marker} | name='{span_name}' id={span_id}{status_info}{depth_info}{time_info}")

    def log_attribute_set(self, key: str, value: Any, span_id: Optional[str] = None) -> None:
        span_info = f" span={span_id}" if span_id else ""
        # Truncate long values
        value_str = str(value)
        if len(value_str) > 100:
            value_str = value_str[:97] + "..."
        self.debug(f"ATTR_SET | {key}={value_str}{span_info}")

    def log_exception(self, exc: Exception, context: Optional[str] = None) -> None:
        context_str = f" context={context}" if context else ""
        self.error(f"EXCEPTION | {type(exc).__name__}: {exc}{context_str}")

    def log_context_update(self, trace_id: Optional[str], span_id: Optional[str], operation: str = "update") -> None:
        """Log OTEL context update operation."""
        self.debug(f"CONTEXT_{operation.upper()} | trace={trace_id} span={span_id}")

    def log_context_read(self, trace_id: str, span_id: str) -> None:
        """Log OTEL context read operation."""
        self.debug(f"CONTEXT_READ | trace={trace_id} span={span_id}")
