# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Utility functions for OpenTelemetry initialization and logging."""

import base64
import json
import logging
import os
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path, PurePath
from typing import Any

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
    OTLPSpanExporter as GrpcOTLPSpanExporter,
)
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter as HttpOTLPSpanExporter,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor, SpanExporter, SpanExportResult
from opentelemetry.sdk.trace.id_generator import IdGenerator
from opentelemetry.trace import Tracer
from opentelemetry.util.types import AttributeValue

logger = logging.getLogger(__name__)

_PRIMITIVE_TYPES = (str, bool, int, float)


class FileSpanExporter(SpanExporter):
    """Exports OTEL spans as JSON lines to a single file."""

    def __init__(self, file_path: str | Path) -> None:
        self._path = Path(file_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def export(self, spans: Sequence) -> SpanExportResult:
        try:
            with open(self._path, "a") as f:
                for span in spans:
                    f.write(span.to_json(indent=None) + "\n")
            return SpanExportResult.SUCCESS
        except Exception:
            logger.exception("FileSpanExporter failed to write %d spans to %s", len(spans), self._path)
            return SpanExportResult.FAILURE

    def shutdown(self) -> None:
        pass


class PerSessionFileExporter(SpanExporter):
    """Routes spans belonging to ``run_id`` into per-session files under ``run_root``.

    In batch mode the parent process registers one exporter per
    :class:`RunConfig` on the global ``TracerProvider``, and OTEL fans
    every emitted span out to every registered processor. Without the
    ``run_id`` filter, every exporter would write every span into its
    own ``run_root`` -- N-way write amplification plus cross-run trace
    leakage. ``exgentic.run.id`` is set as a heritable attribute in
    ``OtelTracingObserver.on_session_enter`` and propagated to every
    descendant span via ``SessionSpanManager.start_span``, so the
    routing key is always present on spans this exporter should own.
    """

    def __init__(self, run_root: str | Path, run_id: str) -> None:
        self._run_root = Path(run_root)
        self._run_id = run_id

    def export(self, spans: Sequence) -> SpanExportResult:
        try:
            for span in spans:
                attrs = getattr(span, "attributes", {}) or {}
                if attrs.get("exgentic.run.id") != self._run_id:
                    continue
                sid = attrs.get("exgentic.session.id")
                if not sid:
                    continue
                path = self._run_root / "sessions" / str(sid) / "otel_spans.jsonl"
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, "a") as f:
                    f.write(span.to_json(indent=None) + "\n")
            return SpanExportResult.SUCCESS
        except Exception:
            logger.exception("PerSessionFileExporter failed to write spans")
            return SpanExportResult.FAILURE

    def shutdown(self) -> None:
        pass


class FilteredSpanExporter(SpanExporter):
    """Wraps a SpanExporter to filter out spans from specific libraries."""

    def __init__(self, wrapped_exporter: SpanExporter) -> None:
        self._wrapped = wrapped_exporter
        # List of instrumentation scope names to exclude
        self._excluded_scopes = {
            "a2a.server.events.event_queue",
            "a2a.server.tasks",
            "a2a",
        }

    def export(self, spans: Sequence) -> SpanExportResult:
        """Filter spans and export only those not from excluded libraries."""
        filtered_spans = []
        for span in spans:
            # Check instrumentation scope
            scope_name = getattr(span, "instrumentation_scope", None)
            if scope_name:
                scope_name_str = getattr(scope_name, "name", "")
                # Skip spans from a2a-python-sdk
                if any(excluded in scope_name_str for excluded in self._excluded_scopes):
                    continue

            # Also check span name for a2a patterns
            span_name = getattr(span, "name", "")
            if span_name and any(pattern in span_name.lower() for pattern in ["enqueue", "a2a"]):
                # Check if this is an exgentic span (has our attributes)
                attrs = getattr(span, "attributes", {})
                if not any(k.startswith("exgentic.") for k in attrs.keys()):
                    continue

            filtered_spans.append(span)

        if not filtered_spans:
            return SpanExportResult.SUCCESS

        return self._wrapped.export(filtered_spans)

    def shutdown(self) -> None:
        self._wrapped.shutdown()


class UrandomIdGenerator(IdGenerator):
    def generate_span_id(self) -> int:
        return int.from_bytes(os.urandom(8), "big")

    def generate_trace_id(self) -> int:
        return int.from_bytes(os.urandom(16), "big")


def init_tracing_from_env(
    service_name: str | None = None,
    use_urandom_ids: bool = True,
    use_simple_processor: bool = True,
) -> Tracer:
    """Initialize OpenTelemetry tracing from environment variables. Idempotent."""
    current_provider = trace.get_tracer_provider()
    if type(current_provider).__name__ != "ProxyTracerProvider":
        return trace.get_tracer(__name__)

    resource = Resource.create(
        {
            "service.name": service_name or os.getenv("OTEL_SERVICE_NAME", "exgentic"),
            "service.version": os.getenv("OTEL_SERVICE_VERSION", "1.0.0"),
            "service.namespace": os.getenv("OTEL_SERVICE_NAMESPACE", "exgentic"),
            "deployment.environment.name": os.getenv("DEPLOYMENT_ENVIRONMENT", "dev"),
        }
    )

    provider = TracerProvider(
        resource=resource,
        id_generator=UrandomIdGenerator() if use_urandom_ids else None,
    )
    trace.set_tracer_provider(provider)

    exporters: list[SpanExporter] = []
    file_path = os.getenv("OTEL_EXPORTER_FILE")
    if file_path:
        exporters.append(FileSpanExporter(file_path))
    if os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
        protocol = os.getenv("OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf").strip().lower()
        exporters.append(GrpcOTLPSpanExporter() if protocol == "grpc" else HttpOTLPSpanExporter())

    processor_class = SimpleSpanProcessor if use_simple_processor else BatchSpanProcessor
    for exporter in exporters:
        # Wrap exporter with filtering to exclude a2a-python-sdk spans
        filtered_exporter = FilteredSpanExporter(exporter)
        provider.add_span_processor(processor_class(filtered_exporter))

    return trace.get_tracer(__name__)


def check_otel_collector_health(timeout: int = 5) -> tuple[bool, str | None]:
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
    """Flush all pending spans. Returns True on success."""
    try:
        provider = trace.get_tracer_provider()
        if type(provider).__name__ == "ProxyTracerProvider":
            return True
        return provider.force_flush(timeout_millis)
    except Exception:
        logger.exception("Failed to flush traces")
        return False


# --- Attribute value conversion ---


def _json_default(o: Any) -> Any:
    """Fallback serializer for json.dumps(default=...)."""
    if hasattr(o, "model_dump"):
        return o.model_dump()
    if hasattr(o, "dict"):
        return o.dict()
    from dataclasses import asdict, is_dataclass

    if is_dataclass(o):
        return asdict(o)
    if isinstance(o, (datetime, date)):
        return o.isoformat()
    if isinstance(o, PurePath):
        return str(o)
    if isinstance(o, (bytes, bytearray, memoryview)):
        return {"__bytes_b64__": base64.b64encode(bytes(o)).decode("ascii")}
    return str(o)


def _to_json(value: Any) -> str | None:
    try:
        return json.dumps(value, default=_json_default, ensure_ascii=False, sort_keys=True)
    except Exception:
        return None


def _to_scalar(value: Any) -> str | int | float | None:
    """Convert a single non-primitive value to a scalar, or None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, PurePath):
        return str(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return base64.b64encode(bytes(value)).decode("ascii")
    return None


def _to_homogeneous_sequence(seq: Sequence[Any]) -> list[str | bool | int | float] | None:
    """Coerce a sequence into a homogeneous primitive list, or None if impossible."""
    result = []
    for v in seq:
        if v is None:
            return None
        if isinstance(v, _PRIMITIVE_TYPES):
            result.append(v)
        elif (scalar := _to_scalar(v)) is not None:
            result.append(scalar)
        else:
            return None

    types = {type(x) for x in result}
    # bool is subclass of int — keep them separate
    if bool in types and int in types:
        return None
    if len(types) <= 1:
        return result
    if types <= {int, float}:
        return [float(x) for x in result]
    return None


def to_otel_attribute_value(value: Any, *, prefer_json: bool = True) -> AttributeValue | None:
    """Convert an arbitrary value to an OTEL AttributeValue, or None to skip."""
    if value is None:
        return None

    if isinstance(value, _PRIMITIVE_TYPES):
        return value

    scalar = _to_scalar(value)
    if scalar is not None:
        return scalar  # type: ignore[return-value]

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray, memoryview)):
        coerced = _to_homogeneous_sequence(value)
        if coerced is not None:
            return coerced  # type: ignore[return-value]
        if prefer_json:
            return _to_json(value) or str(value)

    if isinstance(value, Mapping) or hasattr(value, "__dict__") or hasattr(value, "model_dump"):
        return _to_json(value) or str(value)

    return str(value)


# --- Session logger ---


def get_session_logger(session_root: Path, name: str) -> "OtelLogger":
    """Get a session-specific OTEL logger that writes to <session_root>/otel.log."""
    from ..observers.logging import get_logger

    log_path = session_root / "otel.log"
    file_logger = get_logger(f"{name}.{session_root.name}", str(log_path), console=False)
    return OtelLogger(file_logger, scope=name)


class OtelLogger:
    """Structured logger for OpenTelemetry operations."""

    def __init__(self, logger, scope: str):
        self._logger = logger
        self._scope = scope

    def _fmt(self, message: str) -> str:
        return f"[{self._scope}] {message}"

    def debug(self, message: str) -> None:
        self._logger.debug(self._fmt(message))

    def info(self, message: str) -> None:
        self._logger.info(self._fmt(message))

    def warning(self, message: str) -> None:
        self._logger.warning(self._fmt(message))

    def error(self, message: str) -> None:
        self._logger.error(self._fmt(message))

    def _log_span_event(self, event: str, span_name: str, span_id: str, is_root: bool = False, **extras) -> None:
        parts = [f"name='{span_name}' id={span_id}"]
        for k, v in extras.items():
            if v is not None:
                fmt = f"{v:%Y-%m-%d %H:%M:%S.%f}" if isinstance(v, datetime) else str(v)
                parts.append(f"{k}={fmt}")
        root = " [ROOT]" if is_root else ""
        self.info(f"{event}{root} | {' '.join(parts)}")

    def log_span_start(
        self,
        span_name: str,
        span_id: str,
        trace_id: str,
        parent_span_id: str | None = None,
        is_root: bool = False,
        depth: int | None = None,
        start_time: datetime | None = None,
    ) -> None:
        self._log_span_event(
            "SPAN_START",
            span_name,
            span_id,
            is_root,
            trace=trace_id,
            parent=parent_span_id,
            depth=depth,
            start_time=start_time,
        )

    def log_span_end(
        self,
        span_name: str,
        span_id: str,
        is_root: bool = False,
        status: str | None = None,
        depth: int | None = None,
        end_time: datetime | None = None,
    ) -> None:
        self._log_span_event("SPAN_END", span_name, span_id, is_root, status=status, depth=depth, end_time=end_time)

    def log_span_rename(self, old_name: str, new_name: str, span_id: str) -> None:
        self.info(f"SPAN_RENAME | '{old_name}' -> '{new_name}' id={span_id}")

    def log_attribute_set(self, key: str, value: Any, span_id: str | None = None) -> None:
        span_info = f" span={span_id}" if span_id else ""
        self.debug(f"ATTR_SET | {key}={str(value)[:100]}{span_info}")

    def log_exception(self, exc: Exception, context: str | None = None) -> None:
        ctx = f" context={context}" if context else ""
        self.error(f"EXCEPTION | {type(exc).__name__}: {exc}{ctx}")

    def log_context_update(self, trace_id: str | None, span_id: str | None, operation: str = "update") -> None:
        self.debug(f"CONTEXT_{operation.upper()} | trace={trace_id} span={span_id}")

    def log_context_read(self, trace_id: str, span_id: str) -> None:
        self.debug(f"CONTEXT_READ | trace={trace_id} span={span_id}")
