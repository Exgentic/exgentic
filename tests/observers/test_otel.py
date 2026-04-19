# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Comprehensive tests for OTEL tracing: observer, SessionSpanManager, TraceLogger, context propagation."""

from __future__ import annotations

import contextvars
import json
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import ClassVar
from unittest.mock import MagicMock, patch

import pytest
from exgentic.core.context import (
    Context,
    OtelContext,
    RuntimeConfig,
    get_context,
    set_context,
    set_context_fallback,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult
from opentelemetry.trace import SpanKind

# ---------------------------------------------------------------------------
# In-memory span exporter (not available in this otel SDK version)
# ---------------------------------------------------------------------------


class InMemorySpanExporter(SpanExporter):
    """Collects finished spans in memory for test assertions."""

    def __init__(self):
        self._spans = []
        self._stopped = False

    def export(self, spans):
        if self._stopped:
            return SpanExportResult.FAILURE
        self._spans.extend(spans)
        return SpanExportResult.SUCCESS

    def get_finished_spans(self):
        return list(self._spans)

    def clear(self):
        self._spans.clear()

    def shutdown(self):
        self._stopped = True


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


class MockAction:
    name = "test_action"
    description = "A test action"
    is_message = False
    is_finish = False


class MockSession:
    session_id = "sess-001"
    task_id = "task-001"
    task = "Test task"
    actions: ClassVar[list] = [MockAction()]
    context: ClassVar[dict] = {"key": "value"}


class MockRunConfig:
    benchmark = "test_bench"
    agent = "test_agent"
    subset = None
    model = "gpt-4"
    agent_kwargs: ClassVar[dict] = {}


class MockObservation:
    def to_observation_list(self):
        return [{"type": "text", "content": "hello"}]


class MockAgentPaths:
    agent_dir = "/tmp/agent"


class MockAgent:
    agent_id = "agent-001"
    paths = MockAgentPaths()

    def get_cost(self):
        return {"total": 0.5}


class MockScore:
    success = True
    score = 0.95
    is_finished = True
    session_metrics: ClassVar[dict] = {}
    session_metadata: ClassVar[dict] = {}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def ctx(tmp_path: Path):
    """Set up a minimal Context so get_context() works."""
    c = Context(
        run_id="test-run",
        output_dir=str(tmp_path),
        cache_dir=str(tmp_path),
    )
    set_context(c)
    return c


@pytest.fixture()
def session_root(tmp_path: Path) -> Path:
    root = tmp_path / "test-run" / "sessions" / "sess-001"
    root.mkdir(parents=True, exist_ok=True)
    return root


@pytest.fixture()
def exporter():
    """In-memory span exporter for capturing spans."""
    return InMemorySpanExporter()


@pytest.fixture()
def tracer(exporter):
    """Tracer backed by in-memory exporter for span inspection."""
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    t = provider.get_tracer("test")
    return t


@pytest.fixture()
def span_manager(session_root, tracer):
    """SessionSpanManager backed by in-memory tracer."""
    from exgentic.observers.handlers.otel import SessionSpanManager

    return SessionSpanManager("sess-001", session_root, tracer=tracer)


@pytest.fixture()
def observer(ctx):
    """OtelTracingObserver with registry lookups stubbed."""
    with (
        patch("exgentic.observers.handlers.otel.get_benchmark_entries", return_value={}),
        patch("exgentic.observers.handlers.otel.get_agent_entries", return_value={}),
        patch("exgentic.observers.handlers.otel.get_settings", return_value=MagicMock(otel_record_content=False)),
    ):
        from exgentic.observers.handlers.otel import OtelTracingObserver

        obs = OtelTracingObserver()
        yield obs


def _create_observer_with_tracer(ctx, tmp_path, tracer):
    """Helper: create observer that uses a specific tracer."""
    from exgentic.observers.handlers.otel import OtelTracingObserver

    obs = OtelTracingObserver()
    return obs


def _trigger_session_lifecycle(observer, tracer, ctx, tmp_path, *, record_content=False):
    """Helper: run full session lifecycle and return the observer."""
    from exgentic.observers.handlers.otel import SessionSpanManager

    session = MockSession()
    session_root = tmp_path / "test-run" / "sessions" / session.session_id
    session_root.mkdir(parents=True, exist_ok=True)

    settings = MagicMock(otel_record_content=record_content)

    with (
        patch("exgentic.observers.handlers.otel.get_benchmark_entries", return_value={}),
        patch("exgentic.observers.handlers.otel.get_agent_entries", return_value={}),
        patch("exgentic.observers.handlers.otel.get_settings", return_value=settings),
        patch(
            "exgentic.observers.handlers.otel.to_otel_attribute_value",
            side_effect=lambda v: str(v) if v is not None else None,
        ),
    ):
        # Inject our tracer into the SessionSpanManager constructor
        original_init = SessionSpanManager.__init__

        def patched_init(self, session_id, session_root_path, tracer=tracer):
            original_init(self, session_id, session_root_path, tracer=tracer)

        with patch.object(SessionSpanManager, "__init__", patched_init):
            run_config = MockRunConfig()
            run_config.subset = "test_subset"
            observer.on_run_start(run_config)
            observer.on_session_creation(session)
            observer.on_session_start(session, MockAgent(), MockObservation())

    return observer, session, settings


# ===================================================================
# A. Full lifecycle span hierarchy (docs: span hierarchy section)
# ===================================================================


class TestSpanHierarchy:
    """Verify span tree matches docs: Session(ROOT) → sibling execute_tool and chat spans."""

    def test_session_root_span_created(self, exporter, ctx, tmp_path):
        """on_session_creation creates a root session span."""
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        t = provider.get_tracer("test")

        obs, session, settings = _trigger_session_lifecycle(
            _create_observer_with_tracer(ctx, tmp_path, t), t, ctx, tmp_path
        )

        # End session to flush spans
        with (
            patch("exgentic.observers.handlers.otel.get_settings", return_value=settings),
            patch("exgentic.observers.handlers.otel.flush_traces"),
        ):
            obs.on_session_success(session, MockScore(), MockAgent())

        spans = exporter.get_finished_spans()
        # Should have: session root + initial_observation execute_tool
        assert len(spans) >= 2

    def test_session_span_name_format(self, exporter, ctx, tmp_path):
        """Session span name = '{benchmark} {subset} session' per docs."""
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        t = provider.get_tracer("test")

        obs, session, settings = _trigger_session_lifecycle(
            _create_observer_with_tracer(ctx, tmp_path, t), t, ctx, tmp_path
        )
        with (
            patch("exgentic.observers.handlers.otel.get_settings", return_value=settings),
            patch("exgentic.observers.handlers.otel.flush_traces"),
        ):
            obs.on_session_success(session, MockScore(), MockAgent())

        spans = exporter.get_finished_spans()
        span_names = [s.name for s in spans]
        assert any("session" in name for name in span_names)

    def test_execute_tool_span_kind_is_client(self, exporter, ctx, tmp_path):
        """execute_tool spans have SpanKind.CLIENT per docs."""
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        t = provider.get_tracer("test")

        obs, session, settings = _trigger_session_lifecycle(
            _create_observer_with_tracer(ctx, tmp_path, t), t, ctx, tmp_path
        )
        with (
            patch("exgentic.observers.handlers.otel.get_settings", return_value=settings),
            patch("exgentic.observers.handlers.otel.flush_traces"),
        ):
            obs.on_session_success(session, MockScore(), MockAgent())

        spans = exporter.get_finished_spans()
        tool_spans = [s for s in spans if "execute_tool" in s.name]
        assert len(tool_spans) >= 1
        for ts in tool_spans:
            assert ts.kind == SpanKind.CLIENT

    def test_execute_tool_spans_are_children_of_session(self, exporter, ctx, tmp_path):
        """execute_tool spans must be children of the session root span."""
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        t = provider.get_tracer("test")

        obs, session, settings = _trigger_session_lifecycle(
            _create_observer_with_tracer(ctx, tmp_path, t), t, ctx, tmp_path
        )
        with (
            patch("exgentic.observers.handlers.otel.get_settings", return_value=settings),
            patch("exgentic.observers.handlers.otel.flush_traces"),
        ):
            obs.on_session_success(session, MockScore(), MockAgent())

        spans = exporter.get_finished_spans()
        session_spans = [s for s in spans if "session" in s.name]
        tool_spans = [s for s in spans if "execute_tool" in s.name]

        assert len(session_spans) >= 1
        session_span = session_spans[0]
        session_span_id = session_span.context.span_id

        for ts in tool_spans:
            assert ts.parent is not None
            assert ts.parent.span_id == session_span_id

    def test_execute_tool_spans_are_siblings(self, exporter, ctx, tmp_path):
        """Multiple execute_tool spans are siblings (same parent), not nested."""
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        t = provider.get_tracer("test")

        obs, session, settings = _trigger_session_lifecycle(
            _create_observer_with_tracer(ctx, tmp_path, t), t, ctx, tmp_path
        )

        # Add another step: react_success + step_success
        mock_action = MagicMock()
        mock_action.to_action_list.return_value = [MagicMock(name="browse", id="act-1")]
        mock_action.to_action_list.return_value[0].name = "browse"
        mock_action.to_action_list.return_value[0].id = "act-1"

        with (
            patch("exgentic.observers.handlers.otel.get_settings", return_value=settings),
            patch("exgentic.observers.handlers.otel.flush_traces"),
        ):
            obs.on_react_success(session, mock_action)
            obs.on_step_success(session, MockObservation())
            obs.on_session_success(session, MockScore(), MockAgent())

        spans = exporter.get_finished_spans()
        tool_spans = [s for s in spans if "execute_tool" in s.name]
        assert len(tool_spans) >= 2

        # All tool spans should share the same parent
        parent_ids = {ts.parent.span_id for ts in tool_spans}
        assert len(parent_ids) == 1, "All execute_tool spans should be siblings with the same parent"


# ===================================================================
# B. Heritable attribute inheritance
# ===================================================================


class TestHeritableAttributes:
    """Heritable attributes propagate from session to child spans."""

    def test_heritable_attributes_set_on_root(self, span_manager):
        """set_heritable_attribute sets value on current span."""
        span_manager.start_span("root")
        span_manager.set_heritable_attribute("gen_ai.conversation.id", "sess-001")

        span = span_manager.current_span
        # The attribute should be set (we check via the span's attributes dict)
        assert span.attributes.get("gen_ai.conversation.id") == "sess-001"
        span_manager.end_current_span()

    def test_heritable_attributes_propagate_to_children(self, span_manager):
        """Heritable attributes auto-propagate to new child spans."""
        span_manager.start_span("root")
        span_manager.set_heritable_attribute("exgentic.run.id", "run-123")

        child = span_manager.start_span("child")
        # Child should inherit the attribute
        assert child.attributes.get("exgentic.run.id") == "run-123"

        span_manager.end_current_span()
        span_manager.end_current_span()

    def test_multiple_heritable_attributes_propagate(self, span_manager):
        """All heritable attributes propagate to each new child."""
        span_manager.start_span("root")
        span_manager.set_heritable_attributes(
            **{
                "exgentic.run.id": "run-123",
                "gen_ai.conversation.id": "sess-001",
                "exgentic.agent.slug": "my-agent",
            }
        )

        child = span_manager.start_span("child")
        assert child.attributes.get("exgentic.run.id") == "run-123"
        assert child.attributes.get("gen_ai.conversation.id") == "sess-001"
        assert child.attributes.get("exgentic.agent.slug") == "my-agent"

        span_manager.end_current_span()
        span_manager.end_current_span()

    def test_heritable_attributes_propagate_to_grandchild(self, span_manager):
        """Heritable attrs propagate to grandchildren too."""
        span_manager.start_span("root")
        span_manager.set_heritable_attribute("exgentic.run.id", "run-123")

        span_manager.start_span("child")
        grandchild = span_manager.start_span("grandchild")
        assert grandchild.attributes.get("exgentic.run.id") == "run-123"

        span_manager.end_current_span()
        span_manager.end_current_span()
        span_manager.end_current_span()


# ===================================================================
# C. Content filtering (otel_record_content)
# ===================================================================


class TestContentFiltering:
    """Opt-in content attributes only appear when otel_record_content=True."""

    def test_task_not_recorded_when_disabled(self, ctx, tmp_path):
        """session.task must NOT be an attribute when record_content=False."""
        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        t = provider.get_tracer("test")

        obs = _create_observer_with_tracer(ctx, tmp_path, t)
        obs, session, _ = _trigger_session_lifecycle(obs, t, ctx, tmp_path, record_content=False)

        settings = MagicMock(otel_record_content=False)
        with (
            patch("exgentic.observers.handlers.otel.get_settings", return_value=settings),
            patch("exgentic.observers.handlers.otel.flush_traces"),
        ):
            obs.on_session_success(session, MockScore(), MockAgent())

        spans = exporter.get_finished_spans()
        session_spans = [s for s in spans if "session" in s.name]
        for s in session_spans:
            assert "exgentic.session.task" not in (
                s.attributes or {}
            ), "Task content should not be recorded when otel_record_content=False"

    def test_task_recorded_when_enabled(self, ctx, tmp_path):
        """session.task must be recorded when record_content=True."""
        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        t = provider.get_tracer("test")

        obs = _create_observer_with_tracer(ctx, tmp_path, t)
        obs, session, _ = _trigger_session_lifecycle(obs, t, ctx, tmp_path, record_content=True)

        settings = MagicMock(otel_record_content=True)
        with (
            patch("exgentic.observers.handlers.otel.get_settings", return_value=settings),
            patch("exgentic.observers.handlers.otel.flush_traces"),
        ):
            obs.on_session_success(session, MockScore(), MockAgent())

        spans = exporter.get_finished_spans()
        session_spans = [s for s in spans if "session" in s.name]
        assert any(
            "exgentic.session.task" in (s.attributes or {}) for s in session_spans
        ), "Task content should be recorded when otel_record_content=True"

    def test_tool_result_not_recorded_when_disabled(self, ctx, tmp_path):
        """gen_ai.tool.result must NOT appear when record_content=False."""
        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        t = provider.get_tracer("test")

        obs = _create_observer_with_tracer(ctx, tmp_path, t)
        obs, session, _ = _trigger_session_lifecycle(obs, t, ctx, tmp_path, record_content=False)

        settings = MagicMock(otel_record_content=False)
        with (
            patch("exgentic.observers.handlers.otel.get_settings", return_value=settings),
            patch("exgentic.observers.handlers.otel.flush_traces"),
        ):
            obs.on_session_success(session, MockScore(), MockAgent())

        spans = exporter.get_finished_spans()
        tool_spans = [s for s in spans if "execute_tool" in s.name]
        for s in tool_spans:
            assert "gen_ai.tool.result" not in (s.attributes or {})


# ===================================================================
# D. Session error handling
# ===================================================================


class TestSessionErrorHandling:
    """Error paths: exception recording, proper cleanup."""

    def test_on_session_error_records_exception(self, exporter, ctx, tmp_path):
        """on_session_error must record the exception on the session span."""
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        t = provider.get_tracer("test")

        obs = _create_observer_with_tracer(ctx, tmp_path, t)
        obs, session, settings = _trigger_session_lifecycle(obs, t, ctx, tmp_path)

        error = RuntimeError("test failure")
        session_mock = MagicMock()
        session_mock.session_id = session.session_id
        session_mock.task_id = session.task_id
        session_mock.get_cost.return_value = {}

        with (
            patch("exgentic.observers.handlers.otel.get_settings", return_value=settings),
            patch("exgentic.observers.handlers.otel.flush_traces"),
        ):
            obs.on_session_error(session_mock, error)

        spans = exporter.get_finished_spans()
        session_spans = [s for s in spans if "session" in s.name]
        assert len(session_spans) >= 1
        # Session span should have error events
        session_span = session_spans[0]
        assert len(session_span.events) > 0, "Session span should have exception events"

    def test_on_session_error_cleans_up_managers(self, exporter, ctx, tmp_path):
        """on_session_error must remove the span manager from _span_managers."""
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        t = provider.get_tracer("test")

        obs = _create_observer_with_tracer(ctx, tmp_path, t)
        obs, session, settings = _trigger_session_lifecycle(obs, t, ctx, tmp_path)

        session_mock = MagicMock()
        session_mock.session_id = session.session_id
        session_mock.task_id = session.task_id
        session_mock.get_cost.return_value = {}

        with (
            patch("exgentic.observers.handlers.otel.get_settings", return_value=settings),
            patch("exgentic.observers.handlers.otel.flush_traces"),
        ):
            obs.on_session_error(session_mock, RuntimeError("fail"))

        assert session.session_id not in obs._span_managers

    def test_on_step_error_records_exception(self, span_manager):
        """on_step_error records exception on current span."""
        span_manager.start_span("session")
        span_manager.start_span("execute_tool test")

        exc = ValueError("bad input")
        span_manager.record_exception(exc)

        span = span_manager.current_span
        assert len(span.events) > 0
        span_manager.end_current_span()
        span_manager.end_current_span()

    def test_trailing_execute_tool_span_closed_on_error(self, exporter, ctx, tmp_path):
        """If 2 spans on stack during on_session_error, the trailing execute_tool span is closed."""
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        t = provider.get_tracer("test")

        obs = _create_observer_with_tracer(ctx, tmp_path, t)
        obs, session, settings = _trigger_session_lifecycle(obs, t, ctx, tmp_path)

        # Simulate react_success without step_success (trailing span)
        mock_action = MagicMock()
        mock_action.to_action_list.return_value = [MagicMock(name="act", id="a1")]
        mock_action.to_action_list.return_value[0].name = "act"
        mock_action.to_action_list.return_value[0].id = "a1"

        with (
            patch("exgentic.observers.handlers.otel.get_settings", return_value=settings),
            patch("exgentic.observers.handlers.otel.flush_traces"),
        ):
            obs.on_react_success(session, mock_action)
            # Now stack has 2: session + execute_tool
            session_mock = MagicMock()
            session_mock.session_id = session.session_id
            session_mock.task_id = session.task_id
            session_mock.get_cost.return_value = {}
            obs.on_session_error(session_mock, RuntimeError("crash"))

        spans = exporter.get_finished_spans()
        # All spans should be ended (finished)
        assert len(spans) >= 3  # initial_observation + trailing tool + session


# ===================================================================
# E. OTEL context propagation via env vars
# ===================================================================


class TestOtelContextEnvPropagation:
    """RuntimeConfig round-trip for OTEL trace/span ids."""

    def test_otel_context_round_trip_via_runtime_config(self, tmp_path):
        """RuntimeConfig → to_context preserves trace_id and span_id."""
        config = RuntimeConfig(
            run_id="run-1",
            output_dir=str(tmp_path),
            cache_dir=str(tmp_path),
            otel_trace_id="a" * 32,
            otel_span_id="b" * 16,
        )

        assert config.otel_trace_id == "a" * 32
        assert config.otel_span_id == "b" * 16

        restored = config.to_context()
        assert restored.otel_context is not None
        assert restored.otel_context.trace_id == "a" * 32
        assert restored.otel_context.span_id == "b" * 16

    def test_no_otel_context_omits_otel_fields(self, tmp_path):
        """RuntimeConfig without otel fields must not produce otel_context."""
        config = RuntimeConfig(
            run_id="run-1",
            output_dir=str(tmp_path),
            cache_dir=str(tmp_path),
        )
        assert config.otel_trace_id is None
        assert config.otel_span_id is None

    def test_runtime_config_without_otel_gives_none(self, tmp_path):
        """RuntimeConfig without OTEL fields yields otel_context=None on to_context."""
        config = RuntimeConfig(
            run_id="run-1",
            output_dir=str(tmp_path),
            cache_dir=str(tmp_path),
        )
        ctx = config.to_context()
        assert ctx.otel_context is None

    def test_partial_otel_fields_gives_none(self, tmp_path):
        """If only trace_id (no span_id) is set, otel_context should be None."""
        config = RuntimeConfig(
            run_id="run-1",
            output_dir=str(tmp_path),
            cache_dir=str(tmp_path),
            otel_trace_id="a" * 32,
            # No span_id
        )
        ctx = config.to_context()
        assert ctx.otel_context is None


# ===================================================================
# F. Runner context propagation patterns
# ===================================================================


class TestRunnerContextPropagation:
    """Context propagation via ContextVar (thread), env vars (process/docker), fallback (service)."""

    def test_thread_runner_inherits_context_via_copy_context(self, ctx):
        """contextvars.copy_context() propagates Context to thread."""
        set_context(ctx)
        copied_ctx = contextvars.copy_context()

        # Simulate running in thread context
        def get_in_thread():
            return get_context()

        result = copied_ctx.run(get_in_thread)
        assert result.run_id == ctx.run_id

    def test_thread_runner_inherits_otel_context(self, tmp_path):
        """OTEL context in ContextVar survives copy_context()."""
        otel = OtelContext(trace_id="t" * 32, span_id="s" * 16)
        ctx = Context(
            run_id="run-1",
            output_dir=str(tmp_path),
            cache_dir=str(tmp_path),
            otel_context=otel,
        )
        set_context(ctx)
        copied = contextvars.copy_context()

        result = copied.run(get_context)
        assert result.otel_context is not None
        assert result.otel_context.trace_id == "t" * 32

    def test_process_runner_propagates_via_runtime_config(self, tmp_path):
        """Process runner uses RuntimeConfig for context propagation."""
        otel = OtelContext(trace_id="a" * 32, span_id="b" * 16)
        ctx = Context(
            run_id="run-1",
            output_dir=str(tmp_path),
            cache_dir=str(tmp_path),
            session_id="sess-1",
            otel_context=otel,
        )
        config = RuntimeConfig(
            run_id=ctx.run_id,
            output_dir=ctx.output_dir,
            cache_dir=ctx.cache_dir,
            session_id=ctx.session_id,
            otel_trace_id=ctx.otel_context.trace_id,
            otel_span_id=ctx.otel_context.span_id,
        )

        # Simulate subprocess reading config
        restored = config.to_context()
        assert restored.otel_context.trace_id == "a" * 32
        assert restored.otel_context.span_id == "b" * 16
        assert restored.session_id == "sess-1"

    def test_service_runner_fallback_context(self, tmp_path):
        """set_context_fallback sets _SUBPROCESS_CONTEXT for threads without ContextVar."""
        otel = OtelContext(trace_id="f" * 32, span_id="e" * 16)
        ctx = Context(
            run_id="run-1",
            output_dir=str(tmp_path),
            cache_dir=str(tmp_path),
            otel_context=otel,
        )

        # Clear ContextVar, set fallback
        set_context(None)  # type: ignore[arg-type]
        set_context_fallback(ctx)

        # get_context() should return fallback
        result = get_context()
        assert result.otel_context is not None
        assert result.otel_context.trace_id == "f" * 32

        # Cleanup
        set_context_fallback(None)


# ===================================================================
# G. TraceLogger parent context reconstruction
# ===================================================================


class TestTraceLoggerParentContext:
    """TraceLogger._get_parent_context reconstructs NonRecordingSpan from otel_context."""

    def test_get_parent_context_creates_nonrecording_span(self, tmp_path):
        """_get_parent_context must return context with NonRecordingSpan containing correct ids."""
        from exgentic.integrations.litellm.trace_logger import TraceLogger
        from opentelemetry import trace as trace_api

        otel = OtelContext(trace_id="a" * 32, span_id="b" * 16)
        ctx = Context(
            run_id="run-1",
            output_dir=str(tmp_path),
            cache_dir=str(tmp_path),
            session_id="sess-1",
            otel_context=otel,
        )

        logger = TraceLogger()
        logger._otel_logger = MagicMock()

        with patch.object(logger, "get_context", return_value=ctx):
            parent_ctx = logger._get_parent_context({})

        # Extract span from context
        span = trace_api.get_current_span(parent_ctx)
        span_ctx = span.get_span_context()

        assert format(span_ctx.trace_id, "032x") == "a" * 32
        assert format(span_ctx.span_id, "016x") == "b" * 16
        assert span_ctx.is_remote is True

    def test_write_otel_early_return_when_tracer_none(self):
        """_write_otel exits early when tracer is None."""
        from exgentic.integrations.litellm.trace_logger import TraceLogger

        logger = TraceLogger()
        assert logger._tracer is None

        with patch(
            "exgentic.integrations.litellm.trace_logger.get_settings",
            return_value=MagicMock(otel_enabled=True, otel_record_content=False),
        ):
            logger._write_otel(
                kwargs={"model": "test"},
                response_obj={},
                status="success",
            )

        assert logger._tracer is None

    def test_write_otel_skips_when_disabled(self):
        """_write_otel returns immediately when _otel_enabled() is False."""
        from exgentic.integrations.litellm.trace_logger import TraceLogger

        logger = TraceLogger()
        logger._tracer = MagicMock()  # Would fail if called

        with patch(
            "exgentic.integrations.litellm.trace_logger.get_settings",
            return_value=MagicMock(otel_enabled=False, otel_record_content=False),
        ):
            logger._write_otel(
                kwargs={"model": "test"},
                response_obj={},
                status="success",
            )
        # tracer methods should not have been called
        logger._tracer.start_span.assert_not_called()


# ===================================================================
# H. on_session_success final attributes
# ===================================================================


class TestSessionSuccessAttributes:
    """on_session_success sets score, step count, cost, then cleans up."""

    def test_score_attributes_set(self, exporter, ctx, tmp_path):
        """Score attributes must be on the session span."""
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        t = provider.get_tracer("test")

        obs = _create_observer_with_tracer(ctx, tmp_path, t)
        obs, session, settings = _trigger_session_lifecycle(obs, t, ctx, tmp_path)

        with (
            patch("exgentic.observers.handlers.otel.get_settings", return_value=settings),
            patch("exgentic.observers.handlers.otel.flush_traces"),
        ):
            obs.on_session_success(session, MockScore(), MockAgent())

        spans = exporter.get_finished_spans()
        session_spans = [s for s in spans if "session" in s.name]
        assert len(session_spans) >= 1
        attrs = dict(session_spans[0].attributes)
        assert attrs.get("exgentic.score.success") is True
        assert attrs.get("exgentic.score") == 0.95
        assert attrs.get("exgentic.score.is_finished") is True

    def test_score_extras_flattened_onto_session_span(self, exporter, ctx, tmp_path):
        """session_metrics and session_metadata are flattened onto the session span."""
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        t = provider.get_tracer("test")

        obs = _create_observer_with_tracer(ctx, tmp_path, t)
        obs, session, settings = _trigger_session_lifecycle(obs, t, ctx, tmp_path)

        class RichScore(MockScore):
            session_metrics: ClassVar[dict] = {"reward": 0.75, "db_check_db_match": True}
            session_metadata: ClassVar[dict] = {
                "reward_info": {
                    "reward": 0.75,
                    "nl_assertions": [
                        {"assertion": "agent greeted user", "met": True},
                        {"assertion": "agent confirmed order", "met": False},
                    ],
                },
            }

        with (
            patch("exgentic.observers.handlers.otel.get_settings", return_value=settings),
            patch("exgentic.observers.handlers.otel.flush_traces"),
        ):
            obs.on_session_success(session, RichScore(), MockAgent())

        spans = exporter.get_finished_spans()
        session_spans = [s for s in spans if "session" in s.name]
        attrs = dict(session_spans[0].attributes)

        # Primitive metric values pass through directly.
        assert attrs.get("exgentic.score.metrics.reward") == 0.75
        assert attrs.get("exgentic.score.metrics.db_check_db_match") is True

        # Nested metadata is JSON-serialized so nothing is dropped.
        reward_info_raw = attrs.get("exgentic.score.metadata.reward_info")
        assert reward_info_raw is not None, "rich score metadata must be emitted"
        parsed = json.loads(reward_info_raw)
        assert parsed["reward"] == 0.75
        assert parsed["nl_assertions"][0]["assertion"] == "agent greeted user"
        assert parsed["nl_assertions"][1]["met"] is False

    def test_score_extras_absent_when_empty(self, exporter, ctx, tmp_path):
        """Scores without metrics/metadata emit no extra attributes and don't crash."""
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        t = provider.get_tracer("test")

        obs = _create_observer_with_tracer(ctx, tmp_path, t)
        obs, session, settings = _trigger_session_lifecycle(obs, t, ctx, tmp_path)

        with (
            patch("exgentic.observers.handlers.otel.get_settings", return_value=settings),
            patch("exgentic.observers.handlers.otel.flush_traces"),
        ):
            obs.on_session_success(session, MockScore(), MockAgent())

        spans = exporter.get_finished_spans()
        session_spans = [s for s in spans if "session" in s.name]
        attrs = dict(session_spans[0].attributes)
        assert not any(k.startswith("exgentic.score.metrics.") for k in attrs)
        assert not any(k.startswith("exgentic.score.metadata.") for k in attrs)

    def test_step_counter_set(self, exporter, ctx, tmp_path):
        """exgentic.session.steps reflects the number of steps."""
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        t = provider.get_tracer("test")

        obs = _create_observer_with_tracer(ctx, tmp_path, t)
        obs, session, settings = _trigger_session_lifecycle(obs, t, ctx, tmp_path)

        with (
            patch("exgentic.observers.handlers.otel.get_settings", return_value=settings),
            patch("exgentic.observers.handlers.otel.flush_traces"),
        ):
            obs.on_session_success(session, MockScore(), MockAgent())

        spans = exporter.get_finished_spans()
        session_spans = [s for s in spans if "session" in s.name]
        attrs = dict(session_spans[0].attributes)
        assert "exgentic.session.steps" in attrs
        assert attrs["exgentic.session.steps"] >= 1

    def test_cleanup_after_success(self, exporter, ctx, tmp_path):
        """on_session_success cleans up _span_managers, counters, agents, actions."""
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        t = provider.get_tracer("test")

        obs = _create_observer_with_tracer(ctx, tmp_path, t)
        obs, session, settings = _trigger_session_lifecycle(obs, t, ctx, tmp_path)

        with (
            patch("exgentic.observers.handlers.otel.get_settings", return_value=settings),
            patch("exgentic.observers.handlers.otel.flush_traces"),
        ):
            obs.on_session_success(session, MockScore(), MockAgent())

        assert session.session_id not in obs._span_managers
        assert session.session_id not in obs._session_step_counters


# ===================================================================
# I. Concurrent sessions isolation
# ===================================================================


class TestConcurrentSessions:
    """Each session gets its own SessionSpanManager — no cross-talk."""

    def test_separate_span_managers_per_session(self, observer, ctx, tmp_path):
        """Two sessions get independent span managers."""
        session1 = MockSession()
        session1.session_id = "sess-A"

        session2 = MagicMock()
        session2.session_id = "sess-B"
        session2.task_id = "task-B"
        session2.task = "Task B"
        session2.actions = [MockAction()]
        session2.context = {}

        for sid in [session1.session_id, session2.session_id]:
            (tmp_path / "test-run" / "sessions" / sid).mkdir(parents=True, exist_ok=True)

        with (
            patch("exgentic.observers.handlers.otel.get_settings", return_value=MagicMock(otel_record_content=False)),
            patch("exgentic.observers.handlers.otel.to_otel_attribute_value", return_value=None),
        ):
            observer.on_session_creation(session1)
            observer.on_session_creation(session2)

        assert "sess-A" in observer._span_managers
        assert "sess-B" in observer._span_managers
        assert observer._span_managers["sess-A"] is not observer._span_managers["sess-B"]

    def test_different_trace_ids_per_session(self, ctx, tmp_path):
        """Each session generates a unique trace_id."""
        from exgentic.observers.handlers.otel import SessionSpanManager

        provider = TracerProvider()
        t = provider.get_tracer("test")

        root_a = tmp_path / "a"
        root_a.mkdir()
        root_b = tmp_path / "b"
        root_b.mkdir()

        mgr_a = SessionSpanManager("sess-A", root_a, tracer=t)
        mgr_b = SessionSpanManager("sess-B", root_b, tracer=t)

        mgr_a.start_span("session-A")
        mgr_b.start_span("session-B")

        ctx_a = mgr_a.get_otel_context()
        ctx_b = mgr_b.get_otel_context()

        assert ctx_a.trace_id != ctx_b.trace_id

        mgr_a.end_current_span()
        mgr_b.end_current_span()


# ===================================================================
# J. to_otel_attribute_value edge cases
# ===================================================================


class TestToOtelAttributeValue:
    """to_otel_attribute_value converts arbitrary Python values to OTEL-safe types."""

    def test_none_returns_none(self):
        from exgentic.utils.otel import to_otel_attribute_value

        assert to_otel_attribute_value(None) is None

    def test_string_passthrough(self):
        from exgentic.utils.otel import to_otel_attribute_value

        assert to_otel_attribute_value("hello") == "hello"

    def test_bool_passthrough(self):
        from exgentic.utils.otel import to_otel_attribute_value

        assert to_otel_attribute_value(True) is True
        assert to_otel_attribute_value(False) is False

    def test_int_passthrough(self):
        from exgentic.utils.otel import to_otel_attribute_value

        assert to_otel_attribute_value(42) == 42

    def test_float_passthrough(self):
        from exgentic.utils.otel import to_otel_attribute_value

        assert to_otel_attribute_value(3.14) == 3.14

    def test_decimal_to_float(self):
        from exgentic.utils.otel import to_otel_attribute_value

        result = to_otel_attribute_value(Decimal("1.5"))
        assert isinstance(result, float)
        assert result == 1.5

    def test_datetime_to_iso(self):
        from exgentic.utils.otel import to_otel_attribute_value

        dt = datetime(2026, 1, 1, 12, 0, 0)
        result = to_otel_attribute_value(dt)
        assert isinstance(result, str)
        assert "2026" in result

    def test_date_to_iso(self):
        from exgentic.utils.otel import to_otel_attribute_value

        d = date(2026, 3, 15)
        result = to_otel_attribute_value(d)
        assert isinstance(result, str)
        assert "2026-03-15" in result

    def test_path_to_string(self):
        from exgentic.utils.otel import to_otel_attribute_value

        result = to_otel_attribute_value(Path("/tmp/test"))
        assert isinstance(result, str)
        assert "tmp" in result

    def test_bytes_to_base64(self):
        from exgentic.utils.otel import to_otel_attribute_value

        result = to_otel_attribute_value(b"hello")
        assert isinstance(result, str)
        # base64 of "hello" = "aGVsbG8="
        assert result == "aGVsbG8="

    def test_homogeneous_int_list(self):
        from exgentic.utils.otel import to_otel_attribute_value

        result = to_otel_attribute_value([1, 2, 3])
        assert result == [1, 2, 3]

    def test_homogeneous_string_list(self):
        from exgentic.utils.otel import to_otel_attribute_value

        result = to_otel_attribute_value(["a", "b"])
        assert result == ["a", "b"]

    def test_mixed_int_float_list_upcasts(self):
        from exgentic.utils.otel import to_otel_attribute_value

        result = to_otel_attribute_value([1, 2.5])
        assert result == [1.0, 2.5]

    def test_dict_to_json_string(self):
        import json

        from exgentic.utils.otel import to_otel_attribute_value

        result = to_otel_attribute_value({"key": "val"})
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert parsed["key"] == "val"

    def test_nested_dict_to_json(self):
        import json

        from exgentic.utils.otel import to_otel_attribute_value

        result = to_otel_attribute_value({"a": {"b": 1}})
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert parsed["a"]["b"] == 1

    def test_list_with_none_falls_to_json(self):
        from exgentic.utils.otel import to_otel_attribute_value

        result = to_otel_attribute_value([1, None, 3])
        assert isinstance(result, str)  # JSON fallback


# ===================================================================
# K. SessionSpanManager core operations (existing tests, expanded)
# ===================================================================


class TestSpanManagerOperations:
    """Stack operations, get_otel_context, update_tracing_context."""

    def test_stack_push_pop(self, span_manager):
        """start/end span maintains correct stack."""
        root = span_manager.start_span("root")
        assert span_manager.current_span is root
        assert len(span_manager._span_stack) == 1

        child = span_manager.start_span("child")
        assert span_manager.current_span is child
        assert len(span_manager._span_stack) == 2

        span_manager.end_current_span()
        assert span_manager.current_span is root

        span_manager.end_current_span()
        assert span_manager.current_span is None

    def test_end_empty_stack_no_crash(self, span_manager):
        span_manager.end_current_span()

    def test_get_otel_context_with_span(self, span_manager):
        span_manager.start_span("test")
        otel_ctx = span_manager.get_otel_context()
        assert otel_ctx is not None
        assert isinstance(otel_ctx, OtelContext)
        assert len(otel_ctx.trace_id) == 32
        assert len(otel_ctx.span_id) == 16
        span_manager.end_current_span()

    def test_get_otel_context_without_span(self, span_manager):
        assert span_manager.get_otel_context() is None

    def test_update_tracing_context_sets_context(self, span_manager, ctx):
        span_manager.start_span("ctx-test")
        expected = span_manager.get_otel_context()
        span_manager.update_tracing_context()

        current = get_context()
        assert current.otel_context is not None
        assert current.otel_context.trace_id == expected.trace_id
        assert current.otel_context.span_id == expected.span_id
        span_manager.end_current_span()

    def test_set_attribute(self, span_manager):
        span_manager.start_span("test")
        span_manager.set_attribute("key", "value")
        assert span_manager.current_span.attributes.get("key") == "value"
        span_manager.end_current_span()

    def test_set_attribute_no_span_raises(self, span_manager):
        with pytest.raises(AttributeError):
            span_manager.set_attribute("key", "value")

    def test_update_span_name(self, span_manager):
        span_manager.start_span("old-name")
        span_manager.update_current_span_name("new-name")
        assert span_manager.current_span._name == "new-name"
        span_manager.end_current_span()


# ===================================================================
# L. Litellm callback registration
# ===================================================================


class TestLitellmCallbackRegistration:
    """_configure_callbacks adds a TraceLogger to litellm.callbacks."""

    def test_adds_single_trace_logger(self):
        import litellm
        from exgentic.integrations.litellm.config import _configure_callbacks
        from exgentic.integrations.litellm.trace_logger import SyncTraceLogger

        litellm.callbacks = []
        _configure_callbacks()

        # One CustomLogger handles both sync and async dispatch on modern
        # litellm; registering two caused duplicate OTEL spans.
        assert sum(1 for cb in litellm.callbacks if isinstance(cb, SyncTraceLogger)) == 1

    def test_idempotent(self):
        import litellm
        from exgentic.integrations.litellm.config import _configure_callbacks
        from exgentic.integrations.litellm.trace_logger import SyncTraceLogger

        litellm.callbacks = []
        _configure_callbacks()
        _configure_callbacks()

        assert sum(1 for cb in litellm.callbacks if isinstance(cb, SyncTraceLogger)) == 1


# ===================================================================
# M. Observer on_run_start attribute handling
# ===================================================================


class TestOnRunStart:
    """on_run_start sets _run_attributes correctly."""

    def test_subset_none_becomes_empty(self, observer, ctx):
        run_config = MockRunConfig()
        run_config.subset = None

        with (
            patch("exgentic.observers.handlers.otel.get_benchmark_entries", return_value={}),
            patch("exgentic.observers.handlers.otel.get_agent_entries", return_value={}),
        ):
            observer.on_run_start(run_config)

        assert observer._run_attributes["exgentic.benchmark.subset"] == ""

    def test_subset_preserved(self, observer, ctx):
        run_config = MockRunConfig()
        run_config.subset = "my_subset"

        with (
            patch("exgentic.observers.handlers.otel.get_benchmark_entries", return_value={}),
            patch("exgentic.observers.handlers.otel.get_agent_entries", return_value={}),
        ):
            observer.on_run_start(run_config)

        assert observer._run_attributes["exgentic.benchmark.subset"] == "my_subset"

    def test_model_set_as_heritable(self, observer, ctx):
        run_config = MockRunConfig()
        run_config.model = "gpt-4o"

        with (
            patch("exgentic.observers.handlers.otel.get_benchmark_entries", return_value={}),
            patch("exgentic.observers.handlers.otel.get_agent_entries", return_value={}),
        ):
            observer.on_run_start(run_config)

        assert observer._run_attributes.get("gen_ai.request.model") == "gpt-4o"

    def test_on_session_start_without_creation_raises(self, observer, ctx):
        with pytest.raises(KeyError):
            observer.on_session_start(MockSession(), MockAgent(), MockObservation())


# ===================================================================
# N. Cross-boundary OTEL context propagation (the real integration)
# ===================================================================


class TestCrossBoundaryContextPropagation:
    """End-to-end: parent creates session span → context crosses boundary → child creates LLM span with correct parent.

    Tests the full chain:
    1. SessionSpanManager creates session span, exports OtelContext
    2. Context propagated via env vars / metadata / ContextVar
    3. TraceLogger in child reconstructs parent and creates LLM span
    4. LLM span shares trace_id with session span (same trace)
    """

    def test_env_var_path_preserves_trace_linkage(self, tmp_path):
        """Simulate process/venv/docker: parent span → env vars → child TraceLogger → LLM span shares trace_id."""
        from exgentic.integrations.litellm.trace_logger import TraceLogger
        from opentelemetry import trace as trace_api

        # -- Parent side: create session span, get otel context --
        parent_provider = TracerProvider()
        parent_exporter = InMemorySpanExporter()
        parent_provider.add_span_processor(SimpleSpanProcessor(parent_exporter))
        parent_tracer = parent_provider.get_tracer("parent")

        session_span = parent_tracer.start_span("test_bench test_subset session")
        session_ctx = session_span.get_span_context()
        parent_trace_id = format(session_ctx.trace_id, "032x")
        parent_span_id = format(session_ctx.span_id, "016x")

        # -- Simulate env var transport (what inject_exgentic_env does) --
        otel = OtelContext(trace_id=parent_trace_id, span_id=parent_span_id)
        ctx = Context(
            run_id="run-1",
            output_dir=str(tmp_path),
            cache_dir=str(tmp_path),
            session_id="sess-1",
            otel_context=otel,
        )
        config = RuntimeConfig(
            run_id=ctx.run_id,
            output_dir=ctx.output_dir,
            cache_dir=ctx.cache_dir,
            session_id=ctx.session_id,
            otel_trace_id=parent_trace_id,
            otel_span_id=parent_span_id,
        )

        # -- Child side: restore context from RuntimeConfig --
        child_ctx = config.to_context()
        assert child_ctx.otel_context is not None
        assert child_ctx.otel_context.trace_id == parent_trace_id
        assert child_ctx.otel_context.span_id == parent_span_id

        # -- Child side: TraceLogger reconstructs parent context --
        logger = TraceLogger()
        logger._otel_logger = MagicMock()

        with patch.object(logger, "get_context", return_value=child_ctx):
            parent_otel_ctx = logger._get_parent_context({})

        # Verify the reconstructed parent has correct trace_id and span_id
        reconstructed_span = trace_api.get_current_span(parent_otel_ctx)
        reconstructed_ctx = reconstructed_span.get_span_context()
        assert format(reconstructed_ctx.trace_id, "032x") == parent_trace_id
        assert format(reconstructed_ctx.span_id, "016x") == parent_span_id
        assert reconstructed_ctx.is_remote is True

        # -- Child side: create LLM span as child of reconstructed parent --
        child_provider = TracerProvider()
        child_exporter = InMemorySpanExporter()
        child_provider.add_span_processor(SimpleSpanProcessor(child_exporter))
        child_tracer = child_provider.get_tracer("child")

        llm_span = child_tracer.start_span("chat gpt-4o", context=parent_otel_ctx, kind=SpanKind.CLIENT)
        llm_span.end()

        child_spans = child_exporter.get_finished_spans()
        assert len(child_spans) == 1
        llm = child_spans[0]

        # KEY ASSERTIONS: same trace, correct parent linkage
        assert (
            format(llm.context.trace_id, "032x") == parent_trace_id
        ), "LLM span must share the same trace_id as the session span"
        assert llm.parent is not None, "LLM span must have a parent"
        assert format(llm.parent.span_id, "016x") == parent_span_id, "LLM span's parent must be the session span"
        assert llm.kind == SpanKind.CLIENT

        session_span.end()

    def test_smolagents_context_object_path(self, tmp_path):
        """Simulate smolagents: whole Context object in metadata['context']."""
        from exgentic.integrations.litellm.trace_logger import TraceLogger

        otel = OtelContext(trace_id="ee" * 16, span_id="ff" * 8)
        ctx = Context(
            run_id="run-1",
            output_dir=str(tmp_path),
            cache_dir=str(tmp_path),
            session_id="sess-1",
            otel_context=otel,
        )

        # Smolagents injects the whole Context object
        kwargs = {"context": ctx, "model": "gpt-4o"}

        logger = TraceLogger()
        resolved = logger.get_context(kwargs)
        assert resolved is ctx
        assert resolved.otel_context.trace_id == "ee" * 16

    def test_contextvar_path_for_thread_runner(self, tmp_path):
        """Simulate thread runner: ContextVar propagation via copy_context()."""
        from exgentic.integrations.litellm.trace_logger import TraceLogger

        otel = OtelContext(trace_id="11" * 16, span_id="22" * 8)
        ctx = Context(
            run_id="run-1",
            output_dir=str(tmp_path),
            cache_dir=str(tmp_path),
            session_id="sess-1",
            otel_context=otel,
        )
        set_context(ctx)

        # copy_context simulates what ThreadTransport does
        thread_ctx = contextvars.copy_context()

        def check_in_thread():
            logger = TraceLogger()
            # No metadata, no kwargs context — falls through to ContextVar
            resolved = logger.get_context({})
            assert resolved is not None
            assert resolved.otel_context is not None
            assert resolved.otel_context.trace_id == "11" * 16
            return True

        assert thread_ctx.run(check_in_thread)

    def test_service_runner_fallback_path(self, tmp_path):
        """Simulate service runner: _SUBPROCESS_CONTEXT fallback for uvicorn threads."""
        from exgentic.integrations.litellm.trace_logger import TraceLogger

        otel = OtelContext(trace_id="33" * 16, span_id="44" * 8)
        ctx = Context(
            run_id="run-1",
            output_dir=str(tmp_path),
            cache_dir=str(tmp_path),
            session_id="sess-1",
            otel_context=otel,
        )

        # Service runner sets fallback, clears ContextVar
        set_context(None)  # type: ignore[arg-type]
        set_context_fallback(ctx)

        try:
            logger = TraceLogger()
            resolved = logger.get_context({})
            assert resolved is not None
            assert resolved.otel_context is not None
            assert resolved.otel_context.trace_id == "33" * 16
        finally:
            set_context_fallback(None)

    def test_inject_exgentic_env_includes_otel_vars(self, tmp_path, monkeypatch):
        """inject_exgentic_env(role=...) writes a per-service runtime.json.

        OTEL vars are persisted in runtime.json on disk; the child reads them
        via init_context() using the runtime file path.
        """
        from exgentic.core.context import Role

        monkeypatch.delenv("EXGENTIC_RUNTIME_FILE", raising=False)
        otel = OtelContext(trace_id="aa" * 16, span_id="bb" * 8)
        ctx = Context(
            run_id="run-1",
            output_dir=str(tmp_path),
            cache_dir=str(tmp_path),
            session_id="sess-1",
            otel_context=otel,
        )
        set_context(ctx)

        from exgentic.adapters.runners._utils import inject_exgentic_env

        env: dict[str, str] = {}
        inject_exgentic_env(env, role=Role.AGENT)

        expected_file = str(tmp_path / "run-1" / "sessions" / "sess-1" / "agent" / "runtime.json")
        assert (
            env.get("EXGENTIC_RUNTIME_FILE") == expected_file
        ), "inject_exgentic_env must set EXGENTIC_RUNTIME_FILE for the given role"

    def test_full_chain_session_to_llm_span(self, tmp_path):
        """Full chain: SessionSpanManager → update_tracing_context → env vars → child TraceLogger → LLM span.

        This is the complete end-to-end test simulating what happens when:
        1. OtelTracingObserver creates a session span
        2. Propagates context via update_tracing_context
        3. Runner sends context to child via env vars
        4. Child's TraceLogger creates LLM span linked to session
        """
        from exgentic.observers.handlers.otel import SessionSpanManager

        # Step 1: Parent creates session span
        parent_provider = TracerProvider()
        parent_exporter = InMemorySpanExporter()
        parent_provider.add_span_processor(SimpleSpanProcessor(parent_exporter))
        parent_tracer = parent_provider.get_tracer("parent")

        session_root = tmp_path / "test-run" / "sessions" / "sess-1"
        session_root.mkdir(parents=True)

        base_ctx = Context(run_id="run-1", output_dir=str(tmp_path), cache_dir=str(tmp_path))
        set_context(base_ctx)

        mgr = SessionSpanManager("sess-1", session_root, tracer=parent_tracer)
        mgr.start_span("test_bench test_subset session")
        mgr.update_tracing_context()

        # Step 2: Read context (what runner does before spawning child)
        updated_ctx = get_context()
        assert updated_ctx.otel_context is not None
        parent_trace_id = updated_ctx.otel_context.trace_id
        parent_span_id = updated_ctx.otel_context.span_id

        # Step 3: Serialize via RuntimeConfig (what inject_exgentic_env does)
        config = RuntimeConfig(
            run_id=updated_ctx.run_id,
            output_dir=updated_ctx.output_dir,
            cache_dir=updated_ctx.cache_dir,
            session_id=updated_ctx.session_id,
            otel_trace_id=updated_ctx.otel_context.trace_id,
            otel_span_id=updated_ctx.otel_context.span_id,
        )
        assert config.otel_trace_id is not None
        assert config.otel_span_id is not None

        # Step 4: Child restores context from RuntimeConfig
        child_ctx = config.to_context()
        assert child_ctx.otel_context.trace_id == parent_trace_id
        assert child_ctx.otel_context.span_id == parent_span_id

        # Step 5: Child's TraceLogger creates LLM span

        from exgentic.integrations.litellm.trace_logger import TraceLogger

        logger = TraceLogger()
        logger._otel_logger = MagicMock()

        with patch.object(logger, "get_context", return_value=child_ctx):
            parent_otel_ctx = logger._get_parent_context({})

        child_provider = TracerProvider()
        child_exporter = InMemorySpanExporter()
        child_provider.add_span_processor(SimpleSpanProcessor(child_exporter))
        child_tracer = child_provider.get_tracer("child")

        llm_span = child_tracer.start_span("chat gpt-4o", context=parent_otel_ctx, kind=SpanKind.CLIENT)
        llm_span.end()

        # Step 6: Verify trace linkage
        child_spans = child_exporter.get_finished_spans()
        assert len(child_spans) == 1
        llm = child_spans[0]

        assert (
            format(llm.context.trace_id, "032x") == parent_trace_id
        ), "LLM span trace_id must match session span trace_id"
        assert format(llm.parent.span_id, "016x") == parent_span_id, "LLM span parent must point to session span"

        # Cleanup
        mgr.end_current_span()


# ===================================================================
# O. Callback registration in child process context
# ===================================================================


class TestCallbackRegistrationCrossBoundary:
    """Verify litellm callbacks work when registered in child process context."""

    def test_configure_callbacks_registers_custom_logger_subclasses(self):
        """A single CustomLogger must be in litellm.callbacks, not success_callback."""
        import litellm
        from exgentic.integrations.litellm.config import _configure_callbacks
        from exgentic.integrations.litellm.trace_logger import TraceLogger

        # Simulate fresh child process (no callbacks)
        litellm.callbacks = []
        litellm.success_callback = []
        litellm.failure_callback = []

        _configure_callbacks()

        # Must be in callbacks (not success_callback)
        assert any(isinstance(cb, TraceLogger) for cb in litellm.callbacks)
        # Must NOT be in success_callback
        assert not any(isinstance(cb, TraceLogger) for cb in litellm.success_callback)

    def test_trace_logger_is_custom_logger_subclass(self):
        """TraceLogger must be a CustomLogger subclass for litellm to call log_success_event."""
        from exgentic.integrations.litellm.trace_logger import (
            AsyncTraceLogger,
            SyncTraceLogger,
            TraceLogger,
        )
        from litellm.integrations.custom_logger import CustomLogger

        assert issubclass(TraceLogger, CustomLogger)
        assert issubclass(SyncTraceLogger, CustomLogger)
        assert issubclass(AsyncTraceLogger, CustomLogger)

    def test_trace_logger_has_log_success_event(self):
        """TraceLogger must implement log_success_event (called by litellm for CustomLogger)."""
        from exgentic.integrations.litellm.trace_logger import SyncTraceLogger

        logger = SyncTraceLogger()
        assert hasattr(logger, "log_success_event")
        assert callable(logger.log_success_event)


# ── P. Dependency crossing ──────────────────────────────────────────────


class TestDependencyCrossing:
    """Tests that otel+litellm dependencies are available and wired correctly across boundaries.

    If someone wants OTEL tracing, the full chain must work — not silently fail.
    """

    # -- otel packages are importable (hard requirement) --

    def test_opentelemetry_api_importable(self):
        """opentelemetry-api must be installed for tracing to work."""
        from opentelemetry import trace

        assert trace.get_tracer is not None

    def test_opentelemetry_sdk_importable(self):
        """opentelemetry-sdk must be installed for TracerProvider."""
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor

        assert TracerProvider is not None
        assert SimpleSpanProcessor is not None

    def test_opentelemetry_exporters_importable(self):
        """OTLP exporters must be available for span export."""
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter as GrpcExporter,
        )
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter as HttpExporter,
        )

        assert GrpcExporter is not None
        assert HttpExporter is not None

    # -- litellm + tracing integration is wired --

    def test_litellm_custom_logger_base_importable(self):
        """Litellm CustomLogger base class must be importable for callbacks."""
        from litellm.integrations.custom_logger import CustomLogger

        assert CustomLogger is not None

    def test_trace_logger_is_custom_logger(self):
        """TraceLogger must subclass CustomLogger so litellm invokes it."""
        from exgentic.integrations.litellm.trace_logger import TraceLogger
        from litellm.integrations.custom_logger import CustomLogger

        assert issubclass(TraceLogger, CustomLogger)

    # -- settings propagation: otel_enabled is included in get_env() --

    def test_settings_get_env_includes_otel_enabled(self):
        """get_env() must export EXGENTIC_OTEL_ENABLED so child processes inherit it."""
        from exgentic.utils.settings import ExgenticSettings

        settings = ExgenticSettings(otel_enabled=True)
        env = settings.get_env()
        assert "EXGENTIC_OTEL_ENABLED" in env
        assert env["EXGENTIC_OTEL_ENABLED"] == "true"

    def test_settings_get_env_includes_otel_record_content(self):
        """get_env() must export EXGENTIC_OTEL_RECORD_CONTENT."""
        from exgentic.utils.settings import ExgenticSettings

        settings = ExgenticSettings(otel_record_content=True)
        env = settings.get_env()
        assert "EXGENTIC_OTEL_RECORD_CONTENT" in env
        assert env["EXGENTIC_OTEL_RECORD_CONTENT"] == "true"

    # -- inject_exgentic_env propagates OTEL context vars --

    def test_inject_exgentic_env_includes_otel_context(self, tmp_path, monkeypatch):
        """inject_exgentic_env(role=...) writes a per-service runtime.json.

        OTEL context is propagated via runtime.json on disk; the child reads
        it through init_context() using the runtime file path.
        """
        monkeypatch.delenv("EXGENTIC_RUNTIME_FILE", raising=False)
        from exgentic.core.context import Context, OtelContext, Role, set_context

        ctx = Context(
            session_id="s1",
            run_id="r1",
            output_dir=str(tmp_path),
            cache_dir="/tmp/cache",
            otel_context=OtelContext(trace_id="abc123", span_id="def456"),
        )
        set_context(ctx)

        env: dict[str, str] = {}
        from exgentic.adapters.runners._utils import inject_exgentic_env

        inject_exgentic_env(env, role=Role.BENCHMARK)
        expected = str(tmp_path / "r1" / "sessions" / "s1" / "benchmark" / "runtime.json")
        assert env.get("EXGENTIC_RUNTIME_FILE") == expected

    def test_inject_exgentic_env_propagates_settings(self, tmp_path, monkeypatch):
        """inject_exgentic_env(role=...) sets EXGENTIC_RUNTIME_FILE; settings are in runtime.json."""
        monkeypatch.delenv("EXGENTIC_RUNTIME_FILE", raising=False)
        from exgentic.core.context import Context, Role, set_context

        ctx = Context(
            session_id="s",
            run_id="r",
            output_dir=str(tmp_path),
            cache_dir="/tmp/c",
        )
        set_context(ctx)

        env: dict[str, str] = {}
        from exgentic.adapters.runners._utils import inject_exgentic_env

        inject_exgentic_env(env, role=Role.AGENT)
        expected = str(tmp_path / "r" / "sessions" / "s" / "agent" / "runtime.json")
        assert env.get("EXGENTIC_RUNTIME_FILE") == expected

    # -- init_tracing_from_env produces a working tracer --

    def test_init_tracing_from_env_returns_tracer(self, monkeypatch):
        """init_tracing_from_env must return a real Tracer, not None."""
        # Set endpoint so it doesn't fail on missing env
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
        from exgentic.utils.otel import init_tracing_from_env

        tracer = init_tracing_from_env()
        assert tracer is not None
        # Tracer must be able to create spans
        span = tracer.start_span("test-dep-crossing")
        assert span is not None
        span.end()

    # -- TraceLogger _init_otel creates tracer when context is present --

    def test_trace_logger_init_otel_creates_tracer(self, monkeypatch):
        """When OTEL is enabled and context is available, _init_otel must produce a tracer."""
        import tempfile

        monkeypatch.setenv("EXGENTIC_OTEL_ENABLED", "true")
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
        from exgentic.utils import settings as _sm

        _sm._settings = None

        from exgentic.core.context import Context, OtelContext, set_context
        from exgentic.integrations.litellm.trace_logger import TraceLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = Context(
                session_id="s1",
                run_id="r1",
                output_dir=tmpdir,
                cache_dir="/tmp/cache",
                otel_context=OtelContext(trace_id="aaa", span_id="bbb"),
            )
            set_context(ctx)
            logger = TraceLogger()
            logger._init_otel({})
            assert logger._tracer is not None, "_init_otel must create a tracer when context has OTEL"

        _sm._settings = None

    # -- Full chain: parent context → env vars → child reconstruction --

    def test_otel_context_survives_env_roundtrip(self):
        """OtelContext must survive RuntimeConfig roundtrip without data loss."""
        from exgentic.core.context import Context, OtelContext, RuntimeConfig

        original = Context(
            session_id="sess",
            run_id="run",
            output_dir="/out",
            cache_dir="/cache",
            otel_context=OtelContext(trace_id="t1", span_id="s1"),
        )
        config = RuntimeConfig(
            run_id=original.run_id,
            output_dir=original.output_dir,
            cache_dir=original.cache_dir,
            session_id=original.session_id,
            otel_trace_id="t1",
            otel_span_id="s1",
        )
        restored = config.to_context()
        assert restored.otel_context is not None
        assert restored.otel_context.trace_id == "t1"
        assert restored.otel_context.span_id == "s1"

    def test_parent_span_reconstructed_from_otel_context(self):
        """TraceLogger must reconstruct a NonRecordingSpan from propagated OtelContext."""
        from exgentic.core.context import Context, OtelContext

        trace_id = format(12345678901234567890, "032x")
        span_id = format(9876543210, "016x")
        ctx = Context(
            session_id="s",
            run_id="r",
            output_dir="/o",
            cache_dir="/c",
            otel_context=OtelContext(trace_id=trace_id, span_id=span_id),
        )
        from unittest.mock import MagicMock

        from exgentic.integrations.litellm.trace_logger import TraceLogger

        logger = TraceLogger()
        logger._otel_logger = MagicMock()  # normally set by _init_otel
        parent_ctx = logger._get_parent_context({"context": ctx})
        # Should produce a valid opentelemetry context, not None
        assert parent_ctx is not None


# ── Q. Per-runner OTEL contract ─────────────────────────────────────────


class TestPerRunnerOtelContract:
    """Verify each runner type correctly propagates OTEL context and settings.

    Uses behavioral tests where practical (mocking the propagation functions
    and verifying they're called), and direct functional tests for context
    propagation primitives.
    """

    # -- Thread runner: copies ContextVar --

    def test_contextvar_copy_preserves_otel_context(self):
        """Verify that contextvars.copy_context() actually preserves our OTEL context."""
        import contextvars

        from exgentic.core.context import _CONTEXT, Context, OtelContext, set_context

        ctx = Context(
            session_id="s",
            run_id="r",
            output_dir="/o",
            cache_dir="/c",
            otel_context=OtelContext(trace_id="thread-trace", span_id="thread-span"),
        )
        set_context(ctx)
        copied = contextvars.copy_context()
        child_ctx = copied[_CONTEXT]
        assert child_ctx.otel_context is not None
        assert child_ctx.otel_context.trace_id == "thread-trace"

    # -- inject_exgentic_env: behavioral test --

    def test_inject_exgentic_env_writes_runtime_file(self, tmp_path, monkeypatch):
        """inject_exgentic_env(role=...) writes a runtime.json and sets EXGENTIC_RUNTIME_FILE."""
        from exgentic.adapters.runners._utils import inject_exgentic_env
        from exgentic.core.context import Role, run_scope, session_scope

        with run_scope(run_id="test-run", output_dir=str(tmp_path)):
            with session_scope("test-session"):
                env: dict[str, str] = {}
                inject_exgentic_env(env, role=Role.AGENT)
                assert "EXGENTIC_RUNTIME_FILE" in env
                runtime_path = env["EXGENTIC_RUNTIME_FILE"]
                assert Path(runtime_path).exists(), "runtime.json must be written to disk"
                import json

                data = json.loads(Path(runtime_path).read_text())
                assert data["role"] == "agent"
                assert data["run_id"] == "test-run"
                assert data["session_id"] == "test-session"

    def test_inject_exgentic_env_inherits_parent_runtime(self, monkeypatch):
        """inject_exgentic_env(role=None) copies parent's EXGENTIC_RUNTIME_FILE."""
        from exgentic.adapters.runners._utils import inject_exgentic_env

        monkeypatch.setenv("EXGENTIC_RUNTIME_FILE", "/parent/runtime.json")
        env: dict[str, str] = {}
        inject_exgentic_env(env, role=None)
        assert env.get("EXGENTIC_RUNTIME_FILE") == "/parent/runtime.json"

    # -- init_context: restores OTEL context --

    def test_init_context_restores_otel(self, tmp_path, monkeypatch):
        """init_context must reconstruct OtelContext from runtime.json."""
        import json

        runtime = {
            "run_id": "r",
            "output_dir": str(tmp_path),
            "cache_dir": "/c",
            "otel_trace_id": "restored-trace",
            "otel_span_id": "restored-span",
        }
        (tmp_path / "runtime.json").write_text(json.dumps(runtime))
        monkeypatch.setenv("EXGENTIC_RUNTIME_FILE", str(tmp_path / "runtime.json"))

        from exgentic.core.context import init_context

        ctx = init_context()
        assert ctx.otel_context is not None
        assert ctx.otel_context.trace_id == "restored-trace"
        assert ctx.otel_context.span_id == "restored-span"


# ===================================================================
# SEMANTIC CONVENTIONS VERIFICATION TESTS
# ===================================================================
#
# These tests verify that every attribute documented in
# docs/observability/semantic-conventions.md is actually emitted
# on the correct span type, with correct values, names, and kinds.
# ===================================================================


def _full_lifecycle_spans(
    ctx,
    tmp_path,
    *,
    record_content=False,
    run_config=None,
    score=None,
    agent=None,
    session=None,
    add_react_step=False,
):
    """Run full observer lifecycle and return (session_span, tool_spans, all_spans).

    Creates a TracerProvider + InMemorySpanExporter, runs on_run_start through
    on_session_success, and returns the finished spans for assertion.
    """
    from exgentic.observers.handlers.otel import OtelTracingObserver, SessionSpanManager

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    t = provider.get_tracer("test")

    session = session or MockSession()
    run_config = run_config or MockRunConfig()
    score = score or MockScore()
    agent = agent or MockAgent()
    settings = MagicMock(otel_record_content=record_content)

    obs = OtelTracingObserver()
    original_init = SessionSpanManager.__init__

    def patched_init(self, session_id, session_root_path, tracer=t):
        original_init(self, session_id, session_root_path, tracer=tracer)

    with (
        patch("exgentic.observers.handlers.otel.get_benchmark_entries", return_value={}),
        patch("exgentic.observers.handlers.otel.get_agent_entries", return_value={}),
        patch("exgentic.observers.handlers.otel.get_settings", return_value=settings),
        patch(
            "exgentic.observers.handlers.otel.to_otel_attribute_value",
            side_effect=lambda v: str(v) if v is not None else None,
        ),
        patch("exgentic.observers.handlers.otel.flush_traces"),
        patch.object(SessionSpanManager, "__init__", patched_init),
    ):
        obs.on_run_start(run_config)
        obs.on_session_creation(session)
        obs.on_session_start(session, agent, MockObservation())

        if add_react_step:
            mock_action = MagicMock()
            action_item = MagicMock()
            action_item.name = "browse"
            action_item.id = "act-1"
            action_item.arguments = MagicMock()
            action_item.arguments.model_dump_json.return_value = '{"url": "http://example.com"}'
            mock_action.to_action_list.return_value = [action_item]
            obs.on_react_success(session, mock_action)
            obs.on_step_success(session, MockObservation())

        obs.on_session_success(session, score, agent)

    spans = exporter.get_finished_spans()
    session_span = next((s for s in spans if "session" in s.name), None)
    tool_spans = [s for s in spans if "execute_tool" in s.name]
    return session_span, tool_spans, spans


def _invoke_write_otel(
    *,
    kwargs=None,
    response_obj=None,
    status="success",
    record_content=False,
    session_id="sess-001",
    otel_trace_id="00000000000000000000000000000001",
    otel_span_id="0000000000000001",
):
    """Invoke TraceLogger._write_otel with a real tracer+exporter, return finished spans."""
    from exgentic.integrations.litellm.trace_logger import TraceLogger

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    t = provider.get_tracer("test")

    logger = TraceLogger()
    logger._tracer = t
    logger._otel_logger = MagicMock()

    mock_ctx = MagicMock()
    mock_ctx.session_id = session_id
    mock_ctx.otel_context = MagicMock()
    mock_ctx.otel_context.trace_id = otel_trace_id
    mock_ctx.otel_context.span_id = otel_span_id

    if kwargs is None:
        kwargs = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hello"}],
            "optional_params": {},
            "litellm_params": {"custom_llm_provider": "openai"},
        }
    if response_obj is None:
        response_obj = {
            "id": "chatcmpl-123",
            "model": "gpt-4o-2024-05-13",
            "usage": {"prompt_tokens": 10, "completion_tokens": 20},
            "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "hi"}}],
        }

    settings = MagicMock(otel_enabled=True, otel_record_content=record_content)
    with (
        patch("exgentic.integrations.litellm.trace_logger.get_settings", return_value=settings),
        patch.object(logger, "get_context", return_value=mock_ctx),
    ):
        logger._write_otel(kwargs, response_obj, status)

    return exporter.get_finished_spans()


# ===================================================================
# 1. Session ROOT span semantic conventions
# ===================================================================


class TestSessionSpanSemanticConventions:
    """Verify every attribute documented for the Session (ROOT) span type."""

    def test_session_span_name_exact_format(self, ctx, tmp_path):
        """Span name is '{benchmark} {subset} session' per docs."""
        rc = MockRunConfig()
        rc.benchmark = "tau2"
        rc.subset = "retail"
        session_span, _, _ = _full_lifecycle_spans(ctx, tmp_path, run_config=rc)
        assert session_span.name == "tau2 retail session"

    def test_session_span_name_empty_subset(self, ctx, tmp_path):
        """When subset is None, name uses empty string."""
        rc = MockRunConfig()
        rc.benchmark = "tau2"
        rc.subset = None
        session_span, _, _ = _full_lifecycle_spans(ctx, tmp_path, run_config=rc)
        assert session_span.name == "tau2  session"

    def test_session_span_kind_is_internal(self, ctx, tmp_path):
        """Session span has SpanKind.INTERNAL (default) per docs."""
        session_span, _, _ = _full_lifecycle_spans(ctx, tmp_path)
        # INTERNAL is the default span kind (value 0)
        assert session_span.kind == SpanKind.INTERNAL

    def test_session_span_benchmark_slug_name(self, ctx, tmp_path):
        rc = MockRunConfig()
        rc.benchmark = "tau2"
        session_span, _, _ = _full_lifecycle_spans(ctx, tmp_path, run_config=rc)
        assert session_span.attributes["exgentic.benchmark.slug_name"] == "tau2"

    def test_session_span_benchmark_subset(self, ctx, tmp_path):
        rc = MockRunConfig()
        rc.subset = "retail"
        session_span, _, _ = _full_lifecycle_spans(ctx, tmp_path, run_config=rc)
        assert session_span.attributes["exgentic.benchmark.subset"] == "retail"

    def test_session_span_agent_slug(self, ctx, tmp_path):
        rc = MockRunConfig()
        rc.agent = "tool_calling"
        session_span, _, _ = _full_lifecycle_spans(ctx, tmp_path, run_config=rc)
        assert session_span.attributes["exgentic.agent.slug"] == "tool_calling"

    def test_session_span_run_id(self, ctx, tmp_path):
        session_span, _, _ = _full_lifecycle_spans(ctx, tmp_path)
        assert "exgentic.run.id" in session_span.attributes

    def test_session_span_conversation_id(self, ctx, tmp_path):
        session_span, _, _ = _full_lifecycle_spans(ctx, tmp_path)
        assert session_span.attributes["gen_ai.conversation.id"] == "sess-001"

    def test_session_span_session_id(self, ctx, tmp_path):
        session_span, _, _ = _full_lifecycle_spans(ctx, tmp_path)
        assert session_span.attributes["exgentic.session.id"] == "sess-001"

    def test_session_span_request_model(self, ctx, tmp_path):
        rc = MockRunConfig()
        rc.model = "gpt-4o"
        session_span, _, _ = _full_lifecycle_spans(ctx, tmp_path, run_config=rc)
        assert session_span.attributes["gen_ai.request.model"] == "gpt-4o"

    def test_session_span_model_from_agent_kwargs(self, ctx, tmp_path):
        """gen_ai.request.model falls back to agent_kwargs['model']."""
        rc = MockRunConfig()
        rc.model = None
        rc.agent_kwargs = {"model": "claude-3-opus"}
        session_span, _, _ = _full_lifecycle_spans(ctx, tmp_path, run_config=rc)
        assert session_span.attributes["gen_ai.request.model"] == "claude-3-opus"

    def test_session_span_no_model_when_absent(self, ctx, tmp_path):
        """gen_ai.request.model absent when both model and agent_kwargs empty."""
        rc = MockRunConfig()
        rc.model = None
        rc.agent_kwargs = {}
        session_span, _, _ = _full_lifecycle_spans(ctx, tmp_path, run_config=rc)
        assert "gen_ai.request.model" not in session_span.attributes

    def test_session_span_task_id(self, ctx, tmp_path):
        session_span, _, _ = _full_lifecycle_spans(ctx, tmp_path)
        assert session_span.attributes["exgentic.session.task_id"] == "task-001"

    def test_session_span_action_attributes(self, ctx, tmp_path):
        """Each action in session.actions produces name/description/is_message/is_finish attributes."""
        session_span, _, _ = _full_lifecycle_spans(ctx, tmp_path)
        assert session_span.attributes["exgentic.session.action.test_action.name"] == "test_action"
        assert session_span.attributes["exgentic.session.action.test_action.description"] == "A test action"
        assert session_span.attributes["exgentic.session.action.test_action.is_message"] is False
        assert session_span.attributes["exgentic.session.action.test_action.is_finish"] is False

    def test_session_span_context_attributes(self, ctx, tmp_path):
        """exgentic.context.{key} emitted for each context key."""
        session_span, _, _ = _full_lifecycle_spans(ctx, tmp_path)
        assert session_span.attributes["exgentic.context.key"] == "value"

    def test_session_span_agent_id(self, ctx, tmp_path):
        session_span, _, _ = _full_lifecycle_spans(ctx, tmp_path)
        assert session_span.attributes["exgentic.session.agent.id"] == "agent-001"

    def test_session_span_agent_path(self, ctx, tmp_path):
        session_span, _, _ = _full_lifecycle_spans(ctx, tmp_path)
        assert session_span.attributes["exgentic.session.agent.path"] == "/tmp/agent"

    def test_session_span_score_success(self, ctx, tmp_path):
        session_span, _, _ = _full_lifecycle_spans(ctx, tmp_path)
        assert session_span.attributes["exgentic.score.success"] is True

    def test_session_span_score_value(self, ctx, tmp_path):
        session_span, _, _ = _full_lifecycle_spans(ctx, tmp_path)
        assert session_span.attributes["exgentic.score"] == 0.95

    def test_session_span_score_is_finished(self, ctx, tmp_path):
        session_span, _, _ = _full_lifecycle_spans(ctx, tmp_path)
        assert session_span.attributes["exgentic.score.is_finished"] is True

    def test_session_span_steps(self, ctx, tmp_path):
        session_span, _, _ = _full_lifecycle_spans(ctx, tmp_path)
        assert isinstance(session_span.attributes["exgentic.session.steps"], int)
        assert session_span.attributes["exgentic.session.steps"] >= 1

    def test_session_span_agent_cost_json(self, ctx, tmp_path):
        """exgentic.agent.agent_cost is a parseable JSON string."""
        session_span, _, _ = _full_lifecycle_spans(ctx, tmp_path)
        cost_str = session_span.attributes["exgentic.agent.agent_cost"]
        parsed = json.loads(cost_str)
        assert isinstance(parsed, dict)
        assert parsed["total"] == 0.5

    def test_session_span_session_cost_json(self, ctx, tmp_path):
        """exgentic.session.cost is a parseable JSON string."""

        class CostSession(MockSession):
            def get_cost(self):
                return {"total": 1.0}

        session_span, _, _ = _full_lifecycle_spans(ctx, tmp_path, session=CostSession())
        cost_str = session_span.attributes["exgentic.session.cost"]
        parsed = json.loads(cost_str)
        assert isinstance(parsed, dict)

    def test_session_span_task_filtered_when_disabled(self, ctx, tmp_path):
        """exgentic.session.task absent when record_content=False."""
        session_span, _, _ = _full_lifecycle_spans(ctx, tmp_path, record_content=False)
        assert "exgentic.session.task" not in session_span.attributes

    def test_session_span_task_present_when_enabled(self, ctx, tmp_path):
        """exgentic.session.task present when record_content=True."""
        session_span, _, _ = _full_lifecycle_spans(ctx, tmp_path, record_content=True)
        assert session_span.attributes["exgentic.session.task"] == "Test task"


# ===================================================================
# 2. execute_tool span semantic conventions
# ===================================================================


class TestExecuteToolSpanSemanticConventions:
    """Verify every attribute documented for execute_tool spans."""

    def test_initial_observation_span_name(self, ctx, tmp_path):
        """First tool span named 'execute_tool initial_observation'."""
        _, tool_spans, _ = _full_lifecycle_spans(ctx, tmp_path)
        assert any(s.name == "execute_tool initial_observation" for s in tool_spans)

    def test_execute_tool_span_name_format(self, ctx, tmp_path):
        """Tool span named 'execute_tool {tool_name}' using action.name."""
        _, tool_spans, _ = _full_lifecycle_spans(ctx, tmp_path, add_react_step=True)
        assert any(s.name == "execute_tool browse" for s in tool_spans)

    def test_execute_tool_span_kind_client(self, ctx, tmp_path):
        """All execute_tool spans have SpanKind.CLIENT per docs."""
        _, tool_spans, _ = _full_lifecycle_spans(ctx, tmp_path, add_react_step=True)
        for ts in tool_spans:
            assert ts.kind == SpanKind.CLIENT

    def test_execute_tool_operation_name(self, ctx, tmp_path):
        """gen_ai.operation.name == 'execute_tool' on all tool spans."""
        _, tool_spans, _ = _full_lifecycle_spans(ctx, tmp_path, add_react_step=True)
        for ts in tool_spans:
            assert ts.attributes["gen_ai.operation.name"] == "execute_tool"

    def test_execute_tool_tool_name(self, ctx, tmp_path):
        """gen_ai.tool.name matches action name."""
        _, tool_spans, _ = _full_lifecycle_spans(ctx, tmp_path, add_react_step=True)
        browse_span = next(s for s in tool_spans if "browse" in s.name)
        assert browse_span.attributes["gen_ai.tool.name"] == "browse"

    def test_execute_tool_tool_id(self, ctx, tmp_path):
        """gen_ai.tool.id set on tool spans from react_success."""
        _, tool_spans, _ = _full_lifecycle_spans(ctx, tmp_path, add_react_step=True)
        browse_span = next(s for s in tool_spans if "browse" in s.name)
        assert browse_span.attributes["gen_ai.tool.id"] == "act-1"

    def test_execute_tool_initial_observation_description(self, ctx, tmp_path):
        """Initial observation span has gen_ai.tool.description."""
        _, tool_spans, _ = _full_lifecycle_spans(ctx, tmp_path)
        initial_span = next(s for s in tool_spans if "initial_observation" in s.name)
        assert initial_span.attributes["gen_ai.tool.description"] == "Initial observation from benchmark"

    def test_execute_tool_tool_description_from_session_actions(self, ctx, tmp_path):
        """gen_ai.tool.description looked up from session.actions by name."""

        class ActionWithDesc:
            name = "browse"
            description = "Browse the web"
            is_message = False
            is_finish = False

        class SessionWithActions(MockSession):
            actions: ClassVar = [ActionWithDesc()]

        _, tool_spans, _ = _full_lifecycle_spans(ctx, tmp_path, session=SessionWithActions(), add_react_step=True)
        browse_span = next(s for s in tool_spans if "browse" in s.name)
        assert browse_span.attributes["gen_ai.tool.description"] == "Browse the web"

    def test_execute_tool_parameters_filtered_when_disabled(self, ctx, tmp_path):
        """gen_ai.tool.parameters absent when record_content=False."""
        _, tool_spans, _ = _full_lifecycle_spans(ctx, tmp_path, record_content=False, add_react_step=True)
        browse_span = next(s for s in tool_spans if "browse" in s.name)
        assert "gen_ai.tool.parameters" not in browse_span.attributes

    def test_execute_tool_parameters_present_when_enabled(self, ctx, tmp_path):
        """gen_ai.tool.parameters present as JSON when record_content=True."""
        _, tool_spans, _ = _full_lifecycle_spans(ctx, tmp_path, record_content=True, add_react_step=True)
        browse_span = next(s for s in tool_spans if "browse" in s.name)
        assert "gen_ai.tool.parameters" in browse_span.attributes
        params = json.loads(browse_span.attributes["gen_ai.tool.parameters"])
        assert params["url"] == "http://example.com"

    def test_execute_tool_result_filtered_when_disabled(self, ctx, tmp_path):
        """gen_ai.tool.result absent when record_content=False."""
        _, tool_spans, _ = _full_lifecycle_spans(ctx, tmp_path, record_content=False, add_react_step=True)
        browse_span = next(s for s in tool_spans if "browse" in s.name)
        assert "gen_ai.tool.result" not in browse_span.attributes

    def test_execute_tool_result_present_when_enabled(self, ctx, tmp_path):
        """gen_ai.tool.result present when record_content=True."""
        _, tool_spans, _ = _full_lifecycle_spans(ctx, tmp_path, record_content=True, add_react_step=True)
        browse_span = next(s for s in tool_spans if "browse" in s.name)
        assert "gen_ai.tool.result" in browse_span.attributes

    def test_execute_tool_inherits_conversation_id(self, ctx, tmp_path):
        """execute_tool spans inherit gen_ai.conversation.id from session."""
        _, tool_spans, _ = _full_lifecycle_spans(ctx, tmp_path)
        for ts in tool_spans:
            assert ts.attributes["gen_ai.conversation.id"] == "sess-001"


# ===================================================================
# 3. LLM inference span semantic conventions
# ===================================================================


class TestLLMInferenceSpanSemanticConventions:
    """Verify every attribute documented for LLM inference spans."""

    def test_llm_span_name_chat(self):
        """Span name is 'chat {model}' for chat operations."""
        spans = _invoke_write_otel(
            kwargs={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "hi"}],
                "optional_params": {},
                "litellm_params": {"custom_llm_provider": "openai"},
            }
        )
        assert len(spans) == 1
        assert spans[0].name == "chat gpt-4o"

    def test_llm_span_name_text_completion(self):
        """Span name is 'text_completion {model}' when no messages key."""
        spans = _invoke_write_otel(
            kwargs={
                "model": "text-davinci-003",
                "optional_params": {},
                "litellm_params": {"custom_llm_provider": "openai"},
            }
        )
        assert spans[0].name == "text_completion text-davinci-003"

    def test_llm_span_kind_client(self):
        """LLM inference spans have SpanKind.CLIENT per docs."""
        spans = _invoke_write_otel()
        assert spans[0].kind == SpanKind.CLIENT

    def test_llm_span_operation_name_chat(self):
        spans = _invoke_write_otel()
        assert spans[0].attributes["gen_ai.operation.name"] == "chat"

    def test_llm_span_operation_name_text_completion(self):
        spans = _invoke_write_otel(
            kwargs={
                "model": "text-davinci-003",
                "optional_params": {},
                "litellm_params": {"custom_llm_provider": "openai"},
            }
        )
        assert spans[0].attributes["gen_ai.operation.name"] == "text_completion"

    def test_llm_span_provider_name_openai(self):
        spans = _invoke_write_otel()
        assert spans[0].attributes["gen_ai.provider.name"] == "openai"

    @pytest.mark.parametrize(
        "litellm_provider,expected",
        [
            ("openai", "openai"),
            ("azure", "azure.ai.openai"),
            ("anthropic", "anthropic"),
            ("bedrock", "aws.bedrock"),
            ("vertex_ai", "gcp.vertex_ai"),
            ("gemini", "gcp.gemini"),
            ("cohere", "cohere"),
            ("groq", "groq"),
            ("mistral", "mistral_ai"),
            ("deepseek", "deepseek"),
            ("perplexity", "perplexity"),
            ("watsonx", "ibm.watsonx.ai"),
            ("xai", "x_ai"),
        ],
    )
    def test_llm_span_provider_mapping(self, litellm_provider, expected):
        """All documented provider mappings produce correct gen_ai.provider.name."""
        spans = _invoke_write_otel(
            kwargs={
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
                "optional_params": {},
                "litellm_params": {"custom_llm_provider": litellm_provider},
            }
        )
        assert spans[0].attributes["gen_ai.provider.name"] == expected

    def test_llm_span_provider_unknown_passthrough(self):
        """Unknown provider name passes through unchanged."""
        spans = _invoke_write_otel(
            kwargs={
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "optional_params": {},
                "litellm_params": {"custom_llm_provider": "custom_provider"},
            }
        )
        assert spans[0].attributes["gen_ai.provider.name"] == "custom_provider"

    def test_llm_span_provider_none_becomes_unknown(self):
        """None provider becomes 'unknown'."""
        spans = _invoke_write_otel(
            kwargs={
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "optional_params": {},
                "litellm_params": {"custom_llm_provider": None},
            }
        )
        assert spans[0].attributes["gen_ai.provider.name"] == "unknown"

    def test_llm_span_request_model(self):
        spans = _invoke_write_otel()
        assert spans[0].attributes["gen_ai.request.model"] == "gpt-4o"

    def test_llm_span_conversation_id(self):
        spans = _invoke_write_otel(session_id="sess-xyz")
        assert spans[0].attributes["gen_ai.conversation.id"] == "sess-xyz"

    def test_llm_span_request_max_tokens(self):
        spans = _invoke_write_otel(
            kwargs={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "hi"}],
                "optional_params": {"max_tokens": 100},
                "litellm_params": {"custom_llm_provider": "openai"},
            }
        )
        assert spans[0].attributes["gen_ai.request.max_tokens"] == 100

    def test_llm_span_request_temperature(self):
        spans = _invoke_write_otel(
            kwargs={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "hi"}],
                "optional_params": {"temperature": 0.7},
                "litellm_params": {"custom_llm_provider": "openai"},
            }
        )
        assert spans[0].attributes["gen_ai.request.temperature"] == 0.7

    def test_llm_span_request_top_p(self):
        spans = _invoke_write_otel(
            kwargs={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "hi"}],
                "optional_params": {"top_p": 0.9},
                "litellm_params": {"custom_llm_provider": "openai"},
            }
        )
        assert spans[0].attributes["gen_ai.request.top_p"] == 0.9

    def test_llm_span_request_top_k(self):
        """top_k is cast to float per implementation."""
        spans = _invoke_write_otel(
            kwargs={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "hi"}],
                "optional_params": {"top_k": 40},
                "litellm_params": {"custom_llm_provider": "openai"},
            }
        )
        assert spans[0].attributes["gen_ai.request.top_k"] == 40.0

    def test_llm_span_request_frequency_penalty(self):
        spans = _invoke_write_otel(
            kwargs={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "hi"}],
                "optional_params": {"frequency_penalty": 0.5},
                "litellm_params": {"custom_llm_provider": "openai"},
            }
        )
        assert spans[0].attributes["gen_ai.request.frequency_penalty"] == 0.5

    def test_llm_span_request_presence_penalty(self):
        spans = _invoke_write_otel(
            kwargs={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "hi"}],
                "optional_params": {"presence_penalty": 0.3},
                "litellm_params": {"custom_llm_provider": "openai"},
            }
        )
        assert spans[0].attributes["gen_ai.request.presence_penalty"] == 0.3

    def test_llm_span_request_stop_sequences_list(self):
        spans = _invoke_write_otel(
            kwargs={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "hi"}],
                "optional_params": {"stop": ["END", "STOP"]},
                "litellm_params": {"custom_llm_provider": "openai"},
            }
        )
        assert spans[0].attributes["gen_ai.request.stop_sequences"] == ("END", "STOP")

    def test_llm_span_request_stop_sequences_string(self):
        """Single string stop is wrapped in a list."""
        spans = _invoke_write_otel(
            kwargs={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "hi"}],
                "optional_params": {"stop": "END"},
                "litellm_params": {"custom_llm_provider": "openai"},
            }
        )
        assert spans[0].attributes["gen_ai.request.stop_sequences"] == ("END",)

    def test_llm_span_choice_count_when_not_one(self):
        """gen_ai.request.choice.count set only when n != 1."""
        spans = _invoke_write_otel(
            kwargs={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "hi"}],
                "optional_params": {"n": 3},
                "litellm_params": {"custom_llm_provider": "openai"},
            }
        )
        assert spans[0].attributes["gen_ai.request.choice.count"] == 3

    def test_llm_span_no_choice_count_when_one(self):
        """gen_ai.request.choice.count absent when n == 1."""
        spans = _invoke_write_otel(
            kwargs={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "hi"}],
                "optional_params": {"n": 1},
                "litellm_params": {"custom_llm_provider": "openai"},
            }
        )
        assert "gen_ai.request.choice.count" not in spans[0].attributes

    def test_llm_span_seed(self):
        spans = _invoke_write_otel(
            kwargs={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "hi"}],
                "optional_params": {"seed": 42},
                "litellm_params": {"custom_llm_provider": "openai"},
            }
        )
        assert spans[0].attributes["gen_ai.request.seed"] == 42

    def test_llm_span_response_id(self):
        spans = _invoke_write_otel()
        assert spans[0].attributes["gen_ai.response.id"] == "chatcmpl-123"

    def test_llm_span_response_model(self):
        spans = _invoke_write_otel()
        assert spans[0].attributes["gen_ai.response.model"] == "gpt-4o-2024-05-13"

    def test_llm_span_usage_input_tokens(self):
        spans = _invoke_write_otel()
        assert spans[0].attributes["gen_ai.usage.input_tokens"] == 10

    def test_llm_span_usage_output_tokens(self):
        spans = _invoke_write_otel()
        assert spans[0].attributes["gen_ai.usage.output_tokens"] == 20

    def test_llm_span_finish_reasons(self):
        spans = _invoke_write_otel()
        assert spans[0].attributes["gen_ai.response.finish_reasons"] == ("stop",)

    def test_llm_span_error_type_on_failure(self):
        """error.type set to exception class name on failure."""
        spans = _invoke_write_otel(
            status="failure",
            kwargs={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "hi"}],
                "optional_params": {},
                "litellm_params": {"custom_llm_provider": "openai"},
                "exception": ValueError("bad request"),
            },
            response_obj={},
        )
        assert spans[0].attributes["error.type"] == "ValueError"

    def test_llm_span_error_type_from_dict(self):
        """error.type extracted from dict error info."""
        spans = _invoke_write_otel(
            status="failure",
            kwargs={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "hi"}],
                "optional_params": {},
                "litellm_params": {"custom_llm_provider": "openai"},
            },
            response_obj={"error": {"type": "rate_limit_error"}},
        )
        assert spans[0].attributes["error.type"] == "rate_limit_error"

    def test_llm_span_status_ok_on_success(self):
        from opentelemetry.trace.status import StatusCode

        spans = _invoke_write_otel(status="success")
        assert spans[0].status.status_code == StatusCode.OK

    def test_llm_span_status_error_on_failure(self):
        from opentelemetry.trace.status import StatusCode

        spans = _invoke_write_otel(
            status="failure",
            kwargs={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "hi"}],
                "optional_params": {},
                "litellm_params": {"custom_llm_provider": "openai"},
            },
            response_obj={},
        )
        assert spans[0].status.status_code == StatusCode.ERROR

    def test_llm_span_optional_params_absent_when_not_set(self):
        """Empty optional_params don't produce request attributes."""
        spans = _invoke_write_otel(
            kwargs={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "hi"}],
                "optional_params": {},
                "litellm_params": {"custom_llm_provider": "openai"},
            }
        )
        for attr in [
            "gen_ai.request.max_tokens",
            "gen_ai.request.temperature",
            "gen_ai.request.top_p",
            "gen_ai.request.top_k",
            "gen_ai.request.frequency_penalty",
            "gen_ai.request.presence_penalty",
            "gen_ai.request.stop_sequences",
            "gen_ai.request.choice.count",
            "gen_ai.request.seed",
        ]:
            assert attr not in spans[0].attributes, f"{attr} should be absent"


# ===================================================================
# 4. LLM inference content filtering
# ===================================================================


class TestLLMInferenceContentFiltering:
    """Verify content-filtered attributes on LLM inference spans."""

    def _kwargs_with_tools(self):
        return {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [{"type": "function", "function": {"name": "search", "parameters": {}}}],
            "optional_params": {},
            "litellm_params": {"custom_llm_provider": "openai"},
        }

    def _response_with_choices(self):
        return {
            "id": "chatcmpl-123",
            "model": "gpt-4o",
            "usage": {"prompt_tokens": 10, "completion_tokens": 20},
            "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "result"}}],
        }

    def test_tool_definitions_present_when_enabled(self):
        spans = _invoke_write_otel(
            record_content=True,
            kwargs=self._kwargs_with_tools(),
            response_obj=self._response_with_choices(),
        )
        assert "gen_ai.tool.definitions" in spans[0].attributes
        parsed = json.loads(spans[0].attributes["gen_ai.tool.definitions"])
        assert parsed[0]["function"]["name"] == "search"

    def test_tool_definitions_absent_when_disabled(self):
        spans = _invoke_write_otel(
            record_content=False,
            kwargs=self._kwargs_with_tools(),
            response_obj=self._response_with_choices(),
        )
        assert "gen_ai.tool.definitions" not in spans[0].attributes

    def test_input_messages_present_when_enabled(self):
        spans = _invoke_write_otel(record_content=True)
        assert "gen_ai.input.messages" in spans[0].attributes
        parsed = json.loads(spans[0].attributes["gen_ai.input.messages"])
        assert parsed[0]["role"] == "user"

    def test_input_messages_absent_when_disabled(self):
        spans = _invoke_write_otel(record_content=False)
        assert "gen_ai.input.messages" not in spans[0].attributes

    def test_output_messages_present_when_enabled(self):
        spans = _invoke_write_otel(record_content=True)
        assert "gen_ai.output.messages" in spans[0].attributes
        parsed = json.loads(spans[0].attributes["gen_ai.output.messages"])
        assert parsed[0]["role"] == "assistant"

    def test_output_messages_absent_when_disabled(self):
        spans = _invoke_write_otel(record_content=False)
        assert "gen_ai.output.messages" not in spans[0].attributes


# ===================================================================
# 5. TraceLogger silent failure behavior
# ===================================================================


class TestTraceLoggerSilentFailure:
    """Verify TraceLogger._write_otel swallows exceptions silently."""

    def test_write_otel_emits_warning_on_tracer_error(self, caplog):
        """Exception in start_span logs warning but doesn't propagate."""
        import logging

        from exgentic.integrations.litellm.trace_logger import TraceLogger

        tl = TraceLogger()
        tl._tracer = MagicMock()
        tl._tracer.start_span.side_effect = RuntimeError("tracer broken")
        tl._otel_logger = MagicMock()

        mock_ctx = MagicMock()
        mock_ctx.session_id = "s"
        mock_ctx.otel_context = MagicMock()
        mock_ctx.otel_context.trace_id = "0" * 32
        mock_ctx.otel_context.span_id = "0" * 16

        with (
            patch(
                "exgentic.integrations.litellm.trace_logger.get_settings",
                return_value=MagicMock(otel_enabled=True, otel_record_content=False),
            ),
            patch.object(tl, "get_context", return_value=mock_ctx),
            caplog.at_level(logging.WARNING, logger="exgentic.integrations.litellm.trace_logger"),
        ):
            tl._write_otel(
                {"model": "m", "messages": [], "optional_params": {}, "litellm_params": {}},
                {},
                "success",
            )
        assert "TraceLogger._write_otel failed" in caplog.text

    def test_write_otel_emits_warning_on_context_error(self, caplog):
        """Exception in get_context logs warning but doesn't propagate."""
        import logging

        from exgentic.integrations.litellm.trace_logger import TraceLogger

        tl = TraceLogger()
        tl._tracer = MagicMock()
        tl._otel_logger = MagicMock()

        with (
            patch(
                "exgentic.integrations.litellm.trace_logger.get_settings",
                return_value=MagicMock(otel_enabled=True, otel_record_content=False),
            ),
            patch.object(tl, "get_context", side_effect=RuntimeError("context broken")),
            caplog.at_level(logging.WARNING, logger="exgentic.integrations.litellm.trace_logger"),
        ):
            tl._write_otel(
                {"model": "m", "messages": [], "optional_params": {}, "litellm_params": {}},
                {},
                "success",
            )
        assert "TraceLogger._write_otel failed" in caplog.text

    def test_write_otel_noop_when_init_otel_fails_to_set_tracer(self):
        """If _init_otel fails (context missing), subsequent _write_otel calls silently skip."""
        from exgentic.integrations.litellm.trace_logger import TraceLogger

        logger = TraceLogger()
        # _tracer is None and _init_otel won't set it because context is None
        with (
            patch(
                "exgentic.integrations.litellm.trace_logger.get_settings",
                return_value=MagicMock(otel_enabled=True, otel_record_content=False),
            ),
            patch.object(logger, "get_context", return_value=None),
        ):
            # First call: _tracer is None → calls _init_otel → context=None → warns, tracer stays None
            logger._write_otel(
                {"model": "m", "messages": [], "optional_params": {}, "litellm_params": {}},
                {},
                "success",
            )
            assert logger._tracer is None  # tracer was never set

            # Second call: _tracer is still None → calls _init_otel again → same result
            logger._write_otel(
                {"model": "m", "messages": [], "optional_params": {}, "litellm_params": {}},
                {},
                "success",
            )
            assert logger._tracer is None

    def test_init_otel_skips_when_session_id_none(self):
        """_init_otel returns without setting tracer if context.session_id is None."""
        from exgentic.integrations.litellm.trace_logger import TraceLogger

        logger = TraceLogger()
        mock_ctx = MagicMock()
        mock_ctx.session_id = None
        mock_ctx.otel_context = MagicMock()

        with patch.object(logger, "get_context", return_value=mock_ctx):
            logger._init_otel({"model": "m", "litellm_params": {}})
        assert logger._tracer is None

    def test_init_otel_skips_when_otel_context_none(self):
        """_init_otel returns without setting tracer if context.otel_context is None."""
        from exgentic.integrations.litellm.trace_logger import TraceLogger

        logger = TraceLogger()
        mock_ctx = MagicMock()
        mock_ctx.session_id = "sess-001"
        mock_ctx.otel_context = None

        with patch.object(logger, "get_context", return_value=mock_ctx):
            logger._init_otel({"model": "m", "litellm_params": {}})
        assert logger._tracer is None

    def test_init_otel_skips_when_get_context_returns_none(self):
        """_init_otel returns without setting tracer if get_context returns None entirely."""
        from exgentic.integrations.litellm.trace_logger import TraceLogger

        logger = TraceLogger()
        with patch.object(logger, "get_context", return_value=None):
            logger._init_otel({"model": "m", "litellm_params": {}})
        assert logger._tracer is None

    def test_log_success_event_calls_write_otel(self):
        """log_success_event must invoke _write_otel — if this path is broken, no LLM spans."""
        from exgentic.integrations.litellm.trace_logger import TraceLogger

        logger = TraceLogger()
        with patch.object(logger, "_write_otel") as mock_write, patch.object(logger, "_write_row"):
            logger.log_success_event(
                kwargs={"model": "gpt-4"},
                response_obj={"id": "r1"},
                start_time=None,
                end_time=None,
            )
            mock_write.assert_called_once()
            call_kwargs = mock_write.call_args
            # status is passed as a keyword or positional arg
            assert call_kwargs.kwargs.get("status") == "success" or call_kwargs.args[2] == "success"

    def test_init_tracing_from_env_returns_tracer_even_without_endpoint(self, monkeypatch):
        """init_tracing_from_env always returns a Tracer (never None), even if no collector is reachable.

        This means the failure mode is silent: spans are created but lost to an unreachable exporter.
        """
        # Reset the global tracer provider to simulate a fresh subprocess
        from exgentic.utils.otel import init_tracing_from_env
        from opentelemetry import trace as trace_api

        trace_api._TRACER_PROVIDER = None
        trace_api._TRACER_PROVIDER_SET_ONCE._done = False
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
        monkeypatch.delenv("OTEL_EXPORTER_FILE", raising=False)

        try:
            tracer = init_tracing_from_env(service_name="test-no-endpoint")
            assert tracer is not None  # This is the silent failure: tracer exists, but spans go nowhere reachable
        finally:
            # Cleanup: reset provider so we don't pollute other tests
            trace_api._TRACER_PROVIDER = None
            trace_api._TRACER_PROVIDER_SET_ONCE._done = False


# ===================================================================
# 6. prepare_subprocess_env OTEL var forwarding
# ===================================================================


class TestPrepareSubprocessEnvAllowlist:
    """Verify prepare_subprocess_env forwards only allowlisted env vars.

    Exgentic settings (``EXGENTIC_*``) travel via runtime.json — not env
    vars — so the allowlist only needs to cover model-provider
    credentials and OTEL configuration.
    """

    def test_otel_endpoint_forwarded(self, monkeypatch):
        from exgentic.adapters.runners._utils import prepare_subprocess_env

        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
        env = prepare_subprocess_env()
        assert env.get("OTEL_EXPORTER_OTLP_ENDPOINT") == "http://localhost:4318"

    def test_otel_service_name_forwarded(self, monkeypatch):
        from exgentic.adapters.runners._utils import prepare_subprocess_env

        monkeypatch.setenv("OTEL_SERVICE_NAME", "exgentic")
        env = prepare_subprocess_env()
        assert env.get("OTEL_SERVICE_NAME") == "exgentic"

    def test_otel_traces_endpoint_forwarded(self, monkeypatch):
        from exgentic.adapters.runners._utils import prepare_subprocess_env

        monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://localhost:4318/v1/traces")
        env = prepare_subprocess_env()
        assert env.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT") == "http://localhost:4318/v1/traces"

    def test_provider_api_keys_forwarded(self, monkeypatch):
        """Suffix-match catches providers without explicit prefix entries."""
        from exgentic.adapters.runners._utils import prepare_subprocess_env

        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
        monkeypatch.setenv("FIREWORKS_API_KEY", "fw-key")
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://example.com")
        env = prepare_subprocess_env()
        assert env.get("OPENAI_API_KEY") == "sk-openai"
        assert env.get("FIREWORKS_API_KEY") == "fw-key"
        assert env.get("ANTHROPIC_BASE_URL") == "https://example.com"

    def test_exgentic_vars_not_forwarded(self, monkeypatch):
        """EXGENTIC_* settings travel via runtime.json, not env vars."""
        from exgentic.adapters.runners._utils import prepare_subprocess_env

        monkeypatch.setenv("EXGENTIC_OTEL_ENABLED", "true")
        monkeypatch.setenv("EXGENTIC_CACHE_DIR", "/tmp/cache")
        env = prepare_subprocess_env()
        assert "EXGENTIC_OTEL_ENABLED" not in env
        assert "EXGENTIC_CACHE_DIR" not in env

    def test_system_vars_not_forwarded(self):
        from exgentic.adapters.runners._utils import prepare_subprocess_env

        env = prepare_subprocess_env()
        assert "PATH" not in env
        assert "HOME" not in env
        assert "VIRTUAL_ENV" not in env
        assert "SHELL" not in env

    def test_vscode_prefix_blocked(self, monkeypatch):
        from exgentic.adapters.runners._utils import prepare_subprocess_env

        monkeypatch.setenv("VSCODE_PID", "12345")
        env = prepare_subprocess_env()
        assert "VSCODE_PID" not in env


# ===================================================================
# 7. Heritable attribute completeness
# ===================================================================


class TestHeritableAttributeCompleteness:
    """Verify the exact set of heritable attributes matches docs."""

    DOCUMENTED_HERITABLE: ClassVar = {
        "gen_ai.conversation.id",
        "exgentic.session.id",
        "gen_ai.request.model",
        "exgentic.run.id",
        "exgentic.benchmark.slug_name",
        "exgentic.benchmark.subset",
        "exgentic.benchmark.agent.name",
        "exgentic.agent.slug",
    }

    def test_all_heritable_attributes_on_child_span(self, ctx, tmp_path):
        """All 8 documented heritable attributes appear on execute_tool child spans."""
        rc = MockRunConfig()
        rc.model = "gpt-4o"  # Ensure model is set so gen_ai.request.model is heritable
        _, tool_spans, _ = _full_lifecycle_spans(ctx, tmp_path, run_config=rc)
        assert len(tool_spans) >= 1
        child = tool_spans[0]
        for attr in self.DOCUMENTED_HERITABLE:
            assert attr in child.attributes, f"Heritable attribute {attr} missing from child span"

    def test_non_heritable_not_on_child_span(self, ctx, tmp_path):
        """Session-only attributes do NOT appear on child spans."""
        _, tool_spans, _ = _full_lifecycle_spans(ctx, tmp_path)
        child = tool_spans[0]
        non_heritable = [
            "exgentic.session.task_id",
            "exgentic.score.success",
            "exgentic.score",
            "exgentic.session.steps",
        ]
        for attr in non_heritable:
            assert attr not in child.attributes, f"Non-heritable {attr} should not be on child span"


# ===================================================================
# 8. Exporter tests
# ===================================================================


class TestFileSpanExporter:
    """Direct tests for FileSpanExporter."""

    def test_export_writes_jsonl(self, tmp_path):
        from exgentic.utils.otel import FileSpanExporter

        path = tmp_path / "spans.jsonl"
        exporter = FileSpanExporter(path)

        mock_span = MagicMock()
        mock_span.to_json.return_value = '{"name": "test"}'

        result = exporter.export([mock_span])
        assert result == SpanExportResult.SUCCESS
        assert path.read_text() == '{"name": "test"}\n'

    def test_export_appends(self, tmp_path):
        from exgentic.utils.otel import FileSpanExporter

        path = tmp_path / "spans.jsonl"
        exporter = FileSpanExporter(path)

        span1 = MagicMock()
        span1.to_json.return_value = '{"name": "a"}'
        span2 = MagicMock()
        span2.to_json.return_value = '{"name": "b"}'

        exporter.export([span1])
        exporter.export([span2])
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2

    def test_export_creates_parent_dirs(self, tmp_path):
        from exgentic.utils.otel import FileSpanExporter

        path = tmp_path / "deep" / "nested" / "spans.jsonl"
        FileSpanExporter(path)
        assert path.parent.exists()

    def test_export_returns_failure_on_write_error(self, tmp_path):
        from exgentic.utils.otel import FileSpanExporter

        path = tmp_path / "spans.jsonl"
        exporter = FileSpanExporter(path)

        mock_span = MagicMock()
        mock_span.to_json.side_effect = RuntimeError("serialize error")

        result = exporter.export([mock_span])
        assert result == SpanExportResult.FAILURE


class TestPerSessionFileExporter:
    """Direct tests for PerSessionFileExporter."""

    def test_routes_by_session_id(self, tmp_path):
        from exgentic.utils.otel import PerSessionFileExporter

        exporter = PerSessionFileExporter(tmp_path, run_id="run-1")

        span = MagicMock()
        span.attributes = {
            "exgentic.run.id": "run-1",
            "exgentic.session.id": "sess-abc",
        }
        span.to_json.return_value = '{"name": "test"}'

        result = exporter.export([span])
        assert result == SpanExportResult.SUCCESS

        expected = tmp_path / "sessions" / "sess-abc" / "otel_spans.jsonl"
        assert expected.exists()
        assert '{"name": "test"}' in expected.read_text()

    def test_skips_spans_without_session_id(self, tmp_path):
        from exgentic.utils.otel import PerSessionFileExporter

        exporter = PerSessionFileExporter(tmp_path, run_id="run-1")

        span = MagicMock()
        span.attributes = {"exgentic.run.id": "run-1"}  # no session id
        span.to_json.return_value = '{"name": "orphan"}'

        result = exporter.export([span])
        assert result == SpanExportResult.SUCCESS
        # No session dir should be created
        assert not (tmp_path / "sessions").exists()

    def test_returns_failure_on_write_error(self, tmp_path):
        from exgentic.utils.otel import PerSessionFileExporter

        exporter = PerSessionFileExporter(tmp_path, run_id="run-1")

        span = MagicMock()
        span.attributes = {
            "exgentic.run.id": "run-1",
            "exgentic.session.id": "sess-1",
        }
        span.to_json.side_effect = RuntimeError("boom")

        result = exporter.export([span])
        assert result == SpanExportResult.FAILURE

    # ------------------------------------------------------------------
    # Regression: batch-mode write amplification (Exgentic/exgentic#185)
    # ------------------------------------------------------------------

    def test_drops_foreign_run_spans(self, tmp_path):
        """An exporter scoped to run-A must drop spans tagged with run-B.

        Reproduces #185: in batch mode one PerSessionFileExporter is
        registered per RunConfig on the global TracerProvider, so every
        exporter sees every emitted span. Without this filter each span
        would be written once per config -- N-way write amplification plus
        cross-run trace leakage.
        """
        from exgentic.utils.otel import PerSessionFileExporter

        exporter = PerSessionFileExporter(tmp_path, run_id="run-A")

        foreign = MagicMock()
        foreign.attributes = {
            "exgentic.run.id": "run-B",
            "exgentic.session.id": "sess-1",
        }
        foreign.to_json.return_value = '{"name": "from-B"}'

        result = exporter.export([foreign])
        assert result == SpanExportResult.SUCCESS
        # The exporter must NOT touch its run_root for foreign spans.
        assert not (tmp_path / "sessions").exists()

    def test_drops_spans_with_no_run_id(self, tmp_path):
        """Spans missing ``exgentic.run.id`` don't belong to any run, drop them."""
        from exgentic.utils.otel import PerSessionFileExporter

        exporter = PerSessionFileExporter(tmp_path, run_id="run-A")

        untagged = MagicMock()
        untagged.attributes = {"exgentic.session.id": "sess-1"}
        untagged.to_json.return_value = '{"name": "untagged"}'

        exporter.export([untagged])
        assert not (tmp_path / "sessions").exists()

    def test_batch_mode_fanout_has_no_amplification(self, tmp_path):
        """End-to-end reproduction of the batch-mode write amplification bug.

        Simulates two concurrent RunConfigs (A and B) each owning its own
        ``PerSessionFileExporter`` rooted in a distinct directory, both
        attached to the same global ``TracerProvider`` — so both exporters
        see every emitted span, exactly as in batch mode. We emit one span
        belonging to run A and one belonging to run B, and assert each
        ``run_root`` contains exactly its own span — no cross-contamination,
        no duplication.

        Before the fix, both run_roots would contain both spans (2-way writes
        per span). On the unfiltered implementation this test fails because
        ``run-A``'s tree ends up holding ``run-B``'s span and vice versa.
        """
        from exgentic.utils.otel import PerSessionFileExporter

        root_a = tmp_path / "run-A"
        root_b = tmp_path / "run-B"
        exporter_a = PerSessionFileExporter(root_a, run_id="run-A")
        exporter_b = PerSessionFileExporter(root_b, run_id="run-B")

        span_a = MagicMock()
        span_a.attributes = {
            "exgentic.run.id": "run-A",
            "exgentic.session.id": "sess-A1",
        }
        span_a.to_json.return_value = '{"name": "from-A"}'

        span_b = MagicMock()
        span_b.attributes = {
            "exgentic.run.id": "run-B",
            "exgentic.session.id": "sess-B1",
        }
        span_b.to_json.return_value = '{"name": "from-B"}'

        # Global TracerProvider fan-out: every span hits every exporter.
        for exporter in (exporter_a, exporter_b):
            exporter.export([span_a, span_b])

        # Run A's tree only sees run A's span.
        a_files = list(root_a.rglob("otel_spans.jsonl"))
        assert [p.relative_to(root_a) for p in a_files] == [Path("sessions/sess-A1/otel_spans.jsonl")]
        a_content = a_files[0].read_text()
        assert '"from-A"' in a_content
        assert '"from-B"' not in a_content

        # Run B's tree only sees run B's span.
        b_files = list(root_b.rglob("otel_spans.jsonl"))
        assert [p.relative_to(root_b) for p in b_files] == [Path("sessions/sess-B1/otel_spans.jsonl")]
        b_content = b_files[0].read_text()
        assert '"from-B"' in b_content
        assert '"from-A"' not in b_content


# ===================================================================
# 9. Additional attribute conversion edge cases
# ===================================================================


class TestAttributeConversionEdgeCases:
    """Additional edge cases for to_otel_attribute_value and helpers."""

    def test_bool_int_list_rejected(self):
        """Lists mixing bool and int are not homogeneous (bool is subclass of int)."""
        from exgentic.utils.otel import to_otel_attribute_value

        result = to_otel_attribute_value([True, 1, False])
        # Should fall back to JSON since bool+int can't be homogeneous
        assert isinstance(result, str)

    def test_decimal_in_list(self):
        from exgentic.utils.otel import to_otel_attribute_value

        result = to_otel_attribute_value([Decimal("1.5"), Decimal("2.5")])
        assert result == [1.5, 2.5]

    def test_pydantic_model_dump(self):
        from exgentic.utils.otel import to_otel_attribute_value

        obj = MagicMock()
        obj.model_dump.return_value = {"field": "value"}
        # Remove __dict__ to ensure model_dump path is taken in _json_default
        result = to_otel_attribute_value(obj)
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert parsed["field"] == "value"

    def test_object_with_dict_attr(self):
        """Objects with __dict__ but no model_dump/dict go through JSON with _json_default fallback."""
        from exgentic.utils.otel import to_otel_attribute_value

        class Obj:
            def __init__(self):
                self.x = 1
                self.y = "hello"

        result = to_otel_attribute_value(Obj())
        # Should return a string (JSON or str fallback)
        assert isinstance(result, str)

    def test_prefer_json_false_skips_json_for_sequences(self):
        from exgentic.utils.otel import to_otel_attribute_value

        # Mixed list that can't be homogeneous — with prefer_json=False should return str()
        result = to_otel_attribute_value([{"nested": True}], prefer_json=False)
        # Should not attempt JSON, no result from homogeneous, falls to Mapping/object or str
        assert isinstance(result, str)


# ===================================================================
# 10. SessionSpanManager.depth and sort_session_spans
# ===================================================================


class TestSessionSpanManagerDepth:
    """Tests for the depth property."""

    def test_depth_zero_initially(self, span_manager):
        assert span_manager.depth == 0

    def test_depth_increments_on_start(self, span_manager):
        span_manager.start_span("root")
        assert span_manager.depth == 1
        span_manager.start_span("child")
        assert span_manager.depth == 2

    def test_depth_decrements_on_end(self, span_manager):
        span_manager.start_span("root")
        span_manager.start_span("child")
        span_manager.end_current_span()
        assert span_manager.depth == 1
        span_manager.end_current_span()
        assert span_manager.depth == 0


class TestSortSessionSpans:
    """Tests for _sort_session_spans."""

    def test_sorts_by_start_time(self, ctx, tmp_path):
        session_id = "sess-001"
        session_dir = tmp_path / "test-run" / "sessions" / session_id
        session_dir.mkdir(parents=True)
        spans_file = session_dir / "otel_spans.jsonl"

        # Write spans out of order
        spans = [
            {"name": "second", "start_time": "2026-01-01T00:00:02"},
            {"name": "first", "start_time": "2026-01-01T00:00:01"},
            {"name": "third", "start_time": "2026-01-01T00:00:03"},
        ]
        with open(spans_file, "w") as f:
            for s in spans:
                f.write(json.dumps(s) + "\n")

        # Create observer and sort
        with (
            patch("exgentic.observers.handlers.otel.get_benchmark_entries", return_value={}),
            patch("exgentic.observers.handlers.otel.get_agent_entries", return_value={}),
        ):
            from exgentic.observers.handlers.otel import OtelTracingObserver

            obs = OtelTracingObserver()

        obs._sort_session_spans(session_id)

        # Verify sorted
        with open(spans_file) as f:
            sorted_spans = [json.loads(line) for line in f]
        names = [s["name"] for s in sorted_spans]
        assert names == ["first", "second", "third"]

    def test_handles_corrupt_json_gracefully(self, ctx, tmp_path):
        session_id = "sess-001"
        session_dir = tmp_path / "test-run" / "sessions" / session_id
        session_dir.mkdir(parents=True)
        spans_file = session_dir / "otel_spans.jsonl"

        spans_file.write_text("not valid json\n")

        with (
            patch("exgentic.observers.handlers.otel.get_benchmark_entries", return_value={}),
            patch("exgentic.observers.handlers.otel.get_agent_entries", return_value={}),
        ):
            from exgentic.observers.handlers.otel import OtelTracingObserver

            obs = OtelTracingObserver()

        # Should not raise
        obs._sort_session_spans(session_id)


# ===================================================================
# 11. Cost attribute serialization failure
# ===================================================================


class TestCostAttributeFailure:
    """_set_cost_attr handles missing/broken get_cost gracefully."""

    def test_missing_get_cost_does_not_raise(self, ctx, tmp_path):
        """Session without get_cost method doesn't crash on_session_success."""
        session = MockSession()
        # MockSession has no get_cost — verify on_session_error handles it
        _, _, spans = _full_lifecycle_spans(ctx, tmp_path, session=session)
        # Should complete without error
        assert any("session" in s.name for s in spans)

    def test_get_cost_exception_does_not_raise(self, ctx, tmp_path):
        """If get_cost raises, session still completes."""
        agent = MockAgent()
        agent.get_cost = MagicMock(side_effect=RuntimeError("cost broke"))
        _, _, spans = _full_lifecycle_spans(ctx, tmp_path, agent=agent)
        assert any("session" in s.name for s in spans)


# ===================================================================
# 12. TraceLogger content attributes with tool calls
# ===================================================================


class TestTraceLoggerOutputToolCalls:
    """Verify output messages include tool_call parts."""

    def test_output_messages_include_tool_calls(self):
        response_obj = {
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "function": {"name": "search", "arguments": '{"q": "test"}'},
                            }
                        ],
                    },
                }
            ],
        }
        spans = _invoke_write_otel(response_obj=response_obj, record_content=True)
        assert len(spans) == 1
        output = json.loads(spans[0].attributes["gen_ai.output.messages"])
        parts = output[0]["parts"]
        assert any(p["type"] == "tool_call" and p["name"] == "search" for p in parts)

    def test_output_messages_text_and_tool_calls(self):
        response_obj = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": "Here's what I found",
                        "tool_calls": [
                            {
                                "id": "call_2",
                                "function": {"name": "lookup", "arguments": "{}"},
                            }
                        ],
                    },
                }
            ],
        }
        spans = _invoke_write_otel(response_obj=response_obj, record_content=True)
        output = json.loads(spans[0].attributes["gen_ai.output.messages"])
        parts = output[0]["parts"]
        assert parts[0] == {"type": "text", "content": "Here's what I found"}
        assert parts[1]["type"] == "tool_call"
        assert parts[1]["name"] == "lookup"


class TestTraceLoggerErrorTypeBranches:
    """Cover all branches of _set_error_type."""

    def test_error_type_string_fallback(self):
        """Non-dict, non-Exception error info falls back to str()."""
        spans = _invoke_write_otel(
            kwargs={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "hi"}],
                "optional_params": {},
                "litellm_params": {"custom_llm_provider": "openai"},
                "exception": "some string error",
            },
            status="failure",
        )
        assert spans[0].attributes["error.type"] == "some string error"

    def test_no_error_info_skips_error_type(self):
        """When neither exception nor error is present, error.type is not set."""
        spans = _invoke_write_otel(
            kwargs={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "hi"}],
                "optional_params": {},
                "litellm_params": {"custom_llm_provider": "openai"},
            },
            response_obj={},
            status="failure",
        )
        assert "error.type" not in spans[0].attributes


class TestTraceLoggerResponseEdgeCases:
    """Cover response attribute edge cases."""

    def test_empty_usage_skips_token_attrs(self):
        spans = _invoke_write_otel(response_obj={"id": "x", "model": "m", "usage": None, "choices": []})
        attrs = spans[0].attributes
        assert "gen_ai.usage.input_tokens" not in attrs
        assert "gen_ai.usage.output_tokens" not in attrs

    def test_empty_response_obj_skips_response_attrs(self):
        spans = _invoke_write_otel(response_obj={})
        attrs = spans[0].attributes
        assert "gen_ai.response.id" not in attrs
        assert "gen_ai.response.model" not in attrs


# ===================================================================
# 13. Content recording robustness (GitHub issue #140)
# ===================================================================


class TestContentRecordingRobustness:
    """Verify tool parameters and task recording handle edge cases."""

    def test_tool_parameters_recorded_with_dict_arguments(self, ctx, tmp_path):
        """gen_ai.tool.parameters must be recorded when arguments is a plain dict."""
        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        t = provider.get_tracer("test")

        obs = _create_observer_with_tracer(ctx, tmp_path, t)
        obs, session, _settings = _trigger_session_lifecycle(obs, t, ctx, tmp_path, record_content=True)

        # Create action with dict arguments (no model_dump_json)
        mock_action = MagicMock()
        action_item = MagicMock(spec=[])  # empty spec so no auto-attrs
        action_item.name = "browse"
        action_item.id = "act-dict"
        action_item.arguments = {"url": "http://example.com", "timeout": 30}
        mock_action.to_action_list.return_value = [action_item]

        settings_on = MagicMock(otel_record_content=True)
        with (
            patch("exgentic.observers.handlers.otel.get_settings", return_value=settings_on),
            patch("exgentic.observers.handlers.otel.flush_traces"),
        ):
            obs.on_react_success(session, mock_action)
            obs.on_step_success(session, MockObservation())
            obs.on_session_success(session, MockScore(), MockAgent())

        spans = exporter.get_finished_spans()
        tool_spans = [s for s in spans if "execute_tool browse" in s.name]
        assert len(tool_spans) >= 1, "Should have a browse execute_tool span"
        attrs = tool_spans[0].attributes or {}
        assert "gen_ai.tool.parameters" in attrs, "Dict arguments should be serialized as tool parameters"
        params = json.loads(attrs["gen_ai.tool.parameters"])
        assert params["url"] == "http://example.com"
        assert params["timeout"] == 30

    def test_tool_parameters_recorded_with_pydantic_model(self, ctx, tmp_path):
        """gen_ai.tool.parameters must be recorded when arguments is a Pydantic BaseModel."""
        from pydantic import BaseModel

        class BrowseArgs(BaseModel):
            url: str
            timeout: int = 30

        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        t = provider.get_tracer("test")

        obs = _create_observer_with_tracer(ctx, tmp_path, t)
        obs, session, _settings = _trigger_session_lifecycle(obs, t, ctx, tmp_path, record_content=True)

        mock_action = MagicMock()
        action_item = MagicMock(spec=[])
        action_item.name = "browse"
        action_item.id = "act-pydantic"
        action_item.arguments = BrowseArgs(url="http://example.com", timeout=60)
        mock_action.to_action_list.return_value = [action_item]

        settings_on = MagicMock(otel_record_content=True)
        with (
            patch("exgentic.observers.handlers.otel.get_settings", return_value=settings_on),
            patch("exgentic.observers.handlers.otel.flush_traces"),
        ):
            obs.on_react_success(session, mock_action)
            obs.on_step_success(session, MockObservation())
            obs.on_session_success(session, MockScore(), MockAgent())

        spans = exporter.get_finished_spans()
        tool_spans = [s for s in spans if "execute_tool browse" in s.name]
        assert len(tool_spans) >= 1
        attrs = tool_spans[0].attributes or {}
        assert "gen_ai.tool.parameters" in attrs, "Pydantic model arguments should be serialized"
        params = json.loads(attrs["gen_ai.tool.parameters"])
        assert params["url"] == "http://example.com"
        assert params["timeout"] == 60

    def test_task_not_recorded_when_session_has_no_task_attr(self, ctx, tmp_path):
        """on_session_creation must not fail when session has no task attribute."""
        from exgentic.observers.handlers.otel import OtelTracingObserver, SessionSpanManager

        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        t = provider.get_tracer("test")

        session = MagicMock()
        session.session_id = "sess-notask"
        session.task_id = "task-notask"
        session.actions = [MockAction()]
        session.context = {}
        # Deliberately remove 'task' attribute
        del session.task

        settings_on = MagicMock(otel_record_content=True)

        original_init = SessionSpanManager.__init__

        def patched_init(self, session_id, session_root_path, tracer=t):
            original_init(self, session_id, session_root_path, tracer=tracer)

        session_root = tmp_path / "test-run" / "sessions" / "sess-notask"
        session_root.mkdir(parents=True, exist_ok=True)

        with (
            patch("exgentic.observers.handlers.otel.get_benchmark_entries", return_value={}),
            patch("exgentic.observers.handlers.otel.get_agent_entries", return_value={}),
            patch("exgentic.observers.handlers.otel.get_settings", return_value=settings_on),
            patch.object(SessionSpanManager, "__init__", patched_init),
            patch(
                "exgentic.observers.handlers.otel.to_otel_attribute_value",
                side_effect=lambda v: str(v) if v is not None else None,
            ),
        ):
            obs = OtelTracingObserver()
            run_config = MockRunConfig()
            run_config.subset = "test_subset"
            obs.on_run_start(run_config)
            # This should NOT raise even though session has no 'task'
            obs.on_session_creation(session)

        # Verify span was created without task attribute
        spans = exporter.get_finished_spans()
        for s in spans:
            assert "exgentic.session.task" not in (s.attributes or {})

    def test_serialize_to_json_dict(self):
        """_serialize_to_json handles plain dicts."""
        from exgentic.observers.handlers.otel import _serialize_to_json

        result = _serialize_to_json({"key": "value", "num": 42})
        parsed = json.loads(result)
        assert parsed == {"key": "value", "num": 42}

    def test_serialize_to_json_pydantic(self):
        """_serialize_to_json handles Pydantic models."""
        from exgentic.observers.handlers.otel import _serialize_to_json
        from pydantic import BaseModel

        class MyModel(BaseModel):
            name: str
            count: int

        result = _serialize_to_json(MyModel(name="test", count=5))
        parsed = json.loads(result)
        assert parsed == {"name": "test", "count": 5}
