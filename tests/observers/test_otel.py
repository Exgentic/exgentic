# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Comprehensive tests for OTEL tracing: observer, SessionSpanManager, TraceLogger, context propagation."""

from __future__ import annotations

import contextvars
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
            "exgentic.integrations.litellm.trace_logger._otel_enabled",
            return_value=True,
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
            "exgentic.integrations.litellm.trace_logger._otel_enabled",
            return_value=False,
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
    """_configure_callbacks adds loggers to litellm.callbacks correctly."""

    def test_adds_sync_and_async(self):
        import litellm
        from exgentic.integrations.litellm.config import _configure_callbacks
        from exgentic.integrations.litellm.trace_logger import AsyncTraceLogger, SyncTraceLogger

        litellm.callbacks = []
        _configure_callbacks()

        assert sum(1 for cb in litellm.callbacks if isinstance(cb, SyncTraceLogger)) == 1
        assert sum(1 for cb in litellm.callbacks if isinstance(cb, AsyncTraceLogger)) == 1

    def test_idempotent(self):
        import litellm
        from exgentic.integrations.litellm.config import _configure_callbacks
        from exgentic.integrations.litellm.trace_logger import AsyncTraceLogger, SyncTraceLogger

        litellm.callbacks = []
        _configure_callbacks()
        _configure_callbacks()

        assert sum(1 for cb in litellm.callbacks if isinstance(cb, SyncTraceLogger)) == 1
        assert sum(1 for cb in litellm.callbacks if isinstance(cb, AsyncTraceLogger)) == 1


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

    def test_metadata_path_preserves_trace_linkage(self, tmp_path):
        """Simulate OpenAI agent: otel context injected via litellm metadata → TraceLogger links correctly."""
        from exgentic.integrations.litellm.trace_logger import TraceLogger
        from opentelemetry import trace as trace_api

        # Parent trace/span ids
        parent_trace_id = "ab" * 16  # 32 hex chars
        parent_span_id = "cd" * 8  # 16 hex chars

        # Simulate what OpenAI agent does: inject into litellm metadata
        kwargs = {
            "model": "gpt-4o",
            "litellm_metadata": {
                "exgentic_ctx_run_id": "run-1",
                "exgentic_ctx_output_dir": str(tmp_path),
                "exgentic_ctx_cache_dir": str(tmp_path),
                "exgentic_ctx_session_id": "sess-1",
                "exgentic_ctx_role": "agent",
                "exgentic_ctx_otel_trace_id": parent_trace_id,
                "exgentic_ctx_otel_span_id": parent_span_id,
            },
        }

        logger = TraceLogger()
        logger._otel_logger = MagicMock()

        # _metadata_context should reconstruct Context with OtelContext
        metadata_ctx = logger._metadata_context(kwargs)
        assert metadata_ctx is not None
        assert metadata_ctx.otel_context is not None
        assert metadata_ctx.otel_context.trace_id == parent_trace_id
        assert metadata_ctx.otel_context.span_id == parent_span_id

        # get_context should prefer metadata over ContextVar
        resolved = logger.get_context(kwargs)
        assert resolved.otel_context.trace_id == parent_trace_id

        # _get_parent_context should create correct NonRecordingSpan
        with patch.object(logger, "get_context", return_value=metadata_ctx):
            parent_otel_ctx = logger._get_parent_context(kwargs)

        span = trace_api.get_current_span(parent_otel_ctx)
        span_ctx = span.get_span_context()
        assert format(span_ctx.trace_id, "032x") == parent_trace_id
        assert format(span_ctx.span_id, "016x") == parent_span_id

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
        """inject_exgentic_env sets EXGENTIC_RUNTIME_FILE when session context exists.

        OTEL vars are persisted in runtime.json on disk; the child reads them
        via init_context() using the runtime file path.
        """
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
        inject_exgentic_env(env)

        expected_file = str(tmp_path / "run-1" / "sessions" / "sess-1" / "runtime.json")
        assert (
            env.get("EXGENTIC_RUNTIME_FILE") == expected_file
        ), "inject_exgentic_env must set EXGENTIC_RUNTIME_FILE for session contexts"

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
        """CustomLogger subclasses must be in litellm.callbacks, not success_callback."""
        import litellm
        from exgentic.integrations.litellm.config import _configure_callbacks
        from exgentic.integrations.litellm.trace_logger import AsyncTraceLogger, SyncTraceLogger

        # Simulate fresh child process (no callbacks)
        litellm.callbacks = []
        litellm.success_callback = []
        litellm.failure_callback = []

        _configure_callbacks()

        # Must be in callbacks (not success_callback)
        assert any(isinstance(cb, SyncTraceLogger) for cb in litellm.callbacks)
        assert any(isinstance(cb, AsyncTraceLogger) for cb in litellm.callbacks)
        # Must NOT be in success_callback
        assert not any(isinstance(cb, SyncTraceLogger) for cb in litellm.success_callback)
        assert not any(isinstance(cb, AsyncTraceLogger) for cb in litellm.success_callback)

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

    def test_inject_exgentic_env_includes_otel_context(self, monkeypatch):
        """inject_exgentic_env sets EXGENTIC_RUNTIME_FILE when session context exists.

        OTEL context is propagated via runtime.json on disk; the child reads
        it through init_context() using the runtime file path.
        """
        monkeypatch.delenv("EXGENTIC_RUNTIME_FILE", raising=False)
        from exgentic.core.context import Context, OtelContext, set_context

        ctx = Context(
            session_id="s1",
            run_id="r1",
            output_dir="/tmp/out",
            cache_dir="/tmp/cache",
            otel_context=OtelContext(trace_id="abc123", span_id="def456"),
        )
        set_context(ctx)

        env: dict[str, str] = {}
        from exgentic.adapters.runners._utils import inject_exgentic_env

        inject_exgentic_env(env)
        assert env.get("EXGENTIC_RUNTIME_FILE") == "/tmp/out/r1/sessions/s1/runtime.json"

    def test_inject_exgentic_env_propagates_settings(self, monkeypatch):
        """inject_exgentic_env sets EXGENTIC_RUNTIME_FILE; settings are in runtime.json."""
        monkeypatch.delenv("EXGENTIC_RUNTIME_FILE", raising=False)
        from exgentic.core.context import Context, set_context

        ctx = Context(session_id="s", run_id="r", output_dir="/tmp/o", cache_dir="/tmp/c")
        set_context(ctx)

        env: dict[str, str] = {}
        from exgentic.adapters.runners._utils import inject_exgentic_env

        inject_exgentic_env(env)
        # With session context, settings propagate via runtime.json on disk,
        # not individual env vars.  inject_exgentic_env just sets the dir.
        assert env.get("EXGENTIC_RUNTIME_FILE") == "/tmp/o/r/sessions/s/runtime.json"

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

    Instead of spinning up real runners, we verify the source-level contracts:
    - Which functions each runner calls for env/context propagation
    - That the propagation path is complete (no silent drops)
    """

    # -- Thread runner: copies ContextVar --

    def test_thread_runner_copies_context_vars(self):
        """ThreadTransport uses contextvars.copy_context() which copies OTEL ContextVar."""
        import inspect

        from exgentic.adapters.runners.thread import ThreadTransport

        source = inspect.getsource(ThreadTransport)
        assert "copy_context" in source, "ThreadTransport must use contextvars.copy_context() to propagate ContextVars"

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
        # Verify the copy has the same context
        child_ctx = copied[_CONTEXT]
        assert child_ctx.otel_context is not None
        assert child_ctx.otel_context.trace_id == "thread-trace"

    # -- Process runner: uses EXGENTIC_RUNTIME_FILE --

    def test_process_runner_sets_session_dir(self):
        """PipeTransport.start() must propagate runtime env to the child."""
        import inspect

        from exgentic.adapters.runners.process import PipeTransport

        source = inspect.getsource(PipeTransport.start)
        assert (
            "get_runtime_env" in source
        ), "PipeTransport.start must call get_runtime_env() to propagate context to child"

    def test_process_worker_restores_context(self):
        """Process worker must call try_init_context to restore context."""
        import inspect

        from exgentic.adapters.runners import process

        source = inspect.getsource(process._worker)
        assert "try_init_context" in source, "Process _worker must call try_init_context() to restore context"

    # -- Service runner: sets context fallback --

    def test_service_runner_sets_context_fallback(self):
        """ServiceRunner.start() must set_context_fallback for uvicorn threads."""
        import inspect

        from exgentic.adapters.runners.service import ServiceRunner

        source = inspect.getsource(ServiceRunner.start)
        assert (
            "set_context_fallback" in source
        ), "ServiceRunner must call set_context_fallback() so uvicorn threads see OTEL context"

    # -- Venv runner: injects env vars --

    def test_venv_runner_calls_inject_exgentic_env(self):
        """VenvRunner.start() must call inject_exgentic_env."""
        import inspect

        from exgentic.adapters.runners.venv import VenvRunner

        source = inspect.getsource(VenvRunner.start)
        assert (
            "inject_exgentic_env" in source
        ), "VenvRunner.start must call inject_exgentic_env() to propagate settings+context"

    def test_venv_runner_uses_exgentic_serve(self):
        """VenvRunner must launch child via 'exgentic serve' which calls init_context."""
        import inspect

        from exgentic.adapters.runners.venv import VenvRunner

        source = inspect.getsource(VenvRunner.start)
        assert "serve" in source, "VenvRunner must use 'exgentic serve' CLI which bootstraps context from env"

    # -- Docker runner: injects env vars into container --

    def test_docker_runner_calls_inject_exgentic_env(self):
        """DockerRunner.start() must call inject_exgentic_env."""
        import inspect

        from exgentic.adapters.runners.docker import DockerRunner

        source = inspect.getsource(DockerRunner.start)
        assert (
            "inject_exgentic_env" in source
        ), "DockerRunner.start must call inject_exgentic_env() to propagate settings+context"

    def test_docker_runner_passes_env_to_container(self):
        """DockerRunner must pass env vars via -e flags to docker run."""
        import inspect

        from exgentic.adapters.runners.docker import DockerRunner

        source = inspect.getsource(DockerRunner.start)
        # Must iterate env and add -e flags
        assert '"-e"' in source, "DockerRunner.start must pass env vars with -e to docker run"

    def test_docker_runner_uses_exgentic_serve(self):
        """DockerRunner must launch child via 'exgentic serve'."""
        import inspect

        from exgentic.adapters.runners.docker import DockerRunner

        source = inspect.getsource(DockerRunner.start)
        assert "serve" in source, "DockerRunner must use 'exgentic serve' CLI which bootstraps context from env"

    # -- serve CLI: bootstraps context from env --

    def test_serve_cmd_calls_try_init_context(self):
        """The 'exgentic serve' command must call try_init_context on startup."""
        import inspect

        from exgentic.interfaces.cli.commands.serve import serve_cmd

        # serve_cmd is a Click command; get the underlying function
        source = inspect.getsource(serve_cmd.callback)
        assert "try_init_context" in source, "serve_cmd must call try_init_context() to restore OTEL context in child"

    # -- inject_exgentic_env: propagates all settings --

    def test_inject_exgentic_env_sets_runtime_env(self):
        """inject_exgentic_env must call get_runtime_env for disk-based propagation."""
        import inspect

        from exgentic.adapters.runners._utils import inject_exgentic_env

        source = inspect.getsource(inject_exgentic_env)
        assert "get_runtime_env" in source, "inject_exgentic_env must call get_runtime_env() for disk-based propagation"

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
