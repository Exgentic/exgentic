# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Verify that core types round-trip through cloudpickle via model_dump/model_validate."""

from __future__ import annotations

import cloudpickle as cp
from exgentic.core.types import (
    ActionType,
    BenchmarkResults,
    ModelSettings,
    SessionIndex,
    SessionScore,
    SingleAction,
    SingleObservation,
)


def test_single_observation_roundtrip():
    obs = SingleObservation(result={"key": "value"}, actions=[])
    restored = cp.loads(cp.dumps(obs))
    assert restored.result == {"key": "value"}


def test_session_score_roundtrip():
    score = SessionScore(score=0.95, success=True, is_finished=True)
    restored = cp.loads(cp.dumps(score))
    assert restored.score == 0.95
    assert restored.success is True


def test_action_type_with_class_ref_roundtrip():
    at = ActionType(name="test", description="desc", cls=SingleAction)
    restored = cp.loads(cp.dumps(at))
    assert restored.name == "test"
    assert restored.cls is SingleAction


def test_benchmark_results_roundtrip():
    br = BenchmarkResults(benchmark_name="gsm8k", total_tasks=100, score=0.85)
    restored = cp.loads(cp.dumps(br))
    assert restored.score == 0.85


def test_model_settings_roundtrip():
    ms = ModelSettings(temperature=0.7, max_tokens=1024)
    restored = cp.loads(cp.dumps(ms))
    assert restored.temperature == 0.7
    assert restored.max_tokens == 1024


def test_session_index_roundtrip():
    si = SessionIndex(session_id="abc", task_id="task-1")
    restored = cp.loads(cp.dumps(si))
    assert restored.session_id == "abc"


def test_pickle_uses_model_dump_not_internals():
    """Verify the pickle payload contains model_dump dict, not pydantic internals."""
    score = SessionScore(score=1.0, success=True, is_finished=True)
    data = cp.dumps(score)
    # Should contain _restore function reference, not __pydantic_fields_set__
    assert b"__pydantic_fields_set__" not in data
