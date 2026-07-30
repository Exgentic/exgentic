# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Tests for the GAIA benchmark adapter.

The GAIA dataset (``gaia-benchmark/GAIA``) is gated on the HuggingFace Hub, so
these tests never touch the network: dataset access is patched out and the
adapter's own logic (answer normalization, session lifecycle, scoring and
aggregation) is exercised directly.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import pytest
from exgentic.benchmarks.gaia.gaia_benchmark import (
    GAIA_TASK_COUNTS,
    GAIABenchmark,
    GAIAEvaluator,
    GAIASession,
    exact_match,
    normalize_answer,
)
from exgentic.core.actions import build_action

_MODULE = "exgentic.benchmarks.gaia.gaia_benchmark"


# ---------------------------------------------------------------------------
# Answer normalization / exact match
# ---------------------------------------------------------------------------


class TestNormalizeAnswer:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("The Answer", "answer"),
            ("  extra   spaces  ", "extra spaces"),
            ("Paris, France", "paris france"),
            ("A dog.", "dog"),
            ("42", "42"),
        ],
    )
    def test_normalization(self, raw: str, expected: str) -> None:
        assert normalize_answer(raw) == expected

    @pytest.mark.parametrize(
        ("prediction", "gold"),
        [
            ("The Louvre", "louvre"),
            ("3,4", "34"),
            ("Yes.", "yes"),
        ],
    )
    def test_exact_match_true(self, prediction: str, gold: str) -> None:
        assert exact_match(prediction, gold) is True

    @pytest.mark.parametrize(
        ("prediction", "gold"),
        [
            ("Paris", "London"),
            ("42", "43"),
            ("", "something"),
        ],
    )
    def test_exact_match_false(self, prediction: str, gold: str) -> None:
        assert exact_match(prediction, gold) is False


def test_task_counts_are_self_consistent() -> None:
    """Per-level counts must sum to the ``2023_all`` count."""
    per_level = GAIA_TASK_COUNTS["2023_level1"] + GAIA_TASK_COUNTS["2023_level2"] + GAIA_TASK_COUNTS["2023_level3"]
    assert per_level == GAIA_TASK_COUNTS["2023_all"]


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


_ROW = {
    "Question": "How many studio albums were published by Mercedes Sosa between 2000 and 2009?",
    "Final answer": "3",
    "Level": 1,
}


@pytest.fixture()
def session(tmp_path: Path):
    def _make(row: dict[str, Any] | None = None, session_id: str = "gaia-test") -> GAIASession:
        with mock.patch(f"{_MODULE}._load_gaia_row", return_value=dict(row or _ROW)):
            sess = GAIASession(task_id="0", subset="2023_level1", session_id=session_id)
        sess._paths = SimpleNamespace(  # type: ignore[attr-defined]
            benchmark_results=tmp_path / session_id / "results.json",
            benchmark_config=tmp_path / session_id / "config.json",
            session_log=tmp_path / session_id / "session.log",
            manifest=tmp_path / session_id / "manifest.json",
        )
        return sess

    return _make


class TestGAIASession:
    def test_task_contains_the_real_question(self, session) -> None:
        sess = session()
        assert _ROW["Question"] in sess.task
        assert "submit" in sess.task

    def test_context_only_exposes_level(self, session) -> None:
        assert session().context == {"level": 1}

    def test_context_omits_missing_level(self, session) -> None:
        sess = session({"Question": "q", "Final answer": "a"})
        assert sess.context == {}

    def test_single_submit_action(self, session) -> None:
        actions = session().actions
        assert [a.name for a in actions] == ["submit"]
        assert actions[0].is_finish is True

    def test_unfinished_session_scores_zero_and_not_finished(self, session) -> None:
        score = session().score()
        assert score.score == 0.0
        assert score.success is False
        assert score.is_finished is False

    def test_correct_answer_scores_one(self, session) -> None:
        sess = session()
        sess.step(build_action(sess.actions[0], {"answer": "3"}))
        assert sess.done() is True
        score = sess.score()
        assert score.score == 1.0
        assert score.success is True
        assert score.is_finished is True

    def test_answer_is_matched_after_normalization(self, session) -> None:
        sess = session({"Question": "q", "Final answer": "the Louvre", "Level": 2})
        sess.step(build_action(sess.actions[0], {"answer": "Louvre."}))
        assert sess.score().score == 1.0

    def test_wrong_answer_scores_zero_but_is_finished(self, session) -> None:
        sess = session()
        sess.step(build_action(sess.actions[0], {"answer": "7"}))
        score = sess.score()
        assert score.score == 0.0
        assert score.success is False
        assert score.is_finished is True

    def test_step_after_done_is_a_noop(self, session) -> None:
        sess = session()
        sess.step(build_action(sess.actions[0], {"answer": "3"}))
        assert sess.step(build_action(sess.actions[0], {"answer": "999"})) is None
        assert sess.score().score == 1.0

    def test_none_action_finishes_without_answer(self, session) -> None:
        sess = session()
        assert sess.step(None) is None
        assert sess.done() is True
        assert sess.score().is_finished is False


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


@pytest.fixture()
def run_context(tmp_path: Path):
    """Aggregation writes to the run log, which requires an active run context."""
    from exgentic.core.context import run_scope

    with run_scope(run_id="gaia-test-run", output_dir=str(tmp_path / "outputs")) as ctx:
        yield ctx


def _paths(result_path: Path, session_id: str) -> Any:
    return SimpleNamespace(benchmark_results=result_path, session_id=session_id)


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class TestGAIAEvaluator:
    @pytest.mark.parametrize("subset", sorted(GAIA_TASK_COUNTS))
    def test_list_tasks_uses_known_counts_without_network(self, subset: str) -> None:
        tasks = GAIAEvaluator(subset=subset).list_tasks()
        assert len(tasks) == GAIA_TASK_COUNTS[subset]
        assert tasks[0] == "0"

    def test_aggregate_computes_overall_and_per_level_metrics(self, run_context, tmp_path: Path) -> None:
        a, b, c = (tmp_path / n / "results.json" for n in ("a", "b", "c"))
        _write(a, {"score": 1.0, "success": True, "level": 1})
        _write(b, {"score": 0.0, "success": False, "level": 1})
        _write(c, {"score": 1.0, "success": True, "level": 2})

        ev = GAIAEvaluator(subset="2023_all")
        sessions = [SimpleNamespace(session_id=n) for n in ("a", "b", "c")]
        with mock.patch.object(
            type(ev),
            "get_sessions_paths",
            return_value=[_paths(a, "a"), _paths(b, "b"), _paths(c, "c")],
        ):
            results = ev.aggregate_sessions(sessions)

        assert results.benchmark_name == "gaia"
        assert results.total_tasks == 3
        assert results.score == pytest.approx(2 / 3)
        assert results.metrics["level_1_accuracy"] == pytest.approx(0.5)
        assert results.metrics["level_1_total"] == 2
        assert results.metrics["level_2_accuracy"] == pytest.approx(1.0)
        assert results.metrics["level_2_total"] == 1

    def test_aggregate_raises_on_missing_result_file(self, run_context, tmp_path: Path) -> None:
        missing = tmp_path / "gone" / "results.json"
        ev = GAIAEvaluator()
        with mock.patch.object(type(ev), "get_sessions_paths", return_value=[_paths(missing, "gone")]):
            with pytest.raises(FileNotFoundError, match="Missing benchmark result"):
                ev.aggregate_sessions([SimpleNamespace(session_id="gone")])


# ---------------------------------------------------------------------------
# Benchmark config
# ---------------------------------------------------------------------------


class TestGAIABenchmark:
    def test_defaults(self) -> None:
        bench = GAIABenchmark()
        assert bench.subset == "2023_all"
        assert GAIABenchmark.slug_name == "gaia"

    def test_subset_is_threaded_to_evaluator_and_session(self) -> None:
        bench = GAIABenchmark(subset="2023_level3")
        assert bench._get_evaluator_kwargs() == {"subset": "2023_level3"}
        assert bench._get_session_kwargs() == {"subset": "2023_level3"}

    def test_invalid_subset_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            GAIABenchmark(subset="2024_all")

    def test_registry_entry_resolves(self) -> None:
        from exgentic.interfaces.registry import BENCHMARKS

        entry = BENCHMARKS["gaia"]
        assert entry.module == "exgentic.benchmarks.gaia.gaia_benchmark"
        assert entry.attr == "GAIABenchmark"
        assert set(entry.subsets) == set(GAIA_TASK_COUNTS)
