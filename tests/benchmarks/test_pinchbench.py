# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Tests for the PinchBench benchmark adapter.

PinchBench tasks live in an external skill repo
(https://github.com/pinchbench/skill).  These tests build a small, in-tree
fixture repo with the same layout so nothing here needs the network, a clone,
or an LLM judge (the judge call is patched out).
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import pytest
from exgentic.benchmarks.pinchbench.pinchbench_benchmark import (
    PinchBenchBenchmark,
    PinchBenchEvaluator,
    PinchBenchSession,
    _default_skill_repo_path,
    _extract_grading_code,
    _extract_grading_criteria,
    _judge_max_tokens,
    _load_task_index,
    _parse_judge_json,
    _parse_task_file,
    _run_llm_judge,
)
from exgentic.core.actions import build_action

_MODULE = "exgentic.benchmarks.pinchbench.pinchbench_benchmark"


# ---------------------------------------------------------------------------
# Fixture skill repo
# ---------------------------------------------------------------------------

_AUTOMATED_TASK = """\
---
id: task_echo
name: Echo Check
category: productivity
grading_type: automated
timeout_seconds: 60
workspace_files: []
---

## Prompt

Say "ready" to confirm you can respond.

## Expected Behavior

The agent replies with a confirmation.

## Grading Criteria

- [ ] Agent responded
- [ ] Response mentions ready

## Automated Checks

```python
def grade(transcript: list, workspace_path: str) -> dict:
    text = ""
    for entry in transcript:
        if entry.get("type") == "message":
            for block in entry.get("message", {}).get("content", []):
                text += block.get("text", "")
    return {
        "responded": 1.0 if text.strip() else 0.0,
        "says_ready": 1.0 if "ready" in text.lower() else 0.0,
    }
```
"""

_JUDGE_TASK = """\
---
id: task_blurb
name: Write A Blurb
category: writing
grading_type: llm_judge
timeout_seconds: 120
---

## Prompt

Write a one sentence product blurb.

## Expected Behavior

A short, punchy sentence.

## LLM Judge Rubric

- clarity
- brevity
"""

_HYBRID_TASK = """\
---
id: task_report
name: Workspace Report
category: analysis
grading_type: hybrid
timeout_seconds: 120
grading_weights:
  automated: 0.75
  llm_judge: 0.25
workspace_files:
  - source: cities.csv
    dest: cities.csv
---

## Prompt

Summarise cities.csv.

## Expected Behavior

A summary of the provided CSV.

## Automated Checks

```python
def grade(transcript: list, workspace_path: str) -> dict:
    from pathlib import Path

    return {"input_present": 1.0 if (Path(workspace_path) / "cities.csv").exists() else 0.0}
```

## LLM Judge Rubric

- accuracy
"""

_MANIFEST = """\
run_first:
  - task_echo

core:
  - task_echo

categories:
  productivity:
    - task_echo
  writing:
    - task_blurb
  analysis:
    - task_report
"""


@pytest.fixture()
def skill_dir(tmp_path: Path) -> Path:
    root = tmp_path / "skill"
    tasks = root / "tasks"
    tasks.mkdir(parents=True)
    (tasks / "manifest.yaml").write_text(_MANIFEST, encoding="utf-8")
    (tasks / "task_echo.md").write_text(_AUTOMATED_TASK, encoding="utf-8")
    (tasks / "task_blurb.md").write_text(_JUDGE_TASK, encoding="utf-8")
    (tasks / "task_report.md").write_text(_HYBRID_TASK, encoding="utf-8")
    assets = root / "assets"
    assets.mkdir()
    (assets / "cities.csv").write_text("city,pop\nlisbon,545000\n", encoding="utf-8")
    return root


def _session(skill_dir: Path, index: int, tmp_path: Path, session_id: str) -> PinchBenchSession:
    sess = PinchBenchSession(task_id=str(index), skill_dir=str(skill_dir), session_id=session_id)
    sess._paths = SimpleNamespace(  # type: ignore[attr-defined]
        benchmark_results=tmp_path / session_id / "results.json",
        benchmark_config=tmp_path / session_id / "config.json",
        session_log=tmp_path / session_id / "session.log",
        manifest=tmp_path / session_id / "manifest.json",
    )
    return sess


# ---------------------------------------------------------------------------
# Task file parsing
# ---------------------------------------------------------------------------


class TestTaskParsing:
    def test_parse_task_file_splits_frontmatter_and_sections(self, skill_dir: Path) -> None:
        parsed = _parse_task_file(skill_dir / "tasks" / "task_echo.md")
        assert parsed["metadata"]["id"] == "task_echo"
        assert parsed["metadata"]["grading_type"] == "automated"
        assert parsed["Prompt"].startswith('Say "ready"')
        assert "def grade" in parsed["Automated Checks"]

    def test_parse_task_file_rejects_missing_frontmatter(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.md"
        bad.write_text("## Prompt\n\nno frontmatter\n", encoding="utf-8")
        with pytest.raises(ValueError, match="No YAML frontmatter"):
            _parse_task_file(bad)

    def test_extract_grading_code(self) -> None:
        section = "text\n\n```python\ndef grade(t, w):\n    return {}\n```\n"
        assert _extract_grading_code(section).startswith("def grade")

    def test_extract_grading_code_returns_empty_when_absent(self) -> None:
        assert _extract_grading_code("no code here") == ""

    def test_extract_grading_criteria(self) -> None:
        text = textwrap.dedent(
            """\
            - [ ] first item
            - [x] second item
            - not a checkbox
            """
        )
        assert _extract_grading_criteria(text) == ["first item", "second item"]


class TestTaskIndex:
    def test_index_follows_manifest_categories(self, skill_dir: Path) -> None:
        index = _load_task_index(skill_dir)
        assert [t["task_id"] for t in index] == ["task_echo", "task_blurb", "task_report"]
        assert [t["category"] for t in index] == ["productivity", "writing", "analysis"]

    def test_index_ignores_run_first_and_core_lists(self, skill_dir: Path) -> None:
        """``run_first``/``core`` repeat task ids; they must not duplicate entries."""
        index = _load_task_index(skill_dir)
        assert len(index) == len({t["task_id"] for t in index})

    def test_index_falls_back_to_glob_without_manifest(self, skill_dir: Path) -> None:
        (skill_dir / "tasks" / "manifest.yaml").unlink()
        index = _load_task_index(skill_dir)
        assert {t["task_id"] for t in index} == {"task_echo", "task_blurb", "task_report"}

    def test_index_is_empty_for_missing_skill_dir(self, tmp_path: Path) -> None:
        assert _load_task_index(tmp_path / "nope") == []

    def test_default_skill_dir_honours_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PINCHBENCH_SKILL_DIR", "/custom/skill")
        assert _default_skill_repo_path() == Path("/custom/skill")
        monkeypatch.delenv("PINCHBENCH_SKILL_DIR")
        assert _default_skill_repo_path() == Path("/tmp/pinchbench-source")


# ---------------------------------------------------------------------------
# Judge JSON parsing
# ---------------------------------------------------------------------------


class TestParseJudgeJson:
    def test_plain_json(self) -> None:
        assert _parse_judge_json('{"total": 0.5}') == {"total": 0.5}

    def test_fenced_json(self) -> None:
        assert _parse_judge_json('```json\n{"total": 1.0}\n```') == {"total": 1.0}

    def test_json_embedded_in_prose(self) -> None:
        raw = 'Here you go:\n{"scores": {"clarity": 0.8}, "total": 0.8}\nHope that helps.'
        assert _parse_judge_json(raw)["total"] == 0.8

    def test_unparsable_returns_empty_dict(self) -> None:
        assert _parse_judge_json("not json at all") == {}

    def test_truncated_json_is_not_parsable(self) -> None:
        """What a reasoning model emits when it runs out of budget mid-object."""
        assert _parse_judge_json('```json\n{\n  "scores": {\n    "Criterion 1": 0') == {}


# ---------------------------------------------------------------------------
# LLM judge
# ---------------------------------------------------------------------------


def _judge_response(content: str, finish_reason: str = "stop") -> Any:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason=finish_reason)]
    )


class TestJudgeMaxTokens:
    def test_default_budget_fits_a_reasoning_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("PINCHBENCH_JUDGE_MAX_TOKENS", raising=False)
        assert _judge_max_tokens() == 4096

    @pytest.mark.parametrize("raw", ["not-a-number", "0", "-5", ""])
    def test_invalid_override_falls_back_to_default(self, monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
        monkeypatch.setenv("PINCHBENCH_JUDGE_MAX_TOKENS", raw)
        assert _judge_max_tokens() == 4096

    def test_valid_override_is_used(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PINCHBENCH_JUDGE_MAX_TOKENS", "512")
        assert _judge_max_tokens() == 512


class TestRunLLMJudge:
    def _run(self, response: Any) -> tuple[float, dict[str, float], str]:
        with mock.patch("litellm.completion", return_value=response):
            return _run_llm_judge(prompt="p", expected_behavior="e", rubric="r", agent_output="a", workspace_path="")

    def test_parses_scores_and_total(self) -> None:
        score, breakdown, notes = self._run(
            _judge_response('{"scores": {"tone": 1.0, "clarity": 0.6}, "total": 0.8, "notes": "good"}')
        )
        assert score == pytest.approx(0.8)
        assert breakdown == {"tone": 1.0, "clarity": 0.6}
        assert notes == "good"

    def test_total_falls_back_to_mean_of_criteria(self) -> None:
        score, _breakdown, _notes = self._run(_judge_response('{"scores": {"a": 1.0, "b": 0.0}}'))
        assert score == pytest.approx(0.5)

    def test_budget_is_passed_to_the_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PINCHBENCH_JUDGE_MAX_TOKENS", "2048")
        with mock.patch("litellm.completion", return_value=_judge_response('{"total": 1.0}')) as call:
            _run_llm_judge(prompt="p", expected_behavior="e", rubric="r", agent_output="a", workspace_path="")
        assert call.call_args.kwargs["max_tokens"] == 2048

    def test_truncation_is_reported_distinctly_from_a_bad_verdict(self) -> None:
        """A judge that ran out of tokens must not look like a judge that said 0."""
        score, breakdown, notes = self._run(
            _judge_response('```json\n{\n  "scores": {\n    "tone": 0', finish_reason="length")
        )
        assert score == 0.0
        assert breakdown == {}
        assert "truncated" in notes
        assert "PINCHBENCH_JUDGE_MAX_TOKENS" in notes

    def test_unparsable_verdict_is_reported_as_such(self) -> None:
        _score, _breakdown, notes = self._run(_judge_response("I refuse to grade this."))
        assert notes == "LLM judge returned unparsable response"

    def test_empty_content_does_not_raise(self) -> None:
        score, _breakdown, notes = self._run(_judge_response(None))
        assert score == 0.0
        assert notes

    def test_transport_failure_is_reported(self) -> None:
        with mock.patch("litellm.completion", side_effect=RuntimeError("boom")):
            score, _breakdown, notes = _run_llm_judge(
                prompt="p", expected_behavior="e", rubric="r", agent_output="a", workspace_path=""
            )
        assert score == 0.0
        assert "LLM judge call failed" in notes


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


class TestPinchBenchSession:
    def test_task_prompt_is_the_real_prompt(self, skill_dir: Path, tmp_path: Path) -> None:
        sess = _session(skill_dir, 0, tmp_path, "s-prompt")
        assert 'Say "ready"' in sess.task
        assert "submit" in sess.task

    def test_context_exposes_task_metadata(self, skill_dir: Path, tmp_path: Path) -> None:
        sess = _session(skill_dir, 0, tmp_path, "s-ctx")
        assert sess.context == {
            "category": "productivity",
            "pinchbench_task_id": "task_echo",
            "grading_type": "automated",
            "timeout_seconds": 60,
        }

    def test_out_of_range_task_index_raises(self, skill_dir: Path) -> None:
        with pytest.raises(ValueError, match="out of range"):
            PinchBenchSession(task_id="99", skill_dir=str(skill_dir), session_id="s-oob")

    def test_unfinished_session_is_not_finished(self, skill_dir: Path, tmp_path: Path) -> None:
        score = _session(skill_dir, 0, tmp_path, "s-unfinished").score()
        assert (score.score, score.success, score.is_finished) == (0.0, False, False)

    def test_automated_grading_scores_a_good_answer(self, skill_dir: Path, tmp_path: Path) -> None:
        sess = _session(skill_dir, 0, tmp_path, "s-auto-good")
        sess.step(build_action(sess.actions[0], {"answer": "ready"}))
        score = sess.score()
        assert score.score == 1.0
        assert score.success is True
        assert score.is_finished is True

    def test_automated_grading_partially_scores_a_poor_answer(self, skill_dir: Path, tmp_path: Path) -> None:
        sess = _session(skill_dir, 0, tmp_path, "s-auto-poor")
        sess.step(build_action(sess.actions[0], {"answer": "nope"}))
        score = sess.score()
        assert score.score == pytest.approx(0.5)
        assert score.success is True  # 0.5 is the success threshold

    def test_llm_judge_score_is_used_for_judge_tasks(self, skill_dir: Path, tmp_path: Path) -> None:
        sess = _session(skill_dir, 1, tmp_path, "s-judge")
        sess.step(build_action(sess.actions[0], {"answer": "A blurb."}))
        with mock.patch(f"{_MODULE}._run_llm_judge", return_value=(0.8, {"clarity": 0.8}, "ok")) as judge:
            score = sess.score()
        assert judge.called
        assert score.score == pytest.approx(0.8)
        assert score.success is True

    def test_llm_judge_failure_scores_zero_without_raising(self, skill_dir: Path, tmp_path: Path) -> None:
        sess = _session(skill_dir, 1, tmp_path, "s-judge-fail")
        sess.step(build_action(sess.actions[0], {"answer": "A blurb."}))
        with mock.patch("litellm.completion", side_effect=RuntimeError("no api key")):
            score = sess.score()
        assert score.score == 0.0
        assert score.success is False
        assert score.is_finished is True

    def test_hybrid_grading_applies_declared_weights(self, skill_dir: Path, tmp_path: Path) -> None:
        sess = _session(skill_dir, 2, tmp_path, "s-hybrid")
        sess.step(build_action(sess.actions[0], {"answer": "Lisbon has 545000 people."}))
        with mock.patch(f"{_MODULE}._run_llm_judge", return_value=(0.0, {}, "")):
            score = sess.score()
        # automated=1.0 (workspace file was staged) weighted 0.75, judge=0.0 weighted 0.25
        assert score.score == pytest.approx(0.75)

    def test_workspace_files_are_staged_from_assets(self, skill_dir: Path, tmp_path: Path) -> None:
        sess = _session(skill_dir, 2, tmp_path, "s-workspace")
        assert sess._workspace_dir is not None
        staged = Path(sess._workspace_dir) / "cities.csv"
        assert staged.read_text(encoding="utf-8").startswith("city,pop")
        assert str(sess._workspace_dir) in sess.task

    def test_close_persists_results_and_removes_workspace(self, skill_dir: Path, tmp_path: Path) -> None:
        sess = _session(skill_dir, 0, tmp_path, "s-close")
        sess.step(build_action(sess.actions[0], {"answer": "ready"}))
        workspace = sess._workspace_dir
        sess.close()
        payload = json.loads((tmp_path / "s-close" / "results.json").read_text(encoding="utf-8"))
        assert payload["score"] == 1.0
        assert payload["success"] is True
        assert payload["pinchbench_task_id"] == "task_echo"
        assert payload["category"] == "productivity"
        assert workspace is not None and not Path(workspace).exists()


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


@pytest.fixture()
def run_context(tmp_path: Path):
    """Aggregation writes to the run log, which requires an active run context."""
    from exgentic.core.context import run_scope

    with run_scope(run_id="pinchbench-test-run", output_dir=str(tmp_path / "outputs")) as ctx:
        yield ctx


def _paths(result_path: Path, session_id: str) -> Any:
    return SimpleNamespace(benchmark_results=result_path, session_id=session_id)


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class TestPinchBenchEvaluator:
    def test_list_tasks_returns_every_index(self, skill_dir: Path) -> None:
        assert PinchBenchEvaluator(skill_dir=str(skill_dir)).list_tasks() == ["0", "1", "2"]

    def test_category_filter_returns_full_index_positions(self, skill_dir: Path) -> None:
        ev = PinchBenchEvaluator(skill_dir=str(skill_dir), category_filter="analysis")
        ids = ev.list_tasks()
        assert ids == ["2"]
        # The returned id must address the same task through a Session.
        sess = PinchBenchSession(task_id=ids[0], skill_dir=str(skill_dir), session_id="s-filter")
        assert sess.context["category"] == "analysis"

    def test_unknown_category_filter_yields_no_tasks(self, skill_dir: Path) -> None:
        assert PinchBenchEvaluator(skill_dir=str(skill_dir), category_filter="nope").list_tasks() == []

    def test_aggregate_computes_overall_and_per_category_metrics(
        self, run_context, skill_dir: Path, tmp_path: Path
    ) -> None:
        a, b, c = (tmp_path / n / "results.json" for n in ("a", "b", "c"))
        _write(a, {"score": 1.0, "category": "coding"})
        _write(b, {"score": 0.0, "category": "coding"})
        _write(c, {"score": 0.5, "category": "writing"})

        ev = PinchBenchEvaluator(skill_dir=str(skill_dir))
        sessions = [SimpleNamespace(session_id=n) for n in ("a", "b", "c")]
        with mock.patch.object(
            type(ev),
            "get_sessions_paths",
            return_value=[_paths(a, "a"), _paths(b, "b"), _paths(c, "c")],
        ):
            results = ev.aggregate_sessions(sessions)

        assert results.benchmark_name == "pinchbench"
        assert results.total_tasks == 3
        assert results.score == pytest.approx(0.5)
        assert results.metrics["coding_accuracy"] == pytest.approx(0.5)
        assert results.metrics["coding_total"] == 2
        assert results.metrics["writing_accuracy"] == pytest.approx(0.5)
        assert results.metrics["writing_total"] == 1

    def test_aggregate_raises_on_missing_result_file(self, run_context, skill_dir: Path, tmp_path: Path) -> None:
        missing = tmp_path / "gone" / "results.json"
        ev = PinchBenchEvaluator(skill_dir=str(skill_dir))
        with mock.patch.object(type(ev), "get_sessions_paths", return_value=[_paths(missing, "gone")]):
            with pytest.raises(FileNotFoundError, match="Missing benchmark result"):
                ev.aggregate_sessions([SimpleNamespace(session_id="gone")])


# ---------------------------------------------------------------------------
# Benchmark config
# ---------------------------------------------------------------------------


class TestPinchBenchBenchmark:
    def test_defaults(self) -> None:
        bench = PinchBenchBenchmark()
        assert bench.category == "all"
        assert bench.skill_dir is None
        assert PinchBenchBenchmark.slug_name == "pinchbench"

    def test_all_category_means_no_filter(self) -> None:
        bench = PinchBenchBenchmark(skill_dir="/some/where")
        assert bench._get_evaluator_kwargs() == {"skill_dir": "/some/where", "category_filter": None}
        assert bench._get_session_kwargs() == {"skill_dir": "/some/where", "category_filter": None}

    def test_explicit_category_is_threaded_through(self) -> None:
        bench = PinchBenchBenchmark(category="coding")
        assert bench._get_evaluator_kwargs()["category_filter"] == "coding"
        assert bench._get_session_kwargs()["category_filter"] == "coding"

    def test_registry_entry_resolves(self) -> None:
        from exgentic.interfaces.registry import BENCHMARKS

        entry = BENCHMARKS["pinchbench"]
        assert entry.module == "exgentic.benchmarks.pinchbench.pinchbench_benchmark"
        assert entry.attr == "PinchBenchBenchmark"
