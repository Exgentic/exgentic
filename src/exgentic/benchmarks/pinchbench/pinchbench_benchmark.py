# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""PinchBench adapter for Exgentic.

PinchBench (https://pinchbench.com) is a real-world benchmark for AI coding
agents, created by PinchBench / Kilo Code.  It evaluates agents across
categories such as productivity, research, writing, coding, analysis,
CSV analysis, log analysis, meeting analysis, memory, skills, and
integrations.

Tasks are defined as Markdown files with YAML frontmatter specifying the
task ID, category, grading type (automated / llm_judge / hybrid), timeout,
and optional workspace files.  Grading uses automated Python checks,
LLM-judge rubrics, or a weighted combination of both.

This adapter loads PinchBench tasks from a local clone of the skill repo,
presents each task prompt to an Exgentic agent, collects the agent's final
answer (free-form text), and scores using the automated grading functions
embedded in each task definition.  LLM-judge grading is supported by
delegating to a configurable judge model via LiteLLM.

Attribution: PinchBench is developed by the PinchBench / Kilo Code team
and is available under the MIT license at
https://github.com/pinchbench/skill
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from ...core.actions import ActionsHandler, extract_argument
from ...core.benchmark import Benchmark
from ...core.evaluator import Evaluator
from ...core.session import Session
from ...core.types import (
    Action,
    ActionType,
    BenchmarkResults,
    EmptyObservation,
    FinishAction,
    Observation,
    SessionIndex,
    SessionScore,
    SingleAction,
    SingleObservation,
)
from ...observers.logging import get_logger
from ...utils.paths import get_run_paths
from ...utils.settings import RunnerName

_run_logger: logging.Logger | None = None


def _get_run_logger() -> logging.Logger:
    """Benchmark-level logger that writes into the run's run log."""
    global _run_logger
    if _run_logger is None:
        log_path = get_run_paths().tracker
        _run_logger = get_logger(__name__, str(log_path))
    return _run_logger


# ---------------------------------------------------------------------------
# PinchBench task loading
# ---------------------------------------------------------------------------

# Categories present in the PinchBench manifest.
PINCHBENCH_CATEGORIES: list[str] = [
    "productivity",
    "research",
    "writing",
    "coding",
    "analysis",
    "csv_analysis",
    "log_analysis",
    "meeting_analysis",
    "memory",
    "skills",
    "integrations",
]


def _default_skill_repo_path() -> Path:
    """Return the default path where PinchBench skill repo is expected.

    Order of precedence:
    1. ``PINCHBENCH_SKILL_DIR`` environment variable
    2. ``/tmp/pinchbench-source`` (conventional clone location)
    """
    env = os.environ.get("PINCHBENCH_SKILL_DIR")
    if env:
        return Path(env)
    return Path("/tmp/pinchbench-source")


def _parse_task_file(task_path: Path) -> Dict[str, Any]:
    """Parse a PinchBench task Markdown file into its components.

    Returns a dict with keys: ``metadata`` (YAML frontmatter as dict),
    and section names mapped to their content strings (e.g.
    ``"Prompt"``, ``"Expected Behavior"``, ``"Automated Checks"``,
    ``"LLM Judge Rubric"``, ``"Grading Criteria"``).
    """
    import yaml

    content = task_path.read_text(encoding="utf-8")
    fm_match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", content, re.DOTALL)
    if not fm_match:
        raise ValueError(f"No YAML frontmatter in {task_path}")

    metadata = yaml.safe_load(fm_match.group(1))
    body = fm_match.group(2)

    # Parse markdown sections
    sections: Dict[str, str] = {}
    current_section: str | None = None
    current_lines: list[str] = []
    for line in body.split("\n"):
        header = re.match(r"^##\s+(.+)$", line)
        if header:
            if current_section:
                sections[current_section] = "\n".join(current_lines).strip()
            current_section = header.group(1)
            current_lines = []
        elif current_section is not None:
            current_lines.append(line)
    if current_section:
        sections[current_section] = "\n".join(current_lines).strip()

    return {"metadata": metadata, **sections}


def _extract_grading_code(automated_checks_section: str) -> str:
    """Extract the Python grading function from a ```python``` code block."""
    match = re.search(r"```python\s*(.*?)\s*```", automated_checks_section, re.DOTALL)
    return match.group(1) if match else ""


def _extract_grading_criteria(criteria_text: str) -> list[str]:
    """Extract checklist items from grading criteria section."""
    criteria: list[str] = []
    for line in criteria_text.split("\n"):
        m = re.match(r"^-\s+\[[ x]\]\s+(.+)$", line.strip())
        if m:
            criteria.append(m.group(1))
    return criteria


def _load_task_index(skill_dir: Path) -> list[Dict[str, Any]]:
    """Load all PinchBench tasks and return a list of parsed task dicts.

    Each dict contains: task_id, name, category, grading_type,
    timeout_seconds, workspace_files, prompt, expected_behavior,
    grading_criteria, automated_checks (raw section), llm_judge_rubric
    (raw section), grading_weights, and file_path.
    """
    import yaml

    tasks_dir = skill_dir / "tasks"
    manifest_path = tasks_dir / "manifest.yaml"

    # Determine task ordering from manifest or fallback to glob
    task_ids: list[str] = []
    category_map: Dict[str, str] = {}

    if manifest_path.exists():
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        if "categories" in manifest:
            for category, ids in manifest["categories"].items():
                for tid in ids or []:
                    category_map[tid] = category
                    task_ids.append(tid)
        else:
            task_ids = manifest.get("tasks", [])
    else:
        for p in sorted(tasks_dir.glob("task_*.md")):
            task_ids.append(p.stem)

    # Parse each task file
    tasks: list[Dict[str, Any]] = []
    for tid in task_ids:
        task_file = tasks_dir / f"{tid}.md"
        if not task_file.exists():
            continue
        try:
            parsed = _parse_task_file(task_file)
        except Exception:
            continue

        meta = parsed["metadata"]
        # Use manifest category if available, else frontmatter
        category = category_map.get(tid, meta.get("category", ""))
        tasks.append(
            {
                "task_id": meta.get("id", tid),
                "name": meta.get("name", ""),
                "category": category,
                "grading_type": meta.get("grading_type", "automated"),
                "timeout_seconds": meta.get("timeout_seconds", 120),
                "workspace_files": meta.get("workspace_files") or [],
                "grading_weights": meta.get("grading_weights"),
                "prompt": parsed.get("Prompt", "").strip(),
                "expected_behavior": parsed.get("Expected Behavior", "").strip(),
                "grading_criteria": _extract_grading_criteria(
                    parsed.get("Grading Criteria", "")
                ),
                "automated_checks": parsed.get("Automated Checks"),
                "llm_judge_rubric": parsed.get("LLM Judge Rubric"),
                "file_path": str(task_file),
            }
        )
    return tasks


# ---------------------------------------------------------------------------
# Grading helpers
# ---------------------------------------------------------------------------


def _run_automated_grade(
    grading_code: str,
    transcript: list[Dict[str, Any]],
    workspace_path: str,
) -> Dict[str, float]:
    """Execute a PinchBench automated grading function and return scores.

    The ``grade(transcript, workspace_path)`` function is expected to return
    a dict mapping criterion names to float scores in [0, 1].
    """
    namespace: Dict[str, Any] = {}
    exec(grading_code, namespace)
    grade_func = namespace.get("grade")
    if not callable(grade_func):
        return {}
    result = grade_func(transcript, workspace_path)
    if not isinstance(result, dict):
        return {}
    return {str(k): float(v) for k, v in result.items() if isinstance(v, (int, float))}


def _average_scores(scores: Dict[str, float]) -> float:
    values = list(scores.values())
    return sum(values) / len(values) if values else 0.0


def _run_llm_judge(
    prompt: str,
    expected_behavior: str,
    rubric: str,
    agent_output: str,
    workspace_path: str,
) -> tuple[float, Dict[str, float], str]:
    """Run LLM-judge grading via LiteLLM and return (score, breakdown, notes).

    Falls back to 0.0 if the judge call fails or produces no parseable output.
    """
    judge_prompt = (
        "You are a grading function. Your ONLY job is to output a single JSON object.\n\n"
        "CRITICAL RULES:\n"
        "- Do NOT use any tools\n"
        "- Respond with ONLY a JSON object -- nothing else\n\n"
        "Be a strict evaluator. Reserve 1.0 for genuinely excellent performance. "
        "An average acceptable completion should score around 0.6-0.7.\n\n"
        f"## Task Prompt\n{prompt}\n\n"
        f"## Expected Behavior\n{expected_behavior}\n\n"
        f"## Agent Output\n{agent_output}\n\n"
    )

    # Include workspace file contents if any
    if workspace_path:
        ws = Path(workspace_path)
        if ws.exists():
            file_sections: list[str] = []
            for f in sorted(ws.rglob("*")):
                if not f.is_file():
                    continue
                try:
                    text = f.read_text(encoding="utf-8")
                    rel = f.relative_to(ws)
                    file_sections.append(f"### File: {rel}\n{text[:2000]}")
                except (OSError, UnicodeDecodeError):
                    pass
            if file_sections:
                judge_prompt += "## Workspace Files Created by Agent\n"
                judge_prompt += "\n\n".join(file_sections[:20]) + "\n\n"

    judge_prompt += (
        f"## Grading Rubric\n{rubric}\n\n"
        "Score each criterion from 0.0 to 1.0.\n"
        'The "total" field must be between 0.0 and 1.0 (arithmetic mean of criterion scores).\n\n'
        "Respond with ONLY this JSON structure:\n"
        '{"scores": {"criterion_name": 0.0}, "total": 0.0, "notes": "brief justification"}'
    )

    try:
        import litellm

        response = litellm.completion(
            model=os.environ.get("PINCHBENCH_JUDGE_MODEL", "gpt-4o-mini"),
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=0.0,
            max_tokens=1024,
        )
        raw_text = response.choices[0].message.content.strip()
    except Exception as exc:
        return 0.0, {}, f"LLM judge call failed: {exc}"

    # Parse JSON response
    parsed = _parse_judge_json(raw_text)
    if not parsed:
        return 0.0, {}, f"LLM judge returned unparseable response"

    scores = {}
    if isinstance(parsed.get("scores"), dict):
        for k, v in parsed["scores"].items():
            try:
                scores[str(k)] = float(v)
            except (TypeError, ValueError):
                pass

    total = None
    for key in ("total", "score", "overall_score"):
        if key in parsed:
            try:
                total = float(parsed[key])
                break
            except (TypeError, ValueError):
                pass
    if total is None and scores:
        total = _average_scores(scores)

    notes = str(parsed.get("notes", parsed.get("justification", "")))
    return float(total) if total is not None else 0.0, scores, notes


def _parse_judge_json(raw_text: str) -> Dict[str, Any]:
    """Best-effort JSON extraction from judge response text."""
    raw_text = raw_text.strip()
    # Direct parse
    try:
        parsed = json.loads(raw_text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    # Code block extraction
    code_match = re.search(r"```(?:json)?\s*(.*?)\s*```", raw_text, re.DOTALL)
    if code_match:
        try:
            parsed = json.loads(code_match.group(1))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    # Balanced-brace extraction
    depth = 0
    current: list[str] = []
    candidates: list[str] = []
    for ch in raw_text:
        if ch == "{":
            if depth == 0:
                current = []
            depth += 1
        if depth > 0:
            current.append(ch)
        if ch == "}":
            depth -= 1
            if depth == 0 and current:
                candidates.append("".join(current))
    for candidate in reversed(candidates):
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    return {}


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------


class PinchBenchFinishArgs(BaseModel):
    answer: str = Field(
        ...,
        description="Final answer or output for the task. Provide a concise response.",
    )


class PinchBenchFinishAction(FinishAction):
    name: Literal["submit"] = "submit"
    arguments: PinchBenchFinishArgs


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


class PinchBenchSession(Session):
    """Session for a single PinchBench task evaluation.

    Loads the task definition, sets up any workspace files, presents the
    prompt to the agent, and scores the result using PinchBench's
    automated grading functions and/or LLM-judge rubrics.
    """

    _done: bool
    _final_answer: str | None

    def __init__(
        self,
        task_id: str,
        skill_dir: str | None = None,
        category_filter: str | None = None,
        session_id: str | None = None,
    ) -> None:
        if session_id is not None:
            self._session_id = session_id

        self._skill_dir = Path(skill_dir) if skill_dir else _default_skill_repo_path()
        self._task_index = _load_task_index(self._skill_dir)
        self._task_id_str = task_id

        # Find the task by index (task_id is an integer index into the list)
        idx = int(task_id)
        if idx < 0 or idx >= len(self._task_index):
            raise ValueError(
                f"PinchBench task index {idx} out of range "
                f"(0..{len(self._task_index) - 1})"
            )
        self._task_data = self._task_index[idx]

        self._done = False
        self._final_answer = None
        self._workspace_dir: str | None = None

        # Set up workspace if the task requires files
        self._setup_workspace()

        self._registry = ActionsHandler(logger=self.logger)
        self._registry.add_action(
            name="submit",
            description="Submit the final answer and complete the task.",
            action_cls=PinchBenchFinishAction,
            handler=self._handle_finish,
            is_finish=True,
        )
        super().__init__()

    def _setup_workspace(self) -> None:
        """Create a temporary workspace and copy any required files."""
        workspace_files = self._task_data.get("workspace_files") or []
        self._workspace_dir = tempfile.mkdtemp(prefix="pinchbench_ws_")

        assets_dir = self._skill_dir / "assets"
        for wf in workspace_files:
            src_rel = wf.get("source", "")
            dest_rel = wf.get("dest", "")
            if not src_rel or not dest_rel:
                continue
            src_path = assets_dir / src_rel
            if not src_path.exists():
                # Also try tasks-relative path
                src_path = self._skill_dir / "tasks" / src_rel
            if not src_path.exists():
                # Try root of skill dir
                src_path = self._skill_dir / src_rel
            dest_path = Path(self._workspace_dir) / dest_rel
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            if src_path.exists():
                shutil.copy2(src_path, dest_path)

    @property
    def task(self) -> str:
        prompt = self._task_data["prompt"]
        workspace_note = ""
        if self._workspace_dir:
            workspace_files = self._task_data.get("workspace_files") or []
            if workspace_files:
                file_list = ", ".join(wf.get("dest", "") for wf in workspace_files)
                workspace_note = (
                    f"\n\nWorkspace directory: {self._workspace_dir}\n"
                    f"Available files: {file_list}\n"
                )

        return (
            "Complete the following task. When you are finished, submit your "
            "final answer or output by calling `submit`.\n\n"
            f"Task:\n\n{prompt}"
            f"{workspace_note}"
        )

    @property
    def context(self) -> dict[str, Any]:
        return {
            "category": self._task_data.get("category", ""),
            "pinchbench_task_id": self._task_data.get("task_id", ""),
            "grading_type": self._task_data.get("grading_type", ""),
            "timeout_seconds": self._task_data.get("timeout_seconds", 120),
        }

    @property
    def actions(self) -> list[ActionType]:
        return self._registry.actions

    @property
    def task_id(self) -> str:
        return self._task_id_str

    def _to_observation(
        self, raw: Any, invoking_actions: list[SingleAction] | None = None
    ) -> Observation:
        return SingleObservation(invoking_actions=invoking_actions or [], result=raw)

    def start(self) -> Observation | None:
        return EmptyObservation()

    def step(self, action: Action) -> Observation | None:
        if action is None:
            self._done = True
        if self._done:
            return None
        observation = self._registry.execute(action)
        return observation

    def done(self) -> bool:
        return self._done

    def score(self) -> SessionScore:
        if self._final_answer is None:
            return SessionScore(score=0.0, success=False, is_finished=False)

        grading_type = self._task_data.get("grading_type", "automated")
        auto_score = 0.0
        llm_score = 0.0
        auto_breakdown: Dict[str, float] = {}
        llm_breakdown: Dict[str, float] = {}
        llm_notes = ""

        # Automated grading
        if grading_type in ("automated", "hybrid"):
            checks_section = self._task_data.get("automated_checks") or ""
            grading_code = _extract_grading_code(checks_section)
            if grading_code:
                # Build a minimal transcript-like structure from the agent output
                transcript = [
                    {
                        "type": "message",
                        "message": {
                            "role": "assistant",
                            "content": [{"type": "text", "text": self._final_answer}],
                        },
                    }
                ]
                workspace = self._workspace_dir or ""
                auto_breakdown = _run_automated_grade(
                    grading_code, transcript, workspace
                )
                auto_score = _average_scores(auto_breakdown)

        # LLM judge grading
        if grading_type in ("llm_judge", "hybrid"):
            rubric = self._task_data.get("llm_judge_rubric") or ""
            if not rubric:
                # Fall back to grading criteria as text
                criteria = self._task_data.get("grading_criteria") or []
                rubric = "\n".join(f"- {c}" for c in criteria)
            if rubric:
                llm_score, llm_breakdown, llm_notes = _run_llm_judge(
                    prompt=self._task_data["prompt"],
                    expected_behavior=self._task_data.get("expected_behavior", ""),
                    rubric=rubric,
                    agent_output=self._final_answer,
                    workspace_path=self._workspace_dir or "",
                )

        # Combine scores based on grading type
        if grading_type == "automated":
            final_score = auto_score
        elif grading_type == "llm_judge":
            final_score = llm_score
        elif grading_type == "hybrid":
            weights = self._task_data.get("grading_weights") or {
                "automated": 0.5,
                "llm_judge": 0.5,
            }
            aw = float(weights.get("automated", 0.5))
            lw = float(weights.get("llm_judge", 0.5))
            total_w = aw + lw
            if total_w <= 0:
                aw = lw = 0.5
                total_w = 1.0
            final_score = (auto_score * aw + llm_score * lw) / total_w
        else:
            final_score = auto_score

        success = final_score >= 0.5
        self.logger.info(
            f"PinchBench task {self._task_data.get('task_id', '')} "
            f"(category={self._task_data.get('category', '')}, "
            f"grading={grading_type}): "
            f"auto={auto_score:.3f} llm={llm_score:.3f} "
            f"final={final_score:.3f}"
        )
        if llm_notes:
            self.logger.info(f"  LLM judge notes: {llm_notes[:300]}")

        return SessionScore(
            score=final_score, success=success, is_finished=True
        )

    def close(self) -> None:
        super().close()
        sc = self.score()
        payload: Dict[str, Any] = {
            "score": sc.score,
            "success": bool(sc.success),
            "category": self._task_data.get("category", ""),
            "pinchbench_task_id": self._task_data.get("task_id", ""),
            "grading_type": self._task_data.get("grading_type", ""),
        }
        self.save_results(payload)

        # Cleanup workspace
        if self._workspace_dir and Path(self._workspace_dir).exists():
            shutil.rmtree(self._workspace_dir, ignore_errors=True)

    # -- Action handlers ---------------------------------------------------

    def _handle_finish(self, action: SingleAction) -> None:
        self.logger.info(f"Received final answer: {action}")
        answer = extract_argument(action.arguments, "answer", None)
        self._final_answer = answer
        self._done = True


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class PinchBenchEvaluator(Evaluator):
    """Evaluator for PinchBench -- task discovery and result aggregation."""

    def __init__(
        self,
        skill_dir: str | None = None,
        category_filter: str | None = None,
    ) -> None:
        self._skill_dir = Path(skill_dir) if skill_dir else _default_skill_repo_path()
        self._category_filter = category_filter
        self._task_index = _load_task_index(self._skill_dir)

    def list_tasks(self) -> list[str]:
        """Return task indices, optionally filtered by category."""
        tasks = self._task_index
        if self._category_filter:
            tasks = [
                t for t in tasks if t.get("category") == self._category_filter
            ]
        # Return string indices matching the filtered task list positions
        # in the *full* index (so the Session can look them up).
        full_ids = [t["task_id"] for t in self._task_index]
        result: list[str] = []
        for t in tasks:
            try:
                idx = full_ids.index(t["task_id"])
                result.append(str(idx))
            except ValueError:
                pass
        return result

    def aggregate_sessions(
        self, sessions: list[SessionIndex]
    ) -> BenchmarkResults:
        run_logger = _get_run_logger()
        scores: list[float] = []
        category_scores: Dict[str, list[float]] = {}

        for paths in self.get_sessions_paths(sessions):
            fp = paths.benchmark_results
            try:
                with open(fp, encoding="utf-8-sig") as f:
                    payload = json.load(f)
                s = float(payload["score"])
                scores.append(s)
            except FileNotFoundError as err:
                raise FileNotFoundError(
                    f"Missing benchmark result for session "
                    f"'{paths.session_id}' at {fp}"
                ) from err
            except Exception:
                run_logger.exception(
                    "Failed to load benchmark result for session %s at %s",
                    paths.session_id,
                    fp,
                )
                raise

            # Per-category tracking
            cat = payload.get("category", "unknown")
            category_scores.setdefault(cat, []).append(s)

        avg = sum(scores) / len(scores) if scores else 0.0

        # Build per-category metrics
        metrics: Dict[str, Any] = {}
        for cat, cat_scores in sorted(category_scores.items()):
            cat_avg = (
                sum(cat_scores) / len(cat_scores) if cat_scores else 0.0
            )
            metrics[f"{cat}_accuracy"] = cat_avg
            metrics[f"{cat}_total"] = len(cat_scores)

        return BenchmarkResults(
            benchmark_name="pinchbench",
            total_tasks=len(sessions),
            score=avg,
            metrics=metrics,
        )


# ---------------------------------------------------------------------------
# Benchmark config
# ---------------------------------------------------------------------------


class PinchBenchBenchmark(Benchmark, BaseModel):
    """Benchmark configuration for PinchBench.

    Parameters
    ----------
    skill_dir : str, optional
        Path to a local clone of the PinchBench skill repo
        (https://github.com/pinchbench/skill).  Defaults to
        ``/tmp/pinchbench-source`` or the ``PINCHBENCH_SKILL_DIR``
        environment variable.
    category : str, optional
        Filter to a single PinchBench category (e.g. ``"coding"``,
        ``"csv_analysis"``).  When set, only tasks from that category
        are included.  Defaults to ``"all"`` (run every task).
    """

    display_name: ClassVar[str] = "PinchBench"
    slug_name: ClassVar[str] = "pinchbench"
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def _get_evaluator_class(cls):
        return PinchBenchEvaluator

    @classmethod
    def _get_session_class(cls):
        return PinchBenchSession

    skill_dir: str | None = None
    category: str = "all"
    runner: RunnerName | None = None

    def _get_evaluator_kwargs(self) -> dict[str, Any]:
        cat = self.category if self.category != "all" else None
        return {
            "skill_dir": self.skill_dir,
            "category_filter": cat,
        }

    def _get_session_kwargs(self) -> dict[str, Any]:
        cat = self.category if self.category != "all" else None
        return {
            "skill_dir": self.skill_dir,
            "category_filter": cat,
        }
