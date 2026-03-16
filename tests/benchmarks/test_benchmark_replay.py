# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Replay recorded sessions to verify the execution loop end-to-end.

Each subdirectory under ``recordings/<benchmark_slug>/`` contains:
    trajectory.jsonl  — recorded action/observation events
    session.json      — session manifest (task, context, actions schema)
    results.json      — recorded score / details
    recording.json    — metadata: benchmark slug, task_id, expected score

The tests use ReplayBenchmark + ReplayAgent + ReplaySession so that
**no benchmark third-party dependencies** are required.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from exgentic.agents.replay.replay_agent import ReplayAgent
from exgentic.agents.replay.replay_benchmark import ReplayBenchmark
from exgentic.interfaces.lib.api import evaluate
from exgentic.interfaces.registry import AGENTS, BENCHMARKS, RegistryEntry

RECORDINGS_DIR = Path(__file__).parent / "recordings"

# Register replay agent and benchmark for the duration of these tests.
# They're not in the main registry because they're test-only utilities.
AGENTS["replay"] = RegistryEntry(
    slug_name="replay",
    display_name="Replay Agent",
    module="exgentic.agents.replay.replay_agent",
    attr="ReplayAgent",
    kind="agent",
)
BENCHMARKS["replay"] = RegistryEntry(
    slug_name="replay",
    display_name="Replay Benchmark",
    module="exgentic.agents.replay.replay_benchmark",
    attr="ReplayBenchmark",
    kind="benchmark",
)


def _discover_recordings() -> list[tuple[str, Path]]:
    """Return (benchmark_slug, recording_dir) pairs."""
    recordings = []
    if not RECORDINGS_DIR.exists():
        return recordings
    for bench_dir in sorted(RECORDINGS_DIR.iterdir()):
        if not bench_dir.is_dir():
            continue
        meta_path = bench_dir / "recording.json"
        if not meta_path.exists():
            continue
        recordings.append((bench_dir.name, bench_dir))
    return recordings


_RECORDINGS = _discover_recordings()


@pytest.mark.parametrize(
    "benchmark_slug,recording_dir",
    _RECORDINGS,
    ids=[slug for slug, _ in _RECORDINGS],
)
def test_benchmark_replay(benchmark_slug: str, recording_dir: Path, tmp_path: Path):
    """Replay a recorded session using ReplayBenchmark (no real deps needed)."""
    meta = json.loads((recording_dir / "recording.json").read_text())
    task_id = meta["task_id"]
    expected_score = meta.get("expected_score")

    agent = ReplayAgent(recording=str(recording_dir))
    benchmark = ReplayBenchmark(recording_dir=str(recording_dir))

    results = evaluate(
        benchmark=benchmark,
        agent=agent,
        task_ids=[task_id],
        output_dir=str(tmp_path / "outputs"),
    )

    assert results.total_sessions == 1, f"Expected 1 session, got {results.total_sessions}"
    session = results.session_results[0]

    # The session should complete without error
    assert session.is_finished is not None, (
        f"Session did not finish (status={session.status})"
    )

    # If expected_score is provided, check it
    if expected_score is not None:
        assert session.score == pytest.approx(expected_score, abs=0.01), (
            f"Score mismatch: expected {expected_score}, got {session.score}"
        )
