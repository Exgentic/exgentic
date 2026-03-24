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

Tests are parametrized across runners (direct, venv, and docker) to verify
that isolation runners work end-to-end with benchmark components.
"""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
from exgentic.agents.replay.replay_agent import ReplayAgent
from exgentic.agents.replay.replay_benchmark import ReplayBenchmark
from exgentic.interfaces.lib.api import evaluate
from exgentic.interfaces.registry import AGENTS, BENCHMARKS, RegistryEntry

RECORDINGS_DIR = Path(__file__).parent / "recordings"

# Check runner availability for parametrized tests.
_uv_available = shutil.which("uv") is not None
_docker_available = shutil.which("docker") is not None
if _docker_available:
    try:
        subprocess.run(["docker", "info"], check=True, capture_output=True, timeout=5)
    except Exception:
        _docker_available = False


@pytest.fixture(autouse=True)
def _register_replay_components():
    """Register replay agent and benchmark for the duration of these tests."""
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
    yield
    AGENTS.pop("replay", None)
    BENCHMARKS.pop("replay", None)


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
    "runner",
    [
        "direct",
        pytest.param(
            "venv",
            marks=[
                pytest.mark.skipif(not _uv_available, reason="uv CLI not available"),
                pytest.mark.skipif(
                    sys.version_info < (3, 12),
                    reason="Venv replay tests require Python 3.12+ (CPython 3.11 segfault)",
                ),
            ],
        ),
        pytest.param(
            "docker",
            marks=[
                pytest.mark.skipif(not _docker_available, reason="Docker not available"),
                pytest.mark.skipif(
                    sys.version_info < (3, 12),
                    reason="Docker replay tests require Python 3.12+ (CPython 3.11 segfault)",
                ),
            ],
        ),
    ],
)
@pytest.mark.parametrize(
    "benchmark_slug,recording_dir",
    _RECORDINGS,
    ids=[slug for slug, _ in _RECORDINGS],
)
def test_benchmark_replay(benchmark_slug: str, recording_dir: Path, tmp_path: Path, runner: str, request):
    """Replay a recorded session using ReplayBenchmark (no real deps needed)."""
    # Docker volume mounts on macOS only work under /Users/ (Rancher Desktop
    # / Docker Desktop share that by default).  pytest's tmp_path lives under
    # /var/folders/ which is NOT shared.
    if runner == "docker" and platform.system() == "Darwin":
        out = Path(tempfile.mkdtemp(prefix=".exgentic_test_", dir=Path.home()))
        request.addfinalizer(lambda: shutil.rmtree(out, ignore_errors=True))
    else:
        out = tmp_path

    meta = json.loads((recording_dir / "recording.json").read_text())
    task_id = meta["task_id"]
    expected_score = meta.get("expected_score")

    agent = ReplayAgent(recording=str(recording_dir))
    benchmark = ReplayBenchmark(recording_dir=str(recording_dir), runner=runner)

    results = evaluate(
        benchmark=benchmark,
        agent=agent,
        task_ids=[task_id],
        output_dir=str(out / "outputs"),
    )

    assert results.total_sessions == 1, f"Expected 1 session, got {results.total_sessions}"
    session = results.session_results[0]

    # The session should complete without error
    assert session.is_finished is not None, f"Session did not finish (status={session.status})"

    # If expected_score is provided, check it
    if expected_score is not None:
        assert session.score == pytest.approx(
            expected_score, abs=0.01
        ), f"Score mismatch: expected {expected_score}, got {session.score}"
