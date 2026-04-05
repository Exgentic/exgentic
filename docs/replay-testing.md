# Replay Testing

Replay tests let you verify benchmark and agent integration end-to-end without making real API calls or running external services. A recording captures a live session; the test replays it deterministically.

This is the primary testing pattern for contributors adding new benchmarks or agents.

**Related docs:**
[docs/](./README.md) · [Adding Benchmarks](./adding-benchmarks.md) · [Adding Agents](./adding-agents.md) · [Runners](./runners.md)

---

## How it works

1. Run a live evaluation and capture the session trajectory.
2. Store the trajectory in `tests/benchmarks/recordings/<benchmark_slug>/`.
3. Write a test that replays the trajectory using `ReplayAgent` and `ReplayBenchmark`.
4. The test verifies the session ends with the expected score — no network, no API keys, no benchmark installation required.

Replay tests can be parametrized across all runner types (`direct`, `venv`, `docker`) to verify that runner isolation doesn't change session outcomes.

---

## Recording format

Each recording lives in its own directory:

```
tests/benchmarks/recordings/<benchmark_slug>/
├── recording.json       # Metadata: task_id, benchmark slug, expected score
├── trajectory.jsonl     # Recorded actions and observations
├── session.json         # Session manifest
└── results.json         # Recorded session results
```

### recording.json

```json
{
  "task_id": "retail_1",
  "benchmark_slug": "tau2",
  "expected_score": 1.0
}
```

Set `expected_score` to `null` if you only want to verify that the session completes without asserting on score.

### trajectory.jsonl

Newline-delimited JSON. Each line is one event — either an observation (benchmark → agent) or an action (agent → benchmark):

```json
{"event": "observation", "step": 0, "initial": true, "session_id": "...", "task_id": "retail_1", "observation": {...}, "action": null}
{"event": "action",      "step": 1, "initial": false, "session_id": "...", "task_id": "retail_1", "observation": null, "action": {"name": "search", "arguments": {...}}}
{"event": "observation", "step": 1, "initial": false, "session_id": "...", "task_id": "retail_1", "observation": {...}, "action": null}
```

---

## Creating a recording

Run a live evaluation and save the trajectory. The trajectory file is written automatically to the session output directory:

```
outputs/<run_id>/sessions/<session_id>/trajectory.jsonl
outputs/<run_id>/sessions/<session_id>/results.json
```

Copy the relevant files into your recording directory:

```bash
mkdir -p tests/benchmarks/recordings/my_benchmark

cp outputs/<run_id>/sessions/<session_id>/trajectory.jsonl \
   tests/benchmarks/recordings/my_benchmark/

cp outputs/<run_id>/sessions/<session_id>/results.json \
   tests/benchmarks/recordings/my_benchmark/

# Write recording.json manually
cat > tests/benchmarks/recordings/my_benchmark/recording.json <<EOF
{
  "task_id": "my_task_id",
  "benchmark_slug": "my_benchmark",
  "expected_score": 1.0
}
EOF
```

---

## Writing a replay test

```python
import sys
from pathlib import Path

import pytest

from exgentic.agents.replay.replay_agent import ReplayAgent
from exgentic.agents.replay.replay_benchmark import ReplayBenchmark
from exgentic.interfaces.lib.api import evaluate

RECORDINGS = Path(__file__).parent / "recordings" / "my_benchmark"

# Check for optional runner availability
def _uv_available():
    import shutil
    return shutil.which("uv") is not None

def _docker_available():
    import subprocess
    try:
        subprocess.run(["docker", "info"], capture_output=True, check=True, timeout=5)
        return True
    except Exception:
        return False


@pytest.fixture
def recording_dir():
    return RECORDINGS


@pytest.mark.parametrize("runner", [
    "direct",
    pytest.param(
        "venv",
        marks=pytest.mark.skipif(not _uv_available(), reason="uv not available"),
    ),
    pytest.param(
        "docker",
        marks=pytest.mark.skipif(not _docker_available(), reason="Docker not available"),
    ),
])
def test_my_benchmark_replay(runner, recording_dir, tmp_path):
    recording_json = (recording_dir / "recording.json").read_text()
    recording = json.loads(recording_json)

    agent = ReplayAgent(recording=str(recording_dir))
    benchmark = ReplayBenchmark(recording_dir=str(recording_dir), runner=runner)

    results = evaluate(
        benchmark=benchmark,
        agent=agent,
        task_ids=[recording["task_id"]],
        output_dir=str(tmp_path / "outputs"),
    )

    assert results.total_sessions == 1
    session = results.session_results[0]
    assert session.is_finished is not None

    if recording.get("expected_score") is not None:
        assert session.score == recording["expected_score"]
```

---

## Registering replay components

Replay agent and benchmark are not in the default registry. Register them in a pytest fixture:

```python
import pytest
from exgentic.interfaces.registry import AGENTS, BENCHMARKS, RegistryEntry

@pytest.fixture(autouse=True)
def _register_replay_components():
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
```

Place this fixture in a `conftest.py` in your test directory so it applies automatically.

---

## What replay tests verify

A replay test confirms:

- The session completes (no unhandled exceptions).
- The agent produces the same actions in the same order.
- The benchmark produces the same observations for the same actions.
- The final score matches the expected value (when set in `recording.json`).
- All of the above holds across runner types when parametrized with `direct`, `venv`, and `docker`.

This catches:
- Serialisation bugs (actions/observations can't be pickled for venv/docker).
- Import errors in isolated environments.
- Broken `requirements.txt` or `setup.sh`.
- Regressions in benchmark scoring logic.

---

## Tips

**Keep recordings small.** Pick tasks with short trajectories (3–10 steps). Long recordings slow down tests and are harder to maintain.

**One recording per meaningful scenario.** You don't need to cover every task. Focus on: a successful run, a failed run (if the benchmark has failure paths), and any edge cases specific to your benchmark.

**Update recordings when the protocol changes.** If you change action or observation schemas, existing recordings will fail replay. Re-record from a live run and update the trajectory file.

**Use `direct` for fast iteration.** Only run `venv` and `docker` variants in CI or before merging. Add appropriate `skipif` marks so local test runs stay fast.

---

## See also

- [Adding Benchmarks](./adding-benchmarks.md) — required checklist includes replay test
- [Adding Agents](./adding-agents.md) — required checklist includes functional validation
- [Runners](./runners.md) — how `direct`, `venv`, and `docker` differ
- [Output Format](./output-format.md) — trajectory.jsonl schema
- [CONTRIBUTING.md](../CONTRIBUTING.md) — PR guidelines
- [docs/](./README.md) — documentation index
