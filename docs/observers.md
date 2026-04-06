# Observers and Controllers

Observers let you hook into the evaluation lifecycle to add custom logging, monitoring, alerting, or analytics — without modifying benchmarks or agents. Controllers extend this with the ability to stop a run early.

**Related docs:**
[docs/](./README.md) · [Python API](./python-api.md) · [Output Format](./output-format.md) · [Observability](./observability/quickstart.md)

---

## Observer interface

All observers extend `exgentic.core.orchestrator.observer.Observer`. Every method has a default no-op implementation, so you only override what you need.

```python
from exgentic.core.orchestrator.observer import Observer

class Observer:
    # Run-level callbacks
    def on_run_start(self, run_config) -> None: ...
    def on_run_success(self, results, run_config) -> None: ...
    def on_run_error(self, error) -> None: ...

    # Session-level callbacks
    def on_session_enter(self, session_id: str, task_id: str | None) -> None: ...
    def on_session_creation(self, session) -> None: ...
    def on_session_start(self, session, agent, observation) -> None: ...
    def on_session_scoring(self, session) -> None: ...
    def on_session_success(self, session, score, agent) -> None: ...
    def on_session_error(self, session, error) -> None: ...
    def on_session_reuse(self, task_result) -> None: ...

    # Step-level: agent.react() returned an action
    def on_react_success(self, session, action) -> None: ...
    def on_react_error(self, session, error) -> None: ...

    # Step-level: session.step(action) returned an observation
    def on_step_success(self, session, observation) -> None: ...
    def on_step_error(self, session, error) -> None: ...
```

### Callback parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `session_id` | `str` | Session identifier (available in `on_session_enter`) |
| `task_id` | `str \| None` | Task identifier (available in `on_session_enter`) |
| `run_config` | `RunConfig` | The run configuration |
| `results` | `RunResults` | Aggregated results (available in `on_run_success`) |
| `session` | `Session` | Session object with `session_id`, `task_id`, `paths` |
| `agent` | `Agent` | Agent config object with `get_cost()` |
| `observation` | `Observation \| None` | Observation returned by the benchmark (None on first step) |
| `action` | `Action \| None` | Action returned by the agent (None if agent is done) |
| `score` | `SessionScore` | Score with `score`, `success`, `is_finished`, `session_metrics`, `session_metadata` |
| `task_result` | `SessionResults` | Results for a session that was skipped/reused from cache |
| `error` | `Exception` | The exception that occurred |

---

## Using observers

Pass observers to `evaluate()`, `execute()`, or `aggregate()`:

```python
from exgentic import evaluate

results = evaluate(
    benchmark="tau2",
    agent="tool_calling",
    num_tasks=10,
    observers=[MyObserver()],
)
```

Multiple observers are supported:

```python
observers=[LoggingObserver(), MetricsObserver(), AlertObserver()]
```

---

## Examples

### Print session results as they complete

```python
from exgentic.core.orchestrator.observer import Observer

class PrintObserver(Observer):
    def on_session_success(self, session, score, agent):
        status = "PASS" if score.success else "FAIL"
        print(f"[{status}] {session.task_id}  score={score.score:.2f}  cost=${agent.get_cost().total_cost:.4f}")

    def on_session_error(self, session, error):
        print(f"[ERROR] {session.task_id}  {type(error).__name__}: {error}")
```

### Track costs in real time

```python
from exgentic.core.orchestrator.observer import Observer

class CostTracker(Observer):
    def __init__(self):
        self.total_cost = 0.0

    def on_session_success(self, session, score, agent):
        self.total_cost += agent.get_cost().total_cost
        print(f"Running total: ${self.total_cost:.4f}")
```

### Write a custom log file

```python
import json
from pathlib import Path
from exgentic.core.orchestrator.observer import Observer

class JsonlLogger(Observer):
    def __init__(self, path: str):
        self.path = Path(path)

    def on_session_success(self, session, score, agent):
        entry = {
            "session_id": session.session_id,
            "task_id": session.task_id,
            "success": score.success,
            "score": score.score,
        }
        with self.path.open("a") as f:
            f.write(json.dumps(entry) + "\n")
```

### Send a Slack alert on failure

```python
import requests
from exgentic.core.orchestrator.observer import Observer

class SlackAlerter(Observer):
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url

    def on_session_error(self, session, error):
        requests.post(self.webhook_url, json={
            "text": f":x: Session `{session.task_id}` failed: `{error}`"
        })
```

---

## Controllers

Controllers extend observers with the ability to raise errors that stop the run. Use them to implement early stopping — e.g. abort if too many consecutive failures occur.

```python
from exgentic.core.orchestrator.observer import Observer

class EarlyStopController(Observer):
    def __init__(self, max_failures: int = 3):
        self.max_failures = max_failures
        self.failures = 0

    def on_session_error(self, session, error):
        self.failures += 1
        if self.failures >= self.max_failures:
            raise RuntimeError(f"Stopping: {self.failures} consecutive session errors")
```

Pass controllers via the `controllers` parameter:

```python
results = evaluate(
    benchmark="tau2",
    agent="tool_calling",
    num_tasks=50,
    controllers=[EarlyStopController(max_failures=5)],
)
```

---

## Built-in observers

Exgentic uses these observers internally. They run automatically — you do not need to register them.

| Observer | What it does |
|----------|-------------|
| `ResultsObserver` | Writes `trajectory.jsonl` and `results.json` for every session; computes `RunResults` |
| `LoggerObserver` | Logs run progress to the console |
| `FileLoggerObserver` | Writes `run.log` and per-session logs |
| `OtelTracingObserver` | Emits OpenTelemetry spans (active when `EXGENTIC_OTEL_ENABLED=true`) |
| `DashboardEventsObserver` | Streams events to the live dashboard |
| `WarningsObserver` | Captures and writes `warnings.log` |
| `RecapObserver` | Prints a summary table at the end of a run |

---

## Thread safety

Observers may be called from multiple threads when `max_workers > 1`. If your observer maintains shared state (counters, file handles, accumulators), protect it with a lock:

```python
import threading
from exgentic.core.orchestrator.observer import Observer

class ThreadSafeCounter(Observer):
    def __init__(self):
        self._lock = threading.Lock()
        self.count = 0

    def on_session_success(self, session, score, agent):
        with self._lock:
            self.count += 1
```

---

## See also

- [Python API](./python-api.md) — how to pass observers to `evaluate()`
- [Observability Quick Start](./observability/quickstart.md) — OpenTelemetry tracing (built-in observer)
- [Output Format](./output-format.md) — the data that `ResultsObserver` writes
- [docs/](./README.md) — documentation index
