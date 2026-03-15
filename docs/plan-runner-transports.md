# Runner & Transport Architecture Plan

## Goal

Enable benchmarks and agents to run in different isolation levels via a `runner` parameter,
without changing any benchmark or agent code. The same object can run in-process, in a thread,
in a subprocess, as an HTTP service, or inside a Docker container.

```python
benchmark = GSM8kBenchmark(subset="main", runner="direct")    # same thread
benchmark = GSM8kBenchmark(subset="main", runner="thread")    # separate thread
benchmark = GSM8kBenchmark(subset="main", runner="process")   # separate process
benchmark = GSM8kBenchmark(subset="main", runner="service")   # HTTP service
benchmark = GSM8kBenchmark(subset="main", runner="docker")    # containerized service
```

All return an object implementing the `Benchmark` interface. The orchestrator doesn't know
the difference. Agents use the same mechanism.

---

## Architecture

### Layered Isolation

Each runner adds exactly one layer on top of the previous:

| Runner    | Adds                 | Transport      | Environment       |
|-----------|----------------------|----------------|-------------------|
| `direct`  | nothing              | direct call    | same thread       |
| `thread`  | thread boundary      | queue          | new thread        |
| `process` | process boundary     | pipe (pickle)  | subprocess        |
| `service` | network boundary     | HTTP (JSON)    | process           |
| `docker`  | container boundary   | HTTP (JSON)    | Docker container  |

### Three Core Abstractions

```
Transport       — how proxy and host exchange messages
BenchmarkProxy  — implements Benchmark, forwards calls over Transport
BenchmarkHost   — receives calls from Transport, executes on real Benchmark
```

The same pattern applies to Agent/AgentInstance.

### Separation of Concerns

```
┌───────────────────────────────────────────────────┐
│ Transport (how to communicate)                    │
│                                                   │
│   DirectTransport   — function call               │
│   ThreadTransport   — threading.Queue             │
│   PipeTransport     — multiprocessing pipes       │
│   HTTPTransport     — HTTP requests/responses     │
└───────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────┐
│ Host (receives calls, runs real object)           │
│                                                   │
│   ObjectHost        — generic, wraps any object   │
│   HTTPServer        — serves ObjectHost over HTTP │
└───────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────┐
│ Proxy (implements interface, forwards calls)      │
│                                                   │
│   ObjectProxy       — generic transparent proxy   │
│   BenchmarkProxy    — typed, implements Benchmark │
│   AgentProxy        — typed, implements Agent     │
└───────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────┐
│ Runner (composes Transport + Environment)         │
│                                                   │
│   DirectRunner      — no-op, returns object       │
│   ThreadRunner      — ThreadTransport + thread    │
│   ProcessRunner     — PipeTransport + subprocess  │
│   ServiceRunner     — HTTPTransport + process     │
│   DockerRunner      — HTTPTransport + container   │
└───────────────────────────────────────────────────┘
```

---

## Detailed Design

### Transport Protocol

Every transport implements this interface:

```python
class Transport(ABC):
    @abstractmethod
    def call(self, method: str, *args, **kwargs) -> Any:
        """Call a method on the remote object."""

    @abstractmethod
    def get(self, name: str) -> Any:
        """Get an attribute from the remote object."""

    @abstractmethod
    def set(self, name: str, value: Any) -> None:
        """Set an attribute on the remote object."""

    @abstractmethod
    def close(self) -> None:
        """Shutdown the transport."""
```

This maps directly to the existing `BaseExecuter` interface, which already has
`call`, `get`, `set`, `delete`, `shutdown`. We extend and generalize that pattern.

### ObjectHost

Receives transport messages and executes them on a real object:

```python
class ObjectHost:
    def __init__(self, obj: Any):
        self._obj = obj

    def handle(self, op: str, name: str, *args, **kwargs) -> Any:
        if op == "call":
            return getattr(self._obj, name)(*args, **kwargs)
        elif op == "get":
            return getattr(self._obj, name)
        elif op == "set":
            setattr(self._obj, name, args[0])
        elif op == "del":
            delattr(self._obj, name)
```

### ObjectProxy

Transparent proxy that forwards attribute access/calls to a Transport:

```python
class ObjectProxy:
    def __init__(self, transport: Transport):
        self._transport = transport

    def __getattr__(self, name):
        # Check if callable on remote, return method wrapper or attribute
        ...

    def __setattr__(self, name, value):
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            self._transport.set(name, value)
```

This is essentially the existing `_Proxy` class from `RemoteProcessExecuter.get_proxy()`,
but decoupled from any specific transport.

### Transport Implementations

#### DirectTransport

```python
class DirectTransport(Transport):
    """No isolation. Calls the object directly. Useful as baseline."""

    def __init__(self, obj):
        self._host = ObjectHost(obj)

    def call(self, method, *args, **kwargs):
        return self._host.handle("call", method, *args, **kwargs)
```

#### ThreadTransport

```python
class ThreadTransport(Transport):
    """Runs ObjectHost in a dedicated thread. Queue-based communication."""

    def __init__(self, target_cls, *args, **kwargs):
        self._request_queue = queue.Queue()
        self._response_queue = queue.Queue()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._target_cls = target_cls
        self._args = args
        self._kwargs = kwargs

    def start(self):
        self._thread.start()
        # wait for ready signal

    def _worker(self):
        obj = self._target_cls(*self._args, **self._kwargs)
        host = ObjectHost(obj)
        while True:
            msg = self._request_queue.get()
            if msg is None:
                break
            op, name, args, kwargs = msg
            try:
                result = host.handle(op, name, *args, **kwargs)
                self._response_queue.put(("ok", result))
            except Exception as e:
                self._response_queue.put(("error", e))
```

#### PipeTransport

```python
class PipeTransport(Transport):
    """Runs ObjectHost in a subprocess. Pipe-based with cloudpickle."""
    # Refactors the existing RemoteProcessExecuter logic
    # Uses multiprocessing.get_context("spawn")
    # Serializes via cloudpickle
```

#### HTTPTransport

```python
class HTTPTransport(Transport):
    """Calls over HTTP. Used by both ServiceRunner and DockerRunner."""

    def __init__(self, base_url: str):
        self._base_url = base_url

    def call(self, method, *args, **kwargs):
        resp = httpx.post(f"{self._base_url}/call", json={
            "method": method, "args": serialize(args), "kwargs": serialize(kwargs)
        })
        return deserialize(resp.json()["result"])
```

#### HTTPServer (Host side)

```python
# FastAPI app wrapping ObjectHost
app = FastAPI()

@app.post("/call")
def handle_call(request: CallRequest):
    result = host.handle("call", request.method, *request.args, **request.kwargs)
    return {"result": serialize(result)}

@app.post("/get")
def handle_get(request: GetRequest):
    result = host.handle("get", request.name)
    return {"result": serialize(result)}
```

### Runner Implementations

```python
class DirectRunner:
    def run(self, cls, *args, **kwargs):
        return cls(*args, **kwargs)  # no proxy needed

class ThreadRunner:
    def run(self, cls, *args, **kwargs):
        transport = ThreadTransport(cls, *args, **kwargs)
        transport.start()
        return ObjectProxy(transport)

class ProcessRunner:
    def run(self, cls, *args, **kwargs):
        transport = PipeTransport(cls, *args, **kwargs)
        transport.start()
        return ObjectProxy(transport)

class ServiceRunner:
    def run(self, cls, *args, **kwargs, port=None):
        # Start a subprocess running: python -m exgentic.serve --cls=... --port=...
        transport = HTTPTransport(f"http://localhost:{port}")
        return ObjectProxy(transport)

class DockerRunner:
    def run(self, cls, *args, **kwargs):
        # docker run -p {port}:8080 image_name
        transport = HTTPTransport(f"http://localhost:{port}")
        return ObjectProxy(transport)
```

### Factory Function

```python
RunnerName = Literal["direct", "thread", "process", "service", "docker"]

def with_runner(cls, runner: RunnerName = "direct", **kwargs):
    runners = {
        "direct": DirectRunner,
        "thread": ThreadRunner,
        "process": ProcessRunner,
        "service": ServiceRunner,
        "docker": DockerRunner,
    }
    return runners[runner]().run(cls, **kwargs)
```

---

## Relationship to Existing Code

### What exists today

- `BaseExecuter` with `call/get/set/delete/shutdown` — very similar to our `Transport`
- `InProcessExecuter` — equivalent to our `DirectTransport`
- `RemoteProcessExecuter` — equivalent to our `PipeTransport`
- `_Proxy` (inner class) — equivalent to our `ObjectProxy`
- `ObjectHost` pattern — equivalent to `_worker()` function
- `make_executer()` factory — equivalent to our `with_runner()`
- `ExecuterName = Literal["inprocess", "remote_process"]` — we extend this

### Migration Strategy

We do NOT delete the existing executor code. Instead:

1. Build the new Transport/Proxy/Host system as new modules
2. The new system is a cleaner generalization of the existing pattern
3. Once stable, the existing `BaseExecuter` can optionally be refactored to use
   the new Transport layer under the hood
4. The `executer` field on `Benchmark` and `runner` field can coexist during transition

### New Files

```
src/exgentic/adapters/runners/
├── __init__.py           # RunnerName type, with_runner() factory
├── transport.py          # Transport ABC, ObjectHost, ObjectProxy
├── direct.py             # DirectTransport, DirectRunner
├── thread.py             # ThreadTransport, ThreadRunner
├── process.py            # PipeTransport, ProcessRunner
├── service.py            # HTTPTransport, HTTPServer, ServiceRunner
└── docker.py             # DockerRunner (extends ServiceRunner)

tests/adapters/runners/
├── __init__.py
├── test_direct.py
├── test_thread.py
├── test_process.py
├── test_service.py
└── test_docker.py
```

---

## Serialization Boundary

A key design decision: where do we serialize?

| Transport          | Serialization   | Notes                              |
|--------------------|-----------------|------------------------------------|
| DirectTransport    | None            | Same memory space                  |
| ThreadTransport    | None            | Same memory space, queue passes refs |
| PipeTransport      | cloudpickle     | Cross-process, must serialize      |
| HTTPTransport      | JSON (pydantic) | Cross-network, must serialize      |

For `service` and `docker` runners, all arguments and return values must be
JSON-serializable. Since Benchmark/Agent interfaces already use Pydantic models
and simple types, this should mostly work. Edge cases to handle:

- `ActionType` contains Pydantic model classes (not instances) — need schema export
- `SessionIndex` — simple dataclass, easy to serialize
- `Session` objects cannot cross the HTTP boundary — sessions stay on the host side,
  the proxy holds a session ID and forwards method calls

### Session Proxying

When `create_session()` is called on a remote benchmark:

1. Host creates the real `Session` object, stores it by ID
2. Returns the session ID to the proxy
3. Proxy returns a `SessionProxy` that forwards `start/step/score/done/close`
   calls to the host, tagged with the session ID

```
Proxy side:                           Host side:
session = proxy.create_session(idx)   real_session = benchmark.create_session(idx)
                                      sessions[session_id] = real_session
                                      return session_id
session.start()                    →  sessions[session_id].start()
session.step(action)               →  sessions[session_id].step(action)
session.score()                    →  sessions[session_id].score()
```

---

## Docker Architecture for Benchmarks

### Simple Benchmarks (GSM8K, HotPotQA)

```
Framework (local)
    │ HTTP
    ▼
┌────────────────────┐
│ Benchmark Container │
│ (runs serve())     │
│                    │
│ Sessions run       │
│ inside this        │
│ container          │
└────────────────────┘
```

### Complex Benchmarks (SWE-bench)

```
Framework (local)
    │ HTTP
    ▼
┌────────────────────┐
│ Benchmark Container │  ← has docker.sock mounted
│ (runs serve())     │
│                    │
│ Creates sibling    │──── docker API ────┐
│ session containers │                    │
└────────────────────┘                    │
                                          ▼
                              ┌──────┐ ┌──────┐
                              │ ses1 │ │ ses2 │  (sibling containers)
                              └──────┘ └──────┘
```

The benchmark container has docker.sock mounted and creates session containers
as siblings on the host, not nested. This avoids docker-in-docker.

---

## Implementation Milestones

### Milestone 1: Core Abstractions + DirectTransport

**Files:**
- `src/exgentic/adapters/runners/transport.py` — Transport ABC, ObjectHost, ObjectProxy
- `src/exgentic/adapters/runners/direct.py` — DirectTransport, DirectRunner
- `src/exgentic/adapters/runners/__init__.py` — exports
- `tests/adapters/runners/test_direct.py` — unit tests

**Test with dummy class:**
```python
class Calculator:
    def __init__(self, value=0):
        self.value = value
    def add(self, a, b):
        return a + b
    def accumulate(self, n):
        self.value += n
        return self.value

def test_direct_runner():
    calc = with_runner(Calculator, runner="direct", value=10)
    assert calc.add(2, 3) == 5
    assert calc.accumulate(5) == 15
```

**Done when:** Tests pass with DirectRunner.

### Milestone 2: ThreadTransport

**Files:**
- `src/exgentic/adapters/runners/thread.py` — ThreadTransport, ThreadRunner
- `tests/adapters/runners/test_thread.py`

**Same tests as Milestone 1, plus:**
- Test concurrent access (multiple threads calling the proxy)
- Test that object methods run in a different thread
- Test error propagation across thread boundary
- Test cleanup (transport.close() joins the thread)

**Done when:** Same Calculator tests pass with `runner="thread"`.

### Milestone 3: PipeTransport (Process)

**Files:**
- `src/exgentic/adapters/runners/process.py` — PipeTransport, ProcessRunner
- `tests/adapters/runners/test_process.py`

**Same tests, plus:**
- Test that object runs in a different PID
- Test crash isolation (object raises → proxy gets exception, doesn't crash)
- Test cloudpickle serialization of complex types
- Test context propagation (ContextVar inheritance)

**Done when:** Same Calculator tests pass with `runner="process"`.

### Milestone 4: HTTPTransport + serve()

**Files:**
- `src/exgentic/adapters/runners/service.py` — HTTPTransport, HTTPServer, ServiceRunner
- `tests/adapters/runners/test_service.py`

**Additional considerations:**
- JSON serialization of arguments and return values
- Session management (host holds sessions, proxy holds IDs)
- Error propagation over HTTP (structured error responses)
- Port allocation (auto-find free port)
- Health check endpoint

**Same tests, plus:**
- Test with Pydantic models as arguments
- Test session lifecycle over HTTP
- Test concurrent sessions

**Done when:** Same Calculator tests pass with `runner="service"`.

### Milestone 5: DockerTransport

**Files:**
- `src/exgentic/adapters/runners/docker.py` — DockerRunner
- `tests/adapters/runners/test_docker.py`

**Additional considerations:**
- Dockerfile generation or selection
- Container lifecycle (start, health check, stop)
- Port mapping
- Volume mounts (for output)
- docker.sock mounting (for SWE-bench style benchmarks)

**Done when:** Same Calculator tests pass with `runner="docker"`.

### Milestone 6: Integration

**Files:**
- Modify `src/exgentic/core/benchmark.py` — add `runner` field
- Modify `src/exgentic/core/agent.py` — add `runner` field
- Add `BenchmarkProxy` and `AgentProxy` typed wrappers
- Modify CLI to support `--runner`

**Done when:** `exgentic evaluate --benchmark gsm8k --runner process` works end-to-end.

---

## Testing Strategy

All transports are tested with the **same test suite** against a shared dummy class.
A pytest parametrize fixture cycles through all runners:

```python
@pytest.fixture(params=["direct", "thread", "process", "service"])
def calc(request):
    c = with_runner(Calculator, runner=request.param, value=10)
    yield c
    if hasattr(c, 'close'):
        c.close()

def test_add(calc):
    assert calc.add(2, 3) == 5

def test_accumulate(calc):
    assert calc.accumulate(5) == 15

def test_get_attribute(calc):
    assert calc.value == 10

def test_set_attribute(calc):
    calc.value = 42
    assert calc.value == 42

def test_error_propagation(calc):
    with pytest.raises(ZeroDivisionError):
        calc.divide(1, 0)
```

Docker tests are separate (require Docker daemon) and marked with `@pytest.mark.docker`.

---

## Open Questions

1. **Should `runner` replace `executer` or coexist?** — Recommend coexist during transition,
   deprecate `executer` later.
2. **HTTP serialization format** — JSON with Pydantic `.model_dump()` / `.model_validate()`?
   Or cloudpickle-over-HTTP for richer type support?
3. **Session proxy for HTTP** — Do we proxy individual session methods, or do we expose
   a higher-level benchmark API (list_tasks, create_session, step, score)?
4. **Docker image naming convention** — `exgentic-{benchmark_slug}:latest`?
5. **Async support** — Should transports support async? Some benchmarks use async internally.
