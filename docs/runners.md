# Runners

A runner controls how benchmark and agent code is executed — what process, what environment, and how isolated it is from your host machine.

**Related docs:**
[docs/](./README.md) · [Adding Benchmarks](./adding-benchmarks.md) · [Adding Agents](./adding-agents.md) · [Custom Models](./custom-models.md)

---

## Runner comparison

| Runner | Isolation | Startup | Use when |
|--------|-----------|---------|----------|
| `direct` | None | Instant | Development, unit tests, benchmarks with no conflicting deps |
| `thread` | Thread | Instant | I/O-bound tasks, lightweight isolation |
| `process` | Process | Fast | Memory isolation without venv overhead |
| `venv` | Virtual environment | Slow (first run), fast (cached) | Default for production; each benchmark in its own env |
| `docker` | Container | Slow | Full OS isolation; required for SWE-bench and Docker-in-Docker |

---

## Selecting a runner

### CLI

```bash
exgentic evaluate --benchmark tau2 --agent tool_calling \
  --set benchmark.runner=venv
```

### Config file

```json
{
  "benchmark_kwargs": { "runner": "venv" }
}
```

### Python API

```python
from exgentic import evaluate

evaluate(
    benchmark="tau2",
    agent="tool_calling",
    benchmark_kwargs={"runner": "venv"},
)
```

The default runner is `venv`. For production evaluations, `venv` is recommended.

---

## direct

Executes benchmark and agent code in the same thread as the caller. No isolation.

**Good for:** development, debugging, benchmarks that share a dependency set with your host environment.

**Not good for:** benchmarks that require conflicting dependencies, or when you need to keep your host environment clean.

---

## thread

Runs the benchmark in a separate daemon thread. Communication happens over an in-process queue.

**Good for:** lightweight isolation, I/O-bound benchmarks.

---

## process

Spawns a child process using `multiprocessing.spawn`. Objects are serialised with `cloudpickle`.

**Good for:** memory isolation when you don't need a separate Python environment.

---

## venv

Creates an isolated virtual environment using `uv` and runs the benchmark inside it via a local HTTP service.

On first run, the venv is created and all dependencies are installed — this can take a few minutes. Subsequent runs reuse the cached venv.

### Setup resolution

When a venv runner starts, it:

1. Looks for a `requirements.txt` file starting from the benchmark's Python module directory, walking up the package tree until it finds one with non-comment content.
2. Installs those packages into the venv with `uv pip install`.
3. If a `setup.sh` exists alongside the module, runs it after installation.

To trigger a fresh install (e.g. after updating `requirements.txt`):

```bash
exgentic install --benchmark tau2 --force
```

### Configuration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `venv_dir` | string | `{cache_dir}/venv/` | Where to create the virtual environment |
| `port` | int | auto | Port for the local HTTP service |
| `dependencies` | list[string] | null | Extra packages to install |

---

## docker

Builds a Docker image containing the benchmark and runs it as a container. Communicates over HTTP.

### Prerequisites

Docker must be installed and running:

```bash
docker info   # should succeed
```

### How image building works

1. A `python:3.12-slim` base image is used unless you provide a custom image or Dockerfile.
2. If `requirements.txt` is found, it's copied in and installed during image build.
3. If `setup.sh` is found, it runs at build time.
4. The resulting image is tagged and cached; it's only rebuilt when the setup files change.

### Configuration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `image` | string | null | Use a pre-built image; skips building |
| `dockerfile` | string | null | Path to a custom Dockerfile |
| `port` | int | auto | Host port mapped to the container service |
| `docker_args` | list[string] | null | Extra arguments appended to `docker run` |
| `volumes` | dict | null | Volume mounts: `{"host/path": "container/path"}` |
| `docker_socket` | bool | false | Mount the host Docker socket into the container |
| `dependencies` | list[string] | null | Extra packages to install at build time |

### Docker-in-Docker

Some benchmarks (e.g. SWE-bench) need to launch Docker containers themselves. Enable socket passthrough:

```bash
exgentic evaluate --benchmark swebench --agent tool_calling \
  --set benchmark.runner=docker \
  --set benchmark.docker_socket=true
```

This mounts `/var/run/docker.sock` from the host into the container, giving the benchmark access to the host Docker daemon.

### Volume mounts

```python
evaluate(
    benchmark="my_benchmark",
    benchmark_kwargs={
        "runner": "docker",
        "volumes": {
            "/host/data": "/container/data",
        },
    },
)
```

---

## setup.sh and requirements.txt

Both files are auto-discovered — you do not need to register them explicitly.

The framework walks up from the benchmark or agent's module file looking for these files. The first `requirements.txt` with non-comment, non-empty content wins.

### requirements.txt

Standard pip requirements file. Installed into the venv or Docker image before the benchmark starts.

```
tau2bench>=0.1.0
some-other-dep==1.2.3
```

Git LFS objects are automatically skipped during install (`GIT_LFS_SKIP_SMUDGE=1`).

### setup.sh

Shell script for setup that can't be expressed as pip packages: cloning repositories, compiling binaries, downloading model weights, etc.

```bash
#!/usr/bin/env bash
set -euo pipefail

# Example: clone a dependency
git clone --depth 1 https://github.com/example/repo /opt/repo
```

Place it in the same directory as your benchmark module. It runs after `requirements.txt` is installed.

---

## See also

- [Adding Benchmarks](./adding-benchmarks.md) — how benchmarks declare their setup
- [Adding Agents](./adding-agents.md) — how agents declare their setup
- [Custom Models](./custom-models.md) — configuring the LLM behind the agent
- [docs/](./README.md) — documentation index
