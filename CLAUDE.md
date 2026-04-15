# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is Exgentic?

Exgentic is a universal evaluation framework for AI agents. It provides a standardized interface to evaluate any agent on any benchmark (tau2, gsm8k, swebench, hotpotqa, appworld, bfcl, browsecompplus). The core abstraction is a `Session` loop: a `Benchmark` produces tasks, an `Agent` reacts to observations with actions, and the orchestrator drives the loop until termination.

## Common Commands

```bash
# Install dependencies (always use --frozen to prevent silent upgrades)
uv sync --frozen --extra dev --extra analysis

# Run core tests (no Docker or external services needed)
uv run pytest tests/ --ignore=tests/integrations --ignore=tests/adapters/runners

# Run a single test file
uv run pytest tests/api/test_api_limits.py -v

# Run a single test by name
uv run pytest tests/ -k "test_name" -v

# Runner/transport tests (includes Docker tests)
uv run pytest tests/adapters/runners -v -p no:faulthandler

# Skip tests requiring external services
uv run pytest tests/ -k "not litellm and not mcp"

# Linting (ruff + codespell + custom hooks)
pre-commit run --all-files

# Run an evaluation
uv run exgentic evaluate --benchmark tau2 --agent tool_calling --subset retail --num-tasks 2 \
  --model gpt-4o --set benchmark.user_simulator_model="gpt-4o"

# List benchmarks/agents
uv run exgentic list benchmarks
uv run exgentic list agents
```

## Architecture

### Core Loop (`src/exgentic/core/`)

The evaluation loop is an observe-react-step cycle orchestrated in `core/orchestrator/session.py`:

1. **Benchmark** (`core/benchmark.py`) - Pydantic config that creates `Session` and `Evaluator` instances via runners
2. **Session** (`core/session.py`) - ABC representing one task execution. Owns `task`, `context`, `actions`, `start()`, `step(action)`, `done()`, `score()`, `close()`
3. **Agent** (`core/agent.py`) - Pydantic config that creates `AgentInstance` via runners
4. **AgentInstance** (`core/agent_instance.py`) - ABC with `react(observation) -> Action` and `start(task, context, actions)`
5. **Evaluator** (`core/evaluator.py`) - ABC for task discovery (`list_tasks`), session config, and result aggregation
6. **Controller** (`core/orchestrator/controller.py`) - Hooks for validation (CoreController), limits (LimitController), cleanup
7. **Observer** (`core/orchestrator/observer.py`) - Event handlers for logging, tracing, dashboard updates

### Action System (`core/types/action.py`, `core/actions.py`)

Actions are Pydantic models. `SingleAction` has `name`, `arguments` (BaseModel), and `validation`. `ParallelAction`/`SequentialAction` wrap multiple single actions. `ActionsHandler` is a registry that maps action names to handler callables and performs validation/dispatch.

### Registry (`interfaces/registry.py`)

All benchmarks and agents are registered as `RegistryEntry` in `BENCHMARKS` and `AGENTS` dicts. Entries use lazy loading (`module:attr` strings) to avoid importing heavy deps on the host.

### Runners (`adapters/runners/`)

Runners control isolation level: `direct` (same thread), `thread`, `process`, `service` (HTTP in background thread), `venv` (isolated uv virtualenv), `docker`. `with_runner(cls, runner=...)` wraps any class. Benchmarks and agents default to `venv` runner for dependency isolation.

### Adapters (`adapters/`)

- `adapters/agents/mcp_agent.py` / `mcp_server.py` - MCP protocol integration for tool-calling agents
- `adapters/agents/a2a_executor.py` - Agent-to-Agent (A2A) protocol support
- `adapters/agents/coordinator.py` - Multi-agent coordination
- `adapters/actions/chat.py` - Chat action adapter
- `adapters/schemas/json_schema.py` - JSON schema conversion for action types

### Interfaces

- **CLI** (`interfaces/cli/`) - Click-based CLI, entry point `exgentic.interfaces.cli.main:main`
- **Python API** (`interfaces/lib/api.py`) - `evaluate()`, `execute()`, `aggregate()` functions exposed via `exgentic.__init__`
- **Dashboard** (`interfaces/dashboard/`) - NiceGUI-based web dashboard

### Observers (`observers/`)

Event-driven system with handlers for file logging, session ledger, dashboard events, recap, warnings. Optional OpenTelemetry tracing via `observers/tracing/`.

## Code Conventions

### File Headers

Every Python file under `src/` must start with:
```python
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025, The Exgentic organization and its contributors.
```

### Import Rules (enforced by pre-commit hooks)

- **Inside `src/exgentic/`**: Use relative imports only (`from ..core import ...`), never `from exgentic import ...`
- **Outside `src/`** (tests, examples, misc): Use library imports (`from exgentic import ...`), never `from src import ...`

### Style

- Ruff for linting and formatting. Line length: 120. Google-style docstrings.
- Double quotes, 4-space indent, Unix line endings.
- `asyncio_mode = "strict"` for pytest-asyncio (use `@pytest.mark.asyncio` explicitly).

### Dependency Management

- All dependencies in `pyproject.toml` must have version caps (enforced by `enforce_dependency_caps.py` hook).
- Use `uv sync --frozen` locally. Never plain `uv sync` (prevents silent upgrades).
- To upgrade: `uv lock --upgrade-package <name>`, review diff, commit.

### Commits

- Sign off commits with `-s` flag (DCO requirement).
- PR titles follow [Conventional Commits](https://www.conventionalcommits.org/).
