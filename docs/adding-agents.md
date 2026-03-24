# Adding Agents

This document defines the agent design principles for Exgentic.

It is intentionally opinionated. An agent adapter should not just "work"; it should cleanly separate configuration from execution, isolate heavy dependencies, and adapt to any benchmark contract without requiring the benchmark to change.

Use these existing adapters as reference points:
- `src/exgentic/agents/litellm_tool_calling/litellm_tool_calling_agent.py` + `instance.py` (split pattern)
- `src/exgentic/agents/cli/claude/agent.py` (light pattern, single file)

## Core Principle

The agent adapts to the benchmark contract, not the other way around.

That means:
- the benchmark decides the task, context, actions, step flow, and scoring
- the agent receives these through `get_instance_kwargs()` and must work within them
- the agent should not require benchmark modifications to function
- the agent should not impose protocol-specific assumptions on the benchmark

The default goal should be the thinnest possible agent wrapper.

That means:
- translate the benchmark's actions into whatever protocol your agent uses (tool calls, code generation, CLI commands)
- do not reshape the benchmark contract to match your model's preferred format
- keep the configuration surface small and explicit

## Architecture

Exgentic agents are split into two classes with distinct roles:

### Agent (config, host-side)

`Agent` is a lightweight Pydantic model that holds configuration. It lives on the host and is never sent into an isolated runner. It has no heavy dependencies.

Responsibilities:
- declare `display_name` and `slug_name` as `ClassVar[str]`
- hold user-facing configuration fields (model name, max steps, feature flags)
- implement `get_instance_class()` to resolve the execution class
- implement `get_instance_kwargs()` to translate config + benchmark contract into constructor arguments
- optionally override `setup()` for non-pip setup (Docker builds, npm installs)
- optionally override `model_name` / `get_models_names()` for dashboard metadata

### AgentInstance (execution, venv-side)

`AgentInstance` is the execution class. It runs inside the runner (venv, Docker, or local) and may import heavy third-party libraries.

Responsibilities:
- implement `react(observation) -> Action | None` as the core decision loop
- implement `close()` for resource cleanup
- optionally override `start()` for initialization that happens after construction
- optionally override `get_cost()` to report monetary cost

The agent instance receives a single `session_id` in its constructor, which scopes all logs and artifacts. Additional kwargs come from `get_instance_kwargs()`.

## Key Pattern: Lazy Import for Dependency Isolation

The `get_instance_class()` classmethod must use a lazy import so that heavy dependencies are only loaded inside the runner environment, not on the host.

```python
@classmethod
def get_instance_class(cls):
    from .instance import MyAgentInstance

    return MyAgentInstance
```

This is the same pattern that `Benchmark.get_session_class()` uses. It ensures the host process never imports libraries like `litellm`, `smolagents`, `openai`, or any other agent-specific SDK.

## When to Split Files

**Split into separate files** when your agent depends on heavy third-party libraries:

```
src/exgentic/agents/my_agent/
    __init__.py
    my_agent.py          # Agent subclass (light, no heavy imports)
    instance.py           # AgentInstance subclass (imports litellm, openai, etc.)
    requirements.txt      # Agent-specific pip dependencies
    setup.sh              # Optional non-pip setup
    utils.py              # Optional helpers
```

Examples: `litellm_tool_calling`, `smolagents`, `openai`

**Keep everything in one file** when dependencies are light or already available in the base environment:

```
src/exgentic/agents/my_agent/
    __init__.py
    agent.py              # Both Agent and AgentInstance in one file
```

Example: `cli/claude` (the instance class is in the same file because it only depends on stdlib and core Exgentic types)

The rule is simple: if importing the instance class would pull in packages that are not in the base `exgentic` install, split the files.

## Required Methods

### On the Agent class

#### `get_instance_class()` (classmethod, abstract)

Returns the `AgentInstance` subclass. Must use a lazy import.

```python
@classmethod
def get_instance_class(cls):
    from .instance import MyAgentInstance

    return MyAgentInstance
```

#### `get_instance_kwargs()` (abstract)

Translates the agent's configuration into constructor kwargs for the instance class. Task, context, and actions are passed separately via `start()`, not through the constructor.

```python
def get_instance_kwargs(self, session_id: str) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "model": self.model,
        "max_steps": self.max_steps,
    }
```

The returned dict is passed directly to the instance class constructor. Every key must match a constructor parameter.

### On the AgentInstance class

#### `react(observation) -> Action | None` (abstract)

The core decision loop. Receives an `Observation` (or `None` on the first call) and returns an `Action` to take, or `None` to signal that the agent is done.

```python
def react(self, observation: Observation | None) -> Action | None:
    # Process observation, decide next action
    # Return None when the agent decides it is finished
    ...
```

#### `close()` (abstract)

Cleanup resources. Called when the session ends, whether or not the agent finished normally.

```python
def close(self) -> None:
    # Release connections, flush logs, etc.
    pass
```

#### `start(task, context, actions)` (optional override)

Called after construction but before the first `react()`. Receives the benchmark's task string, context dict, and list of action types. The base implementation stores these as `self.task`, `self.context`, and `self.actions`. Override to perform initialization that depends on these values (e.g., seeding a conversation with the task prompt).

#### `get_cost()` (optional)

Returns a `CostReport` with estimated monetary cost. Default returns an empty report. Override to track API costs.

```python
def get_cost(self) -> CostReport:
    return self._cost_data
```

## Registration

Every agent must be registered in `src/exgentic/interfaces/registry.py` in the `AGENTS` dict.

```python
AGENTS: dict[str, RegistryEntry] = {
    # ...existing entries...
    "my_agent": RegistryEntry(
        slug_name="my_agent",
        display_name="My Agent",
        module="exgentic.agents.my_agent.my_agent",
        attr="MyAgent",
        kind="agent",
    ),
}
```

Requirements:
- `slug_name` must match the `slug_name` ClassVar on the Agent class exactly
- `display_name` must match the `display_name` ClassVar on the Agent class exactly
- `module` is the dotted Python module path to the file containing the Agent class
- `attr` is the class name within that module
- `kind` must be `"agent"`

The registry validates these constraints at load time. Mismatches will raise at startup.

## Setup

### `requirements.txt`

List agent-specific pip dependencies. The runner installs these automatically into the isolated environment.

```
litellm>=1.50.0
```

Place the file in the agent's package directory. The `RunnerMixin` auto-discovers it by walking up from the module file.

### `setup.sh`

Optional script for non-pip setup. Runs after dependencies are installed.

```bash
#!/usr/bin/env bash
set -euo pipefail
# Build Docker images, install npm packages, download models, etc.
```

Place it next to the agent module. The `RunnerMixin` auto-discovers it.

Both files are automatically found by the framework through `RunnerMixin.requirements_txt` and `RunnerMixin.setup_script`. No manual wiring is needed.

## Recommended File Structure

### Split pattern (heavy deps)

```
src/exgentic/agents/my_agent/
    __init__.py
    my_agent.py              # Agent subclass
    instance.py              # AgentInstance subclass
    requirements.txt         # e.g., litellm>=1.50.0
    setup.sh                 # optional
    utils.py                 # optional helpers
```

**my_agent.py** (host-side, no heavy imports):

```python
from __future__ import annotations

from typing import Any, ClassVar

from ...core.agent import Agent
from ...core.types import ActionType, ModelSettings


class MyAgent(Agent):
    display_name: ClassVar[str] = "My Agent"
    slug_name: ClassVar[str] = "my_agent"

    model: str = "gpt-4o"
    max_steps: int = 100

    @classmethod
    def get_instance_class(cls):
        from .instance import MyAgentInstance

        return MyAgentInstance

    @property
    def model_name(self) -> str:
        return self.model

    def get_instance_kwargs(self, session_id: str) -> dict[str, Any]:
        return {
            "session_id": session_id,
            "model": self.model,
            "max_steps": self.max_steps,
        }
```

**instance.py** (runner-side, may import heavy libs):

```python
from __future__ import annotations

from typing import Any, Optional

import some_heavy_library  # only loaded inside the runner

from ...core.agent_instance import AgentInstance
from ...core.types import Action, ActionType, Observation


class MyAgentInstance(AgentInstance):
    def __init__(
        self,
        session_id: str,
        task: str,
        context: dict[str, Any],
        actions: list[ActionType],
        model: str,
        max_steps: int,
    ):
        super().__init__(session_id)
        self.task = task
        self.context = context
        self.actions = actions
        self.model = model
        self.max_steps = max_steps
        self._step_count = 0

    def react(self, observation: Optional[Observation]) -> Optional[Action]:
        self._step_count += 1
        if self._step_count > self.max_steps:
            return None
        # Agent decision logic here
        ...

    def close(self) -> None:
        pass
```

### Light pattern (no heavy deps)

```
src/exgentic/agents/my_agent/
    __init__.py
    agent.py                 # Both Agent and AgentInstance
```

Keep both classes in a single file when the instance has no third-party imports beyond what exgentic already provides.

## Validation Checklist

Before opening a PR for a new agent, validate all of the following.

### Contract validation

- Agent class declares `display_name` and `slug_name` as `ClassVar[str]`
- `get_instance_class()` uses a lazy import
- `get_instance_kwargs()` returns a dict whose keys match the instance constructor
- The instance implements `react()` and `close()`
- The instance calls `super().__init__(session_id)` in its constructor

### Dependency isolation validation

- The Agent file does not import heavy third-party libraries at module level
- Heavy imports only appear inside `get_instance_class()` or in the instance module
- `requirements.txt` lists all agent-specific dependencies

### Registry validation

- `slug_name` in the registry entry matches the class `slug_name` exactly
- `display_name` in the registry entry matches the class `display_name` exactly
- `module` path resolves to the correct file
- `attr` matches the Agent class name
- `kind` is `"agent"`

### Functional validation

- Agent is discoverable through the registry (`load_agent("my_agent")` succeeds)
- Agent works with at least one benchmark end to end
- `react()` correctly returns `None` when the agent decides it is done
- `close()` does not raise
- `get_cost()` returns a valid `CostReport`

### Quality validation

- `py_compile` passes for all changed Python files
- `pre-commit` passes for changed files
- `git diff --check` passes

## Practical Rule of Thumb

When in doubt, ask:

1. Does the agent adapt to the benchmark, or does it require the benchmark to change?
2. Are heavy dependencies isolated behind a lazy import?
3. Is the Agent file importable without installing agent-specific packages?
4. Does `get_instance_kwargs()` faithfully pass the benchmark contract through?
5. Will this agent work with benchmarks that have very different action spaces?
6. Is the configuration surface minimal and explicit?

If the answer to any of those is no, the adapter is probably too coupled or too leaky.
