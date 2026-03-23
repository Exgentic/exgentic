---
name: add-agent
description: Use when adding or updating an agent adapter in the Exgentic repository. Follow the repository agent principles, separate lightweight config from heavy execution logic, isolate third-party dependencies behind lazy imports, adapt to any benchmark contract without requiring benchmark changes, and validate the adapter with representative smoke tests before finishing.
---

# Add Agent

Use this skill when working on agent adapters in the Exgentic repository.

## First read

Start with:
- `docs/adding-agents.md`
- `src/exgentic/core/agent.py`
- `src/exgentic/core/agent_instance.py`
- `src/exgentic/interfaces/registry.py`

Then inspect the most relevant existing adapters:
- `src/exgentic/agents/litellm_tool_calling/litellm_tool_calling_agent.py` + `instance.py` (split pattern: heavy deps in separate file)
- `src/exgentic/agents/cli/claude/agent.py` (light pattern: everything in one file)

## Workflow

1. Decide whether to split files.
   If the agent depends on heavy third-party libraries (litellm, smolagents, openai SDK, etc.), put the Agent in one file and the AgentInstance in `instance.py`. If deps are light, keep both in a single file.

2. Implement the Agent class first.
   Subclass `Agent`. Declare `display_name` and `slug_name` as `ClassVar[str]`. Add user-facing config fields. Implement `get_instance_class()` with a lazy import. Implement `get_instance_kwargs()` to translate config and the benchmark contract into instance constructor kwargs.

3. Implement the AgentInstance class.
   Subclass `AgentInstance`. Accept `session_id` plus the kwargs from `get_instance_kwargs()`. Call `super().__init__(session_id)`. Implement `react()` as the core decision loop and `close()` for cleanup. Optionally override `start()` and `get_cost()`.

4. Add `requirements.txt` for agent-specific dependencies.
   List only packages not already in the base exgentic install. Place the file in the agent's package directory; `RunnerMixin` discovers it automatically.

5. Add `setup.sh` if non-pip setup is needed.
   Place it next to the agent module; `RunnerMixin` discovers it automatically.

6. Register the agent in `src/exgentic/interfaces/registry.py`.
   Add a `RegistryEntry` to the `AGENTS` dict. Ensure `slug_name` and `display_name` match the class exactly.

7. Validate the adapter as an agent, not just as code.
   Check registry loading, dependency isolation, at least one end-to-end benchmark run, cost reporting, and cleanup.

## Non-negotiable rules

- The Agent file must be importable without installing agent-specific packages.
- `get_instance_class()` must use a lazy import to isolate heavy deps.
- `get_instance_kwargs()` must faithfully pass the benchmark contract (task, context, actions, session_id) through to the instance.
- The instance constructor must call `super().__init__(session_id)`.
- `react()` must return `None` when the agent decides it is done.
- `close()` must not raise exceptions.
- `slug_name` and `display_name` in the registry entry must exactly match the class values.
- The agent must adapt to the benchmark, never the other way around.

## Validation

Before finishing, run at least:
- `python -m py_compile` on changed agent files
- `pre-commit run --files ...`
- `git diff --check`

Also confirm:
- `load_agent("slug_name")` succeeds from a Python shell
- The Agent file imports cleanly without agent-specific packages installed
- At least one benchmark runs end to end with the new agent
- `close()` completes without error
- `get_cost()` returns a valid report
