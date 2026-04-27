# RITS Main Migration Plan

## Summary

The old `feat/rits-appworld-structured-outputs` branch diverged from `main` at
`0c399d1`, while latest `origin/main` is `984a3ef`. The valuable changes should
be recreated on top of latest `main`, not cherry-picked wholesale.

RITS support should be one PR, not split across resolver and agent integration
PRs. The resolver, health-check plumbing, and agent call-site integration are a
single behavior: accepting `rits/...` model aliases and making LiteLLM calls work
against RITS endpoints.

## Classified Changes

### Preserve: RITS LLM Support

- `src/exgentic/integrations/litellm/rits_resolver.py` and
  `src/exgentic/integrations/litellm/__init__.py`
  - Old commit: `141cfc2`.
  - Preserve the capability to resolve `rits/<model>` and
    `rits/<provider>/<model>` into LiteLLM-compatible `hosted_vllm/<model>`,
    `api_base`, `api_key`, and RITS API-key headers.
  - Reimplement manually with timeout, validation, and clear errors.

- `src/exgentic/integrations/litellm/health.py`
  - Old commit: `141cfc2`.
  - Preserve forwarding of provider kwargs into `litellm.acompletion()`.
  - Reimplement manually because latest `main` changed health retry behavior.

- `src/exgentic/agents/litellm_tool_calling/instance.py`
  - Old commit: `141cfc2`.
  - Preserve RITS resolution at init, resolved health checks, resolved
    completion kwargs, and local handling for unknown model pricing.
  - Reimplement manually against current `main`.

- `src/exgentic/agents/smolagents/base_instance.py`
  - Old commit: `141cfc2`.
  - Preserve RITS resolution for smolagents model construction and health
    checks, plus local unknown-pricing fallback.
  - Reimplement manually and verify the kwargs expected by smolagents/LiteLLM.

### Possibly Preserve Later

- `src/exgentic/agents/smolagents/base_agent.py`,
  `src/exgentic/agents/smolagents/code_instance.py`, and
  `src/exgentic/agents/smolagents/tool_calling_instance.py`
  - Old commit: `00c64a8`.
  - `use_structured_outputs` configurability is useful, but it is not required
    for RITS support. Port separately unless a RITS model requires it.

- `src/exgentic/benchmarks/tau2/tau2_eval.py`
  - Old commit: `141cfc2`.
  - RITS user-simulator support may be useful, but it is not part of the first
    RITS agent migration pass.

- `src/exgentic/interfaces/cli/commands/serve.py`
  - Old commit: `7184903`.
  - CE-Manager bootstrap may be intentional, but it is external to RITS and
    should be a separate PR if still desired.

### Preserve As Separate Bug Fix

- `src/exgentic/benchmarks/appworld/appworld_eval.py`
  - Old commit: `ef055a5`.
  - Preserve only the `world.save()` before `evaluate_task()` scoring fix.
  - Latest `main` already has the EnvironmentManager root fix and newer
    completion-capture behavior.

### Drop

- `src/exgentic/environment/helpers.py`
  - Old commit: `7184903`.
  - Drop editable installs by default. Latest `main` added atomic venv publishing
    and install locking; editable installs weaken reproducibility.

- `src/exgentic/utils/cost.py`
  - Old commit: `141cfc2`.
  - Drop global conversion of unknown LiteLLM pricing to zero. Keep the current
    `main` behavior of raising for unknown pricing, and handle RITS fallback
    locally in the relevant agents.

- Untracked scratch tests from the old checkout:
  - `tests/api/test_observer_lifecycle.py`
  - `tests/benchmarks/test_appworld_session_lifecycle.py`
  - Do not port as-is; they target different concerns or stale internals.

## Implementation Plan

1. Add a hardened RITS resolver and export it from the LiteLLM integration
   package.
2. Add LiteLLM health-check kwargs forwarding.
3. Integrate RITS resolution into LiteLLM tool-calling and smolagents agents.
4. Keep CE-Manager, editable venv install, Tau2 RITS, AppWorld scoring, and
   smolagents structured-output toggles for follow-up work.

## Test Plan

- Resolver tests for missing env vars, successful direct and provider-prefixed
  model resolution, non-200 responses, malformed payloads, and cache clearing.
- Health tests proving provider kwargs reach `litellm.acompletion()`.
- Agent tests proving RITS is resolved once, health checks and completions use
  resolved kwargs, and unknown pricing does not crash agent cost reporting.

