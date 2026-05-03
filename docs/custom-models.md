# Custom Models

Exgentic routes all LLM calls through [LiteLLM](https://docs.litellm.ai/), which means any provider or deployment LiteLLM supports works out of the box — no code changes required. You pick the model, supply credentials, and optionally tune sampling parameters.

**Related docs:**
[docs/](./README.md) · [CLI Reference](./cli-reference.md) · [Python API](./python-api.md) · [Adding Agents](./adding-agents.md) · [Observability Quick Start](./observability/quickstart.md)

---

## Model string format

The `--model` flag (and the `model` parameter in the Python API) accepts any model string that LiteLLM recognises. The general pattern is:

```
<provider>/<model-name>
```

For OpenAI-native models the provider prefix is optional:

```bash
# These are equivalent
--model gpt-4o
--model openai/gpt-4o
```

For every other provider the prefix is required. See the provider examples below.

---

## Supported providers

### OpenAI

```bash
exgentic evaluate --benchmark gsm8k --agent tool_calling \
  --model gpt-4o
```

Required environment variable:

```bash
export OPENAI_API_KEY=sk-...
```

### Anthropic

```bash
exgentic evaluate --benchmark gsm8k --agent tool_calling \
  --model claude-3-5-sonnet-20241022
```

Required environment variable:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

### Azure OpenAI

```bash
exgentic evaluate --benchmark gsm8k --agent tool_calling \
  --model azure/<your-deployment-name>
```

Required environment variables:

```bash
export AZURE_API_KEY=...
export AZURE_API_BASE=https://<your-resource>.openai.azure.com
export AZURE_API_VERSION=2024-02-01   # or whichever version your deployment uses
```

### AWS Bedrock

```bash
exgentic evaluate --benchmark gsm8k --agent tool_calling \
  --model bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0
```

Required environment variables:

```bash
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_REGION_NAME=us-east-1
```

### Google Vertex AI

```bash
exgentic evaluate --benchmark gsm8k --agent tool_calling \
  --model vertex_ai/gemini-1.5-pro
```

Required environment variables:

```bash
export VERTEXAI_PROJECT=my-gcp-project
export VERTEXAI_LOCATION=us-central1
```

### Ollama (local)

```bash
exgentic evaluate --benchmark gsm8k --agent tool_calling \
  --model ollama/llama3
```

Required: Ollama running locally. Set the base URL if it differs from the default:

```bash
export OLLAMA_API_BASE=http://localhost:11434   # default; only needed if different
```

### Any OpenAI-compatible endpoint

```bash
exgentic evaluate --benchmark gsm8k --agent tool_calling \
  --model openai/<model-name>
```

Override the base URL:

```bash
export OPENAI_API_BASE=http://localhost:8080/v1
export OPENAI_API_KEY=any-non-empty-string       # required by the client even if unused
```

This works with vLLM, LM Studio, LocalAI, Together AI, Fireworks, Anyscale, and any other OpenAI-compatible server.

### LiteLLM proxy

If you run a [LiteLLM proxy server](https://docs.litellm.ai/docs/proxy/quick_start) in front of your models:

```bash
exgentic evaluate --benchmark gsm8k --agent tool_calling \
  --model openai/<model-alias-on-proxy>
```

```bash
export OPENAI_API_BASE=http://localhost:4000
export OPENAI_API_KEY=<your-proxy-key>
```

---

## Sampling parameters

Use `--set agent.model.*` to control sampling. These map to `ModelSettings` and are forwarded to LiteLLM on every completion call.

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--set agent.model.temperature` | float ≥ 0 | `1.0` | Sampling temperature |
| `--set agent.model.top_p` | float 0–1 | `null` | Nucleus sampling |
| `--set agent.model.max_tokens` | int ≥ 0 | `null` | Maximum tokens in response |
| `--set agent.model.reasoning_effort` | string | `null` | Reasoning effort level (o1/o3 models) |
| `--set agent.model.num_retries` | int ≥ 0 | `5` | Retries on transient errors |
| `--set agent.model.retry_after` | float ≥ 0 | `0.5` | Initial retry delay in seconds |
| `--set agent.model.retry_strategy` | string | `exponential_backoff` | `exponential_backoff` or `constant` |

Example — lower temperature and capped output:

```bash
exgentic evaluate --benchmark tau2 --agent tool_calling \
  --model gpt-4o \
  --set benchmark.user_simulator_model="gpt-4o" \
  --set agent.model.temperature=0.2 \
  --set agent.model.max_tokens=2048
```

---

## Python API

```python
from exgentic import evaluate
from exgentic.core.types import ModelSettings

results = evaluate(
    benchmark="tau2",
    agent="tool_calling",
    subset="retail",
    num_tasks=5,
    model="azure/my-gpt-4o-deployment",
    agent_kwargs={
        "model_settings": ModelSettings(
            temperature=0.2,
            max_tokens=2048,
            num_retries=3,
        )
    },
    benchmark_kwargs={"user_simulator_model": "gpt-4o"},
)
```

---

## Non-standard backends (custom headers, custom auth)

Some OpenAI-compatible gateways require auth to ride in a custom HTTP header (e.g. `X-Backend-Auth: <token>`) instead of the standard `Authorization: Bearer <token>`. Setting `OPENAI_API_BASE` alone isn't enough in that case — LiteLLM also needs to know to send the custom header.

For these backends, every Exgentic agent and benchmark accepts a `litellm_params_extra` dict that is forwarded to LiteLLM verbatim. It accepts any key LiteLLM's per-model `litellm_params` block accepts: `api_base`, `api_key`, `extra_headers`, an alias-vs-actual `model` split, etc.

### CLI

```bash
exgentic evaluate --benchmark gsm8k --agent tool_calling \
  --model openai/my-model \
  --set agent.litellm_params_extra='{"api_base":"https://gateway.example/v1","api_key":"$BACKEND_KEY","extra_headers":{"X-Backend-Auth":"$BACKEND_KEY"}}'
```

The value after `=` is parsed as JSON. Quote it carefully in your shell.

For benchmarks that drive their own LLM (e.g. Tau2's user simulator, BrowseComp's judge), use the benchmark's own field:

```bash
# Tau2 user simulator
--set benchmark.user_simulator_litellm_params_extra='{"api_base":"...","extra_headers":{"X-Backend-Auth":"$KEY"}}'
```

### Python API

```python
from exgentic import evaluate

extras = {
    "api_base": "https://gateway.example/v1",
    "api_key": os.environ["BACKEND_KEY"],
    "extra_headers": {"X-Backend-Auth": os.environ["BACKEND_KEY"]},
}

results = evaluate(
    benchmark="gsm8k",
    agent="tool_calling",
    num_tasks=5,
    model="openai/my-model",
    agent_kwargs={"litellm_params_extra": extras},
    # For Tau2: benchmark_kwargs={"user_simulator_litellm_params_extra": extras}
)
```

### Aliasing one model name to another

You can expose a friendly alias to clients while routing to a different underlying model:

```python
agent_kwargs={
    "litellm_params_extra": {
        "model": "hosted_vllm/some-model",  # what LiteLLM actually calls
        "api_base": "...",
        "api_key": "...",
    },
}
# Clients still address the agent as "my-alias" via the `model=` argument.
```

### Where this is wired

The same `litellm_params_extra` dict reaches every LiteLLM call site Exgentic owns:

- The LiteLLM proxy started by CLI agents (Codex, Gemini, Claude Code).
- The direct `litellm.acompletion` calls in `LiteLLMToolCallingAgentInstance` and the model-accessibility health check.
- The smolagents `LiteLLMModel` constructor.
- The openai-agents `LitellmModel` (with `api_base` mapped to its `base_url` constructor kwarg, `extra_headers` to `OpenAIModelSettings.extra_headers`, and any other keys forwarded as `extra_args`).
- The judge model in `BrowseCompEvaluator` and the user simulator in `TAU2Session`.

So the same config works everywhere — pick the agent and benchmark you want, supply the dict once, and you're done.

### Custom-model pricing (cost tracking for non-standard backends)

Exgentic's cost tracking calls `litellm.cost_per_token(model=...)`, which looks the model up in LiteLLM's built-in pricing database. For non-standard backends (your own gateway, an internal deployment, RITS, etc.) the model isn't in that database, so the lookup raises `ValueError("No pricing info found for model '...'")` and your run fails when it tries to record cost.

LiteLLM's native fix is `litellm.register_model()` — register your model once, before constructing the agent, and the lookup succeeds from then on:

```python
import litellm

litellm.register_model({
    "hosted_vllm/granite": {
        "input_cost_per_token": 0.0001,    # USD per token
        "output_cost_per_token": 0.0002,
        "litellm_provider": "hosted_vllm",
    },
})

# Now any agent using "hosted_vllm/granite" will report accurate cost.
agent = LiteLLMToolCallingAgentInstance(
    session_id=...,
    model="hosted_vllm/granite",
    litellm_params_extra={"api_base": "https://gateway.example/v1"},
)
```

If you don't know the rates, set both to `0`:

```python
litellm.register_model({
    "hosted_vllm/granite": {
        "input_cost_per_token": 0,
        "output_cost_per_token": 0,
        "litellm_provider": "hosted_vllm",
    },
})
```

Cost will report as `$0` — your explicit choice, not a silent fallback. The framework deliberately doesn't auto-suppress the pricing-lookup failure: a silent `$0` for an unregistered model would hide a real bill from you.

Reference: [LiteLLM custom pricing docs](https://docs.litellm.ai/docs/proxy/custom_pricing).

> **CLI users**: cost registration currently only works through the Python API (the CLI subprocess has no place to slip in a `register_model` call). If you're driving from the CLI against a non-standard backend, either run via `exgentic.evaluate(...)` from a script that does the registration first, or accept the cost-tracking failure for now. A follow-up will add CLI support if there's demand.

---

## Reasoning models

For models that support reasoning effort (OpenAI o1, o3, etc.):

```bash
exgentic evaluate --benchmark swebench --agent tool_calling \
  --model o3 \
  --set agent.model.reasoning_effort=high \
  --set agent.model.max_tokens=32768
```

Note: temperature is typically fixed at 1 for reasoning models and will be ignored if set.

---

## Cost tracking

Exgentic records token counts and estimated cost for every LiteLLM completion automatically. Results appear in:

- `outputs/<run_id>/results.json` — aggregate cost across all sessions
- `outputs/<run_id>/sessions/<session_id>/results.json` — per-session cost

Cost estimates are calculated using LiteLLM's built-in pricing database. For providers or custom deployments not in the database, cost will show as `0`.

---

## Caching

LiteLLM-level response caching is enabled by default. To disable it for a run:

```bash
export EXGENTIC_LITELLM_CACHING=false
```

The cache directory defaults to `.litellm_cache` in the working directory. To move it:

```bash
export EXGENTIC_LITELLM_CACHE_DIR=/path/to/cache
```

---

## Observability

All LLM inference calls emit OpenTelemetry spans automatically when tracing is enabled. See [Observability Quick Start](./observability/quickstart.md) to set up tracing, and [Semantic Conventions](./observability/semantic-conventions.md) for the full attribute reference.

---

## Further reading

- [LiteLLM providers documentation](https://docs.litellm.ai/docs/providers)
- [Adding a new agent adapter](./adding-agents.md) — relevant when wrapping a framework that manages its own LLM calls
- [Observability Quick Start](./observability/quickstart.md)
