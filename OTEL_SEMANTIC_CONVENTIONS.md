# Exgentic → OpenTelemetry GenAI Mapping

This document maps Exgentic core types and members to OpenTelemetry (OTel) GenAI semantic conventions. It reflects the **actual implementation** in `helix/src/exgentic/observers/handlers/otel.py` and `helix/src/exgentic/integrations/litellm/trace_logger.py`.

## Implemented Trace Model

### Span Hierarchy

```
Session Span (ROOT)
├── execute_tool initial_observation
├── invoke_agent {agent_name}
│   └── chat {model} (LLM inference)
├── execute_tool {tool_name}
├── invoke_agent {agent_name}
│   └── chat {model} (LLM inference)
└── execute_tool {tool_name}
└── ... (continues until session ends)
└── session close
```

## Semantic Mapping Table

This table documents **all attributes actually set in the code**, organized by span type and code scope.

| Span Type | OTel Attribute | Exgentic Source | Type | Requirement | Content Filtered | Code Scope | Notes |
|-----------|----------------|-----------------|------|-------------|------------------|------------|-------|
| **Session (ROOT)** | `exgentic.benchmark.slug_name` | BenchmarkEntry.slug_name | string | Custom | No | OtelTracingObserver.on_run_start | Heritable, from run attributes |
| **Session (ROOT)** | `exgentic.benchmark.subset` | RunConfig.subset | string | Custom | No | OtelTracingObserver.on_run_start | Heritable, from run attributes |
| **Session (ROOT)** | `exgentic.benchmark.agent.name` | AgentEntry.display_name | string | Custom | No | OtelTracingObserver.on_run_start | Heritable, from run attributes |
| **Session (ROOT)** | `exgentic.agent.slug` | RunConfig.agent | string | Custom | No | OtelTracingObserver.on_run_start | Heritable, from run attributes |
| **Session (ROOT)** | `exgentic.run.id` | Context.run_id | string | Custom | No | OtelTracingObserver.on_run_start | Heritable, from run attributes |
| **Session (ROOT)** | `gen_ai.request.model` | RunConfig.model | string | Recommended | No | OtelTracingObserver.on_run_start | Heritable, from run attributes if available |
| **Session (ROOT)** | `gen_ai.conversation.id` | Session.session_id | string | Recommended | No | OtelTracingObserver.on_session_start | Heritable, primary correlation attribute |
| **Session (ROOT)** | `exgentic.session.id` | Session.session_id | string | Custom | No | OtelTracingObserver.on_session_start | Heritable, backwards compatibility |
| **Session (ROOT)** | `exgentic.session.task_id` | Session.task_id | string | Custom | No | OtelTracingObserver.on_session_start | Task identifier |
| **Session (ROOT)** | `exgentic.session.task` | Session.task | string | Opt-In | **Yes** | OtelTracingObserver.on_session_start | Task prompt/description (user content) |
| **Session (ROOT)** | `exgentic.session.action.{name}.name` | ActionType.name | string | Custom | No | OtelTracingObserver.on_session_start | For each action in Session.actions |
| **Session (ROOT)** | `exgentic.session.action.{name}.description` | ActionType.description | string | Custom | No | OtelTracingObserver.on_session_start | For each action in Session.actions |
| **Session (ROOT)** | `exgentic.session.action.{name}.is_message` | ActionType.is_message | bool | Custom | No | OtelTracingObserver.on_session_start | For each action in Session.actions |
| **Session (ROOT)** | `exgentic.session.action.{name}.is_finish` | ActionType.is_finish | bool | Custom | No | OtelTracingObserver.on_session_start | For each action in Session.actions |
| **Session (ROOT)** | `exgentic.context.{key}` | Session.context[key] | string | Custom | No | OtelTracingObserver.on_session_start | For each context key/value |
| **Session (ROOT)** | `exgentic.session.agent.id` | AgentInstance.agent_id | string | Custom | No | OtelTracingObserver.on_session_start | Agent instance ID |
| **Session (ROOT)** | `exgentic.session.agent.path` | AgentInstance.paths.agent_dir | string | Custom | No | OtelTracingObserver.on_session_start | Agent directory path |
| **Session (ROOT)** | `exgentic.score.success` | SessionScore.success | bool | Custom | No | OtelTracingObserver.on_session_success | Set at completion |
| **Session (ROOT)** | `exgentic.score` | SessionScore.score | float | Custom | No | OtelTracingObserver.on_session_success | Set at completion |
| **Session (ROOT)** | `exgentic.score.is_finished` | SessionScore.is_finished | bool | Custom | No | OtelTracingObserver.on_session_success | Set at completion |
| **Session (ROOT)** | `exgentic.session.steps` | Step counter | int | Custom | No | OtelTracingObserver.on_session_success | Set at completion |
| **Session (ROOT)** | `exgentic.agent.agent_cost` | AgentInstance.get_cost() | string (JSON) | Custom | No | OtelTracingObserver.on_session_success | Set at completion |
| **Session (ROOT)** | `exgentic.session.cost` | Session.get_cost() | string (JSON) | Custom | No | OtelTracingObserver.on_session_success | Set at completion |
| **invoke_agent** | `gen_ai.operation.name` | Literal "invoke_agent" | string | Required | No | OtelTracingObserver.on_react_success | Operation type |
| **invoke_agent** | `gen_ai.agent.name` | AgentEntry.display_name | string | Required | No | OtelTracingObserver.on_react_success | Agent display name |
| **invoke_agent** | `gen_ai.agent.id` | AgentInstance.agent_id | string | Recommended | No | OtelTracingObserver.on_react_success | Agent instance ID |
| **invoke_agent** | `gen_ai.request.model` | RunConfig.model | string | Recommended | No | OtelTracingObserver.on_react_success | Inherited from run attributes |
| **invoke_agent** | `gen_ai.conversation.id` | Session.session_id | string | Recommended | No | OtelTracingObserver.on_react_success | Inherited from session |
| **invoke_agent** | `exgentic.step.index` | Step counter | int | Custom | No | OtelTracingObserver._start_next_step | Step number |
| **invoke_agent** | `gen_ai.tool.definitions` | Session.actions | string (JSON) | Recommended | No | OtelTracingObserver._start_next_step | JSON array of tool schemas (metadata) |
| **invoke_agent** | `exgentic.action.repr` | SingleAction (repr) | string | Opt-In | **Yes** | OtelTracingObserver._record_action | Action representation (user content) |
| **invoke_agent** | `exgentic.action` | Action (full) | string | Opt-In | **Yes** | OtelTracingObserver._record_action | Full action object (user content) |
| **invoke_agent** | `exgentic.action.{i}` | SingleAction[i] | string | Opt-In | **Yes** | OtelTracingObserver._record_action | Each action in list (user content) |
| **execute_tool** | `gen_ai.operation.name` | Literal "execute_tool" | string | Required | No | OtelTracingObserver.on_session_start | For initial observation |
| **execute_tool** | `gen_ai.tool.name` | Literal "initial_observation" | string | Required | No | OtelTracingObserver.on_session_start | For initial observation |
| **execute_tool** | `gen_ai.tool.description` | Literal "Initial observation..." | string | Recommended | No | OtelTracingObserver.on_session_start | For initial observation |
| **execute_tool** | `gen_ai.tool.result` | Observation.result | string | Opt-In | **Yes** | OtelTracingObserver.on_session_start | For initial observation (user content) |
| **execute_tool** | `gen_ai.operation.name` | Literal "execute_tool" | string | Required | No | OtelTracingObserver.on_react_success | Set after agent react |
| **execute_tool** | `gen_ai.tool.name` | SingleAction.name | string | Required | No | OtelTracingObserver.on_react_success | Tool/action name |
| **execute_tool** | `gen_ai.tool.id` | SingleAction.id | string | Recommended | No | OtelTracingObserver.on_react_success | Tool invocation ID |
| **execute_tool** | `gen_ai.tool.description` | ActionType.description | string | Recommended | No | OtelTracingObserver.on_react_success | Tool description from Session.actions |
| **execute_tool** | `gen_ai.tool.parameters` | SingleAction.arguments | string (JSON) | Opt-In | **Yes** | OtelTracingObserver.on_react_success | Tool parameters (user content) |
| **execute_tool** | `gen_ai.conversation.id` | Session.session_id | string | Recommended | No | OtelTracingObserver.on_react_success | Inherited from session |
| **execute_tool** | `gen_ai.tool.result` | Observation.result | string | Opt-In | **Yes** | OtelTracingObserver.on_step_success | Tool execution result (user content) |
| **execute_tool** | `exgentic.observation.repr` | Observation (repr) | string | Opt-In | **Yes** | OtelTracingObserver._record_observation | Observation representation (user content) |
| **execute_tool** | `exgentic.observation` | Observation (full) | string | Opt-In | **Yes** | OtelTracingObserver._record_observation | Full observation object (user content) |
| **execute_tool** | `exgentic.observation.{i}` | Observation[i] | string | Opt-In | **Yes** | OtelTracingObserver._record_observation | Each observation in list (user content) |
| **execute_tool** | `gen_ai.tool.name` | Literal "error_recovery" | string | Required | No | OtelTracingObserver.on_react_error | On agent error |
| **LLM Inference** | `gen_ai.operation.name` | Literal "chat"/"text_completion" | string | Required | No | TraceLogger._write_otel | Based on request type |
| **LLM Inference** | `gen_ai.provider.name` | LiteLLM provider | string | Required | No | TraceLogger._write_otel | Standardized provider name |
| **LLM Inference** | `gen_ai.request.model` | Request.model | string | Required | No | TraceLogger._write_otel | Model identifier |
| **LLM Inference** | `error.type` | Exception type | string | Required | No | TraceLogger._write_otel | If operation failed |
| **LLM Inference** | `gen_ai.conversation.id` | Session.session_id | string | Recommended | No | TraceLogger._write_otel | From context if available |
| **LLM Inference** | `gen_ai.request.max_tokens` | Request.max_tokens | int | Recommended | No | TraceLogger._write_otel | If present |
| **LLM Inference** | `gen_ai.request.temperature` | Request.temperature | float | Recommended | No | TraceLogger._write_otel | If present |
| **LLM Inference** | `gen_ai.request.top_p` | Request.top_p | float | Recommended | No | TraceLogger._write_otel | If present |
| **LLM Inference** | `gen_ai.request.top_k` | Request.top_k | float | Recommended | No | TraceLogger._write_otel | If present |
| **LLM Inference** | `gen_ai.request.frequency_penalty` | Request.frequency_penalty | float | Recommended | No | TraceLogger._write_otel | If present |
| **LLM Inference** | `gen_ai.request.presence_penalty` | Request.presence_penalty | float | Recommended | No | TraceLogger._write_otel | If present |
| **LLM Inference** | `gen_ai.request.stop_sequences` | Request.stop | string[] | Recommended | No | TraceLogger._write_otel | If present |
| **LLM Inference** | `gen_ai.request.choice.count` | Request.n | int | Required | No | TraceLogger._write_otel | If != 1 |
| **LLM Inference** | `gen_ai.request.seed` | Request.seed | int | Required | No | TraceLogger._write_otel | If present |
| **LLM Inference** | `gen_ai.response.id` | Response.id | string | Recommended | No | TraceLogger._write_otel | Response ID |
| **LLM Inference** | `gen_ai.response.model` | Response.model | string | Recommended | No | TraceLogger._write_otel | Actual model used |
| **LLM Inference** | `gen_ai.usage.input_tokens` | Response.usage.prompt_tokens | int | Recommended | No | TraceLogger._write_otel | Input token count |
| **LLM Inference** | `gen_ai.usage.output_tokens` | Response.usage.completion_tokens | int | Recommended | No | TraceLogger._write_otel | Output token count |
| **LLM Inference** | `gen_ai.response.finish_reasons` | Response.choices[*].finish_reason | string[] | Recommended | No | TraceLogger._write_otel | Array of finish reasons |
| **LLM Inference** | `gen_ai.input.messages` | Request.messages | string (JSON) | Opt-In | **Yes** | TraceLogger._write_otel | LLM input messages (user content) |
| **LLM Inference** | `gen_ai.output.messages` | Response.choices[*].message | string (JSON) | Opt-In | **Yes** | TraceLogger._write_otel | LLM output messages (user content) |

## Span Details

### Session Span (ROOT)
- **Name**: `{benchmark_name} {subset} session`
- **Kind**: INTERNAL (default)
- **Created**: `OtelTracingObserver.on_session_start`
- **Closed**: `OtelTracingObserver.on_session_success` or `on_session_error`

### invoke_agent Span
- **Name**: `invoke_agent {agent_name}`
- **Kind**: CLIENT
- **Created**: `OtelTracingObserver._start_next_step`
- **Closed**: `OtelTracingObserver.on_react_success` or `on_react_error`
- **OTel Reference**: https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-agent-spans/#invoke-agent-span

### execute_tool Span
- **Name**: `execute_tool {tool_name}` or `execute_tool initial_observation`
- **Kind**: CLIENT
- **Created**: `OtelTracingObserver.on_session_start` (initial), `on_react_success`, or `on_react_error`
- **Closed**: `OtelTracingObserver.on_step_success` or `on_step_error`
- **OTel Reference**: https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/#execute-tool-span

### LLM Inference Span
- **Name**: `{operation} {model}` (e.g., "chat gpt-4")
- **Kind**: CLIENT
- **Created**: `TraceLogger._write_otel`
- **Closed**: `TraceLogger._write_otel` (with end_time)
- **OTel Reference**: https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/#inference

## Attribute Inheritance

Heritable attributes are set on the session span and automatically propagate to all child spans via `SessionSpanManager.set_heritable_attribute()`:

1. **`gen_ai.conversation.id`** - Primary correlation attribute (Session.session_id)
2. **`exgentic.session.id`** - Backwards compatibility (Session.session_id)
3. **`gen_ai.request.model`** - Model name (RunConfig.model, if available)
4. **`exgentic.run.id`** - Run identifier (Context.run_id)
5. **`exgentic.benchmark.slug_name`** - Benchmark identifier (BenchmarkEntry.slug_name)
6. **`exgentic.benchmark.subset`** - Benchmark subset (RunConfig.subset)
7. **`exgentic.benchmark.agent.name`** - Agent display name (AgentEntry.display_name)
8. **`exgentic.agent.slug`** - Agent identifier (RunConfig.agent)

## Tool Definitions Format

Tool definitions follow OpenAI function calling format and are generated from `Session.actions`:

```json
[
  {
    "type": "function",
    "function": {
      "name": "tool_name",
      "description": "Tool description",
      "parameters": {
        "type": "object",
        "properties": {...},
        "required": [...]
      }
    }
  }
]
```

Generated by `OtelTracingObserver._get_tool_definitions()` method (lines 228-259).

## Implementation Notes

### Model Name Resolution
Since `AgentInstance` doesn't expose model settings, the model name is extracted from `RunConfig` in `on_run_start`:
```python
model_name = run_config.model or (run_config.agent_kwargs or {}).get("model")
```

### Cost Attributes
Cost objects (`LiteLLMCostReport` and `UpdatableCostReport`) are serialized to JSON strings for OTEL compatibility:
- `exgentic.agent.agent_cost` - JSON string of agent cost report
- `exgentic.session.cost` - JSON string of session cost report

### Content Recording (Opt-In)

Sensitive content attributes are only recorded when `EXGENTIC_OTEL_RECORD_CONTENT=true`. This follows OpenTelemetry GenAI semantic conventions for opt-in content recording.

**Content Filtered Attributes (marked with "Yes" in table):**

**In otel.py observer:**
- `exgentic.session.task` - Task description (user prompt)
- `exgentic.observation.*` - Observation content (tool results)
- `exgentic.action.*` - Action content (agent decisions)
- `gen_ai.tool.result` - Tool execution results (runtime user data)
- `gen_ai.tool.parameters` - Tool call parameters (runtime user data)

**In trace_logger.py (LLM inference):**
- `gen_ai.input.messages` - LLM input messages (user prompts)
- `gen_ai.output.messages` - LLM output messages (model responses)

**NOT Content Filtered (metadata/schemas):**
- `gen_ai.tool.definitions` - Tool schemas/API definitions (not user content)
- All IDs, names, counts, scores, and other metadata

### Custom Namespace
All Exgentic-specific attributes use the `exgentic.` prefix to distinguish them from standard OTel semantic conventions.

### Backwards Compatibility
- `exgentic.session.id` maintained alongside `gen_ai.conversation.id`
- All custom attributes preserved with `exgentic.` prefix

## References

- [OTel GenAI Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
- [Invoke Agent Spans](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-agent-spans/#invoke-agent-span)
- [Execute Tool Spans](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/#execute-tool-span)
- [Inference Spans](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/#inference)