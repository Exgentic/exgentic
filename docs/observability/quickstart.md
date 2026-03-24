# Observability Quick Start

This guide gets you from zero to traces in five minutes using Jaeger as a local trace collector.

For a full reference of every attribute Exgentic emits, see [Semantic Conventions](./semantic-conventions.md).

---

## Prerequisites

- Docker or Podman installed and running
- `exgentic` installed with the `otel` extra (see below)

---

## Step 1 — Install the OTEL extra

```bash
uv sync --extra otel
```

---

## Step 2 — Start Jaeger

```bash
# Using Docker (or replace 'docker' with 'podman')
docker run -d --name jaeger \
  -e COLLECTOR_OTLP_ENABLED=true \
  -p 16686:16686 \
  -p 4317:4317 \
  -p 4318:4318 \
  jaegertracing/all-in-one:latest
```

Default ports:

| Port  | Service      |
|-------|--------------|
| 16686 | Jaeger UI    |
| 4317  | OTLP gRPC    |
| 4318  | OTLP HTTP    |

---

## Step 3 — Configure environment variables

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
export OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf   # or 'grpc' for port 4317
export EXGENTIC_OTEL_ENABLED=true
```

To include task prompts, tool arguments, and LLM messages in traces (opt-in — may contain sensitive data):

```bash
export EXGENTIC_OTEL_RECORD_CONTENT=true
```

---

## Step 4 — Set up and run an evaluation

```bash
exgentic setup --agent tool_calling
exgentic setup --benchmark tau2

exgentic evaluate \
  --benchmark tau2 \
  --agent tool_calling \
  --model gpt-4o \
  --set benchmark.user_simulator_model="gpt-4o" \
  --task 1 \
  --max-steps 10
```

---

## Step 5 — View traces

Open [http://localhost:16686](http://localhost:16686), select the `exgentic` service, and click **Find Traces**.

---

## Exporting traces

### Via the Jaeger UI

1. Open a trace.
2. Click the **JSON** button in the top-right corner.

### Via the API

```bash
# All recent traces
curl "http://localhost:16686/api/traces?service=exgentic&limit=100" | jq '.' > traces.json

# A specific trace
curl "http://localhost:16686/api/traces/<TRACE_ID>" | jq '.' > trace.json
```

---

## Troubleshooting

### No traces appearing

1. Verify environment variables are set: `env | grep OTEL`
2. Confirm Jaeger is running: `docker ps | grep jaeger`
3. Check Jaeger logs: `docker logs jaeger`
4. Ensure the evaluation completed successfully before looking for traces

### Traces are incomplete

- Wait a few seconds after the evaluation finishes — spans are flushed asynchronously
- Check session logs for OTEL-related errors
- Verify network connectivity to Jaeger

### Jaeger not starting

```bash
# Check for an existing container
docker ps -a | grep jaeger

# Stop and remove, then restart
docker stop jaeger && docker rm jaeger
docker run -d --name jaeger \
  -e COLLECTOR_OTLP_ENABLED=true \
  -p 16686:16686 -p 4317:4317 -p 4318:4318 \
  jaegertracing/all-in-one:latest
```

---

## Cleanup

```bash
docker stop jaeger
docker rm jaeger

# Optional: remove the image
docker rmi jaegertracing/all-in-one:latest
```

---

## Further reading

- [Semantic Conventions](./semantic-conventions.md) — full attribute reference
- [Jaeger documentation](https://www.jaegertracing.io/docs/)
- [OpenTelemetry GenAI conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
