#!/usr/bin/env bash
# Run tau2 retail benchmark with the Claude Code agent.
#
# Usage:
#   scripts/run_tau2_claude_code.sh [NUM_TASKS] [MODEL]
#
# Defaults:
#   NUM_TASKS = 3
#   MODEL     = openai/Azure/gpt-4.1
#
# Prerequisites:
#   - .env in repo root with OPENAI_API_KEY and OPENAI_API_BASE
#   - Jaeger running at localhost:4318 (optional, for OTEL)
#   - podman/docker on PATH (via ~/.zshrc)

set -eo pipefail

cd "$(dirname "$0")/.."

NUM_TASKS="${1:-3}"
MODEL="${2:-openai/Azure/gpt-4.1}"
SUBSET="${SUBSET:-retail}"
MAX_STEPS="${MAX_STEPS:-30}"

# Load API keys
[ -f .env ] && { set -a; . ./.env; set +a; }
# Ensure podman + uv are on PATH (homebrew and local bin)
export PATH="/opt/homebrew/bin:$HOME/.local/bin:$PATH"

export OTEL_EXPORTER_OTLP_ENDPOINT="${OTEL_EXPORTER_OTLP_ENDPOINT:-http://localhost:4318}"
export OTEL_EXPORTER_OTLP_PROTOCOL="${OTEL_EXPORTER_OTLP_PROTOCOL:-http/protobuf}"
export EXGENTIC_OTEL_ENABLED="${EXGENTIC_OTEL_ENABLED:-true}"
export EXGENTIC_OTEL_RECORD_CONTENT="${EXGENTIC_OTEL_RECORD_CONTENT:-true}"

echo "Running tau2 $SUBSET with claude_code agent (model=$MODEL, num_tasks=$NUM_TASKS, max_steps=$MAX_STEPS)"

uv run exgentic evaluate \
  --benchmark tau2 \
  --agent claude_code \
  --subset "$SUBSET" \
  --num-tasks "$NUM_TASKS" \
  --max-steps "$MAX_STEPS" \
  --model "$MODEL" \
  --set benchmark.user_simulator_model="$MODEL"
