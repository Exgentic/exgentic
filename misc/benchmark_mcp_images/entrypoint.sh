#!/bin/bash
set -e

# Ensure HOME is set (Kubernetes may override it)
export HOME=${HOME:-/app}

# Read environment variables with defaults
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8000}
LOG_LEVEL=${LOG_LEVEL:-INFO}

# Validate BENCHMARK_NAME is set
if [ -z "$BENCHMARK_NAME" ]; then
    echo "ERROR: BENCHMARK_NAME environment variable is not set"
    exit 1
fi

# Set TAU2_DATA_DIR to the parent directory
# Temp fix until wwe have a better way to handle this
export TAU2_DATA_DIR="/app/.exgentic/benchmarks/tau2"

echo "Starting Exgentic MCP Server"
echo "Benchmark: $BENCHMARK_NAME"
echo "Host: $HOST"
echo "Port: $PORT"
echo "Log Level: $LOG_LEVEL"

# Build --set arguments from EXGENTIC_SET_* environment variables
SET_ARGS=""
for var in $(env | grep '^EXGENTIC_SET_' | cut -d= -f1); do
    # Extract the key by removing EXGENTIC_SET_ prefix and converting to lowercase
    # Replace first underscore with dot, keep rest as underscores
    key=$(echo "$var" | sed 's/^EXGENTIC_SET_//' | tr '[:upper:]' '[:lower:]' | sed 's/_/./' )
    value="${!var}"
    SET_ARGS="$SET_ARGS --set $key='$value'"
    echo "Setting: $key=$value"
done

# Change to the exgentic directory
cd /app/exgentic

# Run the exgentic MCP server with --set arguments
# --disable-dns-rebinding-protection is added to allow kubernetes to access the MCP
echo "Command: exgentic mcp --benchmark $BENCHMARK_NAME --host $HOST --port $PORT $SET_ARGS --disable-dns-rebinding-protection"
eval exec exgentic mcp --benchmark "$BENCHMARK_NAME" --host "$HOST" --port "$PORT" $SET_ARGS --disable-dns-rebinding-protection

