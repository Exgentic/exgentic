#!/bin/bash
set -e

# Read environment variables with defaults
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8000}
LOG_LEVEL=${LOG_LEVEL:-INFO}

# Validate required environment variables
if [ -z "$AGENT_NAME" ]; then
    echo "ERROR: AGENT_NAME environment variable is not set"
    exit 1
fi

if [ -z "$MCP_URL" ]; then
    echo "ERROR: MCP_URL environment variable is not set"
    exit 1
fi

# Validate MCP_URL format
if [[ ! "$MCP_URL" =~ ^https?:// ]]; then
    echo "ERROR: MCP_URL must start with http:// or https://"
    exit 1
fi

echo "Starting Exgentic A2A Agent"
echo "Agent: $AGENT_NAME"
echo "Host: $HOST"
echo "Port: $PORT"
echo "Log Level: $LOG_LEVEL"
echo "MCP URL: $MCP_URL"

# Build --set arguments from EXGENTIC_SET_* environment variables
SET_ARGS=()
for var in $(env | grep '^EXGENTIC_SET_' | cut -d= -f1); do
    # Extract the key by removing EXGENTIC_SET_ prefix and converting to lowercase
    # Replace first underscore with dot, keep rest as underscores
    key=$(echo "$var" | sed 's/^EXGENTIC_SET_//' | tr '[:upper:]' '[:lower:]' | sed 's/_/./' )
    value="${!var}"
    SET_ARGS+=("--set" "$key=$value")
    echo "Setting: $key=$value"
done

# Change to the exgentic directory
cd /app/exgentic

# Run the exgentic a2a command with --mcp and --set arguments
# --disable-dns-rebinding-protection is added to allow kubernetes to access the service
echo "Command: exgentic a2a --agent $AGENT_NAME --host $HOST --port $PORT --mcp $MCP_URL ${SET_ARGS[*]}"
exec exgentic a2a --agent "$AGENT_NAME" --host "$HOST" --port "$PORT" --mcp "$MCP_URL" "${SET_ARGS[@]}"

# Made with Bob
