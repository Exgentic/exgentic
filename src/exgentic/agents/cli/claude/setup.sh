#!/bin/bash
set -euo pipefail

# Determine container runtime
CONTAINER_CMD=""
if command -v podman >/dev/null 2>&1; then
    CONTAINER_CMD="podman"
    # Start podman machine if needed (macOS/Windows)
    if podman machine list >/dev/null 2>&1; then
        MACHINE_STATUS=$(podman machine list --format "{{.Running}}" 2>/dev/null | head -n 1)
        if [ -z "$MACHINE_STATUS" ]; then
            podman machine init && podman machine start
        elif [ "$MACHINE_STATUS" != "true" ]; then
            podman machine start
        fi
    fi
elif command -v docker >/dev/null 2>&1; then
    CONTAINER_CMD="docker"
else
    echo "Error: Neither Podman nor Docker found." >&2
    exit 1
fi

# Build Claude Code container image (inline — no external Dockerfile needed)
$CONTAINER_CMD build -t exgentic-claude-code:dev -f - . <<'DOCKERFILE'
FROM registry.access.redhat.com/ubi9/nodejs-20
RUN npm install -g @anthropic-ai/claude-code@2.1.7
WORKDIR /work
CMD ["claude","--help"]
DOCKERFILE

echo "Claude Code Agent setup complete (using $CONTAINER_CMD)"
