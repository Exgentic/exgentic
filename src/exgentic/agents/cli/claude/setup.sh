#!/bin/bash
set -euo pipefail

# Use Docker as the container runtime (podman support was removed in #130).
if ! command -v docker >/dev/null 2>&1; then
    echo "Error: Docker not found." >&2
    exit 1
fi
CONTAINER_CMD="docker"

# Build Claude Code container image (inline — no external Dockerfile needed)
$CONTAINER_CMD build -t exgentic-claude-code:dev -f - . <<'DOCKERFILE'
FROM registry.access.redhat.com/ubi9/nodejs-20
RUN npm install -g @anthropic-ai/claude-code@2.1.7
WORKDIR /work
CMD ["claude","--help"]
DOCKERFILE

echo "Claude Code Agent setup complete (using $CONTAINER_CMD)"
