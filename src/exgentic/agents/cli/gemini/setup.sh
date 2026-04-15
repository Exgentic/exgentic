#!/bin/bash
set -euo pipefail

# Use Docker as the container runtime (podman support was removed in #130).
if ! command -v docker >/dev/null 2>&1; then
    echo "Error: Docker not found." >&2
    exit 1
fi
CONTAINER_CMD="docker"

# Build Gemini CLI container image (inline — no external Dockerfile needed)
$CONTAINER_CMD build -t exgentic-gemini:dev -f - . <<'DOCKERFILE'
FROM registry.access.redhat.com/ubi9/nodejs-20
RUN npm install -g @google/gemini-cli@0.25.0
WORKDIR /work
CMD ["gemini","--help"]
DOCKERFILE

echo "Gemini Agent setup complete (using $CONTAINER_CMD)"
