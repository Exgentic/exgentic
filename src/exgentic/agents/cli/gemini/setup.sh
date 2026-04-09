#!/bin/bash
set -euo pipefail

# Require docker CLI (works with Docker Engine or Podman via docker-compatible socket)
if ! command -v docker >/dev/null 2>&1; then
    echo "Error: docker CLI not found. Install Docker or configure Podman's docker-compatible socket." >&2
    exit 1
fi

# Build Gemini CLI container image (inline — no external Dockerfile needed)
docker build -t exgentic-gemini:dev -f - . <<'DOCKERFILE'
FROM registry.access.redhat.com/ubi9/nodejs-20
RUN npm install -g @google/gemini-cli@0.25.0
WORKDIR /work
CMD ["gemini","--help"]
DOCKERFILE

echo "Gemini Agent setup complete"
