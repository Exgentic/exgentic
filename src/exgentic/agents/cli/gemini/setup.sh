#!/bin/bash
set -euo pipefail

# Determine container runtime (override with EXGENTIC_CONTAINER_CMD=docker|podman)
_try_start_podman_machine() {
    if podman machine list >/dev/null 2>&1; then
        MACHINE_STATUS=$(podman machine list --format "{{.Running}}" 2>/dev/null | head -n 1)
        if [ -z "$MACHINE_STATUS" ]; then
            podman machine init && podman machine start
        elif [ "$MACHINE_STATUS" != "true" ]; then
            podman machine start
        fi
    fi
}

_podman_works() {
    command -v podman >/dev/null 2>&1 || return 1
    _try_start_podman_machine
    podman info >/dev/null 2>&1
}

CONTAINER_CMD=""
if [ -n "${EXGENTIC_CONTAINER_CMD:-}" ]; then
    if ! command -v "$EXGENTIC_CONTAINER_CMD" >/dev/null 2>&1; then
        echo "Error: EXGENTIC_CONTAINER_CMD=$EXGENTIC_CONTAINER_CMD not found." >&2
        exit 1
    fi
    CONTAINER_CMD="$EXGENTIC_CONTAINER_CMD"
elif _podman_works; then
    CONTAINER_CMD="podman"
elif command -v docker >/dev/null 2>&1; then
    CONTAINER_CMD="docker"
else
    echo "Error: Neither Podman nor Docker found." >&2
    exit 1
fi

# Build Gemini CLI container image (inline — no external Dockerfile needed)
$CONTAINER_CMD build -t exgentic-gemini:dev -f - . <<'DOCKERFILE'
FROM registry.access.redhat.com/ubi9/nodejs-20
RUN npm install -g @google/gemini-cli@0.25.0
WORKDIR /work
CMD ["gemini","--help"]
DOCKERFILE

echo "Gemini Agent setup complete (using $CONTAINER_CMD)"
