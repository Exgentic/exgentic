#!/bin/bash
# Install Claude Code Agent dependencies

echo "Installing Claude Code Agent..."

# Check if we're in the project root
if [ ! -f "pyproject.toml" ]; then
    echo "Error: setup.sh must be run from the root directory of the Exgentic project"
    exit 1
fi

# Install the base dependencies with CLI extra
if command -v uv >/dev/null 2>&1; then
    echo "Using uv for installation..."
    uv pip install -e ".[cli]"
else
    echo "Using pip for installation..."
    python -m pip install -e ".[cli]"
fi

if [ $? -ne 0 ]; then
    echo "Claude Code Agent installation failed"
    exit 1
fi

# Determine which container runtime to use
CONTAINER_CMD=""
if command -v podman >/dev/null 2>&1; then
    CONTAINER_CMD="podman"
    echo "Using Podman for containerized execution..."
    
    # Check if Podman machine is needed (macOS/Windows) and initialize/start if necessary
    if podman machine list >/dev/null 2>&1; then
        # Podman machine is supported on this platform
        MACHINE_STATUS=$(podman machine list --format "{{.Running}}" 2>/dev/null | head -n 1)
        
        if [ -z "$MACHINE_STATUS" ]; then
            # No machine exists, initialize one
            echo "Initializing Podman machine..."
            podman machine init
            if [ $? -ne 0 ]; then
                echo "Error: Failed to initialize Podman machine"
                exit 1
            fi
            echo "Starting Podman machine..."
            podman machine start
            if [ $? -ne 0 ]; then
                echo "Error: Failed to start Podman machine"
                exit 1
            fi
        elif [ "$MACHINE_STATUS" != "true" ]; then
            # Machine exists but is not running
            echo "Starting Podman machine..."
            podman machine start
            if [ $? -ne 0 ]; then
                echo "Error: Failed to start Podman machine"
                exit 1
            fi
        else
            echo "Podman machine is already running"
        fi
    fi
elif command -v docker >/dev/null 2>&1; then
    CONTAINER_CMD="docker"
    echo "Using Docker for containerized execution..."
else
    echo "Error: Neither Podman nor Docker found."
    echo "Please install Podman or Docker to use Claude Code Agent."
    echo ""
    echo "Installation instructions:"
    echo "  macOS:  brew install podman && podman machine init && podman machine start"
    echo "  Linux:  sudo apt install podman  (or)  sudo dnf install podman"
    echo "  Docker: https://docs.docker.com/get-docker/"
    exit 1
fi

# Build the Claude Code container image
echo "Building Claude Code container image..."
DOCKERFILE_PATH="src/exgentic/agents/dockerfiles/claude_code"

if [ ! -f "$DOCKERFILE_PATH/Dockerfile" ]; then
    echo "Error: Dockerfile not found at $DOCKERFILE_PATH/Dockerfile"
    exit 1
fi

$CONTAINER_CMD build -t exgentic-claude-code:dev -f "$DOCKERFILE_PATH/Dockerfile" "$DOCKERFILE_PATH"

if [ $? -eq 0 ]; then
    echo "Claude Code container image built successfully"
    
    # Verify the image
    echo "Verifying Claude Code installation..."
    $CONTAINER_CMD run --rm exgentic-claude-code:dev claude --version
    
    if [ $? -eq 0 ]; then
        echo "✓ Claude Code Agent setup complete"
        echo ""
        echo "The agent will run in an isolated container environment using $CONTAINER_CMD"
    else
        echo "Warning: Image built but verification failed"
    fi
else
    echo "Error: Failed to build Claude Code container image"
    exit 1
fi

# Made with Bob
