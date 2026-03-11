#!/bin/bash
# Install SmolAgents dependencies

echo "Installing SmolAgents..."

# Check if we're in the project root
if [ ! -f "pyproject.toml" ]; then
    echo "Error: setup.sh must be run from the root directory of the Exgentic project"
    exit 1
fi

# Install the agent dependencies with smolagents extra
if command -v uv >/dev/null 2>&1; then
    echo "Using uv for installation..."
    uv pip install -e ".[smolagents]"
else
    echo "Using pip for installation..."
    python -m pip install -e ".[smolagents]"
fi

if [ $? -eq 0 ]; then
    echo "SmolAgents installed successfully"
else
    echo "SmolAgents installation failed"
    exit 1
fi

# Made with Bob
