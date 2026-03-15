#!/bin/bash
# Install LiteLLM Tool Calling Agent dependencies

echo "Installing LiteLLM Tool Calling Agent..."

# Check if we're in the project root
if [ ! -f "pyproject.toml" ]; then
    echo "Error: setup.sh must be run from the root directory of the Exgentic project"
    exit 1
fi

# Install the agent dependencies
if command -v uv >/dev/null 2>&1; then
    echo "Using uv for installation..."
    uv pip install -e .
else
    echo "Using pip for installation..."
    python -m pip install -e .
fi

if [ $? -eq 0 ]; then
    echo "LiteLLM Tool Calling Agent installed successfully"
else
    echo "LiteLLM Tool Calling Agent installation failed"
    exit 1
fi

# Made with Bob
