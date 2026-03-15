#!/bin/bash
# Install OpenAI MCP Agent dependencies

echo "Installing OpenAI MCP Agent..."

# Check if we're in the project root
if [ ! -f "pyproject.toml" ]; then
    echo "Error: setup.sh must be run from the root directory of the Exgentic project"
    exit 1
fi

# Install the agent dependencies with openaimacp extra
if command -v uv >/dev/null 2>&1; then
    echo "Using uv for installation..."
    uv pip install -e ".[openaimacp]"
else
    echo "Using pip for installation..."
    python -m pip install -e ".[openaimacp]"
fi

if [ $? -eq 0 ]; then
    echo "OpenAI MCP Agent installed successfully"
else
    echo "OpenAI MCP Agent installation failed"
    exit 1
fi

# Made with Bob
