#!/bin/bash
# Install Codex Agent dependencies

echo "Installing Codex Agent..."

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
    echo "Codex Agent installation failed"
    exit 1
fi

# Install Codex CLI
echo "Installing Codex CLI..."
if command -v npm >/dev/null 2>&1; then
    npm install -g codex-cli
    if [ $? -eq 0 ]; then
        echo "Codex CLI installed successfully"
    else
        echo "Warning: Codex CLI installation failed"
        echo "You may need to install it manually with: npm install -g codex-cli"
    fi
else
    echo "Warning: npm not found. Please install Node.js and npm to use Codex CLI"
    echo "After installing npm, run: npm install -g codex-cli"
fi

echo "Codex Agent setup complete"
echo "Note: Additional Codex-specific configuration may be required"

# Made with Bob
