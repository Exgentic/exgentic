#!/bin/bash
# Install Gemini Agent dependencies

echo "Installing Gemini Agent..."

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
    echo "Gemini Agent installation failed"
    exit 1
fi

# Install Gemini CLI
echo "Installing Gemini CLI..."
if command -v npm >/dev/null 2>&1; then
    npm install -g @google/gemini-cli
    if [ $? -eq 0 ]; then
        echo "Gemini CLI installed successfully"
    else
        echo "Warning: Gemini CLI installation failed"
        echo "You may need to install it manually with: npm install -g @google/generative-ai-cli"
    fi
else
    echo "Warning: npm not found. Please install Node.js and npm to use Gemini CLI"
    echo "After installing npm, run: npm install -g @google/gemini-cli"
fi

echo "Gemini Agent setup complete"
echo "Note: Additional Gemini-specific configuration may be required"

# Made with Bob
