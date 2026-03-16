#!/bin/bash
set -euo pipefail

if command -v npm >/dev/null 2>&1; then
    npm install -g @google/gemini-cli
else
    echo "Warning: npm not found. Install Node.js, then run: npm install -g @google/gemini-cli"
fi

echo "Gemini Agent setup complete"
