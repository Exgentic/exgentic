#!/bin/bash
set -euo pipefail

if command -v npm >/dev/null 2>&1; then
    npm install -g codex-cli
else
    echo "Warning: npm not found. Install Node.js, then run: npm install -g codex-cli"
fi

echo "Codex Agent setup complete"
