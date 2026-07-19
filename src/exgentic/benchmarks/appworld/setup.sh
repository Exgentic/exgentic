#!/usr/bin/env bash
set -euo pipefail

if ! command -v git-lfs >/dev/null 2>&1; then
    echo "Error: git-lfs is required but not installed. Install it first: brew install git-lfs (macOS) or apt-get install git-lfs (Linux)" >&2
    exit 1
fi

APPWORLD_ROOT="."
export APPWORLD_ROOT

# Install into the active virtualenv when one exists (venv runner). When no
# virtualenv is active (e.g. the Docker image / --local into system Python),
# uv refuses to install unless told to target the system environment, so pass
# --system in that case.
uv_pip_install() {
    if [ -n "${VIRTUAL_ENV:-}" ]; then
        uv pip install "$@"
    else
        uv pip install --system "$@"
    fi
}

TMPDIR="$(mktemp -d)"
git lfs install >/dev/null 2>&1 || true
git clone https://github.com/StonyBrookNLP/appworld.git "$TMPDIR/appworld"
cd "$TMPDIR/appworld"
git checkout edc960129fa6889c2b381715ecd108982029f6d1
git lfs pull

uv_pip_install "."

python -m appworld.cli install

cd - >/dev/null 2>&1 || true
rm -rf "$TMPDIR"
python -m appworld.cli download data --root "."
