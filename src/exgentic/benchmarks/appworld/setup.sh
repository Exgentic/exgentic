#!/usr/bin/env bash
set -euo pipefail

# Usage: setup.sh [--no-tests]
#   --no-tests   Skip running 'appworld verify tests' at the end

RUN_TESTS=true
for arg in "$@"; do
    case "$arg" in
        --no-tests) RUN_TESTS=false ;;
        *) echo "Unknown argument: $arg" >&2; exit 1 ;;
    esac
done

# Require Git LFS for bundled assets
if ! git lfs version >/dev/null 2>&1; then
  echo "Error: Git LFS not found. Install Git LFS and re-run." >&2
  exit 1
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
export APPWORLD_ROOT="$SCRIPT_DIR"

TMPDIR="$(mktemp -d)"
git lfs install >/dev/null 2>&1 || true
git clone https://github.com/StonyBrookNLP/appworld.git "$TMPDIR/appworld"
cd "$TMPDIR/appworld"

git checkout edc960129fa6889c2b381715ecd108982029f6d1
git lfs pull

if command -v uv >/dev/null 2>&1; then
    uv pip install .
else
    python -m pip install .
fi

python -m appworld.cli install

cd - >/dev/null 2>&1 || true
rm -rf "$TMPDIR"
python -m appworld.cli download data --root "$APPWORLD_ROOT"

if [ "$RUN_TESTS" = true ]; then
    python -m appworld.cli verify tests --root "$APPWORLD_ROOT"
else
    echo "Skipping appworld verify tests (--no-tests specified)"
fi
