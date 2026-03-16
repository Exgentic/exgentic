#!/usr/bin/env bash
set -euo pipefail

# Usage: setup.sh [--no-tests]
RUN_TESTS=true
for arg in "$@"; do
    case "$arg" in
        --no-tests) RUN_TESTS=false ;;
        *) echo "Unknown argument: $arg" >&2; exit 1 ;;
    esac
done

if ! git lfs version >/dev/null 2>&1; then
  echo "Error: Git LFS not found. Install Git LFS and re-run." >&2
  exit 1
fi

APPWORLD_ROOT="${EXGENTIC_CACHE_DIR:-.exgentic_cache}/appworld"
mkdir -p "$APPWORLD_ROOT"
export APPWORLD_ROOT

TMPDIR="$(mktemp -d)"
git lfs install >/dev/null 2>&1 || true
git clone https://github.com/StonyBrookNLP/appworld.git "$TMPDIR/appworld"
cd "$TMPDIR/appworld"
git checkout edc960129fa6889c2b381715ecd108982029f6d1
git lfs pull

uv pip install "."

python -m appworld.cli install

cd - >/dev/null 2>&1 || true
rm -rf "$TMPDIR"
python -m appworld.cli download data --root "$APPWORLD_ROOT"

if [ "$RUN_TESTS" = true ]; then
    python -m appworld.cli verify tests --root "$APPWORLD_ROOT"
else
    echo "Skipping appworld verify tests (--no-tests specified)"
fi
