#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.
set -euo pipefail

TAU2_REPO="https://github.com/sierra-research/tau2-bench.git"
TAU2_REF="v0.1.3"
if [ -d "tau2/domains" ]; then
    echo "tau2 data already present — skipping download."
    exit 0
fi

TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

echo "Cloning tau2-bench data files..."
git clone --depth 1 --branch "$TAU2_REF" --filter=blob:none --sparse "$TAU2_REPO" "$TMPDIR/tau2-bench"
cd "$TMPDIR/tau2-bench"
git sparse-checkout set data
cd - >/dev/null 2>&1

cp -r "$TMPDIR/tau2-bench/data/." "./"

echo "tau2 data installed to env_dir"
