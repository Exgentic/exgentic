#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/release.sh <version> [--push]

Creates an annotated release tag on the current main HEAD.
The Git tag is the single source of truth for package versioning.

Examples:
  scripts/release.sh 0.1.1
  scripts/release.sh 0.1.1 --push
EOF
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
  usage
  exit 1
fi

version="$1"
push_changes="${2:-}"

if [[ ! "$version" =~ ^[0-9]+(\.[0-9]+){2}([A-Za-z0-9._-]+)?$ ]]; then
  echo "Version must look like 1.2.3 or 1.2.3rc1" >&2
  exit 1
fi

if [[ -n "$push_changes" && "$push_changes" != "--push" ]]; then
  usage
  exit 1
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

current_branch="$(git branch --show-current)"
if [[ "$current_branch" != "main" ]]; then
  echo "Release tags must be created from main. Current branch: $current_branch" >&2
  exit 1
fi

if [[ -n "$(git status --short)" ]]; then
  echo "Working tree is dirty. Commit or stash changes before releasing." >&2
  exit 1
fi

git fetch origin main --tags
local_head="$(git rev-parse HEAD)"
remote_head="$(git rev-parse origin/main)"
if [[ "$local_head" != "$remote_head" ]]; then
  echo "Local main is not at origin/main. Pull or push before releasing." >&2
  exit 1
fi

tag="v$version"
if git rev-parse "$tag" >/dev/null 2>&1; then
  echo "Tag $tag already exists." >&2
  exit 1
fi

git tag -a "$tag" -m "Release $tag"

if [[ "$push_changes" == "--push" ]]; then
  git push origin "$tag"
fi

echo "Created release tag $tag at $local_head"
if [[ "$push_changes" != "--push" ]]; then
  echo "Push when ready with:"
  echo "  git push origin $tag"
fi
