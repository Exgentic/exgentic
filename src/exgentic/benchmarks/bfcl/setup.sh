#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_ROOT="${SCRIPT_DIR}/installation"
REPO_DIR="${INSTALL_ROOT}/gorilla"
BFCL_DIR="${REPO_DIR}/berkeley-function-call-leaderboard"
GORILLA_URL="https://github.com/ShishirPatil/gorilla.git"
GORILLA_REF="7ad0134c665944819f88bc50862108d94015968b" # pragma: allowlist secret

pip_install() {
  if command -v uv >/dev/null 2>&1; then
    uv pip install "$@"
  else
    python -m pip install "$@"
  fi
}

mkdir -p "${INSTALL_ROOT}"

if [ ! -d "${REPO_DIR}/.git" ]; then
  rm -rf "${REPO_DIR}"
  git clone "${GORILLA_URL}" "${REPO_DIR}"
fi

git -C "${REPO_DIR}" fetch --depth 1 origin "${GORILLA_REF}"
git -C "${REPO_DIR}" checkout --force "${GORILLA_REF}"

pip_install -e "${BFCL_DIR}"
pip_install soundfile
