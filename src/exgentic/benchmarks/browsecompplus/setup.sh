#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# 1. Detect java in PATH
###############################################################################
BM25_AVAILABLE=false
if command -v java >/dev/null 2>&1; then
  JAVA_VERSION_RAW="$(java -version 2>&1 | head -n1)" || true
  JAVA_MAJOR="$(echo "$JAVA_VERSION_RAW" | sed -E 's/.*"([0-9]+).*/\1/')"
  if [[ "$JAVA_MAJOR" =~ ^[0-9]+$ ]] && [ "$JAVA_MAJOR" -ge 21 ]; then
    BM25_AVAILABLE=true
  else
    echo "[WARNING] Java 21+ not detected. BM25 searcher will not be available."
  fi
else
  echo "[WARNING] Java not found. BM25 searcher will not be available."
fi

BENCH_ROOT="."

###############################################################################
# 2. Detect GPU availability
###############################################################################
GPU_AVAILABLE=false
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  GPU_AVAILABLE=true
fi

###############################################################################
# 3. Install BrowseCompPlus packages
###############################################################################
GIT_SSH_URL="https://github.com/lilacheden/BrowseComp-Plus/"
GIT_REF="mac-support-and-packaging"

if [ "$GPU_AVAILABLE" = true ]; then
  uv pip install "git+${GIT_SSH_URL}@${GIT_REF}#egg=browsecomp-plus[gpu]"
else
  uv pip install "git+${GIT_SSH_URL}@${GIT_REF}"
fi

uv pip install --upgrade "mcp>=1.24" "transformers>=4.53.2,<5.0" \
  "pillow>=12.1.1" "fastmcp>=2.14.0" "fastapi-sso>=0.19.0" "openai>=2.9.0"
uv pip uninstall gradio 2>/dev/null || true

if [ "$GPU_AVAILABLE" = true ]; then
  uv pip install --no-build-isolation flash-attn
fi

# In Docker builds, skip data/index downloads — they'll be mounted as volumes.
if [ "${EXGENTIC_DOCKER_BUILD:-}" = "1" ]; then
  echo "Docker build: skipping data and index downloads (will be mounted at runtime)"
  exit 0
fi

###############################################################################
# 4. Download + decrypt dataset
###############################################################################
DATA_DIR="${BENCH_ROOT}/data"
QUERIES_DIR="${BENCH_ROOT}/topics-qrels"
mkdir -p "${DATA_DIR}" "${QUERIES_DIR}"

if [ -f "${DATA_DIR}/browsecomp_plus_decrypted.jsonl" ] && [ -f "${QUERIES_DIR}/queries.tsv" ]; then
  echo "Dataset already exists, skipping download."
else
  python -m scripts_build_index.decrypt_dataset \
      --output "${DATA_DIR}/browsecomp_plus_decrypted.jsonl" \
      --generate-tsv "${QUERIES_DIR}/queries.tsv"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIGHT_JSONL="${DATA_DIR}/browsecomp_plus_decrypted_docids.jsonl"
if [ ! -f "${LIGHT_JSONL}" ]; then
  PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}" python -m make_light_dataset \
    --input "${DATA_DIR}/browsecomp_plus_decrypted.jsonl" \
    --output "${LIGHT_JSONL}"
fi

###############################################################################
# 5. Download indexes
###############################################################################
uv pip install -U hf_transfer 2>/dev/null || true

mkdir -p "${BENCH_ROOT}/indexes"
cd "${BENCH_ROOT}/indexes"

if [ "$BM25_AVAILABLE" = true ]; then
  HF_HUB_ENABLE_HF_TRANSFER=1 hf download Tevatron/browsecomp-plus-indexes --repo-type=dataset --include="bm25/*" --local-dir .
fi

HF_HUB_ENABLE_HF_TRANSFER=1 hf download Tevatron/browsecomp-plus-indexes --repo-type=dataset --include="qwen3-embedding-8b/*" --local-dir .

echo "BrowseCompPlus setup complete"
