#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# 1. Detect java in PATH
###############################################################################
BM25_AVAILABLE=true
if ! command -v java >/dev/null 2>&1; then
  echo "[WARNING] Java not found. BM25 searcher will not be available."
  echo "To enable it, install Java 21+:"
  echo "  - macOS (Homebrew): brew install openjdk@21"
  echo "  - Linux (Debian/Ubuntu): sudo apt install openjdk-21-jdk"
  echo "  - Conda: conda install -c conda-forge openjdk=21"
  BM25_AVAILABLE=false
else
  JAVA_VERSION_RAW="$(java -version 2>&1 | head -n1)"
  JAVA_MAJOR="$(echo "$JAVA_VERSION_RAW" | sed -E 's/.*\"([0-9]+).*/\1/')"

  if [ "$JAVA_MAJOR" -lt 21 ]; then
    echo "[WARNING] Java detected but version is too old:"
    echo "  $JAVA_VERSION_RAW"
    echo "BrowseComp-Plus requires Java 21+ for BM25 searcher (Pyserini/Anserini)."
    echo "BM25 searcher will not be available."
    BM25_AVAILABLE=false
  else
    echo "[setup] Java 21+ detected: $JAVA_VERSION_RAW"
  fi
fi

# Locate benchmark root (this directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_ROOT="${SCRIPT_DIR}/assets"

echo "[BrowseCompPlus setup] Benchmark root: ${BENCH_ROOT}"

###############################################################################
# 2. Detect GPU availability
###############################################################################
GPU_AVAILABLE=false
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  echo "[BrowseCompPlus setup] NVIDIA GPU detected"
  GPU_AVAILABLE=true
else
  echo "[BrowseCompPlus setup] No GPU detected, installing CPU-only version"
fi

###############################################################################
# 3. Install BrowseCompPlus package from github fork
###############################################################################
GIT_SSH_URL="https://github.com/lilacheden/BrowseComp-Plus/"
GIT_REF="mac-support-and-packaging"
pip_install() {
  if command -v uv >/dev/null 2>&1; then
    uv pip install "$@"
  else
    python -m pip install "$@"
  fi
}

pip_uninstall() {
  if command -v uv >/dev/null 2>&1; then
    uv pip uninstall "$@"
  else
    python -m pip uninstall "$@"
  fi
}


echo "[BrowseCompPlus setup] Installing BrowseCompPlus from git+${GIT_SSH_URL}@${GIT_REF}"
if [ "$GPU_AVAILABLE" = true ]; then
  pip_install "git+${GIT_SSH_URL}@${GIT_REF}#egg=browsecomp-plus[gpu]"
else
  pip_install "git+${GIT_SSH_URL}@${GIT_REF}"
fi

# update packages
echo "[BrowseCompPlus setup] Updating packages versions"
pip_install --upgrade "mcp>=1.24"
pip_install --upgrade "transformers>=4.53.2,<5.0"
pip_install --upgrade "pillow>=12.1.1"
pip_install --upgrade "fastmcp>=2.14.0"
pip_install --upgrade "fastapi-sso>=0.19.0"
pip_install --upgrade "openai>=2.9.0"
pip_uninstall "gradio"


###############################################################################
# 3b. Install flash-attn if GPU is available
###############################################################################
if [ "$GPU_AVAILABLE" = true ]; then
  echo "[BrowseCompPlus setup] GPU detected, installing flash-attn..."
  pip_install --no-build-isolation flash-attn
else
  echo "[BrowseCompPlus setup] Skipping flash-attn (no GPU detected)"
fi

###############################################################################
# 4. Download + decrypt dataset into this benchmark directory
###############################################################################
DATA_DIR="${BENCH_ROOT}/data"
QUERIES_DIR="${BENCH_ROOT}/topics-qrels"

mkdir -p "${DATA_DIR}"
mkdir -p "${QUERIES_DIR}"

python_run() {
  if command -v uv >/dev/null 2>&1; then
    uv run --active python -m "$@" 
  else
    python -m "$@"
  fi
}

if [ -f "${DATA_DIR}/browsecomp_plus_decrypted.jsonl" ] && [ -f "${QUERIES_DIR}/queries.tsv" ]; then
  echo "[BrowseCompPlus setup] Dataset already exists, skipping download."
else
  echo "[BrowseCompPlus setup] Downloading & decrypting dataset..."
  python_run scripts_build_index.decrypt_dataset \
      --output "${DATA_DIR}/browsecomp_plus_decrypted.jsonl" \
      --generate-tsv "${QUERIES_DIR}/queries.tsv"

  echo "[BrowseCompPlus setup] Dataset written to:"
  echo "  ${DATA_DIR}/browsecomp_plus_decrypted.jsonl"
  echo "  ${QUERIES_DIR}/queries.tsv"
fi

# ------------------------------------------------------------
# Create lightweight JSONL with only docids
# ------------------------------------------------------------
LIGHT_JSONL="${DATA_DIR}/browsecomp_plus_decrypted_docids.jsonl"

if [ -f "${LIGHT_JSONL}" ]; then
  echo "[BrowseCompPlus setup] Light docids JSONL already exists, skipping."
else
  echo "[BrowseCompPlus setup] Creating lightweight docid-only JSONL..."
  PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}" python_run make_light_dataset  \
    --input  "${DATA_DIR}/browsecomp_plus_decrypted.jsonl" \
    --output "${LIGHT_JSONL}"
  echo "[BrowseCompPlus setup] Light dataset written to: ${LIGHT_JSONL}"
fi


###############################################################################
# 5. Download indexes
###############################################################################
echo "[BrowseCompPlus setup] Downloading indexes into ${BENCH_ROOT}/indexes..."
cd "${BENCH_ROOT}"

# Try to install hf_transfer for faster downloads (non-blocking)
echo "[BrowseCompPlus setup] Installing hf_transfer for faster downloads..."
pip_install -U hf_transfer 2>/dev/null || echo "[BrowseCompPlus setup] hf_transfer installation skipped (continuing with standard download)"

mkdir -p "${BENCH_ROOT}/indexes"
cd "${BENCH_ROOT}/indexes"

# Clean up stale lock files for the specific indexes we're downloading
# This prevents hanging on interrupted downloads - hf download will skip existing files automatically
echo "[BrowseCompPlus setup] Cleaning up stale download locks for browsecomp-plus indexes..."
if [ -d ".cache/huggingface/download" ]; then
  find .cache/huggingface/download/bm25 -type f \( -name "*.lock" -o -name "*.incomplete" \) -delete 2>/dev/null || true
  find .cache/huggingface/download/qwen3-embedding-8b -type f \( -name "*.lock" -o -name "*.incomplete" \) -delete 2>/dev/null || true
fi

# Download indexes - hf download automatically skips files that already exist
# Using HF_HUB_ENABLE_HF_TRANSFER=1 for faster downloads (only affects this command)
if [ "$BM25_AVAILABLE" = true ]; then
  echo "[BrowseCompPlus setup] Downloading BM25 index (skips existing files)..."
  HF_HUB_ENABLE_HF_TRANSFER=1 hf download Tevatron/browsecomp-plus-indexes --repo-type=dataset --include="bm25/*" --local-dir .
fi

echo "[BrowseCompPlus setup] Downloading Qwen3-embedding-8b index (skips existing files)..."
HF_HUB_ENABLE_HF_TRANSFER=1 hf download Tevatron/browsecomp-plus-indexes --repo-type=dataset --include="qwen3-embedding-8b/*" --local-dir .

echo "[BrowseCompPlus setup] ALL DONE"
