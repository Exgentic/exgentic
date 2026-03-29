#!/usr/bin/env bash
# setup_environments.sh - Create multiple UV virtual environments for Exgentic
#
# Usage: ./setup_environments.sh [benchmark1 benchmark2 ...]
#
# This script creates:
# 1. A 'core' environment with all optional dependencies
# 2. Benchmark-specific environments for specified benchmarks (or all if none specified)
# 3. Generates requirements.txt files for each environment
#
# Examples:
#   ./setup_environments.sh                    # Create all environments
#   ./setup_environments.sh gsm8k hotpotqa     # Create only gsm8k and hotpotqa
#   ./setup_environments.sh core               # Create only core environment

set -euo pipefail

# Tracking arrays for summary report
SETUP_PASSED=()
SETUP_FAILED=()

# Get the project root (two levels up from this script in misc/security/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VENV_DIR="${PROJECT_ROOT}/.venvs"
REQUIREMENTS_DIR="${SCRIPT_DIR}/requirements"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if uv is installed
if ! command -v uv >/dev/null 2>&1; then
    log_error "uv is not installed. Install it with: pip install uv"
    exit 1
fi

# Parse command line arguments
REQUESTED_BENCHMARKS=()
if [ $# -gt 0 ]; then
    REQUESTED_BENCHMARKS=("$@")
fi
ALL_BENCHMARKS=("gsm8k" "hotpotqa" "appworld" "browsecompplus" "swebench" "tau2")

# Determine which benchmarks to create
if [ ${#REQUESTED_BENCHMARKS[@]} -eq 0 ]; then
    # No arguments provided, create all
    CREATE_CORE=true
    BENCHMARKS_TO_CREATE=("${ALL_BENCHMARKS[@]}")
    log_info "No benchmarks specified, creating all environments"
elif [ "${REQUESTED_BENCHMARKS[0]}" = "core" ] && [ ${#REQUESTED_BENCHMARKS[@]} -eq 1 ]; then
    # Only 'core' requested
    CREATE_CORE=true
    BENCHMARKS_TO_CREATE=()
    log_info "Creating only core environment"
else
    # Specific benchmarks requested
    CREATE_CORE=false
    BENCHMARKS_TO_CREATE=()
    
    # Check if 'core' is in the list
    for arg in "${REQUESTED_BENCHMARKS[@]}"; do
        if [ "$arg" = "core" ]; then
            CREATE_CORE=true
        else
            # Validate benchmark name
            if [[ " ${ALL_BENCHMARKS[@]} " =~ " ${arg} " ]]; then
                BENCHMARKS_TO_CREATE+=("$arg")
            else
                log_warning "Unknown benchmark: $arg (skipping)"
            fi
        fi
    done
    
    if [ "$CREATE_CORE" = true ]; then
        log_info "Creating core environment and benchmarks: ${BENCHMARKS_TO_CREATE[*]}"
    else
        log_info "Creating benchmark environments: ${BENCHMARKS_TO_CREATE[*]}"
    fi
fi

log_info "Starting environment setup..."
log_info "Project root: ${PROJECT_ROOT}"
log_info "Virtual environments will be created in: ${VENV_DIR}"

mkdir -p "${VENV_DIR}"

# 1. Create 'core' environment with all optional dependencies
if [ "$CREATE_CORE" = true ]; then
    log_info "Creating 'core' environment with all optional dependencies..."

    uv venv "${VENV_DIR}/core" --python 3.11

    # Activate and install
    source "${VENV_DIR}/core/bin/activate"

    # Install project with all optional dependencies
    log_info "Installing base project with all optional dependencies..."
    uv pip install -e "${PROJECT_ROOT}[smolagents,openaimacp,otel,cli,dev]"

    # Generate requirements.txt for core environment
    log_info "Generating requirements.txt for core environment..."
    mkdir -p "${REQUIREMENTS_DIR}/core"
    uv pip freeze | grep -v "^-e " | grep -v " @ file://" | grep -v "github.ibm.com" | grep -v " @ git+https://" | grep -v " @ git+ssh://" > "${REQUIREMENTS_DIR}/core/requirements.txt"
    core_pkg_count=$(wc -l < "${REQUIREMENTS_DIR}/core/requirements.txt" | tr -d ' ')
    log_success "Saved ${core_pkg_count} packages to ${REQUIREMENTS_DIR}/core/requirements.txt"
    
    deactivate
    log_success "Core environment created at ${VENV_DIR}/core"
else
    log_info "Skipping core environment creation"
fi

# 2. Create benchmark-specific environments
if [ ${#BENCHMARKS_TO_CREATE[@]} -gt 0 ]; then
    BENCHMARKS=("${BENCHMARKS_TO_CREATE[@]}")
else
    BENCHMARKS=()
fi

if [ ${#BENCHMARKS[@]} -gt 0 ]; then
for benchmark in "${BENCHMARKS[@]}"; do
    log_info "Creating '${benchmark}' environment..."
    
    SETUP_SCRIPT="${PROJECT_ROOT}/src/exgentic/benchmarks/${benchmark}/setup.sh"
    
    if [ ! -f "${SETUP_SCRIPT}" ]; then
        log_warning "Setup script not found: ${SETUP_SCRIPT}, skipping..."
        continue
    fi
    
    # Create virtual environment
    uv venv "${VENV_DIR}/${benchmark}" --python 3.11
    
    # Activate environment
    source "${VENV_DIR}/${benchmark}/bin/activate"
    
    # Install base project with all optional dependencies
    log_info "Installing base project with all optional dependencies for ${benchmark}..."
    uv pip install -e "${PROJECT_ROOT}[smolagents,openaimacp,otel,cli,dev]"
    
    # Run benchmark-specific setup script
    log_info "Running setup script for ${benchmark}..."
    
    # Change to project root before running setup script
    # (some scripts expect to be run from project root)
    cd "${PROJECT_ROOT}"
    
    SETUP_OK=true
    if bash "${SETUP_SCRIPT}"; then
        log_success "${benchmark} setup completed successfully"
        SETUP_PASSED+=("${benchmark}")
    else
        log_warning "${benchmark} setup script encountered issues (exit code: $?)"
        SETUP_FAILED+=("${benchmark}")
        SETUP_OK=false
    fi
    
    # Generate requirements.txt only when setup succeeded
    if [ "${SETUP_OK}" = true ]; then
        log_info "Generating requirements.txt for ${benchmark} environment..."
        mkdir -p "${REQUIREMENTS_DIR}/${benchmark}"
        uv pip freeze | grep -v "^-e " | grep -v " @ file://" | grep -v "github.ibm.com" | grep -v " @ git+https://" | grep -v " @ git+ssh://" > "${REQUIREMENTS_DIR}/${benchmark}/requirements.txt"
        bench_pkg_count=$(wc -l < "${REQUIREMENTS_DIR}/${benchmark}/requirements.txt" | tr -d ' ')
        log_success "Saved ${bench_pkg_count} packages to ${REQUIREMENTS_DIR}/${benchmark}/requirements.txt"
    else
        log_warning "Skipping requirements.txt generation for ${benchmark} due to setup failure"
    fi
    
    deactivate
    log_success "${benchmark} environment created at ${VENV_DIR}/${benchmark}"
done
fi

# Generate README for requirements
log_info "Generating requirements README..."
cat > "${REQUIREMENTS_DIR}/README.md" << 'EOF'
# Requirements Files

This directory contains frozen requirements for each virtual environment.

## Directories

Each environment has its own directory with a `requirements.txt` file:

- `core/requirements.txt` - Core environment with all optional dependencies
- `gsm8k/requirements.txt` - GSM8K benchmark environment
- `hotpotqa/requirements.txt` - HotpotQA benchmark environment
- `appworld/requirements.txt` - AppWorld benchmark environment
- `browsecompplus/requirements.txt` - BrowseComp+ benchmark environment
- `swebench/requirements.txt` - SWE-bench benchmark environment
- `tau2/requirements.txt` - TAU-2 benchmark environment

## Usage

To recreate an environment from a requirements file:

```bash
# Create a new virtual environment
uv venv .venv --python 3.11

# Activate it
source .venv/bin/activate

# Install from requirements
uv pip install -r misc/security/requirements/core/requirements.txt
```

## Notes

- These files are generated automatically by `misc/security/setup_environments.sh`
- They represent the exact package versions installed in each environment
- Only PyPI packages are included (no git or local installs)
- Regenerate by re-running the setup script

## Generation Date

EOF

log_success "Requirements README created at ${REQUIREMENTS_DIR}/README.md"

# Summary
echo ""
log_success "Environment setup completed!"
echo ""
echo "Created environments:"
if [ "$CREATE_CORE" = true ] && [ -d "${VENV_DIR}/core" ]; then
    echo "  ${VENV_DIR}/core - Core environment with all optional dependencies"
    if [ -f "${REQUIREMENTS_DIR}/core/requirements.txt" ]; then
        pkg_count=$(wc -l < "${REQUIREMENTS_DIR}/core/requirements.txt" | tr -d ' ')
        echo "    Requirements: ${REQUIREMENTS_DIR}/core/requirements.txt (${pkg_count} packages)"
    fi
fi
if [ ${#BENCHMARKS[@]} -gt 0 ]; then
for benchmark in "${BENCHMARKS[@]}"; do
    if [ -d "${VENV_DIR}/${benchmark}" ]; then
        echo "  ${VENV_DIR}/${benchmark} - ${benchmark} benchmark environment"
        if [ -f "${REQUIREMENTS_DIR}/${benchmark}/requirements.txt" ]; then
            pkg_count=$(wc -l < "${REQUIREMENTS_DIR}/${benchmark}/requirements.txt" | tr -d ' ')
            echo "    Requirements: ${REQUIREMENTS_DIR}/${benchmark}/requirements.txt (${pkg_count} packages)"
        fi
    fi
done
fi
echo ""
echo "To activate an environment, use:"
if [ "$CREATE_CORE" = true ]; then
    echo "  source ${VENV_DIR}/core/bin/activate"
fi
if [ ${#BENCHMARKS[@]} -gt 0 ]; then
for benchmark in "${BENCHMARKS[@]}"; do
    if [ -d "${VENV_DIR}/${benchmark}" ]; then
        echo "  source ${VENV_DIR}/${benchmark}/bin/activate"
    fi
done
fi
echo ""
echo "Note: Some benchmarks may require additional prerequisites:"
echo "  - appworld, swebench: Git LFS"
echo "  - browsecompplus: Java 21+, SSH access to IBM GitHub"
echo ""
echo "Usage examples:"
echo "  ./misc/security/setup_environments.sh                    # Create all environments"
echo "  ./misc/security/setup_environments.sh gsm8k hotpotqa     # Create specific benchmarks"
echo "  ./misc/security/setup_environments.sh core               # Create only core environment"

# Setup script pass/fail summary
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Setup Script Results"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ ${#SETUP_PASSED[@]} -gt 0 ]; then
    for b in "${SETUP_PASSED[@]}"; do
        echo -e "  ${GREEN}✔ PASSED${NC}  ${b}"
    done
else
    echo "  (no benchmark setup scripts ran)"
fi
if [ ${#SETUP_FAILED[@]} -gt 0 ]; then
    for b in "${SETUP_FAILED[@]}"; do
        echo -e "  ${RED}✘ FAILED${NC}  ${b}"
    done
fi
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"