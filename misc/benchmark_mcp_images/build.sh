#!/bin/bash
set -e

# Script to build exgentic benchmark images for tau2 and gsm8k
# Automatically detects whether to use docker or podman
# Optionally pushes to GitHub Container Registry with --push flag

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored messages
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Detect container runtime (docker or podman)
detect_runtime() {
    if command -v docker &> /dev/null; then
        echo "docker"
    elif command -v podman &> /dev/null; then
        echo "podman"
    else
        print_error "Neither docker nor podman is installed!"
        print_error "Please install one of them to continue."
        exit 1
    fi
}

# Push image to GitHub Container Registry
push_to_ghcr() {
    local benchmark=$1
    local runtime=$2
    local tag=$3
    local local_image="localhost/exgentic-mcp-${benchmark}:${tag}"
    local ghcr_image="ghcr.io/exgentic/exgentic-mcp-${benchmark}:${tag}"
    
    print_step "Pushing ${benchmark} to GitHub Container Registry..."
    
    # Check if already logged in to ghcr.io
    if ! $runtime info 2>/dev/null | grep -q "ghcr.io"; then
        print_warn "Not authenticated with ghcr.io"
        
        # Check for environment variables
        if [ -z "$GITHUB_TOKEN" ] || [ -z "$GITHUB_USERNAME" ]; then
            print_error "GITHUB_TOKEN and GITHUB_USERNAME environment variables required for push"
            print_error "Please set them or authenticate manually with:"
            print_error "  echo \$GITHUB_TOKEN | $runtime login ghcr.io -u \$GITHUB_USERNAME --password-stdin"
            return 1
        fi
        
        print_info "Authenticating with ghcr.io..."
        if echo "$GITHUB_TOKEN" | $runtime login ghcr.io -u "$GITHUB_USERNAME" --password-stdin 2>/dev/null; then
            print_info "✓ Successfully authenticated with ghcr.io"
        else
            print_error "✗ Failed to authenticate with ghcr.io"
            return 1
        fi
    fi
    
    # Tag for GHCR
    print_info "Tagging image for GHCR: ${ghcr_image}"
    if $runtime tag "${local_image}" "${ghcr_image}"; then
        print_info "✓ Successfully tagged image"
    else
        print_error "✗ Failed to tag image"
        return 1
    fi
    
    # Push to GHCR
    print_info "Pushing to GHCR (this may take several minutes)..."
    if $runtime push "${ghcr_image}"; then
        print_info "✓ Successfully pushed ${ghcr_image}"
        print_info "View at: https://github.com/orgs/Exgentic/packages/container/package/exgentic-mcp-${benchmark}"
        return 0
    else
        print_error "✗ Failed to push ${ghcr_image}"
        return 1
    fi
}

# Build image for a specific benchmark
build_benchmark() {
    local benchmark=$1
    local runtime=$2
    local image_name="localhost/exgentic-mcp-${benchmark}"
    local tag=$3
    local use_cache=$4
    local should_push=$5
    
    print_info "Building ${image_name}:${tag} using ${runtime}..."
    if [ "$use_cache" = "false" ]; then
        print_info "Building without cache (default)"
    else
        print_info "Building with cache enabled"
    fi
    
    # Build the image with the benchmark name as build arg
    local build_cmd="$runtime build"
    if [ "$use_cache" = "false" ]; then
        build_cmd="$build_cmd --no-cache"
    fi
    
    if $build_cmd \
        --build-arg BENCHMARK_NAME="${benchmark}" \
        -t "${image_name}:${tag}" \
        -f Dockerfile \
        .; then
        print_info "✓ Successfully built ${image_name}:${tag}"
        
        # Push to GHCR if requested
        if [ "$should_push" = "true" ]; then
            echo ""
            if push_to_ghcr "$benchmark" "$runtime" "$tag"; then
                return 0
            else
                print_warn "Build succeeded but push failed"
                return 1
            fi
        fi
        return 0
    else
        print_error "✗ Failed to build ${image_name}:${tag}"
        return 1
    fi
}

# Main script
main() {
    local script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    cd "$script_dir"
    
    print_info "Exgentic Benchmark Image Builder"
    print_info "================================="
    
    # Detect container runtime
    RUNTIME=$(detect_runtime)
    print_info "Detected container runtime: ${RUNTIME}"
    
    # Parse command line arguments
    BENCHMARK=""
    TAG="latest"
    USE_CACHE="false"  # Default: do not use cache for consistency
    PUSH_TO_GHCR="false"  # Default: do not push to GHCR
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --tag)
                TAG="$2"
                shift 2
                ;;
            --use-cache)
                USE_CACHE="true"
                shift
                ;;
            --push)
                PUSH_TO_GHCR="true"
                shift
                ;;
            --help|-h)
                cat << EOF
Usage: $0 BENCHMARK [--tag TAG] [--use-cache] [--push]

Build exgentic benchmark Docker/Podman image.

Arguments:
  BENCHMARK      Benchmark name (required, positional: tau2 or gsm8k)
  --tag TAG      Image tag (optional, default: latest)
  --use-cache    Use Docker cache during build (optional, default: no cache for consistency)
  --push         Push to GitHub Container Registry after build (optional)

Examples:
  $0 tau2                           # Build without cache (default)
  $0 gsm8k --tag v1.0.0             # Build v1.0.0 without cache
  $0 tau2 --use-cache               # Build with cache enabled
  $0 gsm8k --push                   # Build and push to GHCR
  $0 tau2 --tag v1.0.0 --push       # Build v1.0.0 and push to GHCR
  $0 gsm8k --tag v1.0.0 --use-cache --push  # Build v1.0.0 with cache and push

Available benchmarks:
  - tau2
  - gsm8k

Push to GHCR:
  When using --push, you must set these environment variables:
    GITHUB_USERNAME - Your GitHub username
    GITHUB_TOKEN    - GitHub Personal Access Token with 'write:packages' permission
  
  Example:
    export GITHUB_USERNAME="your-username"
    export GITHUB_TOKEN="ghp_xxxxxxxxxxxx"
    $0 tau2 --push

The script automatically detects whether to use docker or podman.
By default, builds do not use cache to ensure consistency.
EOF
                exit 0
                ;;
            -*)
                print_error "Unknown option: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
            *)
                if [ -z "$BENCHMARK" ]; then
                    BENCHMARK="$1"
                    shift
                else
                    print_error "Unexpected argument: $1"
                    echo "Use --help for usage information"
                    exit 1
                fi
                ;;
        esac
    done
    
    # Validate benchmark name is provided
    if [ -z "$BENCHMARK" ]; then
        print_error "Benchmark name is required!"
        echo ""
        echo "Usage: $0 BENCHMARK [--tag TAG] [--use-cache] [--push]"
        echo ""
        echo "Available benchmarks: tau2, gsm8k"
        echo "Example: $0 tau2 --tag v1.0.0 --push"
        echo "Use --help for more information"
        exit 1
    fi
    
    BENCHMARKS=("$BENCHMARK")
    
    print_info "Building benchmark: ${BENCHMARK}"
    print_info "Image tag: ${TAG}"
    if [ "$USE_CACHE" = "true" ]; then
        print_info "Cache: enabled"
    else
        print_info "Cache: disabled (default)"
    fi
    if [ "$PUSH_TO_GHCR" = "true" ]; then
        print_info "Push to GHCR: enabled"
        # Validate GITHUB credentials if push is requested
        if [ -z "$GITHUB_USERNAME" ] || [ -z "$GITHUB_TOKEN" ]; then
            print_error "GITHUB_USERNAME and GITHUB_TOKEN environment variables are required when using --push"
            print_error ""
            print_error "Please set them before running:"
            print_error "  export GITHUB_USERNAME=\"your-username\""
            print_error "  export GITHUB_TOKEN=\"ghp_xxxxxxxxxxxx\""
            print_error ""
            print_error "To create a token: https://github.com/settings/tokens"
            print_error "Required scope: write:packages"
            exit 1
        fi
    else
        print_info "Push to GHCR: disabled"
    fi
    echo ""
    
    # Build each benchmark
    SUCCESS_COUNT=0
    FAIL_COUNT=0
    PUSH_SUCCESS_COUNT=0
    PUSH_FAIL_COUNT=0
    
    for benchmark in "${BENCHMARKS[@]}"; do
        if build_benchmark "$benchmark" "$RUNTIME" "$TAG" "$USE_CACHE" "$PUSH_TO_GHCR"; then
            ((SUCCESS_COUNT++))
            if [ "$PUSH_TO_GHCR" = "true" ]; then
                ((PUSH_SUCCESS_COUNT++))
            fi
        else
            ((FAIL_COUNT++))
            if [ "$PUSH_TO_GHCR" = "true" ]; then
                ((PUSH_FAIL_COUNT++))
            fi
        fi
        echo ""
    done
    
    # Summary
    print_info "Build Summary"
    print_info "============="
    print_info "Builds successful: ${SUCCESS_COUNT}"
    if [ $FAIL_COUNT -gt 0 ]; then
        print_error "Builds failed: ${FAIL_COUNT}"
    fi
    
    if [ "$PUSH_TO_GHCR" = "true" ]; then
        print_info "Pushes successful: ${PUSH_SUCCESS_COUNT}"
        if [ $PUSH_FAIL_COUNT -gt 0 ]; then
            print_error "Pushes failed: ${PUSH_FAIL_COUNT}"
        fi
    fi
    
    if [ $FAIL_COUNT -gt 0 ]; then
        exit 1
    else
        print_info "All operations completed successfully!"
        echo ""
        print_info "Built images:"
        for benchmark in "${BENCHMARKS[@]}"; do
            echo "  - localhost/exgentic-mcp-${benchmark}:${TAG}"
            if [ "$PUSH_TO_GHCR" = "true" ]; then
                echo "  - ghcr.io/exgentic/exgentic-mcp-${benchmark}:${TAG}"
            fi
        done
        
        if [ "$PUSH_TO_GHCR" = "true" ]; then
            echo ""
            print_info "View packages at:"
            for benchmark in "${BENCHMARKS[@]}"; do
                echo "  https://github.com/orgs/Exgentic/packages/container/package/exgentic-mcp-${benchmark}"
            done
        fi
    fi
}

main "$@"

# Made with Bob
