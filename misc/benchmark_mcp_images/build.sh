#!/bin/bash
set -e

# Script to build exgentic benchmark MCP images.
# Automatically detects whether to use docker or podman.
# Optionally pushes to GitHub Container Registry with --push flag.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../build_lib.sh"

main() {
    local script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    cd "$script_dir"

    print_info "Exgentic Benchmark Image Builder"
    print_info "================================="

    RUNTIME=$(detect_runtime)
    print_info "Detected container runtime: ${RUNTIME}"

    check_git_clean "${script_dir}/Dockerfile" || exit 1

    BENCHMARK=""
    TAG="latest"
    USE_CACHE="false"
    PUSH_TO_GHCR="false"
    MULTIPLATFORM="false"

    while [[ $# -gt 0 ]]; do
        case $1 in
            --tag)           TAG="$2"; shift 2 ;;
            --use-cache)     USE_CACHE="true"; shift ;;
            --push)          PUSH_TO_GHCR="true"; shift ;;
            --multiplatform) MULTIPLATFORM="true"; shift ;;
            --help|-h)
                cat << EOF
Usage: $0 BENCHMARK [--tag TAG] [--use-cache] [--push] [--multiplatform]

Build exgentic benchmark Docker/Podman image.

Arguments:
  BENCHMARK        Benchmark name (required, positional: tau2 or gsm8k)
  --tag TAG        Image tag (optional, default: latest)
  --use-cache      Use Docker cache during build (optional, default: no cache)
  --push           Push to GitHub Container Registry after build (optional)
  --multiplatform  Build for both linux/amd64 and linux/arm64 using docker buildx.
                   Requires Docker with buildx. Without --push, only the native
                   platform image is loaded locally.

Examples:
  $0 tau2                                       # Build native platform without cache
  $0 gsm8k --tag v1.0.0                         # Build v1.0.0 without cache
  $0 tau2 --use-cache                           # Build with cache enabled
  $0 gsm8k --push                               # Build and push to GHCR
  $0 tau2 --tag v1.0.0 --push                   # Build v1.0.0 and push to GHCR
  $0 gsm8k --multiplatform --push               # Build amd64+arm64 and push to GHCR
  $0 gsm8k --tag v1.0.0 --multiplatform --push  # Build v1.0.0 multiplatform and push

Push to GHCR:
  When using --push, set these environment variables:
    GITHUB_USERNAME - Your GitHub username
    GITHUB_TOKEN    - GitHub Personal Access Token with 'write:packages' permission
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
                    BENCHMARK="$1"; shift
                else
                    print_error "Unexpected argument: $1"
                    echo "Use --help for usage information"
                    exit 1
                fi
                ;;
        esac
    done

    if [ -z "$BENCHMARK" ]; then
        print_error "Benchmark name is required!"
        echo "Usage: $0 BENCHMARK [--tag TAG] [--use-cache] [--push] [--multiplatform]"
        echo "Available benchmarks: tau2, gsm8k"
        exit 1
    fi

    print_info "Building benchmark: ${BENCHMARK}"
    print_info "Image tag: ${TAG}"
    [ "$USE_CACHE" = "true" ] && print_info "Cache: enabled" || print_info "Cache: disabled (default)"
    [ "$MULTIPLATFORM" = "true" ] && print_info "Platform: linux/amd64 + linux/arm64 (multiplatform)" || print_info "Platform: native ($(uname -m))"

    if [ "$PUSH_TO_GHCR" = "true" ]; then
        print_info "Push to GHCR: enabled"
        if [ -z "$GITHUB_USERNAME" ] || [ -z "$GITHUB_TOKEN" ]; then
            print_error "GITHUB_USERNAME and GITHUB_TOKEN are required when using --push"
            print_error "To create a token: https://github.com/settings/tokens (scope: write:packages)"
            exit 1
        fi
        login_to_ghcr "$RUNTIME" || exit 1
    else
        print_info "Push to GHCR: disabled"
    fi
    echo ""

    prune_build_space "$RUNTIME"

    SUCCESS_COUNT=0
    FAIL_COUNT=0

    if build_image "$BENCHMARK" "exgentic-mcp" "BENCHMARK_NAME" "$RUNTIME" "$TAG" "$USE_CACHE" "$PUSH_TO_GHCR" "$MULTIPLATFORM"; then
        # Plain arithmetic assignment, not a standalone ((expr++)) command: under
        # `set -e`, ((x++)) with x=0 evaluates to the pre-increment value (0),
        # which bash treats as command failure and exits the script immediately
        # -- even though the build just succeeded. $((x + 1)) has no such trap.
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
    echo ""

    print_info "Build Summary"
    print_info "============="
    print_info "Builds successful: ${SUCCESS_COUNT}"
    [ $FAIL_COUNT -gt 0 ] && print_error "Builds failed: ${FAIL_COUNT}"

    if [ $FAIL_COUNT -gt 0 ]; then
        exit 1
    fi

    print_info "All operations completed successfully!"
    echo ""
    print_info "Built images:"
    echo "  - localhost/exgentic-mcp-${BENCHMARK}:${TAG}"
    if [ "$PUSH_TO_GHCR" = "true" ]; then
        echo "  - ghcr.io/exgentic/exgentic-mcp-${BENCHMARK}:${TAG}"
        echo ""
        print_info "View at: https://github.com/orgs/Exgentic/packages/container/package/exgentic-mcp-${BENCHMARK}"
    fi
}

main "$@"

# Made with Bob
