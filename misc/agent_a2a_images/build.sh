#!/bin/bash
set -e

# Script to build exgentic A2A agent images.
# Automatically detects whether to use docker or podman.
# Optionally pushes to GitHub Container Registry with --push flag.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../build_lib.sh"

main() {
    local script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    cd "$script_dir"

    print_info "Exgentic A2A Agent Image Builder"
    print_info "================================="

    RUNTIME=$(detect_runtime)
    print_info "Detected container runtime: ${RUNTIME}"

    check_git_clean "${script_dir}/Dockerfile" || exit 1

    AGENT=""
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
Usage: $0 AGENT_NAME [--tag TAG] [--use-cache] [--push] [--multiplatform]

Build exgentic A2A agent Docker/Podman image.

Arguments:
  AGENT_NAME       Agent name (required, positional)
  --tag TAG        Image tag (optional, default: latest)
  --use-cache      Use Docker cache during build (optional, default: no cache)
  --push           Push to GitHub Container Registry after build (optional)
  --multiplatform  Build for both linux/amd64 and linux/arm64 using docker buildx.
                   Requires Docker with buildx. Without --push, only the native
                   platform image is loaded locally.

Examples:
  $0 tool_calling                                       # Build native platform without cache
  $0 tool_calling --tag v1.0.0                          # Build v1.0.0 without cache
  $0 tool_calling --use-cache                           # Build with cache enabled
  $0 tool_calling --push                                # Build and push to GHCR
  $0 tool_calling --tag v1.0.0 --push                   # Build v1.0.0 and push to GHCR
  $0 tool_calling --multiplatform --push                # Build amd64+arm64 and push to GHCR
  $0 tool_calling --tag v1.0.0 --multiplatform --push   # Build v1.0.0 multiplatform and push

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
                if [ -z "$AGENT" ]; then
                    AGENT="$1"; shift
                else
                    print_error "Unexpected argument: $1"
                    echo "Use --help for usage information"
                    exit 1
                fi
                ;;
        esac
    done

    if [ -z "$AGENT" ]; then
        print_error "Agent name is required!"
        echo "Usage: $0 AGENT_NAME [--tag TAG] [--use-cache] [--push] [--multiplatform]"
        exit 1
    fi

    print_info "Building agent: ${AGENT}"
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

    if build_image "$AGENT" "exgentic-a2a" "AGENT_NAME" "$RUNTIME" "$TAG" "$USE_CACHE" "$PUSH_TO_GHCR" "$MULTIPLATFORM"; then
        ((SUCCESS_COUNT++))
    else
        ((FAIL_COUNT++))
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
    echo "  - localhost/exgentic-a2a-${AGENT}:${TAG}"
    if [ "$PUSH_TO_GHCR" = "true" ]; then
        echo "  - ghcr.io/exgentic/exgentic-a2a-${AGENT}:${TAG}"
        echo ""
        print_info "View at: https://github.com/orgs/Exgentic/packages/container/package/exgentic-a2a-${AGENT}"
    fi
}

main "$@"

# Made with Bob
