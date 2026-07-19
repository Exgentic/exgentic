#!/bin/bash
# Shared build library for exgentic Docker image build scripts.
# Source this file; do not execute it directly.

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
print_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }
print_step()  { echo -e "${BLUE}[STEP]${NC} $1"; }

# Fail if there are uncommitted changes or if the current branch doesn't match
# the branch checked out inside the Dockerfile (git checkout <branch> line).
# Pass the path to the Dockerfile as the first argument.
check_git_clean() {
    local dockerfile=${1:-Dockerfile}

    if ! git rev-parse --is-inside-work-tree &>/dev/null; then
        print_error "Not inside a git repository"
        return 1
    fi

    if ! git diff --quiet || ! git diff --cached --quiet; then
        print_error "There are uncommitted changes. Commit or stash them before building."
        git status --short
        return 1
    fi

    local current_branch
    current_branch=$(git rev-parse --abbrev-ref HEAD)

    # Extract the branch the Dockerfile checks out (first "git checkout <branch>" line)
    local dockerfile_branch
    dockerfile_branch=$(grep -m1 'git checkout ' "$dockerfile" 2>/dev/null | awk '{print $NF}')

    if [ -n "$dockerfile_branch" ] && [ "$current_branch" != "$dockerfile_branch" ]; then
        print_error "Current branch '${current_branch}' does not match the Dockerfile branch '${dockerfile_branch}'."
        print_error "Switch to '${dockerfile_branch}' or update the Dockerfile before building."
        return 1
    fi

    local remote_ref
    remote_ref=$(git rev-parse --abbrev-ref --symbolic-full-name "@{u}" 2>/dev/null || true)

    if [ -z "$remote_ref" ]; then
        print_error "Current branch '${current_branch}' has no upstream. Push it before building."
        return 1
    fi

    local unpushed
    unpushed=$(git rev-list "${remote_ref}..HEAD" --count)
    if [ "$unpushed" -gt 0 ]; then
        print_error "There are ${unpushed} unpushed commit(s) on branch '${current_branch}'. Push before building."
        return 1
    fi

    print_info "Git working tree is clean and up to date with remote (branch: ${current_branch})."
}

# Reclaim disk space before building: prune the buildx cache and dangling
# images, then report how much was freed and the remaining usage. Multi-platform
# builds export the image to a tarball and re-import it, which needs a large slab
# of temporary disk; a full VM is the most common cause of those builds failing.
prune_build_space() {
    local runtime=$1

    print_step "Reclaiming disk space before build..."

    if [ "$runtime" = "docker" ]; then
        print_info "Pruning buildx build cache..."
        docker buildx prune -f 2>&1 | tail -1
    fi

    print_info "Pruning dangling images..."
    $runtime image prune -f 2>&1 | tail -1

    print_info "Disk usage after prune:"
    $runtime system df 2>&1
    echo ""
}

# Detect container runtime (docker preferred over podman)
detect_runtime() {
    if command -v docker &> /dev/null; then
        echo "docker"
    elif command -v podman &> /dev/null; then
        echo "podman"
    else
        print_error "Neither docker nor podman is installed!"
        exit 1
    fi
}

# Login to GitHub Container Registry using GITHUB_USERNAME / GITHUB_TOKEN env vars.
login_to_ghcr() {
    local runtime=$1
    print_info "Authenticating with ghcr.io..."
    if echo "$GITHUB_TOKEN" | $runtime login ghcr.io -u "$GITHUB_USERNAME" --password-stdin 2>/dev/null; then
        print_info "✓ Successfully authenticated with ghcr.io"
    else
        print_error "✗ Failed to authenticate with ghcr.io"
        return 1
    fi
}

# Build (and optionally push) a single image.
#
# Usage:
#   build_image <name> <image_prefix> <build_arg_name> <runtime> <tag> <use_cache> <should_push> <multiplatform>
#
# Example:
#   build_image gsm8k exgentic-mcp BENCHMARK_NAME docker latest false false true
build_image() {
    local name=$1
    local image_prefix=$2
    local build_arg_name=$3
    local runtime=$4
    local tag=$5
    local use_cache=$6
    local should_push=$7
    local multiplatform=$8

    local local_image="localhost/${image_prefix}-${name}:${tag}"
    local ghcr_image="ghcr.io/exgentic/${image_prefix}-${name}:${tag}"

    print_info "Building ${local_image} using ${runtime}..."
    if [ "$use_cache" = "false" ]; then
        print_info "Building without cache (default)"
    else
        print_info "Building with cache enabled"
    fi

    if [ "$multiplatform" = "true" ] && [ "$runtime" != "docker" ]; then
        print_error "Multi-platform builds require Docker with buildx (not podman)"
        return 1
    fi

    # Ensure buildx builder exists when using docker
    if [ "$runtime" = "docker" ]; then
        if ! docker buildx inspect multiplatform-builder &>/dev/null; then
            print_info "Creating multiplatform-builder buildx instance..."
            docker buildx create --name multiplatform-builder --use --bootstrap
        else
            docker buildx use multiplatform-builder
        fi
    fi

    # Assemble build command
    local build_cmd
    if [ "$runtime" = "docker" ]; then
        build_cmd="docker buildx build"
    else
        build_cmd="podman build"
    fi
    if [ "$use_cache" = "false" ]; then
        build_cmd="$build_cmd --no-cache"
    fi

    # Platform and output flags
    if [ "$multiplatform" = "true" ] && [ "$should_push" = "true" ]; then
        build_cmd="$build_cmd --platform linux/amd64,linux/arm64 --network=host --push -t ${ghcr_image}"
    elif [ "$multiplatform" = "true" ]; then
        local native_platform
        native_platform=$(uname -m)
        if [ "$native_platform" = "arm64" ] || [ "$native_platform" = "aarch64" ]; then
            native_platform="linux/arm64"
        else
            native_platform="linux/amd64"
        fi
        print_warn "Multi-platform build without --push: loading native platform (${native_platform}) only"
        print_warn "Use --push to publish both platforms to GHCR"
        build_cmd="$build_cmd --platform ${native_platform} --load -t ${local_image}"
    else
        # Single-platform: --load is docker-specific
        if [ "$runtime" = "docker" ]; then
            build_cmd="$build_cmd --load"
        fi
        build_cmd="$build_cmd -t ${local_image}"
    fi

    if $build_cmd "--build-arg" "${build_arg_name}=${name}" -f Dockerfile .; then
        if [ "$multiplatform" = "true" ] && [ "$should_push" = "true" ]; then
            print_info "✓ Successfully built and pushed multi-platform ${ghcr_image}"
            print_info "View at: https://github.com/orgs/Exgentic/packages/container/package/${image_prefix}-${name}"
        else
            print_info "✓ Successfully built ${local_image}"
            if [ "$should_push" = "true" ]; then
                print_info "Tagging and pushing to GHCR..."
                if $runtime tag "${local_image}" "${ghcr_image}" && $runtime push "${ghcr_image}"; then
                    print_info "✓ Successfully pushed ${ghcr_image}"
                    print_info "View at: https://github.com/orgs/Exgentic/packages/container/package/${image_prefix}-${name}"
                else
                    print_error "✗ Failed to push ${ghcr_image}"
                    return 1
                fi
            fi
        fi
        return 0
    else
        print_error "✗ Failed to build ${local_image}"
        return 1
    fi
}
