FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Build arguments
ARG AGENT_NAME
ARG RELEASE_VERSION="main"
# Source revision baked into the image, supplied by build_lib.sh
ARG GIT_COMMIT="unknown"
ARG GIT_BRANCH="unknown"

# Validate AGENT_NAME is provided
RUN if [ -z "$AGENT_NAME" ]; then \
        echo "ERROR: AGENT_NAME build argument is required"; \
        echo "Usage: docker build --build-arg AGENT_NAME=<agent> -t exgentic-a2a-<agent> ."; \
        exit 1; \
    fi

# Install system dependencies (git and git-lfs are needed for cloning)
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

# Configure git to handle large repositories and avoid HTTP/2 issues
RUN git config --global http.postBuffer 524288000 && \
    git config --global http.version HTTP/1.1 && \
    git config --global core.compression 0

# Set working directory
WORKDIR /app

# Clone the Exgentic repository
RUN git clone https://github.com/Exgentic/exgentic.git /app/exgentic

# Checkout the feature/mcp-command branch
WORKDIR /app/exgentic
RUN git checkout feature/mcp-command

# Set UV_SYSTEM_PYTHON before any uv operations to ensure uv uses system Python
ENV UV_SYSTEM_PYTHON=1

# Install Exgentic and its dependencies using uv
RUN uv pip install --system --no-cache -e .

# Setup the agent - UV_SYSTEM_PYTHON is already set above
RUN exgentic install --agent ${AGENT_NAME}

# Copy entrypoint script
WORKDIR /app
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Set environment variables
ENV AGENT_NAME=${AGENT_NAME} \
    HOST=0.0.0.0 \
    PORT=8000 \
    LOG_LEVEL=INFO \
    PRODUCTION_MODE=True \
    RELEASE_VERSION=${RELEASE_VERSION}

# Create non-root user and set permissions
RUN useradd -m -u 1001 exgentic && \
    chown -R 1001:0 /app && \
    chmod -R g+rwX /app
    
USER 1001

# Set HOME environment variable to /app
ENV HOME=/app


# Record the source revision as standard OCI image labels. These overwrite the
# labels inherited from the uv base image so `docker inspect` reflects Exgentic.
# NOTE: config LABELs are per-platform and are NOT shown on GHCR's package page;
# build_lib.sh also writes matching index-level annotations, which is what GHCR
# and multi-arch consumers read.
LABEL org.opencontainers.image.source="https://github.com/Exgentic/exgentic" \
      org.opencontainers.image.revision="${GIT_COMMIT}" \
      org.opencontainers.image.version="${GIT_BRANCH}" \
      org.opencontainers.image.title="exgentic-a2a-${AGENT_NAME}" \
      org.opencontainers.image.description="Exgentic A2A agent: ${AGENT_NAME}" \
      org.opencontainers.image.url="https://github.com/Exgentic/exgentic"

# Expose the default port
EXPOSE 8000
# Set the entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]