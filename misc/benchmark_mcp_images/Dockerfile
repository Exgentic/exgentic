FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Build arguments
ARG BENCHMARK_NAME
ARG RELEASE_VERSION="main"

# Validate BENCHMARK_NAME is provided
RUN if [ -z "$BENCHMARK_NAME" ]; then \
        echo "ERROR: BENCHMARK_NAME build argument is required"; \
        echo "Usage: docker build --build-arg BENCHMARK_NAME=<benchmark> -t exgentic-mcp-<benchmark> ."; \
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
RUN uv pip install --system --no-cache datasets # to allow direct (and not venv) access to GSK8k and another simple benchmarks

# Set HOME to /app before installing benchmark so data goes to the right location
ENV HOME=/app

# Setup the benchmark - this will install to /app/.exgentic/
RUN exgentic install --benchmark ${BENCHMARK_NAME}

# Copy entrypoint script
WORKDIR /app
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Set environment variables
ENV BENCHMARK_NAME=${BENCHMARK_NAME} \
    HOST=0.0.0.0 \
    PORT=8000 \
    LOG_LEVEL=INFO \
    PRODUCTION_MODE=True \
    RELEASE_VERSION=${RELEASE_VERSION}

# Create non-root user with /app as home directory and set permissions
# Using UID 1000 to match Kubernetes securityContext runAsUser
RUN useradd -u 1000 -d /app exgentic && \
    chown -R 1000:0 /app && \
    chmod -R g+rwX /app
    
USER 1000

# Set HOME after switching user to ensure it's not overridden
ENV HOME=/app

# Expose the default port
EXPOSE 8000

# Set the entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]
