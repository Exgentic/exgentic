# Exgentic A2A Agent Wrapper

A Docker-based wrapper that runs [Exgentic](https://github.com/Exgentic/exgentic) agents using the A2A (Agent-to-Agent) protocol. This wrapper clones the Exgentic repository, installs a specific agent at build time, and exposes it via the A2A interface.  These agents are used for evaluation against the [Exgentic benchmarks mcp servers](mcp/README.md).  The test harness used to evaluate the agents against the benchmarks is found in the [workload-harness repo](https://github.com/kagenti/workload-harness/blob/main/exgentic_a2a_runner/README.md).

## Overview

This wrapper provides access to Exgentic agents through the A2A protocol. Each Docker image is built with a specific agent pre-installed, making it easy to deploy and run different agents in isolated environments.

## Features

- **Agent-Specific Images**: Each Docker image contains a single agent
- **Build-Time Setup**: Agents are installed during image build for faster startup
- **Flexible Configuration**: HOST and PORT configurable at runtime via environment variables
- **Consistent Interface**: Follows A2A protocol standards
- **Security**: Runs as non-root user (UID 1001)
- **Production Ready**: Includes proper error handling and logging
- **Build Script**: Convenient build.sh script with docker/podman auto-detection

## Prerequisites

- Docker or Podman installed on your system
- Internet connection (for cloning repository and downloading agent data)
- Sufficient disk space (agent data can be large)

## Quick Start

### Using the Build Script (Recommended)

The easiest way to build an agent image:

```bash
cd a2a/exgentic_agent

# Build an agent image
./build.sh tool_calling
```

The script will:
- Auto-detect docker or podman
- Build the image with proper tagging
- Provide colored output and progress information


### Run the Agent

```bash
docker run -p 8000:8000 \
  -e MCP_URL=http://host.containers.internal:8000/mcp \
  -e EXGENTIC_SET_AGENT_MODEL='openai/gpt-4o' \
  -e OPENAI_API_KEY \
  -e OPENAI_API_BASE \
  exgentic-a2a-tool_calling:latest

```

The agent will start on `http://0.0.0.0:8000`

### Test the Agent

```bash
# Check if agent is running
curl http://localhost:8000/health

# List available capabilities
curl http://localhost:8000/capabilities
```

## Build Script Usage

The `build.sh` script provides a convenient way to build agent images:

```bash
# Basic usage
./build.sh AGENT_NAME [--tag TAG] [--use-cache] [--push]

# Examples
./build.sh tool_calling                    # Build without cache (default)
./build.sh tool_calling --tag v1.0.0       # Build v1.0.0 without cache
./build.sh tool_calling --tag dev          # Build with 'dev' tag
./build.sh tool_calling --use-cache        # Build with cache enabled
./build.sh tool_calling --push             # Build and push to GitHub Container Registry
./build.sh tool_calling --tag v1.0.0 --push  # Build v1.0.0 and push to GHCR
./build.sh tool_calling --tag v1.0.0 --use-cache --push  # Build v1.0.0 with cache and push

# Get help
./build.sh --help
```

**Features:**
- Automatically detects docker or podman
- Colored output for better readability
- Build summary with success/failure counts
- Builds without cache by default for consistency
- Optional cache usage with `--use-cache` flag
- Optional push to GitHub Container Registry with `--push` flag

### Pushing to GitHub Container Registry

When using the `--push` flag, you must set these environment variables:

```bash
export GITHUB_USERNAME="your-github-username"
export GITHUB_TOKEN="ghp_xxxxxxxxxxxx"
```

To create a GitHub Personal Access Token:
1. Go to: https://github.com/settings/tokens
2. Click "Generate new token (classic)"
3. Select the `write:packages` scope
4. Generate and copy the token

**Example with push:**
```bash
export GITHUB_USERNAME="your-username"
export GITHUB_TOKEN="ghp_xxxxxxxxxxxx"
./build.sh tool_calling --push
```

The image will be pushed to: `ghcr.io/exgentic/exgentic-a2a-tool_calling:latest`

View the package at: https://github.com/orgs/Exgentic/packages/container/package/exgentic-a2a-tool_calling

## Configuration

### Build Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `AGENT_NAME` | Yes | - | The agent to install (e.g., tool_calling, agent1, agent2) |
| `RELEASE_VERSION` | No | main | Version tag for tracking |

### Runtime Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `HOST` | No | 0.0.0.0 | Server host address |
| `PORT` | No | 8000 | Server port |
| `LOG_LEVEL` | No | INFO | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `AGENT_NAME` | No | (from build) | Agent name (set during build) |
| `EXGENTIC_SET_*` | No | - | Runtime configuration parameters (see below) |

### Runtime Configuration with --set Parameters

You can pass runtime configuration parameters to the `exgentic a2a` command using environment variables with the `EXGENTIC_SET_` prefix. These will be converted to `--set` arguments.

**Format**: `EXGENTIC_SET_<CATEGORY>_<PARAMETER>=<value>`
- The `<CATEGORY>` will be separated from `<PARAMETER>` with a dot
- The rest of the underscores in `<PARAMETER>` remain as underscores
- Everything is converted to lowercase
- Example: `EXGENTIC_SET_AGENT_MODEL` → `--set agent.model`

**Common Parameters**:
- `EXGENTIC_SET_AGENT_MODEL` - Set the agent model (e.g., `openai/gpt-4o`)
- `EXGENTIC_SET_AGENT_MAX_STEPS` - Set maximum steps
- `EXGENTIC_SET_AGENT_TEMPERATURE` - Set temperature
- `EXGENTIC_SET_AGENT_TIMEOUT` - Set timeout
- Any other agent-specific configuration parameter

### Custom Configuration Examples


**Set multiple runtime parameters:**
```bash
docker run -p 8000:8000 \
  -e EXGENTIC_SET_AGENT_MODEL='openai/gpt-4o' \
  -e EXGENTIC_SET_AGENT_MAX_STEPS='50' \
  -e EXGENTIC_SET_AGENT_TEMPERATURE='0.7' \
  exgentic-a2a-tool_calling:latest
```

### API Credentials

When using external models, you need to provide API credentials as environment variables:

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes (for OpenAI models) | Your OpenAI API key |
| `OPENAI_API_BASE` | No | Custom API base URL (if using a proxy or alternative endpoint) |

**Example with OpenAI credentials:**
```bash
docker run -p 8000:8000 \
  -e OPENAI_API_KEY='your-api-key-here' \
  -e EXGENTIC_SET_AGENT_MODEL='openai/gpt-4o' \
  exgentic-a2a-tool_calling:latest
```

**Example with custom API base:**
```bash
docker run -p 8000:8000 \
  -e OPENAI_API_KEY='your-api-key-here' \
  -e OPENAI_API_BASE='https://custom-endpoint.example.com/v1' \
  -e EXGENTIC_SET_AGENT_MODEL='openai/gpt-4o' \
  exgentic-a2a-tool_calling:latest
```

## Build Process

```
1. Clone Exgentic repository (HTTPS)
   ↓
2. Checkout feature/mcp-command branch
   ↓
3. Install Exgentic and dependencies
   ↓
4. Run: exgentic install --agent $AGENT_NAME
   ↓
5. Configure entrypoint and permissions
   ↓
6. Create image: exgentic-a2a-{agent}:latest
```

## Runtime Process

```
1. Container starts with entrypoint.sh
   ↓
2. Read environment variables (HOST, PORT)
   ↓
3. Execute: exgentic a2a --agent $AGENT_NAME --host $HOST --port $PORT
   ↓
4. A2A agent listens on configured HOST:PORT
```

## Image Naming Convention

Images follow the pattern: `exgentic-a2a-{agent}:latest`

Examples:
- `exgentic-a2a-tool_calling:latest`
- `exgentic-a2a-agent1:latest`
- `exgentic-a2a-agent2:latest`


## Repository Information

- **Source**: https://github.com/Exgentic/exgentic.git
- **Branch**: feature/mcp-command
- **Protocol**: HTTPS (public repository)

## A2A Protocol

This wrapper implements the Agent-to-Agent (A2A) protocol, which allows agents to communicate and collaborate programmatically.

### Key Features

- **Agent Discovery**: Agents expose their capabilities via A2A
- **Type Safety**: Strong typing for parameters and returns
- **Documentation**: Built-in documentation for each capability
- **Error Handling**: Standardized error responses
- **Transport**: HTTP transport with streamable support

## Security Considerations

- Runs as non-root user (UID 1001)
- No SSH keys or secrets in image
- Public repository access only
- Minimal attack surface
- Production-ready configuration

## Performance Notes

- **Build Time**: 5-15 minutes depending on agent size
- **Image Size**: Varies by agent (typically 1-5 GB)
- **Startup Time**: Fast (agent already installed)
- **Memory**: Depends on agent requirements

## Files in This Directory

- `Dockerfile` - Multi-stage build configuration
- `entrypoint.sh` - Container startup script
- `build.sh` - Convenient build script with auto-detection
- `.dockerignore` - Files to exclude from build context
- `.env.example` - Basic environment variable template
- `.env.advanced` - Advanced configuration example
- `README.md` - This file

## Contributing

When adding support for new agents:

1. Verify the agent exists in the Exgentic repository
2. Test the build process using `./build.sh agent_name`
3. Document any special requirements
4. Update this README with examples

## License

See the repository's LICENSE file for details.

## Support

For issues related to:
- **This A2A wrapper**: Open an issue in the agent-examples repository
- **Exgentic agents**: Refer to the Exgentic repository
- **A2A protocol**: See the Agent-to-Agent protocol documentation

## Related Resources

- [Exgentic Repository](https://github.com/Exgentic/exgentic)
- [Agent-to-Agent Protocol](https://github.com/Exgentic/exgentic)
- [Docker Documentation](https://docs.docker.com/)
- [Podman Documentation](https://podman.io/)
- [Other A2A Agents](../README.md)