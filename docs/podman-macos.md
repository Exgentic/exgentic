# Using Podman instead of Docker (macOS)

On macOS (especially Apple Silicon), you can use [Podman](https://podman.io/) as a drop-in replacement for Docker. This is useful if you prefer not to install Docker Desktop or need Rosetta-based x86_64 emulation for benchmarks like SWE-bench.

**Related docs:**
[Runners](./runners.md) · [CLI Reference](./cli-reference.md) · [docs/](./README.md)

---

## 1. Install Podman and the Docker CLI

```bash
brew install podman docker
```

This installs Podman as the container engine and the `docker` CLI client (without Docker Desktop). The `docker` CLI is needed because Exgentic's docker runner invokes it directly.

## 2. Create and start a Podman machine

```bash
podman machine init --rootful
podman machine set --rootful
podman machine start
```

The `--rootful` flag is required for Docker socket compatibility and for benchmarks that need privileged operations (e.g. Docker-in-Docker for SWE-bench).

## 3. Route the `docker` CLI through Podman

The `docker` runner in Exgentic calls the `docker` CLI. To make it use Podman's backend, point `DOCKER_HOST` to Podman's socket:

```bash
export DOCKER_HOST=unix://$(podman machine inspect --format '{{.ConnectionInfo.PodmanSocket.Path}}')
```

Add this to your `~/.zshrc` or `~/.bashrc` to make it permanent.

Alternatively, you can create a symlink:

```bash
sudo ln -sf $(podman machine inspect --format '{{.ConnectionInfo.PodmanSocket.Path}}') /var/run/docker.sock
```

## 4. Verify

```bash
docker info   # should show Podman as the backend
```

You can now use the `docker` runner as usual — Exgentic doesn't need any Podman-specific configuration.

---

## Apple Silicon and x86_64 benchmarks

SWE-bench containers are x86_64. On Apple Silicon, Podman uses Rosetta 2 for emulation, which is significantly faster than QEMU. Make sure Rosetta is installed:

```bash
softwareupdate --install-rosetta --agree-to-license
```

Rosetta is enabled by default in recent Podman versions. You can verify by running an x86_64 container:

```bash
docker run --platform linux/amd64 --rm alpine uname -m
# should output: x86_64
```

---

## See also

- [Runners](./runners.md) — all runner types and configuration
- [CLI Reference](./cli-reference.md) — every command and flag
- [docs/](./README.md) — documentation index
