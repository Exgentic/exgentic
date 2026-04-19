# Using Podman instead of Docker (macOS)

Use [Podman](https://podman.io/) as a drop-in replacement for Docker Desktop on macOS. After setup, all `docker` commands are routed through Podman — no changes needed in Exgentic.

**Related docs:**
[Runners](./runners.md) · [CLI Reference](./cli-reference.md) · [docs/](./README.md)

---

## 1. Install Podman and the Docker CLI

```bash
brew install podman docker
```

`podman` is the container engine. `docker` is the CLI client that Exgentic invokes — installing it via Homebrew gives you the CLI without Docker Desktop.

## 2. Set up Rosetta (Apple Silicon only)

SWE-bench and other benchmarks use x86_64 containers. On Apple Silicon, Podman uses Rosetta 2 for fast emulation:

```bash
softwareupdate --install-rosetta --agree-to-license
```

## 3. Create and start a Podman machine

```bash
podman machine init --rootful
podman machine start
```

The `--rootful` flag is needed for Docker socket compatibility and Docker-in-Docker (used by SWE-bench).

## 4. Point `docker` CLI at Podman

```bash
export DOCKER_HOST=unix://$(podman machine inspect --format '{{.ConnectionInfo.PodmanSocket.Path}}')
```

Add this line to your `~/.zshrc` (or `~/.bashrc`) to make it permanent.

## 5. Verify

```bash
docker info
```

You should see `podman` in the output. If this works, you're done — Exgentic's `docker` runner will work as-is.

To confirm x86_64 emulation (Apple Silicon):

```bash
docker run --platform linux/amd64 --rm alpine uname -m
# expected output: x86_64
```

---

## Troubleshooting: x86_64 emulation segfaults on Apple Silicon

x86_64 containers may segfault when importing heavy Python packages (pyarrow, pandas). This happens because Podman silently falls back to QEMU even when Rosetta is configured.

**Check if Rosetta is active:**

```bash
podman machine ssh "cat /proc/sys/fs/binfmt_misc/rosetta 2>/dev/null || echo NOT ACTIVE"
```

- If it says `enabled` with interpreter `/mnt/rosetta` — you're fine.
- If it says `NOT ACTIVE` — Rosetta is not running, apply the fix below.

**Fix:**

```bash
podman machine ssh "sudo touch /etc/containers/enable-rosetta"
podman machine stop
podman machine start
```

**Verify:**

```bash
# Should say "enabled" with interpreter /mnt/rosetta
podman machine ssh "cat /proc/sys/fs/binfmt_misc/rosetta"

# Quick test — this segfaults under QEMU, works under Rosetta
docker run --rm python:3.12-slim python3 -c "import struct; print('OK')"
```

**Why:** Podman 5.7+ has a bug where `Rosetta: true` in machine config doesn't actually enable Rosetta inside the VM. The trigger file `/etc/containers/enable-rosetta` is missing, so the activation service skips. Creating it and restarting fixes it. See [containers/podman#28181](https://github.com/containers/podman/issues/28181).

---

## See also

- [Runners](./runners.md) — all runner types and configuration
- [CLI Reference](./cli-reference.md) — every command and flag
- [docs/](./README.md) — documentation index
