# How to contribute to Exgentic

Thank you for your interest in contributing!

## Development Setup

```bash
# Install dependencies using the pinned lock file — never plain `uv sync`
uv sync --frozen --extra dev --extra analysis

# To intentionally upgrade a specific package:
uv lock --upgrade-package <package-name>
# Review the uv.lock diff carefully before committing
```

> **Security note:** Always use `uv sync --frozen` locally. Running plain `uv sync` may silently
> upgrade packages and introduce untested or malicious versions. Dependency upgrades should be
> explicit, reviewed, and go through a PR.

## How to Contribute

1. Fork the [repository](https://github.com/exgentic/exgentic).
2. Create a new branch for your changes.
3. Sign your commits using the `-s` flag (see [Legal](#legal))
4. Submit a pull request to the `main` branch with a clear title and description.
Reference any issues fixed, for example `Fixes #1234`.
Ensure your PR title follows [semantic commit conventions](https://www.conventionalcommits.org/).
5. A maintainer will review your PR and may request changes.

## Legal

### License

This project is licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE).

Each source code file must include the following SPDX headers at the top of the file:

**For Python files:**
```python
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025, The Exgentic organization and its contributors.

"""Module docstring here."""
import ...
```

**For other file types:** Use the appropriate comment syntax for that language.

### Developer Certificate of Origin (DCO)

We require all commits to be **signed off** to indicate agreement with the [DCO](DCO.txt).

By signing off a commit, you certify:

> “I have the right to submit this contribution under the Apache License, Version 2.0 (or the open source license indicated in the file), and understand this project and my contribution are public.”

### How to sign off your commits

The easiest way is to use the `-s` flag when committing:

```bash
git commit -s -m "Fix: Correct spelling in README"
```

This uses your Git configuration. Make sure your name and email are set:
```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

Alternatively you can manually sign your commit by adding this line to the commit message:
```
Signed-off-by: Your Name <your.email@example.com>
```

## Development Environment Setup

For detailed instructions on setting up your local development environment, see [DEVELOPMENT.md](./DEVELOPMENT.md).
