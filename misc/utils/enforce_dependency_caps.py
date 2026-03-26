#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2024-present Exgentic Team
# SPDX-License-Identifier: MIT
"""Enforce that all direct dependencies in pyproject.toml have upper version bounds.

This script prevents supply chain attacks by ensuring no dependency can auto-upgrade
to an arbitrary future version. All dependencies must be capped at the next major
version (e.g., >=1.0.0,<2).

Exit codes:
    0: All dependencies have upper bounds
    1: One or more dependencies lack upper bounds
"""

import re
import sys
from pathlib import Path


def check_dependency_caps(pyproject_path: Path) -> list[str]:
    """Check all dependencies in pyproject.toml for upper version bounds.

    Args:
        pyproject_path: Path to pyproject.toml file

    Returns:
        List of dependency lines that lack upper bounds (empty if all are capped)
    """
    content = pyproject_path.read_text()
    uncapped = []

    # Pattern to match dependency specifications
    # Matches: "package>=1.0.0" or "package>=1.0.0,!=1.2.3" but not "package>=1.0.0,<2"
    dep_pattern = re.compile(r'^\s*"([a-zA-Z0-9_-]+)([><=!,.\d\s]+)"', re.MULTILINE)

    for match in dep_pattern.finditer(content):
        full_line = match.group(0).strip()
        package_name = match.group(1)
        version_spec = match.group(2)

        # Check if there's an upper bound (< or <=)
        if "<" not in version_spec:
            uncapped.append(full_line)

    return uncapped


def main() -> int:
    """Main entry point."""
    pyproject_path = Path("pyproject.toml")

    if not pyproject_path.exists():
        print("Error: pyproject.toml not found", file=sys.stderr)
        return 1

    uncapped = check_dependency_caps(pyproject_path)

    if uncapped:
        print("❌ Dependencies without upper version bounds found:", file=sys.stderr)
        print(file=sys.stderr)
        for dep in uncapped:
            print(f"  {dep}", file=sys.stderr)
        print(file=sys.stderr)
        print(
            "All dependencies must have upper bounds (e.g., >=1.0.0,<2) to limit supply chain attack exposure.",
            file=sys.stderr,
        )
        print("See SECURITY.md for the dependency management policy.", file=sys.stderr)
        return 1

    print("✅ All dependencies have upper version bounds")
    return 0


if __name__ == "__main__":
    sys.exit(main())
