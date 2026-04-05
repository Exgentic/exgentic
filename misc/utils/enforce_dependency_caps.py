# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

#!/usr/bin/env python3
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


def _extract_dependency_lines(content: str) -> list[str]:
    """Extract lines that belong to dependency sections in pyproject.toml.

    Scopes extraction to [project.dependencies] and
    [project.optional-dependencies.*] sections only, so that version-like
    strings in other sections (e.g. build-system.requires) are ignored.
    """
    lines: list[str] = []
    in_dep_section = False
    in_dep_array = False

    for line in content.splitlines():
        stripped = line.strip()

        # Detect section headers
        if stripped.startswith("["):
            in_dep_section = stripped in ("[project]",) or stripped.startswith("[project.optional-dependencies")
            in_dep_array = False
            continue

        if not in_dep_section:
            continue

        # Inside a relevant section, look for dependency array starts
        if "dependencies" in stripped and "=" in stripped and "[" in stripped:
            in_dep_array = True
            continue
        # Also handle bare list continuation under optional-dependencies groups
        if stripped.startswith('"') and in_dep_section and not in_dep_array:
            # We're likely in an optional-dep group list
            in_dep_array = True

        if in_dep_array:
            if stripped == "]":
                in_dep_array = False
                continue
            lines.append(line)

    return lines


def check_dependency_caps(pyproject_path: Path) -> list[str]:
    """Check all dependencies in pyproject.toml for upper version bounds.

    Args:
        pyproject_path: Path to pyproject.toml file

    Returns:
        List of dependency lines that lack upper bounds (empty if all are capped)
    """
    content = pyproject_path.read_text()
    uncapped = []

    # Only check lines inside dependency sections
    dep_lines = _extract_dependency_lines(content)

    # Pattern to match dependency specifications (supports extras like [extra])
    # Matches: "package>=1.0.0" or "package[extra]>=1.0.0,!=1.2.3" but not "package>=1.0.0,<2"
    dep_pattern = re.compile(
        r'^\s*"([a-zA-Z0-9_-]+(?:\[[a-zA-Z0-9_,\s-]+\])?)([><=!,.\d\s]+)"',
    )

    for line in dep_lines:
        match = dep_pattern.match(line)
        if match:
            full_line = match.group(0).strip()
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
