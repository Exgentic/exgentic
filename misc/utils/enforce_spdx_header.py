# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

#!/usr/bin/env python3
"""Ensure SPDX+copyright header exists at top of Python files."""

from __future__ import annotations

import re
import sys
from pathlib import Path

HEADER_LINES = [
    "# SPDX-License-Identifier: Apache-2.0",
    "# Copyright (C) 2026, The Exgentic organization and its contributors.",
]
HEADER_TEXT = "\n".join(HEADER_LINES) + "\n\n"

COPYRIGHT_RE = re.compile(r"^# Copyright \(C\) (?P<year>\d{4}), The Exgentic organization and its contributors\.$")

SKIP_DIRS = {
    ".git",
    ".venv",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
}


def should_skip(path: Path) -> bool:
    return bool(set(path.parts) & SKIP_DIRS)


def update_file(path: Path) -> bool:
    raw = path.read_bytes()
    original = raw.decode("utf-8", errors="surrogateescape")
    if not original:
        path.write_text(HEADER_TEXT, encoding="utf-8")
        return True

    lines = original.splitlines()
    if len(lines) >= 2 and lines[0] == HEADER_LINES[0] and COPYRIGHT_RE.match(lines[1]):
        if lines[1] != HEADER_LINES[1]:
            lines[1] = HEADER_LINES[1]
            updated = "\n".join(lines) + ("\n" if original.endswith("\n") else "")
            path.write_text(updated, encoding="utf-8", errors="surrogateescape")
            return True
        return False

    updated = HEADER_TEXT + original
    path.write_text(updated, encoding="utf-8", errors="surrogateescape")
    return True


def main(argv: list[str]) -> int:
    changed = False
    for filename in argv:
        path = Path(filename)
        if not path.is_file() or should_skip(path):
            continue
        if update_file(path):
            changed = True
    if changed:
        print("SPDX headers updated. Re-run pre-commit.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
