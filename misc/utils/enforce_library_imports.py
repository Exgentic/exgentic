# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def check_imports(file_path):
    success = True
    with open(file_path, encoding="utf-8") as file:
        for lineno, line in enumerate(file, start=1):
            if "from src import" in line or "from src." in line or "import src" in line:
                logger.error(f"Non library import: {file_path}:{lineno}: {line.strip()[:30]}...")
                success = False
    return success


def main():
    success = True
    for file_path in sys.argv[1:]:
        if not check_imports(file_path):
            success = False

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
