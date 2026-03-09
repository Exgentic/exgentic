# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

import os
from pathlib import Path

if "TAU2_DATA_DIR" not in os.environ:
    os.environ["TAU2_DATA_DIR"] = str(Path(__file__).resolve().parent / "installation" / "tau2-bench" / "data")
