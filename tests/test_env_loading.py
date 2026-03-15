# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch


def test_dotenv_loads_settings_and_env_vars() -> None:
    with TemporaryDirectory() as tmpdir:
        env_path = Path(tmpdir) / ".env"
        env_path.write_text(
            "EXGENTIC_LOG_LEVEL=DEBUG\nWATSONX_API_KEY=abc123\n"  # pragma: allowlist secret
        )

        with patch.dict(os.environ, {"EXGENTIC_DOTENV_PATH": str(env_path)}, clear=False):
            os.environ.pop("EXGENTIC_LOG_LEVEL", None)
            os.environ.pop("WATSONX_API_KEY", None)

            sys.modules.pop("exgentic.utils.settings", None)
            settings_module = importlib.import_module("exgentic.utils.settings")

            settings_module.get_settings.cache_clear()
            settings = settings_module.get_settings()

            assert settings.log_level == "DEBUG"
            assert os.environ.get("WATSONX_API_KEY") == "abc123"
            assert settings.dotenv_path == str(env_path)
