# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Thin import shim around Gorilla's BFCL package."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

EXGENTIC_BFCL_MODEL = "exgentic-proxy-fc"


@lru_cache(maxsize=1)
def load_bfcl_symbols() -> dict[str, Any]:
    from bfcl_eval.constants.enums import Language, ModelStyle
    from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING, ModelConfig
    from bfcl_eval.constants.type_mappings import GORILLA_TO_OPENAPI
    from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker
    from bfcl_eval.eval_checker.eval_runner import (
        ast_file_runner,
        multi_turn_runner,
        relevance_file_runner,
    )
    from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import (
        multi_turn_checker,
        multi_turn_irrelevance_checker,
    )
    from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (
        execute_multi_turn_func_call,
    )
    from bfcl_eval.model_handler.api_inference.openai_completion import OpenAICompletionsHandler
    from bfcl_eval.model_handler.base_handler import BaseHandler
    from bfcl_eval.model_handler.utils import convert_to_tool
    from bfcl_eval.utils import load_dataset_entry, load_ground_truth_entry

    if EXGENTIC_BFCL_MODEL not in MODEL_CONFIG_MAPPING:
        MODEL_CONFIG_MAPPING[EXGENTIC_BFCL_MODEL] = ModelConfig(
            model_name=EXGENTIC_BFCL_MODEL,
            display_name="Exgentic Proxy (FC)",
            url="https://github.com/Exgentic/exgentic",
            org="Exgentic",
            license="Apache 2.0",
            model_handler=OpenAICompletionsHandler,
            input_price=None,
            output_price=None,
            is_fc_model=True,
            underscore_to_dot=True,
        )

    return {
        "BaseHandler": BaseHandler,
        "Language": Language,
        "ModelStyle": ModelStyle,
        "OpenAICompletionsHandler": OpenAICompletionsHandler,
        "GORILLA_TO_OPENAPI": GORILLA_TO_OPENAPI,
        "convert_to_tool": convert_to_tool,
        "load_dataset_entry": load_dataset_entry,
        "load_ground_truth_entry": load_ground_truth_entry,
        "ast_checker": ast_checker,
        "multi_turn_checker": multi_turn_checker,
        "multi_turn_irrelevance_checker": multi_turn_irrelevance_checker,
        "execute_multi_turn_func_call": execute_multi_turn_func_call,
        "ast_file_runner": ast_file_runner,
        "multi_turn_runner": multi_turn_runner,
        "relevance_file_runner": relevance_file_runner,
        "proxy_model_name": EXGENTIC_BFCL_MODEL,
    }
