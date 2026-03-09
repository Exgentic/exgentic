# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

import exgentic

from tests.api.fixtures.test_agent import TestAgent
from tests.api.fixtures.test_benchmark import TestBenchmark


def test_top_level_registry_class_imports_via_from_import():
    scope: dict[str, object] = {}
    exec("from exgentic import TestAgent, TestBenchmark", {}, scope)
    assert scope["TestAgent"] is TestAgent
    assert scope["TestBenchmark"] is TestBenchmark


def test_top_level_registry_class_imports_via_attribute_access():
    assert exgentic.TestAgent is TestAgent
    assert exgentic.TestBenchmark is TestBenchmark


def test_unknown_top_level_export_raises_attribute_error():
    try:
        _ = exgentic.NotARealExport
    except AttributeError as exc:
        assert "NotARealExport" in str(exc)
    else:
        raise AssertionError("Expected AttributeError for unknown top-level export.")
