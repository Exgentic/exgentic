# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

# Make package importable
sys.path.insert(0, os.path.abspath("src"))
from exgentic.utils.cost import _load_custom_pricing, litellm_tokens_cost  # noqa: E402


@pytest.fixture
def temp_home(tmp_path, monkeypatch):
    """Create a temporary home directory for testing."""
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path


@pytest.fixture(autouse=True)
def reset_pricing_loaded():
    """Reset the _pricing_loaded flag before each test."""
    import exgentic.utils.cost as cost_module

    cost_module._pricing_loaded = False
    yield
    cost_module._pricing_loaded = False


def test_load_custom_pricing_file_not_exists(temp_home):
    """Test that loading works when pricing file doesn't exist."""
    _load_custom_pricing()
    # Should not raise any errors


def test_load_custom_pricing_valid_file(temp_home):
    """Test loading a valid custom pricing file."""
    pricing_dir = temp_home / ".exgentic"
    pricing_dir.mkdir()
    pricing_file = pricing_dir / "pricing.json"

    custom_pricing = {
        "my-custom-model": {
            "input_cost_per_token": 0.0001,
            "output_cost_per_token": 0.0002,
            "litellm_provider": "openai",
        }
    }

    pricing_file.write_text(json.dumps(custom_pricing))

    # Mock litellm.register_model to verify it's called
    with patch("litellm.register_model") as mock_register:
        _load_custom_pricing()
        mock_register.assert_called_once_with(custom_pricing)


def test_load_custom_pricing_invalid_json(temp_home, caplog):
    """Test handling of invalid JSON in pricing file."""
    pricing_dir = temp_home / ".exgentic"
    pricing_dir.mkdir()
    pricing_file = pricing_dir / "pricing.json"

    pricing_file.write_text("{ invalid json }")

    _load_custom_pricing()
    assert "Failed to parse custom pricing file" in caplog.text


def test_load_custom_pricing_not_dict(temp_home, caplog):
    """Test handling when pricing file is not a JSON object."""
    pricing_dir = temp_home / ".exgentic"
    pricing_dir.mkdir()
    pricing_file = pricing_dir / "pricing.json"

    pricing_file.write_text(json.dumps(["not", "a", "dict"]))

    _load_custom_pricing()
    assert "is not a JSON object" in caplog.text


def test_load_custom_pricing_invalid_model_entry(temp_home, caplog):
    """Test handling when a model entry is not a dict."""
    pricing_dir = temp_home / ".exgentic"
    pricing_dir.mkdir()
    pricing_file = pricing_dir / "pricing.json"

    custom_pricing = {
        "valid-model": {
            "input_cost_per_token": 0.0001,
            "output_cost_per_token": 0.0002,
        },
        "invalid-model": "not a dict",
    }

    pricing_file.write_text(json.dumps(custom_pricing))

    with patch("litellm.register_model") as mock_register:
        _load_custom_pricing()
        # Should only register the valid model
        mock_register.assert_called_once()
        assert "invalid-model" in caplog.text
        assert "is not a dict" in caplog.text


def test_load_custom_pricing_called_once():
    """Test that custom pricing is only loaded once."""
    with patch("exgentic.utils.cost.Path.home") as mock_home:
        mock_home.return_value = Path(tempfile.mkdtemp())
        pricing_dir = mock_home.return_value / ".exgentic"
        pricing_dir.mkdir()
        pricing_file = pricing_dir / "pricing.json"
        pricing_file.write_text(json.dumps({}))

        with patch("litellm.register_model") as mock_register:
            _load_custom_pricing()
            _load_custom_pricing()
            _load_custom_pricing()
            # Should only be called once despite multiple invocations
            assert mock_register.call_count <= 1


def test_litellm_tokens_cost_loads_custom_pricing(temp_home):
    """Test that litellm_tokens_cost calls _load_custom_pricing."""
    pricing_dir = temp_home / ".exgentic"
    pricing_dir.mkdir()
    pricing_file = pricing_dir / "pricing.json"

    # Create a custom model with known pricing
    custom_pricing = {
        "test-custom-model": {
            "input_cost_per_token": 0.0001,
            "output_cost_per_token": 0.0002,
            "litellm_provider": "openai",
        }
    }

    pricing_file.write_text(json.dumps(custom_pricing))

    # This should load custom pricing and then try to get cost
    # It will fail because litellm won't actually have the model registered in test,
    # but we can verify the loading was attempted
    with patch("exgentic.utils.cost._load_custom_pricing") as mock_load:
        try:
            litellm_tokens_cost(100, 100, "test-custom-model")
        except ValueError:
            pass  # Expected to fail in test environment
        mock_load.assert_called_once()


def test_error_message_mentions_pricing_json():
    """Test that the error message mentions the pricing.json file."""
    with pytest.raises(ValueError) as exc_info:
        litellm_tokens_cost(100, 100, "totally-nonexistent-model-xyz")

    error_message = str(exc_info.value)
    assert "~/.exgentic/pricing.json" in error_message
    assert "name_map" in error_message
