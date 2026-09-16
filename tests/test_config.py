"""Tests for settings loading.

Each test passes an explicit env mapping, so none of them read the developer's
real .env or mutate os.environ.
"""

import pytest
from pydantic import ValidationError

from customer_service.config import Settings

MINIMAL = {"GEMINI_API_KEY": "test-gemini-key"}


def test_defaults_apply_when_only_the_key_is_set():
    settings = Settings.from_env(MINIMAL)
    assert settings.router_model == "gemini-3.5-flash-lite"
    assert settings.agent_model == "gemini-3.8-flash"
    assert settings.router_confidence_threshold == 0.7
    assert settings.max_agent_turns == 6


def test_env_values_override_defaults_and_are_coerced():
    settings = Settings.from_env(
        MINIMAL | {"ROUTER_CONFIDENCE_THRESHOLD": "0.9", "MAX_AGENT_TURNS": "3"}
    )
    assert settings.router_confidence_threshold == 0.9
    assert settings.max_agent_turns == 3


def test_missing_api_key_fails_at_startup():
    with pytest.raises(ValidationError):
        Settings.from_env({})


def test_threshold_outside_zero_to_one_is_rejected():
    with pytest.raises(ValidationError):
        Settings.from_env(MINIMAL | {"ROUTER_CONFIDENCE_THRESHOLD": "1.5"})


def test_zero_agent_turns_is_rejected():
    """A budget of zero turns would escalate every conversation immediately."""
    with pytest.raises(ValidationError):
        Settings.from_env(MINIMAL | {"MAX_AGENT_TURNS": "0"})


def test_api_key_is_not_exposed_by_repr():
    """SecretStr keeps the key out of logs, tracebacks and error messages."""
    settings = Settings.from_env(MINIMAL)
    assert "test-gemini-key" not in repr(settings)
    assert settings.gemini_api_key.get_secret_value() == "test-gemini-key"
