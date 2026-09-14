"""Settings loaded from the environment. See .env.example for the contract.

Every value the system can be tuned with lives here, validated once at startup.
The alternative - reading os.environ where it's needed - means a typo in a
variable name surfaces as a default silently taking effect, hours in.
"""

import os
from collections.abc import Mapping
from functools import lru_cache

from dotenv import load_dotenv
from pydantic import BaseModel, Field, SecretStr


class Settings(BaseModel):
    """Validated configuration for one run of the system."""

    anthropic_api_key: SecretStr

    router_model: str = "claude-haiku-4-5"
    agent_model: str = "claude-opus-5"

    router_confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    max_agent_turns: int = Field(default=6, gt=0)

    log_level: str = "INFO"

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Settings":
        """Build settings from a mapping, defaulting to the process environment.

        Passing `env` explicitly is what makes this testable without touching
        os.environ or the developer's real .env file.
        """
        if env is None:
            load_dotenv()  # no-op if .env is absent; never overrides real env vars
            env = os.environ

        provided = {
            field: env[name]
            for field, name in ((f, f.upper()) for f in cls.model_fields)
            if name in env
        }
        return cls(**provided)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """The process-wide settings, parsed once.

    Cached so config errors surface at the first call rather than on every
    request, and so .env is read once.
    """
    return Settings.from_env()
