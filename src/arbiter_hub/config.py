"""Settings, read from the environment.

Everything secret lives here and nowhere else. The rule the rest of the package
relies on is that a secret is read once, at startup, and never logged, never
rendered into a template, and never returned from an endpoint.
"""

from __future__ import annotations

import secrets
from functools import lru_cache
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ["Settings", "get_settings"]


class Settings(BaseSettings):
    """Configuration for one deployment."""

    model_config = SettingsConfigDict(
        env_prefix="ARBITER_HUB_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # -- identity of this deployment ---------------------------------------
    app_name: str = "arbiter hub"
    base_url: str = "http://localhost:8000"
    environment: Literal["development", "production"] = "development"

    # -- storage ------------------------------------------------------------
    database_url: str = "sqlite:///./arbiter_hub.sqlite"

    # -- session signing ----------------------------------------------------
    # Generated per process when unset, which is fine for a laptop and wrong for
    # a deployment: every restart would sign everyone out. Production refuses to
    # start without an explicit one, see the validator below.
    secret_key: str = Field(default_factory=lambda: secrets.token_urlsafe(48))
    session_cookie: str = "arbiter_hub_session"
    session_max_age_seconds: int = 60 * 60 * 24 * 14

    # -- GitHub OAuth -------------------------------------------------------
    # Register at https://github.com/settings/developers with the callback set
    # to {base_url}/auth/github/callback.
    github_client_id: str = ""
    github_client_secret: str = ""

    # -- development escape hatch ------------------------------------------
    # Signs you in as a fake user without GitHub, so the app can be run and
    # tested without registering an OAuth application. It is an authentication
    # bypass, so it is off by default and cannot be switched on in production.
    dev_auth: bool = False

    # -- limits -------------------------------------------------------------
    max_payload_bytes: int = 4 * 1024 * 1024
    ingest_rate_per_minute: int = 60

    @property
    def github_configured(self) -> bool:
        return bool(self.github_client_id and self.github_client_secret)

    @property
    def github_callback_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/auth/github/callback"

    @property
    def secure_cookies(self) -> bool:
        return self.environment == "production" or self.base_url.startswith("https://")

    @model_validator(mode="after")
    def _production_is_actually_safe(self) -> Settings:
        """Refuse to start a production deployment in an unsafe shape.

        Every one of these has been a real incident somewhere. Failing at
        startup is the cheapest possible moment to find out.
        """
        if self.environment != "production":
            return self
        problems = []
        if self.dev_auth:
            problems.append(
                "ARBITER_HUB_DEV_AUTH is on, which lets anyone sign in as anyone"
            )
        if not self.github_configured:
            problems.append("GitHub OAuth is not configured, so nobody can sign in")
        if "ARBITER_HUB_SECRET_KEY" not in _env_keys():
            problems.append(
                "ARBITER_HUB_SECRET_KEY is unset, so sessions would be signed with a "
                "key that changes on every restart"
            )
        if self.base_url.startswith("http://"):
            problems.append(f"base_url is not https ({self.base_url})")
        if problems:
            raise ValueError(
                "refusing to start in production:\n  - " + "\n  - ".join(problems)
            )
        return self


def _env_keys() -> set[str]:
    import os

    return set(os.environ)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
