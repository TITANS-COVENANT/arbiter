"""Tokens, slugs, and the small security primitives.

Two rules the rest of the package depends on:

An API token is stored as a SHA-256 hash and shown to the user exactly once.
There is no "reveal token" button anywhere, because a dashboard that can show
you your own token can show it to anyone who gets hold of your session.

Every comparison of a secret uses :func:`hmac.compare_digest`. The tokens here
are 256 bits of randomness so a timing attack is not the realistic threat, but
the habit is cheap and the alternative is remembering which comparisons matter.
"""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets

__all__ = [
    "TOKEN_PREFIX",
    "generate_token",
    "hash_token",
    "slugify",
    "token_prefix",
    "verify_token",
]

TOKEN_PREFIX = "arb_"
_SLUG_STRIP = re.compile(r"[^a-z0-9]+")
_RESERVED_SLUGS = {
    "api", "auth", "login", "logout", "static", "new", "settings", "admin",
    "dev", "health", "docs", "openapi", "p", "orgs", "about", "help",
}


def generate_token() -> str:
    """A fresh CI token. 32 bytes of urandom, url-safe, prefixed for greppability.

    The prefix is not decoration: it lets secret scanners and log filters
    recognise one of these on sight, and it makes an accidentally committed
    token findable.
    """
    return TOKEN_PREFIX + secrets.token_urlsafe(32)


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def verify_token(token: str, expected_hash: str) -> bool:
    return hmac.compare_digest(hash_token(token), expected_hash)


def token_prefix(token: str) -> str:
    """The visible fragment shown in the UI so a token can be told apart.

    Short enough to be useless on its own: with 256 bits of entropy behind it,
    eight characters identify which token without meaningfully narrowing a
    guess.
    """
    return token[: len(TOKEN_PREFIX) + 8]


def slugify(value: str, fallback: str = "project") -> str:
    """URL-safe slug, rejecting the paths the router already uses."""
    slug = _SLUG_STRIP.sub("-", value.strip().lower()).strip("-")
    slug = slug[:60].strip("-")
    if not slug or slug in _RESERVED_SLUGS:
        slug = f"{slug or fallback}-{secrets.token_hex(3)}"
    return slug
