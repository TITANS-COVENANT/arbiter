"""Persistence for runs, verdicts and gate history."""

from .db import SCHEMA_VERSION, Store

__all__ = ["SCHEMA_VERSION", "Store"]
