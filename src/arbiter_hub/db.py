"""Engine, session factory, and schema creation."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from .config import Settings, get_settings
from .models import Base

__all__ = ["SessionLocal", "build_engine", "create_all", "get_db", "reset_state", "session_scope"]

_engine: Engine | None = None
SessionLocal: sessionmaker[Session] | None = None


def build_engine(settings: Settings | None = None) -> Engine:
    """Create the engine, with the SQLite-specific care SQLite needs."""
    settings = settings or get_settings()
    url = settings.database_url
    kwargs: dict[str, object] = {"future": True, "pool_pre_ping": True}
    if url.startswith("sqlite"):
        # check_same_thread is required because FastAPI runs sync endpoints in a
        # threadpool, so a connection can legitimately move between threads.
        kwargs["connect_args"] = {"check_same_thread": False}
    engine = create_engine(url, **kwargs)

    if url.startswith("sqlite"):

        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(dbapi_connection, _record):
            cursor = dbapi_connection.cursor()
            # Foreign keys are off by default in SQLite, which would silently
            # defeat every ondelete="CASCADE" in models.py.
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.close()

    return engine


def _ensure() -> sessionmaker[Session]:
    global _engine, SessionLocal
    if SessionLocal is None:
        _engine = build_engine()
        SessionLocal = sessionmaker(bind=_engine, autoflush=False, expire_on_commit=False)
    return SessionLocal


def create_all(engine: Engine | None = None) -> None:
    """Create any missing tables.

    Fine for getting started and for a single-instance deployment. A schema that
    has to change under a running service wants a migration tool; this project
    does not pretend otherwise.
    """
    Base.metadata.create_all(bind=engine or _engine or build_engine())


def reset_state() -> None:
    """Drop the cached engine. Used by tests, which build their own."""
    global _engine, SessionLocal
    _engine = None
    SessionLocal = None


def get_db() -> Iterator[Session]:
    """FastAPI dependency."""
    factory = _ensure()
    session = factory()
    try:
        yield session
    finally:
        session.close()


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transaction for code outside a request, such as the CLI."""
    factory = _ensure()
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
