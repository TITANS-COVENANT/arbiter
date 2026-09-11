"""Request dependencies: who is asking, and are they allowed.

Access control lives here and only here. Every route that touches a project
resolves it through :func:`require_project`, which starts from the signed-in
user's memberships rather than from the URL. Getting that backwards, by loading
the project first and checking ownership afterwards, is how one forgotten check
turns into cross-tenant data access.
"""

from __future__ import annotations

import hmac
import secrets
import time
from collections import deque

from fastapi import Depends, Form, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from .config import Settings, get_settings
from .db import get_db
from .models import ApiToken, Membership, Org, Project, User, utcnow
from .security import hash_token

__all__ = [
    "RateLimiter",
    "csrf_token",
    "current_user",
    "require_csrf",
    "require_project",
    "require_token_project",
    "require_user",
    "user_orgs",
]

_CSRF_KEY = "csrf"


def current_user(request: Request, db: Session = Depends(get_db)) -> User | None:
    """The signed-in user, or None."""
    user_id = request.session.get("user_id")
    if not user_id:
        return None
    return db.get(User, int(user_id))


def require_user(user: User | None = Depends(current_user)) -> User:
    """Signed-in user, or a redirect to the login page."""
    if user is None:
        raise HTTPException(
            status.HTTP_303_SEE_OTHER, "sign in first", headers={"Location": "/login"}
        )
    return user


def user_orgs(db: Session, user: User) -> list[Org]:
    return list(
        db.scalars(
            select(Org).join(Membership).where(Membership.user_id == user.id).order_by(Org.name)
        )
    )


def require_project(
    org_slug: str,
    project_slug: str,
    user: User = Depends(require_user),
    db: Session = Depends(get_db),
) -> Project:
    """Resolve a project the user is actually a member of.

    The join to Membership is the access check. A project that exists but
    belongs to someone else is a 404 rather than a 403, so the URL space does
    not confirm which project names are taken in other orgs.
    """
    project = db.scalar(
        select(Project)
        .join(Org, Project.org_id == Org.id)
        .join(Membership, Membership.org_id == Org.id)
        .where(
            Membership.user_id == user.id,
            Org.slug == org_slug,
            Project.slug == project_slug,
        )
    )
    if project is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "project not found")
    return project


class RateLimiter:
    """Fixed-window limiter, in memory.

    Honest about its limits: this is per process, so a deployment behind several
    workers multiplies the effective allowance, and a restart forgets
    everything. It is here to stop one misconfigured CI loop from filling the
    database, not to resist a determined attacker. Anything stronger belongs in
    front of the app.
    """

    def __init__(self, per_minute: int) -> None:
        self.per_minute = per_minute
        self._hits: dict[str, deque[float]] = {}

    def check(self, key: str, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        window = self._hits.setdefault(key, deque())
        cutoff = now - 60.0
        while window and window[0] < cutoff:
            window.popleft()
        if len(window) >= self.per_minute:
            return False
        window.append(now)
        return True

    def reset(self) -> None:
        self._hits.clear()


_ingest_limiter: RateLimiter | None = None


def get_ingest_limiter(settings: Settings = Depends(get_settings)) -> RateLimiter:
    global _ingest_limiter
    if _ingest_limiter is None:
        _ingest_limiter = RateLimiter(settings.ingest_rate_per_minute)
    return _ingest_limiter


def require_token_project(
    request: Request,
    db: Session = Depends(get_db),
    limiter: RateLimiter = Depends(get_ingest_limiter),
) -> Project:
    """Authenticate a CI request by project token.

    Looked up by hash, never by plaintext comparison against a list, so the
    query is an index hit and the stored value is useless if the database leaks.
    """
    header = request.headers.get("authorization", "")
    scheme, _, raw = header.partition(" ")
    if scheme.lower() != "bearer" or not raw:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "send the project token as: Authorization: Bearer arb_...",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = db.scalar(select(ApiToken).where(ApiToken.token_hash == hash_token(raw.strip())))
    if token is None or not token.is_active:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "unknown or revoked token")
    if not limiter.check(f"token:{token.id}"):
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS, "too many runs submitted, slow down"
        )
    token.last_used_at = utcnow()
    project = db.get(Project, token.project_id)
    if project is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "project no longer exists")
    db.commit()
    return project


def csrf_token(request: Request) -> str:
    """Per-session CSRF token, minted on first use."""
    token = request.session.get(_CSRF_KEY)
    if not token:
        token = secrets.token_urlsafe(24)
        request.session[_CSRF_KEY] = token
    return token


def require_csrf(request: Request, csrf: str = Form("")) -> None:
    """Reject a form post whose token does not match the session.

    SameSite=Lax already blocks cross-site form posts in current browsers. This
    is the second lock, for the cases where that turns out not to be true.
    """
    expected = request.session.get(_CSRF_KEY)
    if not expected or not csrf or not hmac.compare_digest(expected, csrf):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "form expired, please try again")
