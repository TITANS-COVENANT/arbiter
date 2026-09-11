"""Signing in with GitHub.

The design decision worth stating: **the GitHub access token is used once, to
read the profile, and then thrown away.** It is never written to the database.

This product shape does not need to touch anyone's repositories, so holding a
credential that could would be storing risk in exchange for nothing. The scopes
requested are ``read:user`` and ``user:email``, which is the least GitHub will
give you and still tell you who signed in. If a later version needs to post
commit statuses, that is a GitHub App with its own installation token, not a
broadening of this one.
"""

from __future__ import annotations

import hmac
import secrets
from typing import Any
from urllib.parse import urlencode

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from .config import Settings, get_settings
from .db import get_db
from .models import Membership, Org, Role, User, utcnow
from .security import slugify

__all__ = ["ensure_personal_org", "login_user", "router"]

router = APIRouter()

_AUTHORIZE = "https://github.com/login/oauth/authorize"
_TOKEN = "https://github.com/login/oauth/access_token"
_API = "https://api.github.com"
_SCOPES = "read:user user:email"
_STATE_KEY = "oauth_state"
_NEXT_KEY = "oauth_next"


def ensure_personal_org(session: Session, user: User) -> Org:
    """Every user owns an org from their first sign-in.

    Projects hang off orgs rather than users so that sharing a project later is
    adding a membership row, not a schema migration.
    """
    existing = session.scalar(
        select(Org).join(Membership).where(Membership.user_id == user.id).order_by(Org.id)
    )
    if existing is not None:
        return existing
    org = Org(slug=slugify(user.login, fallback="org"), name=user.display_name, is_personal=True)
    session.add(org)
    session.flush()
    session.add(Membership(user_id=user.id, org_id=org.id, role=Role.OWNER))
    session.flush()
    return org


def login_user(request: Request, user: User) -> None:
    """Record the signed-in user in the session cookie.

    The session is regenerated on login so that a session fixated before sign-in
    cannot be reused afterwards.
    """
    request.session.clear()
    request.session["user_id"] = user.id
    request.session["login_at"] = utcnow().isoformat()


def _safe_next(value: str | None) -> str:
    """Only allow same-site relative redirects.

    ``//evil.example`` is a protocol-relative absolute URL that a naive
    startswith("/") check would happily send a freshly authenticated user to.
    """
    if not value or not value.startswith("/") or value.startswith("//"):
        return "/"
    return value


def _upsert_user(session: Session, profile: dict[str, Any], email: str) -> User:
    github_id = int(profile["id"])
    user = session.scalar(select(User).where(User.github_id == github_id))
    if user is None:
        user = User(github_id=github_id)
        session.add(user)
    user.login = str(profile.get("login") or f"user{github_id}")[:100]
    user.name = str(profile.get("name") or "")[:200]
    user.avatar_url = str(profile.get("avatar_url") or "")[:500]
    if email:
        user.email = email[:320]
    user.last_seen_at = utcnow()
    session.flush()
    return user


@router.get("/login")
def login(
    request: Request,
    next: str = "/",
    settings: Settings = Depends(get_settings),
) -> RedirectResponse:
    """Start the OAuth dance."""
    if not settings.github_configured:
        if settings.dev_auth:
            return RedirectResponse(f"/auth/dev?next={_safe_next(next)}", status_code=303)
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "GitHub sign-in is not configured on this deployment. Set "
            "ARBITER_HUB_GITHUB_CLIENT_ID and ARBITER_HUB_GITHUB_CLIENT_SECRET, or "
            "turn on ARBITER_HUB_DEV_AUTH for local use.",
        )
    state = secrets.token_urlsafe(24)
    request.session[_STATE_KEY] = state
    request.session[_NEXT_KEY] = _safe_next(next)
    query = urlencode(
        {
            "client_id": settings.github_client_id,
            "redirect_uri": settings.github_callback_url,
            "scope": _SCOPES,
            "state": state,
            "allow_signup": "true",
        }
    )
    return RedirectResponse(f"{_AUTHORIZE}?{query}", status_code=303)


@router.get("/auth/github/callback")
async def github_callback(
    request: Request,
    code: str = "",
    state: str = "",
    error: str = "",
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> RedirectResponse:
    """Finish the dance: validate state, swap the code, sign the user in."""
    if error:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"GitHub declined the sign-in: {error}")

    # Single-use state. Popping it means a replayed callback fails even if the
    # original was intercepted.
    expected = request.session.pop(_STATE_KEY, None)
    destination = _safe_next(request.session.pop(_NEXT_KEY, "/"))
    if not expected or not state or not hmac.compare_digest(expected, state):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "sign-in state did not match; start again from the login page",
        )
    if not code:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "GitHub did not return a code")

    async with httpx.AsyncClient(timeout=15.0) as client:
        token_response = await client.post(
            _TOKEN,
            headers={"Accept": "application/json"},
            data={
                "client_id": settings.github_client_id,
                "client_secret": settings.github_client_secret,
                "code": code,
                "redirect_uri": settings.github_callback_url,
            },
        )
        if token_response.status_code >= 400:
            raise HTTPException(status.HTTP_502_BAD_GATEWAY, "GitHub rejected the token exchange")
        token_body = token_response.json()
        access_token = token_body.get("access_token")
        if not access_token:
            # Deliberately does not echo the body, which can carry detail we
            # have no business rendering back to a browser.
            raise HTTPException(status.HTTP_502_BAD_GATEWAY, "GitHub did not return a token")

        headers = {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/vnd.github+json",
        }
        profile_response = await client.get(f"{_API}/user", headers=headers)
        if profile_response.status_code >= 400:
            raise HTTPException(status.HTTP_502_BAD_GATEWAY, "could not read your GitHub profile")
        profile = profile_response.json()

        email = str(profile.get("email") or "")
        if not email:
            emails_response = await client.get(f"{_API}/user/emails", headers=headers)
            if emails_response.status_code < 400:
                for entry in emails_response.json():
                    if entry.get("primary") and entry.get("verified"):
                        email = str(entry.get("email") or "")
                        break

    # The access token goes out of scope here and is never persisted.
    user = _upsert_user(db, profile, email)
    ensure_personal_org(db, user)
    db.commit()
    login_user(request, user)
    return RedirectResponse(destination, status_code=303)


@router.get("/auth/dev")
def dev_login(
    request: Request,
    next: str = "/",
    login: str = "dev",
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> RedirectResponse:
    """Sign in without GitHub, for local development only.

    This is an authentication bypass. It is refused unless explicitly enabled,
    and :class:`~arbiter_hub.config.Settings` refuses to start a production
    deployment that has it on.
    """
    if not settings.dev_auth:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "not found")
    handle = slugify(login, fallback="dev")
    user = db.scalar(select(User).where(User.login == handle))
    if user is None:
        # Negative ids cannot collide with a real GitHub account id.
        lowest = db.scalar(select(User.github_id).order_by(User.github_id).limit(1)) or 0
        user = User(
            github_id=min(lowest, 0) - 1,
            login=handle,
            name=handle.replace("-", " ").title(),
            email=f"{handle}@example.invalid",
        )
        db.add(user)
        db.flush()
    ensure_personal_org(db, user)
    db.commit()
    login_user(request, user)
    return RedirectResponse(_safe_next(next), status_code=303)


@router.post("/logout")
def logout(request: Request) -> RedirectResponse:
    request.session.clear()
    return RedirectResponse("/", status_code=303)


@router.get("/logout")
def logout_get() -> HTMLResponse:
    """Signing out is a state change, so it does not happen on a GET.

    A bare GET /logout can be triggered by any image tag on any page, which is
    only an annoyance, but the fix costs one form.
    """
    return HTMLResponse(
        '<form method="post" action="/logout"><button type="submit">Sign out</button></form>',
        status_code=405,
    )
