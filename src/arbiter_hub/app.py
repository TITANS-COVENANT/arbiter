"""Application factory and middleware."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.sessions import SessionMiddleware

from . import __version__
from .api import health_router
from .api import router as api_router
from .auth import router as auth_router
from .config import Settings, get_settings
from .db import build_engine, create_all
from .views import render
from .views import router as views_router

__all__ = ["create_app"]

STATIC_DIR = Path(__file__).parent / "static"

# Everything is served from this origin. htmx is vendored into static/ rather
# than pulled from a CDN precisely so this can stay 'self' and nothing else.
_CSP = (
    "default-src 'self'; "
    "script-src 'self'; "
    "style-src 'self' 'unsafe-inline'; "
    "img-src 'self' https://avatars.githubusercontent.com data:; "
    "form-action 'self'; "
    "frame-ancestors 'none'; "
    "base-uri 'none'; "
    "object-src 'none'"
)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Headers that cost nothing and close whole categories of problem."""

    def __init__(self, app, hsts: bool) -> None:
        super().__init__(app)
        self.hsts = hsts

    async def dispatch(self, request: Request, call_next):
        response: Response = await call_next(request)
        response.headers.setdefault("Content-Security-Policy", _CSP)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault(
            "Permissions-Policy", "geolocation=(), microphone=(), camera=()"
        )
        if self.hsts:
            response.headers.setdefault(
                "Strict-Transport-Security", "max-age=31536000; includeSubDomains"
            )
        return response


class MaxBodySizeMiddleware(BaseHTTPMiddleware):
    """Reject oversized uploads before they are parsed.

    A gate result for a large suite is a few hundred kilobytes. Anything far
    past that is either a mistake or someone trying to make the JSON parser do
    the work, and rejecting on Content-Length is cheaper than either.
    """

    def __init__(self, app, max_bytes: int) -> None:
        super().__init__(app)
        self.max_bytes = max_bytes

    async def dispatch(self, request: Request, call_next):
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > self.max_bytes:
            return JSONResponse(
                {"detail": f"payload too large (limit {self.max_bytes} bytes)"},
                status_code=413,
            )
        return await call_next(request)


def _wants_html(request: Request) -> bool:
    if request.url.path.startswith(("/api/", "/health")):
        return False
    return "text/html" in request.headers.get("accept", "")


def create_app(settings: Settings | None = None, *, init_db: bool = True) -> FastAPI:
    """Build the application."""
    settings = settings or get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
    )
    app.state.settings = settings

    if init_db:
        create_all(build_engine(settings))

    app.add_middleware(MaxBodySizeMiddleware, max_bytes=settings.max_payload_bytes)
    app.add_middleware(SecurityHeadersMiddleware, hsts=settings.secure_cookies)
    app.add_middleware(
        SessionMiddleware,
        secret_key=settings.secret_key,
        session_cookie=settings.session_cookie,
        max_age=settings.session_max_age_seconds,
        same_site="lax",
        https_only=settings.secure_cookies,
    )

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    app.include_router(health_router)
    app.include_router(auth_router)
    app.include_router(api_router)
    app.include_router(views_router)

    @app.exception_handler(HTTPException)
    async def _http_exception(request: Request, exc: HTTPException) -> Response:
        # A redirect raised from a dependency, such as require_user sending an
        # anonymous visitor to the login page.
        location = (exc.headers or {}).get("Location")
        if location and 300 <= exc.status_code < 400:
            return Response(status_code=exc.status_code, headers={"Location": location})
        if _wants_html(request):
            return render(
                request,
                "error.html",
                {"user": None, "status": exc.status_code, "detail": exc.detail},
                status_code=exc.status_code,
            )
        return JSONResponse({"detail": exc.detail}, status_code=exc.status_code,
                            headers=exc.headers)

    @app.get("/robots.txt", include_in_schema=False)
    def robots() -> Response:
        return Response("User-agent: *\nDisallow: /p/\nDisallow: /api/\n", media_type="text/plain")

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon() -> Response:
        return Response(status_code=204)

    return app


def build() -> FastAPI:
    """Entry point for `uvicorn arbiter_hub.app:build --factory`."""
    return create_app()
