"""Posting a gate result to an arbiter hub.

Two rules shape this module.

**Publishing must never fail a build.** The gate's exit code is a statement about
the candidate. A hub that is down, slow, or misconfigured says nothing about the
candidate, so every failure here is reported and swallowed. A team whose
deployments start failing because a dashboard fell over will delete the
dashboard, and they will be right to.

**The token is never printed.** Not in the error path, not in a debug line, not
in the request echo. It arrives from an environment variable and goes into one
header.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any

__all__ = ["PublishResult", "RunContext", "detect_context", "publish_result"]

_PR_REF = re.compile(r"^refs/pull/(\d+)/")


@dataclass
class RunContext:
    """Where this run happened, as far as the environment will admit."""

    commit_sha: str = ""
    branch: str = ""
    pr_number: int | None = None
    ci_url: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "commit_sha": self.commit_sha[:64],
            "branch": self.branch[:200],
            "ci_url": self.ci_url[:500],
        }
        if self.pr_number is not None:
            payload["pr_number"] = self.pr_number
        return payload


@dataclass
class PublishResult:
    """What happened, so the CLI can say so without deciding anything."""

    ok: bool
    url: str = ""
    error: str = ""
    extras: dict[str, Any] = field(default_factory=dict)


def _int_or_none(value: str | None) -> int | None:
    return int(value) if value and value.isdigit() else None


def detect_context(env: dict[str, str] | None = None) -> RunContext:
    """Work out the commit, branch and pull request from CI environment variables.

    Supports GitHub Actions and GitLab CI, with explicit ARBITER_* variables
    taking priority so anything else can be wired up without a code change.
    """
    env = dict(os.environ if env is None else env)
    context = RunContext()

    if env.get("GITHUB_ACTIONS"):
        context.commit_sha = env.get("GITHUB_SHA", "")
        context.branch = env.get("GITHUB_REF_NAME", "")
        match = _PR_REF.match(env.get("GITHUB_REF", ""))
        if match:
            context.pr_number = int(match.group(1))
        server = env.get("GITHUB_SERVER_URL", "https://github.com")
        repo = env.get("GITHUB_REPOSITORY", "")
        run_id = env.get("GITHUB_RUN_ID", "")
        if repo and run_id:
            context.ci_url = f"{server}/{repo}/actions/runs/{run_id}"
    elif env.get("GITLAB_CI"):
        context.commit_sha = env.get("CI_COMMIT_SHA", "")
        context.branch = env.get("CI_COMMIT_REF_NAME", "")
        context.pr_number = _int_or_none(env.get("CI_MERGE_REQUEST_IID"))
        context.ci_url = env.get("CI_JOB_URL", "")

    # Explicit overrides win, so an unsupported CI can still report properly.
    context.commit_sha = env.get("ARBITER_COMMIT_SHA", context.commit_sha)
    context.branch = env.get("ARBITER_BRANCH", context.branch)
    context.pr_number = _int_or_none(env.get("ARBITER_PR_NUMBER")) or context.pr_number
    context.ci_url = env.get("ARBITER_CI_URL", context.ci_url)

    if context.ci_url and not context.ci_url.startswith(("http://", "https://")):
        context.ci_url = ""
    return context


def publish_result(
    result: dict[str, Any],
    *,
    base_url: str,
    token: str,
    context: RunContext | None = None,
    timeout_s: float = 20.0,
) -> PublishResult:
    """Post a gate result. Returns what happened; never raises.

    The caller is expected to print the outcome and carry on regardless.
    """
    import httpx

    if not token:
        return PublishResult(
            ok=False,
            error=(
                "no publish token. Pass --publish-token or set ARBITER_HUB_TOKEN "
                "(the flag is preferred in a shell, the variable in CI)"
            ),
        )
    endpoint = base_url.rstrip("/")
    if not endpoint.endswith("/api/v1/runs"):
        endpoint = f"{endpoint}/api/v1/runs"

    body = {"result": result, "context": (context or detect_context()).to_dict()}
    try:
        response = httpx.post(
            endpoint,
            json=body,
            headers={"Authorization": f"Bearer {token}", "User-Agent": "arbiter-cli"},
            timeout=timeout_s,
        )
    except Exception as exc:
        return PublishResult(ok=False, error=f"{type(exc).__name__}: {exc}")

    if response.status_code == 401:
        return PublishResult(ok=False, error="the hub rejected the token")
    if response.status_code >= 400:
        # Truncated, because a hub could return anything and this ends up in
        # someone's CI log.
        return PublishResult(
            ok=False, error=f"hub returned HTTP {response.status_code}: {response.text[:200]}"
        )
    try:
        payload = response.json()
    except ValueError:
        return PublishResult(ok=False, error="the hub did not return JSON")
    return PublishResult(ok=True, url=str(payload.get("url", "")), extras=payload)
