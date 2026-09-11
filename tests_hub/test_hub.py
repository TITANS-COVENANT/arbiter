"""Hub tests, weighted towards the things that would be expensive to get wrong.

A rendering bug is visible the first time someone loads the page. A tenancy bug
is invisible until it is a disclosure, so most of what follows is about access
control rather than about HTML.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from arbiter_hub.analytics import task_stats
from arbiter_hub.config import Settings, get_settings
from arbiter_hub.deps import RateLimiter
from arbiter_hub.ingest import GateRunPayload, RunContext, store_run
from arbiter_hub.models import ApiToken, Project, utcnow
from arbiter_hub.security import generate_token, hash_token, slugify, token_prefix, verify_token

from .conftest import make_project, make_token, make_user, sample_result


class TestSecurityPrimitives:
    def test_tokens_are_unique_and_prefixed(self):
        tokens = {generate_token() for _ in range(200)}
        assert len(tokens) == 200
        assert all(t.startswith("arb_") for t in tokens)

    def test_hash_is_not_reversible_to_the_token(self):
        token = generate_token()
        assert hash_token(token) != token
        assert token not in hash_token(token)

    def test_verify_accepts_only_the_right_token(self):
        token = generate_token()
        digest = hash_token(token)
        assert verify_token(token, digest)
        assert not verify_token(generate_token(), digest)

    def test_visible_prefix_reveals_little(self):
        token = generate_token()
        prefix = token_prefix(token)
        assert token.startswith(prefix)
        assert len(prefix) < len(token) / 2

    def test_reserved_slugs_cannot_collide_with_routes(self):
        for reserved in ("api", "login", "static", "settings", "new"):
            assert slugify(reserved) != reserved

    def test_slugify_handles_awkward_input(self):
        assert slugify("Support Agent Evals!") == "support-agent-evals"
        assert slugify("   ") != ""
        assert "/" not in slugify("a/b/c")


class TestSettings:
    def test_production_refuses_dev_auth(self, monkeypatch):
        monkeypatch.setenv("ARBITER_HUB_ENVIRONMENT", "production")
        monkeypatch.setenv("ARBITER_HUB_DEV_AUTH", "1")
        monkeypatch.setenv("ARBITER_HUB_BASE_URL", "https://hub.example.com")
        monkeypatch.setenv("ARBITER_HUB_GITHUB_CLIENT_ID", "x")
        monkeypatch.setenv("ARBITER_HUB_GITHUB_CLIENT_SECRET", "y")
        monkeypatch.setenv("ARBITER_HUB_SECRET_KEY", "z")
        get_settings.cache_clear()
        with pytest.raises(ValueError, match="sign in as anyone"):
            Settings()

    def test_production_refuses_plain_http(self, monkeypatch):
        monkeypatch.setenv("ARBITER_HUB_ENVIRONMENT", "production")
        monkeypatch.setenv("ARBITER_HUB_BASE_URL", "http://hub.example.com")
        monkeypatch.setenv("ARBITER_HUB_GITHUB_CLIENT_ID", "x")
        monkeypatch.setenv("ARBITER_HUB_GITHUB_CLIENT_SECRET", "y")
        monkeypatch.setenv("ARBITER_HUB_SECRET_KEY", "z")
        with pytest.raises(ValueError, match="not https"):
            Settings()

    def test_production_refuses_an_ephemeral_secret_key(self, monkeypatch):
        monkeypatch.setenv("ARBITER_HUB_ENVIRONMENT", "production")
        monkeypatch.setenv("ARBITER_HUB_BASE_URL", "https://hub.example.com")
        monkeypatch.setenv("ARBITER_HUB_GITHUB_CLIENT_ID", "x")
        monkeypatch.setenv("ARBITER_HUB_GITHUB_CLIENT_SECRET", "y")
        monkeypatch.delenv("ARBITER_HUB_SECRET_KEY", raising=False)
        with pytest.raises(ValueError, match="SECRET_KEY"):
            Settings()

    def test_development_defaults_are_permissive(self):
        # _env_file=None because a developer's local .env must not decide
        # whether the shipped defaults are safe.
        defaults = Settings(_env_file=None)
        assert defaults.environment == "development"
        assert defaults.dev_auth is False
        assert defaults.github_configured is False

    def test_cookies_are_secure_over_https(self, monkeypatch):
        monkeypatch.setenv("ARBITER_HUB_BASE_URL", "https://hub.example.com")
        assert Settings().secure_cookies


class TestAuth:
    def test_dev_login_is_off_unless_enabled(self, client: TestClient, monkeypatch):
        monkeypatch.setenv("ARBITER_HUB_DEV_AUTH", "0")
        get_settings.cache_clear()
        assert client.get("/auth/dev", follow_redirects=False).status_code == 404

    def test_dev_login_signs_you_in(self, client: TestClient):
        response = client.get("/auth/dev?login=alice", follow_redirects=False)
        assert response.status_code == 303
        assert client.get("/").status_code == 200

    def test_anonymous_visitors_see_the_landing_page(self, client: TestClient):
        body = client.get("/").text
        assert "Sign in with GitHub" in body
        assert "Projects" not in body.split("<main>")[-1]

    def test_protected_pages_redirect_when_signed_out(self, client: TestClient):
        response = client.get("/projects/new", follow_redirects=False)
        assert response.status_code == 303
        assert response.headers["location"] == "/login"

    def test_logout_is_not_a_get(self, client: TestClient):
        client.get("/auth/dev?login=alice")
        assert client.get("/logout", follow_redirects=False).status_code == 405

    def test_logout_clears_the_session(self, client: TestClient):
        client.get("/auth/dev?login=alice")
        csrf = _csrf(client)
        client.post("/logout", data={"csrf": csrf}, follow_redirects=False)
        assert client.get("/projects/new", follow_redirects=False).status_code == 303

    def test_login_without_oauth_configured_explains_itself(
        self, client: TestClient, monkeypatch
    ):
        monkeypatch.setenv("ARBITER_HUB_DEV_AUTH", "0")
        get_settings.cache_clear()
        response = client.get("/login", follow_redirects=False)
        assert response.status_code == 503
        assert "GITHUB_CLIENT_ID" in response.text

    def test_oauth_callback_rejects_a_mismatched_state(self, client: TestClient):
        response = client.get("/auth/github/callback?code=x&state=forged", follow_redirects=False)
        assert response.status_code == 400
        assert "state" in response.text.lower()


def _csrf(client: TestClient) -> str:
    """Scrape the CSRF token out of a rendered form."""
    body = client.get("/projects/new").text
    marker = 'name="csrf" value="'
    start = body.index(marker) + len(marker)
    return body[start : body.index('"', start)]


class TestProjects:
    def test_create_and_view(self, client: TestClient):
        client.get("/auth/dev?login=alice")
        csrf = _csrf(client)
        orgs = client.get("/projects/new").text
        org_id = orgs.split('<option value="')[1].split('"')[0]
        response = client.post(
            "/projects/new",
            data={"name": "Support evals", "org_id": org_id, "csrf": csrf},
            follow_redirects=True,
        )
        assert response.status_code == 200
        assert "Support evals" in response.text

    def test_a_name_is_required(self, client: TestClient):
        client.get("/auth/dev?login=alice")
        csrf = _csrf(client)
        org_id = client.get("/projects/new").text.split('<option value="')[1].split('"')[0]
        response = client.post(
            "/projects/new", data={"name": "  ", "org_id": org_id, "csrf": csrf}
        )
        assert response.status_code == 400

    def test_a_malformed_repo_is_rejected(self, client: TestClient):
        client.get("/auth/dev?login=alice")
        csrf = _csrf(client)
        org_id = client.get("/projects/new").text.split('<option value="')[1].split('"')[0]
        response = client.post(
            "/projects/new",
            data={"name": "X", "org_id": org_id, "repo_full_name": "not-a-repo", "csrf": csrf},
        )
        assert response.status_code == 400
        assert "owner/name" in response.text

    def test_csrf_is_required(self, client: TestClient):
        client.get("/auth/dev?login=alice")
        org_id = client.get("/projects/new").text.split('<option value="')[1].split('"')[0]
        response = client.post("/projects/new", data={"name": "X", "org_id": org_id})
        assert response.status_code == 400

    def test_cannot_create_inside_someone_elses_org(self, client: TestClient, session: Session):
        bob = make_user(session, "bob")
        bob_org_id = bob.memberships[0].org_id
        session.commit()

        client.get("/auth/dev?login=alice")
        csrf = _csrf(client)
        response = client.post(
            "/projects/new", data={"name": "Sneaky", "org_id": str(bob_org_id), "csrf": csrf}
        )
        assert response.status_code == 404


class TestTenancyIsolation:
    """The tests that matter most. A leak here is a disclosure, not a bug report."""

    def test_another_users_project_is_not_visible(self, client: TestClient, session: Session):
        bob = make_user(session, "bob")
        project = make_project(session, bob, "secret-evals")
        org_slug = bob.memberships[0].org.slug
        session.commit()

        client.get("/auth/dev?login=alice")
        assert client.get(f"/p/{org_slug}/{project.slug}").status_code == 404
        assert client.get(f"/p/{org_slug}/{project.slug}/settings").status_code == 404
        assert client.get(f"/p/{org_slug}/{project.slug}/runs/1").status_code == 404

    def test_another_users_run_is_not_reachable_through_your_own_project(
        self, client: TestClient, session: Session
    ):
        """The run id is a global sequence, so the project scope has to be enforced."""
        bob = make_user(session, "bob")
        bob_project = make_project(session, bob, "bob-evals")
        run = store_run(
            session,
            bob_project,
            GateRunPayload.model_validate(sample_result()),
            RunContext(),
        )
        alice = make_user(session, "alice")
        alice_project = make_project(session, alice, "alice-evals")
        alice_org = alice.memberships[0].org.slug
        session.commit()

        client.get("/auth/dev?login=alice")
        response = client.get(f"/p/{alice_org}/{alice_project.slug}/runs/{run.id}")
        assert response.status_code == 404

    def test_a_token_cannot_write_to_another_project(
        self, client: TestClient, session: Session
    ):
        bob = make_user(session, "bob")
        bob_project = make_project(session, bob, "bob-evals")
        bob_token = make_token(session, bob_project)
        alice = make_user(session, "alice")
        make_project(session, alice, "alice-evals")
        session.commit()

        response = client.post(
            "/api/v1/runs",
            json={"result": sample_result()},
            headers={"Authorization": f"Bearer {bob_token}"},
        )
        assert response.status_code == 201
        # It landed in Bob's project, which is the only one the token names.
        assert response.json()["url"].endswith("/p/bob-org/bob-evals/runs/1")


class TestIngestApi:
    def test_a_token_is_required(self, client: TestClient):
        assert client.post("/api/v1/runs", json={"result": sample_result()}).status_code == 401

    def test_an_unknown_token_is_rejected(self, client: TestClient):
        response = client.post(
            "/api/v1/runs",
            json={"result": sample_result()},
            headers={"Authorization": "Bearer arb_nope"},
        )
        assert response.status_code == 401

    def test_a_revoked_token_is_rejected(self, client: TestClient, session: Session):
        alice = make_user(session, "alice")
        project = make_project(session, alice)
        token = make_token(session, project)
        session.commit()

        ok = client.post(
            "/api/v1/runs",
            json={"result": sample_result()},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert ok.status_code == 201

        row = session.query(ApiToken).filter(ApiToken.token_hash == hash_token(token)).one()
        row.revoked_at = utcnow()
        session.commit()

        after = client.post(
            "/api/v1/runs",
            json={"result": sample_result()},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert after.status_code == 401

    def test_a_run_round_trips(self, client: TestClient, session: Session):
        alice = make_user(session, "alice")
        project = make_project(session, alice)
        token = make_token(session, project)
        session.commit()
        headers = {"Authorization": f"Bearer {token}"}

        created = client.post(
            "/api/v1/runs",
            json={
                "result": sample_result(),
                "context": {"commit_sha": "deadbeef", "branch": "main", "pr_number": 42},
            },
            headers=headers,
        )
        assert created.status_code == 201
        body = created.json()
        assert body["verdict"] == "fail"
        assert body["n_flagged"] == 1

        fetched = client.get(f"/api/v1/runs/{body['id']}", headers=headers).json()
        assert fetched["pr_number"] == 42
        assert fetched["commit_sha"] == "deadbeef"
        assert len(fetched["tasks"]) == 2

    def test_whoami_identifies_the_project(self, client: TestClient, session: Session):
        alice = make_user(session, "alice")
        project = make_project(session, alice)
        token = make_token(session, project)
        session.commit()
        body = client.get(
            "/api/v1/projects/me", headers={"Authorization": f"Bearer {token}"}
        ).json()
        assert body["project"] == project.slug

    def test_an_unknown_verdict_is_rejected(self, client: TestClient, session: Session):
        alice = make_user(session, "alice")
        token = make_token(session, make_project(session, alice))
        session.commit()
        payload = sample_result()
        payload["verdict"] = "probably-fine"
        response = client.post(
            "/api/v1/runs",
            json={"result": payload},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 422

    def test_a_javascript_ci_url_is_rejected(self, client: TestClient, session: Session):
        """It is rendered as an href, so the scheme has to be checked at the door."""
        alice = make_user(session, "alice")
        token = make_token(session, make_project(session, alice))
        session.commit()
        response = client.post(
            "/api/v1/runs",
            json={"result": sample_result(), "context": {"ci_url": "javascript:alert(1)"}},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 422

    def test_unknown_fields_do_not_break_ingestion(self, client: TestClient, session: Session):
        """A newer arbiter adding a field must not take the hub down."""
        alice = make_user(session, "alice")
        token = make_token(session, make_project(session, alice))
        session.commit()
        payload = sample_result()
        payload["some_future_field"] = {"nested": True}
        payload["tasks"][0]["another_new_one"] = 1
        response = client.post(
            "/api/v1/runs",
            json={"result": payload},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 201

    def test_an_oversized_body_is_refused(self, client: TestClient, session: Session):
        alice = make_user(session, "alice")
        token = make_token(session, make_project(session, alice))
        session.commit()
        response = client.post(
            "/api/v1/runs",
            content=b"{}",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "Content-Length": str(50 * 1024 * 1024),
            },
        )
        assert response.status_code == 413


class TestTokensUi:
    def test_a_token_is_shown_once_and_never_again(self, client: TestClient, session: Session):
        alice = make_user(session, "alice")
        project = make_project(session, alice)
        org_slug = alice.memberships[0].org.slug
        session.commit()

        client.get("/auth/dev?login=alice")
        url = f"/p/{org_slug}/{project.slug}"
        csrf = _csrf(client)
        first = client.post(f"{url}/tokens", data={"name": "ci", "csrf": csrf},
                            follow_redirects=True)
        assert "arb_" in first.text
        again = client.get(f"{url}/settings")
        assert "Copy this token now" not in again.text

    def test_revoking_works_from_the_ui(self, client: TestClient, session: Session):
        alice = make_user(session, "alice")
        project = make_project(session, alice)
        org_slug = alice.memberships[0].org.slug
        session.commit()

        client.get("/auth/dev?login=alice")
        url = f"/p/{org_slug}/{project.slug}"
        csrf = _csrf(client)
        client.post(f"{url}/tokens", data={"name": "ci", "csrf": csrf}, follow_redirects=True)
        token_row = session.query(ApiToken).filter(ApiToken.project_id == project.id).one()
        client.post(f"{url}/tokens/{token_row.id}/revoke", data={"csrf": csrf},
                    follow_redirects=True)
        session.expire_all()
        assert session.get(ApiToken, token_row.id).revoked_at is not None


class TestDeletion:
    def test_deleting_needs_the_slug_typed_back(self, client: TestClient, session: Session):
        alice = make_user(session, "alice")
        project = make_project(session, alice)
        org_slug = alice.memberships[0].org.slug
        session.commit()

        client.get("/auth/dev?login=alice")
        url = f"/p/{org_slug}/{project.slug}"
        slug, project_id = project.slug, project.id
        csrf = _csrf(client)
        refused = client.post(f"{url}/delete", data={"confirm": "wrong", "csrf": csrf})
        assert refused.status_code == 400
        assert session.get(Project, project_id) is not None

        client.post(f"{url}/delete", data={"confirm": slug, "csrf": csrf}, follow_redirects=True)
        # expunge rather than expire: refreshing an identity-mapped row that the
        # other session deleted raises instead of returning None.
        session.expunge_all()
        assert session.get(Project, project_id) is None


class TestRateLimit:
    def test_allows_then_blocks(self):
        limiter = RateLimiter(per_minute=3)
        assert all(limiter.check("k", now=100.0) for _ in range(3))
        assert not limiter.check("k", now=100.0)

    def test_window_slides(self):
        limiter = RateLimiter(per_minute=2)
        limiter.check("k", now=0.0)
        limiter.check("k", now=0.0)
        assert not limiter.check("k", now=1.0)
        assert limiter.check("k", now=61.0)

    def test_keys_are_independent(self):
        limiter = RateLimiter(per_minute=1)
        assert limiter.check("a", now=0.0)
        assert limiter.check("b", now=0.0)
        assert not limiter.check("a", now=0.0)


class TestAnalytics:
    @staticmethod
    def _seed(session: Session, project: Project, pattern: list[bool], task_id: str) -> None:
        for flagged in pattern:
            payload = sample_result(verdict="fail" if flagged else "pass", flagged_task=task_id)
            payload["tasks"][0]["flagged"] = flagged
            store_run(session, project, GateRunPayload.model_validate(payload), RunContext())
        session.flush()

    def test_chronic_is_not_called_flaky(self, session: Session):
        alice = make_user(session, "alice")
        project = make_project(session, alice)
        # Clean for two runs, then broken and left broken.
        self._seed(session, project, [False, False] + [True] * 10, "broken")
        stats = {s.task_id: s for s in task_stats(session, project)}
        assert stats["broken"].is_chronic
        assert not stats["broken"].is_flaky
        assert stats["broken"].label == "chronic"

    def test_flaky_is_not_called_chronic(self, session: Session):
        alice = make_user(session, "alice")
        project = make_project(session, alice)
        self._seed(session, project, [True, False] * 6, "wobbly")
        stats = {s.task_id: s for s in task_stats(session, project)}
        assert stats["wobbly"].is_flaky
        assert not stats["wobbly"].is_chronic

    def test_a_stable_task_is_neither(self, session: Session):
        alice = make_user(session, "alice")
        project = make_project(session, alice)
        self._seed(session, project, [False] * 10, "calm")
        stats = {s.task_id: s for s in task_stats(session, project)}
        assert stats["calm"].label == "stable"
        assert stats["calm"].flag_rate == 0.0

    def test_one_change_is_not_flakiness(self, session: Session):
        """A task that broke once and stayed broken has flipped, but is not flaky."""
        session_project = make_project(session, make_user(session, "alice"))
        self._seed(session, session_project, [False, True, True, True, True, True], "regressed")
        stats = {s.task_id: s for s in task_stats(session, session_project)}
        assert not stats["regressed"].is_flaky

    def test_two_runs_are_not_enough_to_judge(self, session: Session):
        project = make_project(session, make_user(session, "alice"))
        self._seed(session, project, [True, False], "unknown")
        stats = {s.task_id: s for s in task_stats(session, project)}
        assert not stats["unknown"].is_flaky
        assert not stats["unknown"].is_chronic


class TestMeta:
    def test_health_touches_the_database(self, client: TestClient):
        assert client.get("/health").json()["status"] == "ok"

    def test_security_headers_are_present(self, client: TestClient):
        headers = client.get("/").headers
        assert "Content-Security-Policy" in headers
        assert headers["X-Frame-Options"] == "DENY"
        assert headers["X-Content-Type-Options"] == "nosniff"

    def test_csp_does_not_allow_remote_scripts(self, client: TestClient):
        """htmx is vendored precisely so this can stay self-only."""
        csp = client.get("/").headers["Content-Security-Policy"]
        assert "script-src 'self'" in csp
        assert "unsafe-eval" not in csp

    def test_robots_keeps_project_pages_out_of_search(self, client: TestClient):
        assert "Disallow: /p/" in client.get("/robots.txt").text
