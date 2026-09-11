"""Reporting a run to a hub.

The property under test throughout is that nothing here can change a build's
fate. A gate's exit code is a statement about the candidate, and a dashboard
being unreachable is not evidence about the candidate.
"""

from __future__ import annotations

import pytest

from arbiter.publish import RunContext, detect_context, publish_result


class TestContextDetection:
    def test_empty_environment_yields_nothing(self):
        context = detect_context({})
        assert context.commit_sha == ""
        assert context.pr_number is None

    def test_github_actions_push(self):
        context = detect_context(
            {
                "GITHUB_ACTIONS": "true",
                "GITHUB_SHA": "abc123",
                "GITHUB_REF_NAME": "main",
                "GITHUB_REF": "refs/heads/main",
                "GITHUB_SERVER_URL": "https://github.com",
                "GITHUB_REPOSITORY": "acme/agent",
                "GITHUB_RUN_ID": "42",
            }
        )
        assert context.commit_sha == "abc123"
        assert context.branch == "main"
        assert context.pr_number is None
        assert context.ci_url == "https://github.com/acme/agent/actions/runs/42"

    def test_github_actions_pull_request(self):
        context = detect_context(
            {
                "GITHUB_ACTIONS": "true",
                "GITHUB_REF": "refs/pull/1234/merge",
                "GITHUB_REF_NAME": "1234/merge",
            }
        )
        assert context.pr_number == 1234

    def test_gitlab_merge_request(self):
        context = detect_context(
            {
                "GITLAB_CI": "true",
                "CI_COMMIT_SHA": "def456",
                "CI_COMMIT_REF_NAME": "feature",
                "CI_MERGE_REQUEST_IID": "77",
                "CI_JOB_URL": "https://gitlab.example.com/job/1",
            }
        )
        assert context.commit_sha == "def456"
        assert context.pr_number == 77

    def test_explicit_variables_win(self):
        """So an unsupported CI can report properly without a code change."""
        context = detect_context(
            {
                "GITHUB_ACTIONS": "true",
                "GITHUB_SHA": "from-github",
                "ARBITER_COMMIT_SHA": "explicit",
                "ARBITER_PR_NUMBER": "9",
            }
        )
        assert context.commit_sha == "explicit"
        assert context.pr_number == 9

    def test_a_non_http_ci_url_is_dropped(self):
        """It ends up in an href, so the scheme is filtered before it is sent."""
        assert detect_context({"ARBITER_CI_URL": "javascript:alert(1)"}).ci_url == ""

    def test_a_non_numeric_pr_number_is_ignored(self):
        assert detect_context({"ARBITER_PR_NUMBER": "not-a-number"}).pr_number is None

    def test_serialisation_caps_lengths(self):
        payload = RunContext(commit_sha="x" * 200, branch="y" * 500).to_dict()
        assert len(payload["commit_sha"]) == 64
        assert len(payload["branch"]) == 200

    def test_absent_pr_number_is_omitted_entirely(self):
        assert "pr_number" not in RunContext().to_dict()


class TestPublishing:
    def test_a_missing_token_is_reported_not_raised(self):
        outcome = publish_result({}, base_url="http://hub.invalid", token="")
        assert not outcome.ok
        assert "token" in outcome.error

    def test_an_unreachable_hub_does_not_raise(self):
        """The whole point. A dead hub must not be able to fail someone's build."""
        outcome = publish_result(
            {"verdict": "pass"},
            base_url="http://127.0.0.1:1",
            token="arb_x",
            timeout_s=0.5,
        )
        assert not outcome.ok
        assert outcome.error

    @pytest.mark.parametrize(
        "given",
        ["http://hub.example", "http://hub.example/", "http://hub.example/api/v1/runs"],
    )
    def test_the_endpoint_is_derived_consistently(self, given, monkeypatch):
        """Users will paste the base URL or the full endpoint; both should work."""
        seen: dict[str, str] = {}

        class _Response:
            status_code = 201

            @staticmethod
            def json() -> dict[str, str]:
                return {"url": "http://hub.example/p/o/p/runs/1"}

        def fake_post(url, **kwargs):
            seen["url"] = url
            return _Response()

        import httpx

        monkeypatch.setattr(httpx, "post", fake_post)
        outcome = publish_result({"verdict": "pass"}, base_url=given, token="arb_x")
        assert outcome.ok
        assert seen["url"] == "http://hub.example/api/v1/runs"

    def test_the_token_never_appears_in_an_error(self, monkeypatch):
        secret = "arb_super_secret_value"

        class _Response:
            status_code = 500
            text = "internal error"

        import httpx

        monkeypatch.setattr(httpx, "post", lambda url, **kwargs: _Response())
        outcome = publish_result({"verdict": "pass"}, base_url="http://h", token=secret)
        assert not outcome.ok
        assert secret not in outcome.error

    def test_a_rejected_token_says_so_plainly(self, monkeypatch):
        class _Response:
            status_code = 401
            text = "nope"

        import httpx

        monkeypatch.setattr(httpx, "post", lambda url, **kwargs: _Response())
        outcome = publish_result({"verdict": "pass"}, base_url="http://h", token="arb_x")
        assert "rejected the token" in outcome.error

    def test_a_non_json_response_is_handled(self, monkeypatch):
        class _Response:
            status_code = 201
            text = "<html>"

            @staticmethod
            def json():
                raise ValueError("not json")

        import httpx

        monkeypatch.setattr(httpx, "post", lambda url, **kwargs: _Response())
        outcome = publish_result({"verdict": "pass"}, base_url="http://h", token="arb_x")
        assert not outcome.ok
        assert "JSON" in outcome.error
