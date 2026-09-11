# arbiter hub

The web service your CI reports verdicts into. Sign in with GitHub, create a
project, mint a token, add two flags to the job you already have.

## Why it exists

`arbiter gate` answers one question well: did this build regress. What it cannot
answer, because it only ever sees one run, is everything that needs a history.

- Has the task that just went red been flipping for a month?
- Is this actually new, or has it been failing since the sprint before last?
- Which tasks cost the most replicates to decide?

The hub keeps every run and stores per-task rows you can group across runs, which
is what makes those questions cheap to answer.

## The distinction the whole thing is built around

Two tasks can both be "unreliable" and need completely opposite responses.

A **flaky** task alternates between flagged and clear without the build changing
underneath it. Its signal is unstable. Either the seed is not reaching everything
random in your agent, or `max_replicates` is too low for it, or it should not be
gating anything.

A **chronic** task is flagged in nearly every run and does not flip. That is not
flakiness. That is a regression that shipped and stayed, and it will keep failing
the gate until someone fixes it or the expectation changes.

Lumping both into one list called "unreliable tests" is how a real regression
gets filed under flakiness and ignored, so they are counted separately and
labelled differently. The rule is in
[`analytics.py`](../src/arbiter_hub/analytics.py):

```python
is_flaky   = runs >= 3 and flip_rate >= 0.25 and flag_rate < 0.9
is_chronic = runs >= 3 and flag_rate >= 0.7  and flip_rate < 0.25
```

Three runs is the floor because one flip out of two runs is a change, not a
pattern.

## Running it locally

```bash
pip install -e ".[web]"
arbiter-hub demo                 # seed a project with synthetic history
arbiter-hub serve --dev          # http://127.0.0.1:8000
```

`--dev` turns on `/auth/dev`, which signs you in without GitHub. It is an
authentication bypass, so it is off by default and the settings refuse to start a
production deployment that has it on.

The `demo` command seeds a history containing exactly one chronic task, one
flaky task, and eight stable ones, so you can see whether the classification is
doing anything.

## Connecting your CI

```yaml
- name: eval gate
  env:
    ARBITER_HUB_TOKEN: ${{ secrets.ARBITER_HUB_TOKEN }}
  run: |
    arbiter gate evals/suite.yaml \
      --store .arbiter/runs.sqlite \
      --publish https://your-hub.example.com
```

The exit code still comes from the gate. Publishing is additive and cannot change
it: a hub that is down, slow or misconfigured says nothing about your candidate
build, so `publish_result` reports every failure and swallows it. A team whose
deploys start failing because a dashboard fell over will delete the dashboard,
and they would be right to.

Failures are printed even under `--quiet`. Reporting that has silently stopped
working is worse than reporting that never worked, because the dashboard keeps
showing last week's history as though it were current.

### Commit and branch detection

Picked up automatically from GitHub Actions and GitLab CI. For anything else, or
to override, set `ARBITER_COMMIT_SHA`, `ARBITER_BRANCH`, `ARBITER_PR_NUMBER` or
`ARBITER_CI_URL`.

## Deploying

```bash
export ARBITER_HUB_ENVIRONMENT=production
export ARBITER_HUB_BASE_URL=https://hub.example.com
export ARBITER_HUB_SECRET_KEY=$(python -c "import secrets;print(secrets.token_urlsafe(48))")
export ARBITER_HUB_DATABASE_URL=postgresql+psycopg://user:pass@host/arbiter_hub
export ARBITER_HUB_GITHUB_CLIENT_ID=...
export ARBITER_HUB_GITHUB_CLIENT_SECRET=...

arbiter-hub init-db
uvicorn arbiter_hub.app:build --factory --host 0.0.0.0 --port 8000
```

Register the OAuth app at <https://github.com/settings/developers> with the
callback set to `{base_url}/auth/github/callback`.

Production **refuses to start** if dev auth is on, GitHub OAuth is unconfigured,
`SECRET_KEY` is unset, or `base_url` is not https. Every one of those has been a
real incident somewhere, and failing at startup is the cheapest moment to find
out.

Schema creation is `create_all`, which is fine for a single instance and for
getting started. A schema that has to change under a running service wants a
migration tool, and this project does not pretend otherwise.

## Security

**The GitHub token is used once and thrown away.** Scopes requested are
`read:user` and `user:email`, and the access token is never written to the
database. This product does not need to touch your repositories, so holding a
credential that could would be storing risk for nothing.

**API tokens are stored as SHA-256 hashes and shown once.** There is no reveal
button, because a dashboard that can show you your own token can show it to
whoever gets into your session.

**Access control lives in one place.** Every project route resolves through
`require_project`, which starts from the signed-in user's memberships rather than
from the URL. A project belonging to someone else returns 404 rather than 403, so
the URL space does not confirm which names are taken in other orgs.

**Sessions** are signed cookies, httponly, SameSite=Lax, and Secure whenever the
base URL is https. The session is regenerated on login so a fixated session
cannot be reused. Forms carry a CSRF token as a second lock behind SameSite.

**Content-Security-Policy is `self` only.** htmx is vendored into `static/`
rather than loaded from a CDN specifically so that it can be.

**Ingest is validated, not trusted.** It arrives from a machine we do not control
over a token that may have leaked, so unknown fields are ignored, strings are
length-capped, the task list is bounded, and `ci_url` must be http or https
because it is rendered as an href.

**Rate limiting is in-process and honest about it.** It stops one misconfigured
CI loop from filling the database. It is not a defence against a determined
attacker, and anything stronger belongs in front of the app.

### What it does not do yet

- No GitHub App, so no commit statuses and no required-check integration.
- No team invitations. Orgs exist in the schema and every user gets one, but
  there is no UI for adding a second member.
- No billing, and no per-account quotas beyond the rate limit.
- `create_all` rather than migrations.

## API

Authenticate with `Authorization: Bearer arb_...`.

| | |
|---|---|
| `POST /api/v1/runs` | Submit a gate result |
| `GET /api/v1/runs/{id}` | Read one back, scoped to the token's project |
| `GET /api/v1/projects/me` | What this token can write to |
| `GET /health` | Liveness, including a database round trip |

`POST /api/v1/runs` takes exactly what `arbiter gate --json` writes, wrapped:

```json
{
  "result": { "verdict": "fail", "tasks": [ ... ] },
  "context": { "commit_sha": "abc123", "branch": "main", "pr_number": 42 }
}
```

Keeping the wire format identical to the CLI's own output means there is no
translation layer to drift. `projects/me` exists so that a CI job failing to
publish can be diagnosed with one curl rather than by guessing whether the token
or the URL is wrong.

Interactive docs at `/api/docs`.

## Configuration

Every setting is an environment variable prefixed `ARBITER_HUB_`, or a line in
`.env`.

| Variable | Default | Notes |
|---|---|---|
| `ENVIRONMENT` | `development` | `production` enables the startup safety checks |
| `BASE_URL` | `http://localhost:8000` | Used in links and the OAuth callback |
| `DATABASE_URL` | `sqlite:///./arbiter_hub.sqlite` | Any SQLAlchemy URL |
| `SECRET_KEY` | generated per process | Required in production, or restarts sign everyone out |
| `GITHUB_CLIENT_ID` / `GITHUB_CLIENT_SECRET` | empty | From your OAuth app |
| `DEV_AUTH` | `false` | Authentication bypass. Local only |
| `INGEST_RATE_PER_MINUTE` | `60` | Per token |
| `MAX_PAYLOAD_BYTES` | `4194304` | Rejected on Content-Length |
