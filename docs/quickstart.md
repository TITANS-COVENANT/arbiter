# Quickstart

## Install

```bash
pip install -e ".[dev]"
```

## See it work on the simulated agent

No API keys needed. The built-in fake agent has a known pass rate, and one task
in the starter suite is deliberately broken.

```bash
arbiter init demo.yaml
arbiter gate demo.yaml
```

Exit code 1, and a table naming `tool_error_recovery` as regressed. Then ask what
changed:

```bash
arbiter diff demo.yaml tool_error_recovery
```

You get the failure modes the candidate introduced, and a step-by-step alignment
of one replicate where the baseline passed and the candidate failed at the same
seed.

## Wire up your own agent

A target is anything that takes a task and a seed and says whether it passed.
Three ways to provide one.

### In-process Python

```python
# my_evals.py
def candidate(task_input: dict, seed: int) -> dict:
    result = run_my_agent(task_input["prompt"], seed=seed)
    return {
        "passed": result.answer == task_input["expected"],
        "cost_usd": result.cost,
        "steps": [
            {"kind": "tool_call", "name": call.tool, "ok": call.ok, "args": call.args}
            for call in result.tool_calls
        ],
        "error": result.failure_reason,
    }
```

```yaml
candidate:
  kind: python
  ref: my_evals:candidate
  concurrency: 8
```

`passed` is the only required field. Everything else improves the report:
`cost_usd` drives the budget, `steps` drive the trajectory diff, `error` drives
failure-mode clustering.

### A subprocess

Reads one JSON object on stdin, writes one on stdout. Anything else your program
prints must go to stderr.

```yaml
candidate:
  kind: subprocess
  command: ["python", "run_eval.py"]
  env: {MODEL: "candidate"}
```

### An HTTP endpoint

```yaml
candidate:
  kind: http
  url: https://staging.internal/eval
  headers: {Authorization: "Bearer ${TOKEN}"}
  rate_limit_per_s: 4
```

## Use the seed

This is the part that determines what the gate costs you, and it is easy to skip
by accident.

Pass the seed everywhere your agent uses randomness: the sampling seed, any
shuffling, any tie-break, any synthetic data generation. The more of your
agent's nondeterminism the seed pins down, the more often the two builds behave
identically on a replicate, and the fewer replicates you need.

The report tells you how well you are doing. It prints the share of replicates
where the two builds disagreed. If that number is near zero the two builds are
identical or your target is ignoring the seed. If it is high on tasks you did not
change, the seed is not controlling much, and the gate will cost more.

Measured on the simulated agent, moving from no seed control to tight seed
control cut replicates by a third and took per-task power from 0.88 to 1.00.

## Size the run before paying for it

```bash
arbiter plan --tasks 120 --baseline-rate 0.9 --mde 0.15 --cost 0.03 --seconds 8
```

The second table is the one to act on. It reports the e-value a single task must
reach given your suite size, how many disagreeing replicates that takes, and
therefore what `max_replicates` should be.

Set it too low and the gate cannot flag anything. It will tell you when that
happens rather than passing quietly:

```
note underpowered: flagging one task needs about 37 disagreeing replicates, the
builds disagreed on 27.5% of runs, so about 133 replicates per task.
max_replicates is 120. Raise it, raise alpha, or lower odds_ratio.
```

That one came from running `examples/suite.yaml` with the cap left at 120. Raising
it to 200 was enough, and the note went away.

## Put it in CI

```yaml
- name: eval gate
  run: |
    arbiter gate evals/suite.yaml \
      --store .arbiter/runs.sqlite \
      --markdown $GITHUB_STEP_SUMMARY \
      --junit arbiter-junit.xml
```

Exit codes: `0` passed, `1` regressions found or rejected by policy, `2` the gate
could not run reliably.

Cache `.arbiter/runs.sqlite` between runs. This is not an optimisation to get to
later. Without it every run re-executes the baseline from scratch, which roughly
doubles the bill for no statistical benefit.

There is a complete workflow in [`examples/github-actions.yml`](../examples/github-actions.yml).

## Tuning, in the order worth trying

| symptom | what to change |
|---|---|
| Gate never flags anything | `max_replicates` is too low. Run `arbiter plan`. |
| Costs too much | Improve seed coupling first. Then raise `mde` or `odds_ratio`. |
| Too many tasks come back inconclusive | Raise `max_replicates`, or accept it: with FDR control this is the normal state for a clean build. |
| A regression you care about is missed | Lower `odds_ratio`, or lower `futility_confidence` to stop abandoning tasks early. |
| Small suite, want more sensitivity | `correction: none`. Costs you a false flag on roughly 9% of clean builds. |
| Runs are slow | Raise `concurrency` and `batch_size`. Neither affects the verdict. |

## Check the error rates yourself

Do not take the README's word for it:

```bash
arbiter simulate --trials 100 --tasks 25 --regressed 0
```

`--regressed 0` means nothing actually changed between the builds, so every flag
is a false one. That is the number worth knowing before you let a gate block
anyone's pull request.
