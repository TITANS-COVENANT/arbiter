# How arbiter is put together

## The loop

Everything happens in one loop in [`gate/decide.py`](../src/arbiter/gate/decide.py),
and it is short on purpose:

```
while tasks remain undecided and budget remains:
    picks = allocator.select(task_states, batch_size)     # which tasks next
    assign each pick the next unused replicate index
    seed  = hash(task_id, replicate, salt)                # same seed for both builds
    run baseline and candidate cells concurrently         # cached cells skipped
    feed each complete pair to that task's sequential test
    charge the budget
apply suite-level correction to the collected e-values
emit a verdict and an exit code
```

Two things in that sketch carry most of the design.

**The seed is derived from a hash, not a counter.** Both builds get the same seed
for the same cell, which is what makes the comparison paired. And because the
seed comes from `hash(task_id, replicate, salt)`, adding a task to the suite does
not shift any other task's seeds, so yesterday's cached baseline runs stay valid.
A counter would have invalidated the entire cache every time someone appended a
test case.

**Cached cells are skipped, not re-run.** A pull request changes the candidate,
not the baseline. The baseline's replicate 7 on `refund_flow` is the same run it
was yesterday. Reusing those is usually a bigger saving than the sequential
testing is.

## The pieces

```
config.py            suite YAML -> validated models, and the variant hash
errors.py            the one distinction that matters: task failed vs harness failed

stats/               pure standard library, no third-party imports
  special.py         incomplete beta, normal quantile, written out rather than imported
  sprt.py            Wald's sequential test, e-values, operating characteristic
  paired.py          sequential McNemar on disagreeing replicates
  confseq.py         always-valid confidence sequence for scalar scores
  bayes.py           beta-binomial posterior and P(candidate is worse)
  multiple.py        Benjamini-Hochberg, e-BH, Holm, p-to-e calibration
  power.py           sample-size and cost planning

runner/
  types.py           Task, RunOutcome, Step, and the seed derivation
  adapters.py        python / subprocess / http ways to reach a build
  engine.py          concurrency, rate limiting, retries, timeouts

scheduler/
  allocator.py       which task gets the next replicate
  budget.py          replicate, dollar and wall-clock ceilings

gate/
  task_test.py       one task's evidence, and why it stopped
  decide.py          the loop above, plus correction and the verdict

store/db.py          SQLite: runs, verdicts, gate history, and the reuse cache
diff/                alignment and failure-mode clustering for flagged tasks
report/              console, Markdown, JUnit XML
sim/                 a synthetic agent with known truth, and the Monte Carlo harness
cli.py               plan, gate, diff, history, simulate, init
```

### Why `stats/` has no dependencies

The error guarantees are the product. Someone evaluating whether to trust a
verdict should be able to read the code that produces it in one sitting, without
a numerical stack in the way. So the incomplete beta function is a continued
fraction evaluated by Lentz's method in
[`stats/special.py`](../src/arbiter/stats/special.py) rather than a SciPy import,
and the whole package is testable against known values.

### Task failure versus harness failure

A task that runs and fails is evidence. A timeout, a crashed subprocess or a 503
is not: a flaky network is not a statement about the candidate build. The engine
separates them, and the second kind never reaches a statistical test.

They are counted, though, and if more than `max_infra_error_rate` of attempts
could not be run the gate returns `error` with exit code 2 rather than a verdict.
A gate that goes green because most of its runs never happened is worse than no
gate.

### Stop reasons

Every task ends with one of these, and the distinction drives the verdict:

| reason | meaning |
|---|---|
| `boundary` | the sequential test decided, either way |
| `futility` | it cannot reach the evidence threshold within `max_replicates` |
| `max-replicates` | it used its whole per-task allotment |
| `budget` | a suite-level ceiling cut it short |

The first three are complete answers. `budget` is not, and it is the only one
that triggers the `on_inconclusive` policy. Collapsing "I looked as hard as I said
I would and found nothing" together with "I ran out of money" is how a gate ends
up either warning on every clean build or going quiet on a truncated one.

## Variant identity and cache invalidation

`SuiteConfig.variant_id()` hashes the target definition, the seed salt and the
task list. Change the model, the prompt, the command or a task's input and the
hash moves, so stored runs stop being used.

It deliberately excludes the statistical policy and the budget. Tightening alpha
should reuse the runs you already paid for, not throw them away.

## Extending it

**A new way to reach a build.** Implement the `Target` protocol in
[`runner/adapters.py`](../src/arbiter/runner/adapters.py): one `async def run(task, seed)`
returning a `RunOutcome`, plus `aclose`. Add it to `build_target`.

**A new scheduling policy.** Implement `select(states, batch_size)` from
[`scheduler/allocator.py`](../src/arbiter/scheduler/allocator.py). The states
expose `replicates`, `resolved`, `priority` and `expected_remaining()`. You cannot
break the error guarantee this way, only the cost.

**A new statistical mode.** The contract `gate/decide.py` needs from a task test
is `observe`, `resolved`, `e_value`, `anytime_p`, `priority` and
`expected_remaining()`. Emit a genuine e-value and e-BH will handle it; emit an
anytime-valid p-value and run it through `p_to_e`.

## Concurrency

Both builds run concurrently inside a batch, each behind its own semaphore and
optional token bucket, so a rate-limited baseline does not throttle the candidate.
Retries use exponential backoff with full jitter. Synchronous Python targets are
pushed to a thread so a slow one cannot stall the event loop and starve the other
build.
