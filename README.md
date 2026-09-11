# arbiter

**A statistical CI gate for noisy agent evals.**

[![ci](https://github.com/TITANS-COVENANT/arbiter/actions/workflows/ci.yml/badge.svg)](https://github.com/TITANS-COVENANT/arbiter/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.11%2B-blue)](pyproject.toml)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A task that passed yesterday is red today. You re-run it and it goes green. Was
that your change, or was that the coin?

Nothing in the test output answers that. An agent eval is a random variable
wearing the costume of an assertion, and the usual ways of coping with that are
all bad in different ways. Trust a single run and you will block clean pull
requests about as often as you catch real breakage. Run everything fifty times
instead and you are paying for it on every push, most of which changed nothing.
The third option, re-running until it goes green, is how regressions reach
production.

Scale makes it worse. At a 5% threshold, a two-hundred-task suite will light up
about ten tasks on a build where nothing changed at all. It takes people roughly a
week to work out that the red means nothing, and after that the gate is
decoration.

arbiter answers the question properly. Both builds run each task at the same seed.
Each task stops as soon as its own evidence is decisive, so an obvious break costs
a handful of runs. And the flagging decision is made across the whole suite at
once, which is what keeps two hundred tasks from crying wolf.

Measured against a simulated agent whose true pass rates are known by
construction, 25 tasks per suite:

| | |
|---|---|
| Clean build, 200 trials | **zero** false flags |
| One task in 25 regressed, 120 trials | caught on **98%** of builds |
| Cost against a fixed-sample design at the same guarantee | **40% fewer runs** |
| The same suite with correction switched off | 9.2% of clean builds flag something |

Every number in this README comes out of `benchmarks/bench_gate.py`, which needs
no API key because the agent under test is the simulated one. Run it and check.
The full table is [further down](#measured).

---

## Try it

The repository ships a small retrieval agent whose candidate build has a real bug
in it, so there is something to find before you wire up anything of your own.

```bash
git clone https://github.com/TITANS-COVENANT/arbiter && cd arbiter
pip install -e .
arbiter gate examples/suite.yaml
```

```
┌───────────────────────────── arbiter · northwind-qa ──────────────────────────────┐
│ Regressions found                                                                 │
│ 3 flagged / 6 tasks  ·  576 replicates run  ·  $6.20  ·  34s                      │
└───────────────────────────────────────────────────────────────────────────────────┘
Flagged as regressed                                                                 
┌──────────────────┬──────────┬───────────┬───────┬───────┬──────┬─────────┬────────┐
│ task             │ baseline │ candidate │ broke │ fixed │ reps │ e-value │  adj p │
├──────────────────┼──────────┼───────────┼───────┼───────┼──────┼─────────┼────────┤
│ ceo_tenure       │      94% │        0% │    15 │     0 │   16 │     438 │ 0.0103 │
│ revenue_per_head │      88% │        0% │    14 │     0 │   16 │     292 │ 0.0103 │
│ current_ceo      │      95% │       36% │    14 │     1 │   22 │     146 │ 0.0137 │
└──────────────────┴──────────┴───────────┴───────┴───────┴──────┴─────────┴────────┘
cleared outright             3
no evidence of a regression  0
```

Then ask what changed:

```bash
arbiter diff examples/suite.yaml ceo_tenure
```

```
┌─────────────────────────────────── ceo_tenure ────────────────────────────────────┐
│ baseline 15/16 (94%)   candidate 0/16 (0%)                                        │
└───────────────────────────────────────────────────────────────────────────────────┘
Failure modes                                                      
┌──────────┬───────────┬──────────────────────────────────────────┐
│ baseline │ candidate │ mode                                     │
├──────────┼───────────┼──────────────────────────────────────────┤
│        0 │        11 │ missing supporting fact: ceo (new)       │
│        0 │         5 │ missing supporting fact: ceo_start (new) │
│        1 │         0 │ answer did not match expected value      │
└──────────┴───────────┴──────────────────────────────────────────┘
Replicate 0 (baseline passed, candidate failed, same seed)
  = plan
  = search (different arguments)
  = read (different arguments)
  ~ message:answer:ok -> error:incomplete_context:err
```

Two failure modes the candidate invented, and the step where the run went wrong.

Run it a second time and it reuses every stored replicate, finishing in under a
second without touching the target at all.

## What it is doing

There are three problems tangled up in that question, and they need different
answers. The full treatment, including which claims are proved and which are only
measured, is in [docs/statistics.md](docs/statistics.md).

### Compare the builds on the same seed, and look only where they disagree

Most of the variance in an eval suite is between tasks, not between builds. Task
41 is hard and fails half the time under both builds. Task 12 is easy and passes
under both. Estimate each build's pass rate separately and subtract, and you have
paid for that task-level noise twice, usually swamping the effect you were looking
for.

So both builds run the same task at the same seed, and only the replicates where
they disagree count. Under the hypothesis that your change did nothing, a
disagreement is a fair coin. That is McNemar's test, and conditioning on the
disagreements removes task difficulty from the problem because it was common to
both.

A useful side effect: the null becomes an exact Bernoulli(1/2). There is no
baseline pass rate to estimate and no plug-in that quietly goes wrong when your
estimate is off.

### Stop each task when it has decided, not after N runs

Some tasks resolve in four replicates. Some are genuinely borderline and would eat
the whole budget without ever resolving. A fixed sample size spends the same on
both.

arbiter runs Wald's sequential probability ratio test on the disagreeing
replicates and stops at a boundary crossing. A task that cannot reach
significance within its cap is abandoned early, which is where most of the saving
on a clean build comes from.

In the run above, the three broken tasks resolved in 16, 16 and 22 replicates.
The healthy ones needed 62, 74 and 98 before they could be positively cleared.
Finding breakage is cheap; proving the absence of it is not. For something
standing between a developer and a merge, that is the right way round.

### Correct across the suite with e-values, not p-values

arbiter bounds the **false discovery rate**: of the tasks it flags, at most
`alpha` are flagged in error.

The procedure is [e-BH](https://arxiv.org/abs/2009.02824) (Wang and Ramdas, 2022),
which takes e-values instead of p-values. This is not decoration. Benjamini-Hochberg
needs the p-values to be independent or positively dependent, and tasks in one
suite share a model, a prompt template, a tool sandbox and a bad afternoon on
someone's API. They are dependent in ways nobody can characterise, so BH's
precondition is unverifiable. e-BH holds under arbitrary dependence, which leaves
no assumption to get wrong.

The sequential tests emit e-values natively, because a likelihood ratio under the
null is a non-negative martingale starting at 1, and optional stopping gives
exactly the property e-BH needs. `tests/test_stats_core.py` checks that by
simulation, because if it stops holding the guarantee is void and nothing else in
the tool matters.

### The consequence worth knowing before you tune anything

Suite-level control raises the bar for each task. With 200 tasks at `alpha=0.05`,
the strictest rung of e-BH wants an e-value of 4000, not 20.

Which means a task cannot be stopped at its own Wald boundary. That leaves it
holding an e-value around 18, and the correction step then discards it. Do that to
every task and the gate goes permanently blind, because everything decides and
nothing survives. So the per-task stopping boundary is set from what the
correction is going to demand. The step-up structure can only ever flag more than
that boundary already justifies, which is why a lone regression buried in 200
tasks is expensive to prove while six of them are cheap: each one makes the others
easier to believe.

This is why `max_replicates` matters more than it looks, and why `arbiter plan`
tells you what to set it to. If the cap is too low to ever flag anything, the run
says so rather than passing quietly:

```
note underpowered: flagging one task needs about 37 disagreeing replicates, the
builds disagreed on 27.5% of runs, so about 133 replicates per task.
max_replicates is 120. Raise it, raise alpha, or lower odds_ratio.
```

## Install

Not on PyPI yet, so install from a checkout as above, or straight from git:

```bash
pip install "arbiter-eval @ git+https://github.com/TITANS-COVENANT/arbiter"
```

Python 3.11 or newer. The statistics package has no third-party dependencies at
all; the CLI pulls in typer, rich, pydantic, pyyaml and httpx.

## Wire up your agent

A target takes a task and a seed and says whether it passed. In-process Python,
a subprocess speaking JSON, or an HTTP endpoint.

```python
def candidate(task_input: dict, seed: int) -> dict:
    result = run_my_agent(task_input["prompt"], seed=seed)
    return {
        "passed": result.answer == task_input["expected"],
        "cost_usd": result.cost,
        "error": result.failure_reason,
        "steps": [
            {"kind": "tool_call", "name": c.tool, "ok": c.ok, "args": c.args}
            for c in result.tool_calls
        ],
    }
```

`passed` is the only field required. `cost_usd` feeds the budget, `error` feeds
failure-mode clustering, and `steps` feed the trajectory diff.

**Use the seed everywhere your agent is random.** Not just the sampling
temperature, but anything that shuffles, breaks a tie, or generates a fixture.
This is the biggest single lever on what the gate costs you and the easiest thing
to skip without noticing. On the simulator, going from an unseeded target to a
tightly seeded one cut replicates by a third and took per-task power from 0.87 to
1.00. Every run reports the share of replicates where the two builds disagreed, so
you can see where you stand.

Full walkthrough in [docs/quickstart.md](docs/quickstart.md).

## In CI

```yaml
- name: eval gate
  run: |
    arbiter gate evals/suite.yaml \
      --store .arbiter/runs.sqlite \
      --markdown $GITHUB_STEP_SUMMARY \
      --junit arbiter-junit.xml
```

Exit codes: `0` passed, `1` regressions found or rejected by policy, `2` the gate
could not run reliably. That third one exists because a gate that goes green
because most of its runs never happened is worse than no gate, so a run whose
harness failed more than `max_infra_error_rate` of the time refuses to give a
verdict at all.

Cache `.arbiter/` between runs. This is not a nice-to-have. Your pull request
changes the candidate, not the baseline, so the baseline's replicate 7 on
`refund_flow` is the same run it was yesterday. Without the cache every run
re-executes the baseline from scratch and roughly doubles the bill for no
statistical benefit. Seeds are derived from `hash(task_id, replicate, salt)`
rather than a counter precisely so that adding a task to your suite does not
invalidate every stored run.

A complete workflow, including PR comments and annotated failures, is in
[examples/github-actions.yml](examples/github-actions.yml).

## Commands

| | |
|---|---|
| `arbiter plan` | What will this cost, and what should `max_replicates` be |
| `arbiter gate SUITE` | Run the comparison, set an exit code |
| `arbiter diff SUITE TASK` | What changed in one task, with a trajectory alignment |
| `arbiter history SUITE` | What previous runs decided |
| `arbiter simulate` | Check the error rates against an agent with known truth |
| `arbiter init` | Write a starter suite file |

## Configuration

```yaml
stats:
  alpha: 0.05               # false discovery rate across the suite
  beta: 0.10                # miss rate for a regression of the declared size
  mde: 0.15                 # smallest pass-rate drop worth catching (planning)
  odds_ratio: 3.0           # flag when new failures outnumber fixes 3:1
  min_replicates: 4         # warmup before the scheduler starts being clever
  max_replicates: 200       # per-task cap; run `arbiter plan` to size it
  correction: e-bh          # e-bh | bh | holm | none
  futility_confidence: 0.01 # how cautiously to abandon hopeless tasks
  mode: binary              # binary | score

budget:
  batch_size: 32
  max_cost_usd: 25.0
  max_seconds: 900

gate:
  allocator: round-robin         # round-robin | cheapest-to-close | successive-halving
  on_inconclusive: warn          # pass | warn | fail
  max_infra_error_rate: 0.25
```

For continuous outcomes, `mode: score` runs an always-valid confidence sequence on
the paired score differences instead, using Robbins' mixture boundary. You can
watch that interval and stop whenever it excludes your tolerance without inflating
the error rate. It is about 1.9x wider than a fixed-sample interval, which is the
price of being allowed to peek.

### Tuning, in the order worth trying

| symptom | change |
|---|---|
| Never flags anything | `max_replicates` is too low. Run `arbiter plan`. |
| Costs too much | Improve seed coupling first, then raise `mde` or `odds_ratio`. |
| Everything comes back inconclusive | Raise `max_replicates`, or accept it: on a clean build under FDR control this is normal. |
| Misses a regression you care about | Lower `odds_ratio`, or lower `futility_confidence`. |
| Small suite, want sensitivity | `correction: none`, at the cost measured in the table above. |
| Too slow | Raise `concurrency` and `batch_size`. Neither touches the verdict. |

## Measured

| scenario | trials | FDR | clean builds falsely flagged | regressions caught | replicates | vs. fixed sample |
|---|---|---|---|---|---|---|
| Nothing changed between the builds | 200 | 0.000 | 0.0% | n/a | 2,711 | +40% |
| One task in twenty-five got worse | 120 | 0.000 | 0.0% | 98% of tasks, 98% of builds | 2,775 | +38% |
| Three tasks in twenty-five got worse | 120 | 0.000 | 0.0% | 99% of tasks, 100% of builds | 2,894 | +36% |
| Three tasks slipped from 0.90 to 0.80, half the usual effect | 80 | 0.000 | 0.0% | 37% of tasks, 61% of builds | 3,095 | +31% |
| Allocator: equal spend across every task | 60 | 0.000 | 0.0% | 99% of tasks, 100% of builds | 2,900 | +36% |
| Allocator: spend where a decision is closest | 60 | 0.000 | 0.0% | 96% of tasks, 100% of builds | 3,029 | +33% |
| Allocator: successive halving | 60 | 0.000 | 0.0% | 98% of tasks, 100% of builds | 3,094 | +31% |
| Seeds ignored by the target: no coupling between builds | 60 | 0.004 | 1.7% | 87% of tasks, 100% of builds | 3,500 | +22% |
| Target honours seeds tightly: 90% coupling | 60 | 0.000 | 0.0% | 100% of tasks, 100% of builds | 2,336 | +48% |
| No suite-level correction, nothing changed between builds | 120 | 0.092 | 9.2% | n/a | 4,346 | -77% |
| Benjamini-Hochberg on anytime p-values, nothing changed | 120 | 0.000 | 0.0% | n/a | 2,725 | +39% |

Read the bottom four rows rather than the top ones. They are where the surprises
are.

**Correction is not optional.** Switch it off and 9.2% of clean builds flag
something. That is a false alarm every eleven pushes, which is about how long it
takes a team to stop reading the output. It is also the cheapest row in the
table, which is the trap: turning correction off looks like a 77% saving right up
to the point where nobody believes the gate.

**Seeding is the biggest lever you actually control.** A target that ignores its
seed needs 3,500 replicates and catches 87% of regressed tasks. One that honours
it needs 2,336 and catches 100%. Same statistics, better inputs, and the only cost
is threading a parameter through your agent.

**The clever allocator lost.** Spending greedily on whichever task is closest to
deciding used 3,029 replicates at 0.956 power, against 2,900 at 0.994 for plain
round-robin. So round-robin is the default. The reason is worth knowing before
you switch it: a greedy schedule defers the tasks furthest from a decision, and on
a build that really did regress, those are the not-yet-obvious regressions. It
deprioritises what you are hunting. Greedy still wins when per-task costs differ a
lot, or when a budget ceiling is going to cut the run short anyway.

**Power runs out below the effect you declared.** At half the configured effect,
per-task power falls to 0.37 and only 61% of bad builds get blocked. Nothing is
broken here; you asked it to detect a 0.15 drop and this is a 0.10 drop. Declare a
smaller `mde` and pay for it.

One caveat over all of this. These come from a simulated agent, so what they
measure is whether the implementation delivers what the theory promises, which is
the part that can break quietly. They are not a claim about your suite. Run
`arbiter simulate --regressed 0` with your own task count and error budget before
letting this block anyone's pull request.

## What it does not do

**Prove two builds are equivalent.** When arbiter reports "no evidence of a
regression", that is the entire claim. Some tasks do get positively cleared and
those are counted separately, but most clean tasks stop for futility, which means
only that a regression of the size you asked about could not have shown up within
your budget.

**Catch regressions smaller than the one you declared.** At half the declared
effect, per-task power in the table above falls to around a third. Detecting an
arbitrarily small effect needs an arbitrarily large sample, so you have to name
the threshold and live with it.

**Decide whether a flagged task matters.** It tells you the pass rate moved, by
how much, and where the trajectories diverged. Whether that is acceptable is your
call.

**Grade your agent.** You supply the pass/fail or the score. arbiter has no
opinion about what good looks like, only about whether it moved.

## Repository layout

```
src/arbiter/
  stats/       sequential tests, confidence sequences, FDR control (stdlib only)
  runner/      target adapters, concurrency, retries, seeds
  scheduler/   replicate allocation and budget ceilings
  gate/        per-task decisions and the verdict
  store/       SQLite run store and the baseline reuse cache
  diff/        trajectory alignment and failure-mode clustering
  report/      console, Markdown, JUnit XML
  sim/         synthetic agent with known truth, and the Monte Carlo harness
docs/          statistics, architecture, quickstart
benchmarks/    the script every number above came from
```

[docs/architecture.md](docs/architecture.md) covers the loop and how to extend it.

## Development

```bash
pip install -e ".[dev]"
pytest -q                                    # 222 tests, about 45s
ruff check src tests benchmarks examples
mypy
python benchmarks/bench_gate.py --scale 0.15 # quick pass
```

Several tests are Monte Carlo against the simulator and check the guarantees
rather than the plumbing. See [CONTRIBUTING.md](CONTRIBUTING.md) before changing
anything in `stats/` or `gate/`.

## Reading

- Wald (1945), *Sequential Tests of Statistical Hypotheses*
- McNemar (1947), on discordant pairs
- Robbins (1970), on the mixture boundary behind the confidence sequence
- Benjamini and Hochberg (1995), on false discovery rate control
- [Howard, Ramdas, McAuliffe, Sekhon (2021)](https://arxiv.org/abs/1810.08240),
  *Time-uniform, nonparametric, nonasymptotic confidence sequences*
- [Wang and Ramdas (2022)](https://arxiv.org/abs/2009.02824), *False discovery
  rate control with e-values*

## License

MIT
