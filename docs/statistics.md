# What arbiter guarantees, and what it does not

This is the document to read before trusting a verdict. It says exactly which
claims are backed by a proof, which are backed by a simulation, and which are
heuristics that only affect cost.

## The problem in one paragraph

An agent eval is a random variable. Run task 41 twice against the same build and
you may get a pass and a fail. So when a pull request turns one green task red,
there are two explanations and the test result alone cannot separate them: the
change broke something, or the coin landed differently. Teams resolve this by
running everything N times and eyeballing the delta, which is expensive, or by
re-running until it goes green, which is worse.

## Three separate problems, three separate tools

### 1. Task difficulty swamps the effect

Most of the variance in a suite is between tasks, not between builds. Task 41 is
hard and fails half the time under both builds. Task 12 is easy and passes under
both. If you estimate the candidate's pass rate and the baseline's pass rate
independently and subtract, you pay for that task-level variance twice, and it is
usually much larger than the change you are hunting.

So arbiter runs both builds on the same task at the same seed, and looks only at
the replicates where they disagree. Under the null hypothesis that the change did
nothing, a disagreement is equally likely to fall either way. That is McNemar's
test. Conditioning on the disagreements removes the task's difficulty from the
problem entirely, because it is common to both arms.

The useful consequence: the null becomes an exact Bernoulli(1/2), with no
nuisance parameter. There is no baseline rate to estimate, no plug-in, and
nothing that goes subtly wrong when your estimate of the baseline is off.

Replicates where the two builds agree are discarded. They are not wasted exactly,
since they told you about the task, but they carry no information about the
*difference*, which is the thing being tested.

### 2. A fixed sample size is the wrong shape

Some tasks resolve after four replicates. Some are genuinely borderline and would
absorb the entire budget without ever resolving. A fixed N spends the same on
both.

arbiter runs Wald's sequential probability ratio test on the disagreeing
replicates. After each one it updates a log-likelihood ratio and stops as soon as
the ratio crosses a boundary. The construction is in
[`stats/sprt.py`](../src/arbiter/stats/sprt.py) and
[`stats/paired.py`](../src/arbiter/stats/paired.py).

You have to say how big a regression is worth catching. This is not a limitation
that better statistics would remove: detecting an arbitrarily small effect
requires an arbitrarily large sample, so an honest tool makes you name the
threshold. In arbiter you name it as an odds ratio on disagreements.
`odds_ratio: 3.0` means "flag it when new failures outnumber newly fixed ones
three to one".

### 3. Two hundred tasks means two hundred chances to be wrong

Test 200 tasks at alpha 0.05 and about ten will flag on a build where nothing
changed. This is the failure that actually kills eval gates in practice. People
learn the red is meaningless, and they stop reading it.

arbiter controls the **false discovery rate** across the suite: of the tasks it
flags, at most `alpha` of them are flagged in error, in expectation.

The procedure is **e-BH** (Wang and Ramdas, 2022), applied to e-values rather
than p-values. This matters for a specific reason. Benjamini-Hochberg needs the
p-values to be independent or positively dependent. Tasks in one eval suite share
a model, a prompt template, a tool sandbox and a bad afternoon on someone's API,
so they are dependent in ways nobody can characterise, and BH's precondition is
unverifiable. e-BH controls FDR under **arbitrary** dependence. There is no
assumption left to violate.

The sequential tests emit e-values natively, which is why this composes. The
likelihood ratio of a sequential test *is* an e-value:

- Under the null, the per-replicate expectation of the likelihood ratio is 1.
- So the running product is a non-negative martingale starting at 1.
- Optional stopping gives `E[e_value] <= 1` at any stopping time, which is the
  definition of an e-value.

`tests/test_stats_core.py::test_e_value_expectation_is_bounded_by_one_under_the_null`
checks this by simulation, because if it stops being true the FDR guarantee is
void and nothing else in the tool matters.

## The consequence people are surprised by

Suite-level control raises the bar for every individual task. With `m` tasks at
level `alpha`, the strictest rung of e-BH demands an e-value of `m / alpha`. For
200 tasks at 0.05 that is 4000, not 20.

This has a direct implication for how the gate must be built. **A task cannot be
stopped at its own Wald boundary.** That boundary leaves it holding an e-value
around 18, and correction will then reject it. Stop every task there and the gate
is permanently blind: everything decides, and nothing survives correction.

So the per-task stopping boundary is set from what the correction will demand,
not from the per-task alpha. e-BH's step-up structure can then only ever flag
*more* than that boundary justifies: once several tasks look bad, the bar for each
of them drops. A single regression hiding in 200 tasks is expensive to prove. Six
regressions are cheap, because they corroborate each other.

If your suite is small, or you are willing to trade false alarms for sensitivity,
`correction: none` uses the plain Ville threshold of `1 / alpha`. The benchmark
measures what that costs you: 9.2% of clean builds get at least one false flag.

## What is guaranteed, precisely

**Guaranteed by construction.**

- Per task, the probability of flagging a regression when the change did nothing
  is at most the effective alpha. This follows from Ville's inequality on the
  likelihood-ratio martingale, and it holds at every sample size, so you may look
  at a partial result whenever you like.
- Across the suite, the false discovery rate among flagged tasks is at most
  `alpha`, under arbitrary dependence between tasks.

**Guaranteed only up to the effect size you named.** The miss rate is bounded by
`beta` for a regression at least as large as the odds ratio you configured.
Smaller regressions are missed more often, by design. The benchmark's
`subtle_regression` scenario shows this: at half the declared effect, per-task
power falls to 0.37.

**Not guaranteed, and cannot be.** Absence of a flag is not proof of equivalence.
When arbiter reports "no evidence of a regression", that is the whole claim. Some
tasks do get positively cleared, and those are reported separately as *cleared
outright*, but most clean tasks stop for futility, which means only that no
regression of the size you asked about could have shown up within your budget.

## The heuristics, and why they are safe

Three parts of the system are not backed by a proof. All three affect only cost
or power, never the false-flag rate, and it is worth being explicit about why.

**Futility stopping.** A task is abandoned when a one-sided upper bound on the
disagreement rate says it cannot reach the boundary in its remaining budget. This
can cause a miss, and the cost is bounded: it adds at most
`futility_confidence` to the per-task miss probability. It cannot cause a false
flag, because it never flags anything.

**The allocator.** Which task gets the next replicate is a scheduling decision.
It is safe for the same reason: FDR control constrains the set of *rejections*,
and a schedule cannot manufacture a rejection. A bad schedule wastes money and
loses power.

It can lose more power than you would guess, which is why the default is the
unclever one. Spending greedily on whichever task is closest to deciding sounds
obviously right and measured worse on both cost and power, because on a build
that really did regress, the tasks furthest from deciding are the
regressed-but-not-yet-obvious ones. A greedy schedule deprioritises exactly what
you are hunting. The numbers are in the README table.

**Estimated remaining cost.** Used to rank tasks, computed from Wald's
approximation with a Laplace-smoothed plug-in estimate. Wrong estimates make the
schedule worse, not the verdict.

## Measured behaviour

Numbers from `benchmarks/bench_gate.py`, against a simulated agent whose true
pass rates are known by construction. See [the README](../README.md) for the
table and [`sim/agent.py`](../src/arbiter/sim/agent.py) for how the truth is
generated.

The one result worth restating here: on 200 trials of a build where nothing
changed, across 25 tasks each, the gate raised zero false flags. The theory
allows up to 5%. Getting zero rather than 5% is expected, because e-BH is
conservative and the futility rule stops many tasks before they can drift.

## Scalar scores

Not every eval is pass/fail. For rubric scores and other continuous outcomes,
`mode: score` runs an always-valid confidence sequence on the paired differences,
using Robbins' Gaussian mixture boundary. The interval holds simultaneously at
every sample size, so you can stop whenever it excludes your tolerance without
inflating the error rate.

The price is width: about 1.9x a fixed-sample interval at the sample size the
mixture is tuned for. That is the cost of being allowed to peek, and the law of
the iterated logarithm says it cannot be avoided.

Score-mode evidence reaches the same correction step through a p-to-e calibrator,
`e = kappa * p^(kappa-1)`. Some sharpness is lost in the round trip, which is why
binary mode emits e-values directly instead.

## References

- Wald (1945), *Sequential Tests of Statistical Hypotheses*. The SPRT and its
  operating characteristic.
- McNemar (1947). The paired test on discordant pairs.
- Robbins (1970), *Statistical methods related to the law of the iterated
  logarithm*. The mixture boundary behind the confidence sequence.
- Benjamini and Hochberg (1995). FDR control.
- Howard, Ramdas, McAuliffe, Sekhon (2021), *Time-uniform, nonparametric,
  nonasymptotic confidence sequences*.
- Wang and Ramdas (2022), *False discovery rate control with e-values*. e-BH, and
  the dependence-free guarantee arbiter relies on.
- Vovk and Wang (2021), *E-values: Calibration, combination and applications*.
  The p-to-e calibrator.
