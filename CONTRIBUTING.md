# Contributing

## Setup

```bash
python -m pip install -e ".[dev]"
pytest -q
ruff check src tests benchmarks examples
mypy
```

The full test suite takes about 45 seconds. Several tests are Monte Carlo and use
fixed seeds, so they are deterministic, but they are not fast.

## The one rule that matters

`src/arbiter/stats/` imports nothing outside the standard library, and it should
stay that way. The error guarantees are what the project is for, and someone
deciding whether to trust a verdict needs to be able to read the code that
produces it without a numerical stack in the way. If you need a special function,
write it and test it against known values, as
[`special.py`](src/arbiter/stats/special.py) does for the incomplete beta.

## Changing anything statistical

Changes to `stats/` or `gate/` need a test that would fail if the guarantee
broke, not just one that shows the code runs. The existing pattern is Monte Carlo
against a known truth:

- `test_false_positive_rate_stays_under_alpha` drives a stream at the null rate
  and counts flags.
- `test_e_value_expectation_is_bounded_by_one_under_the_null` checks the
  martingale property that e-BH depends on. If that one fails, the suite-level
  FDR guarantee is void and nothing else in the tool means anything.
- `test_e_bh_controls_fdr_under_dependence` builds a suite with a shared shock so
  the tasks are genuinely dependent, then measures the realised false discovery
  rate.

If you are adding a stopping rule or a scheduling policy, note that neither can
break type-I error control: FDR control constrains the set of rejections, and a
schedule cannot manufacture a rejection. Say so in the docstring, and test the
cost rather than the correctness.

## Benchmarks

```bash
python benchmarks/bench_gate.py --out benchmarks/results.json
python benchmarks/bench_gate.py --scale 0.15    # a quick pass while iterating
```

Every number in the README comes from this script. If you change the allocator,
the futility rule or the stopping boundary, re-run it and update the table,
including when the result is worse. The per-batch allocation cap in
[`allocator.py`](src/arbiter/scheduler/allocator.py) exists because removing it
measured 19% more expensive, and that is the kind of thing the table is for.

## Style

Ruff settings are in `pyproject.toml`, line length 100. Beyond that:

Comments should say why, not what. There are several places in this codebase
where the obvious implementation is wrong in a way that is not obvious, such as
the seed being derived from a hash rather than a counter so that adding a task
does not invalidate every cached baseline run. Those are worth a sentence. Code
that reads plainly is not.
