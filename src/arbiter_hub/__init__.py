"""arbiter hub: the service your CI reports eval verdicts into.

`arbiter gate` already decides whether a build regressed and prints a verdict on
the pull request. What it cannot do from inside one CI job is answer the
questions that need a history:

* has this task been unreliable for a month, or did it break today?
* is the thing that just went red actually new?
* which tasks cost the most replicates to decide, and would a better seed fix it?

So the hub keeps every run, stores per-task rows you can group across runs, and
separates a task that keeps flipping (a flaky eval, fix or retire it) from one
that is flagged nearly every time (a regression nobody got round to).

Your API keys and your agent never come near it. The gate runs in your CI, and
only the verdict is posted here.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
