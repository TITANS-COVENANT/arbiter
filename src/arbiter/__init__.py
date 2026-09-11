"""arbiter: a statistical CI gate for noisy agent evals.

The short version of what this package does:

1. Run every eval task under both the baseline and the candidate build, at
   identical seeds, so the comparison is paired and task difficulty cancels out.
2. Watch each task's evidence accumulate and stop that task the moment it has
   decided, instead of running a fixed number of replicates.
3. Spend whatever budget is left on the tasks closest to deciding.
4. Correct across the suite with e-values so that a two-hundred-task run does
   not flag ten regressions by chance.
5. Emit a verdict, an exit code, and a diff of what actually changed.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
