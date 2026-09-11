"""Exception types.

The distinction that matters is between a task failing (evidence, feed it to the
test) and the harness failing (not evidence, do not let it look like a
regression). :class:`TargetError` is the second kind.

Budget exhaustion is deliberately not an exception. A run that ran out of money
still has a partial answer worth reporting, so it comes back as a verdict with
tasks marked incomplete rather than as a raised error.
"""

from __future__ import annotations

__all__ = ["ArbiterError", "ConfigError", "TargetError"]


class ArbiterError(Exception):
    """Base class for everything this package raises on purpose."""


class ConfigError(ArbiterError, ValueError):
    """The suite definition is wrong.

    Also a ValueError, because pydantic raises those for field-level problems and
    a caller should not have to catch two things to validate one file.
    """


class TargetError(ArbiterError):
    """The target could not be run, as opposed to running and failing.

    A timeout, a crashed subprocess, a 503 from the endpoint. These are counted
    and reported separately, and never fed to a statistical test, because a flaky
    network is not evidence that the candidate build got worse.
    """

