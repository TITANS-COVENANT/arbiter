"""Ways to invoke the thing under test.

arbiter does not want to know what your agent is written in. It needs to hand a
target a task and a seed and get back "did it pass", plus whatever trajectory
the target cares to report. Three adapters cover essentially everyone: an
in-process Python callable, a subprocess that speaks JSON, and an HTTP endpoint.
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
import json
import os
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from ..errors import TargetError
from .types import RunOutcome, Task

if TYPE_CHECKING:  # config imports runner.types, so keep this one-way at runtime
    from ..config import TargetConfig

__all__ = ["HttpTarget", "PythonTarget", "SubprocessTarget", "Target", "build_target"]


@runtime_checkable
class Target(Protocol):
    """Anything that can run a task at a given seed."""

    async def run(self, task: Task, seed: int) -> RunOutcome: ...

    async def aclose(self) -> None: ...


def _coerce(result: Any) -> RunOutcome:
    """Accept the several shapes a target might reasonably return."""
    if isinstance(result, RunOutcome):
        return result
    if isinstance(result, bool):
        return RunOutcome(passed=result)
    if isinstance(result, dict):
        return RunOutcome.from_dict(result)
    raise TargetError(
        f"target returned {type(result).__name__}; expected RunOutcome, dict or bool"
    )


class PythonTarget:
    """Calls ``module:callable`` in-process.

    The callable receives ``(task_input, seed)`` and may be sync or async. Sync
    callables are pushed to a thread so that one slow target cannot stall the
    event loop and starve the other build's runs.
    """

    def __init__(self, cfg: TargetConfig) -> None:
        assert cfg.ref is not None
        self.cfg = cfg
        module_name, _, attr = cfg.ref.partition(":")
        if not attr:
            raise TargetError(f"python target ref must be 'module:callable', got '{cfg.ref}'")
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:  # pragma: no cover - depends on user layout
            raise TargetError(f"cannot import '{module_name}': {exc}") from exc
        try:
            self.fn = getattr(module, attr)
        except AttributeError as exc:
            raise TargetError(f"'{module_name}' has no attribute '{attr}'") from exc
        if not callable(self.fn):
            raise TargetError(f"'{cfg.ref}' is not callable")
        self.is_async = inspect.iscoroutinefunction(self.fn)

    async def run(self, task: Task, seed: int) -> RunOutcome:
        kwargs = dict(self.cfg.options)
        if self.is_async:
            result = await self.fn(task.input, seed, **kwargs)
        else:
            result = await asyncio.to_thread(self.fn, task.input, seed, **kwargs)
        return _coerce(result)

    async def aclose(self) -> None:
        return None


class SubprocessTarget:
    """Runs a command, writes the task as JSON on stdin, reads JSON from stdout.

    Anything that is not the final JSON object goes to stderr and is captured
    into the error field, so a target that prints logs to stdout will fail
    loudly rather than corrupting results.
    """

    def __init__(self, cfg: TargetConfig) -> None:
        assert cfg.command is not None
        self.cfg = cfg

    async def run(self, task: Task, seed: int) -> RunOutcome:
        assert self.cfg.command is not None
        payload = json.dumps({"task": task.to_dict(), "seed": seed, **self.cfg.options})
        env = {**os.environ, **self.cfg.env, "ARBITER_SEED": str(seed)}
        proc = await asyncio.create_subprocess_exec(
            *self.cfg.command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        try:
            stdout, stderr = await proc.communicate(payload.encode("utf-8"))
        except asyncio.CancelledError:  # pragma: no cover - cancellation path
            proc.kill()
            raise
        if proc.returncode != 0:
            tail = stderr.decode("utf-8", "replace").strip()[-2000:]
            raise TargetError(f"target exited {proc.returncode}: {tail}")
        text = stdout.decode("utf-8", "replace").strip()
        if not text:
            raise TargetError("target produced no output on stdout")
        try:
            return _coerce(json.loads(text))
        except json.JSONDecodeError as exc:
            raise TargetError(f"target stdout was not JSON: {text[:400]}") from exc

    async def aclose(self) -> None:
        return None


class HttpTarget:
    """Posts the task to an endpoint and reads a JSON body back."""

    def __init__(self, cfg: TargetConfig) -> None:
        assert cfg.url is not None
        import httpx

        self.cfg = cfg
        self._client = httpx.AsyncClient(
            timeout=cfg.timeout_s,
            headers=cfg.headers,
            limits=httpx.Limits(max_connections=max(cfg.concurrency, 1)),
        )

    async def run(self, task: Task, seed: int) -> RunOutcome:
        assert self.cfg.url is not None
        body = {"task": task.to_dict(), "seed": seed, **self.cfg.options}
        response = await self._client.post(self.cfg.url, json=body)
        if response.status_code >= 400:
            raise TargetError(f"target returned HTTP {response.status_code}: {response.text[:400]}")
        try:
            return _coerce(response.json())
        except ValueError as exc:
            raise TargetError(f"target response was not JSON: {response.text[:400]}") from exc

    async def aclose(self) -> None:
        await self._client.aclose()


def build_target(cfg: TargetConfig) -> Target:
    if cfg.kind == "python":
        return PythonTarget(cfg)
    if cfg.kind == "subprocess":
        return SubprocessTarget(cfg)
    return HttpTarget(cfg)
