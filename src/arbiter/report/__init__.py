"""Rendering gate results for humans, for pull requests, and for CI."""

from .console import (
    render_console,
    render_diff_console,
    render_paired_plan_console,
    render_plan_console,
)
from .junit import render_junit
from .markdown import render_diff_markdown, render_markdown

__all__ = [
    "render_console",
    "render_diff_console",
    "render_diff_markdown",
    "render_junit",
    "render_markdown",
    "render_paired_plan_console",
    "render_plan_console",
]
