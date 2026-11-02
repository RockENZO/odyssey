"""Workspace boundary for the file and search tools.

The boundary is opt-in at the library level (no root set means no restriction, which keeps
the tools usable from scripts and tests) and enabled by the REPL, which roots the agent at
the project directory. It covers the ``read``, ``write``, ``edit``, ``glob`` and ``grep``
tools. It does NOT sandbox ``bash``: a shell command can still touch any path the user can.
"""
from __future__ import annotations

import os
from pathlib import Path

ALLOW_OUTSIDE_ENV = "ODYSSEY_ALLOW_OUTSIDE_WORKSPACE"

_root: Path | None = None


class WorkspaceError(Exception):
    """A tool tried to reach a path outside the workspace root."""


def set_workspace_root(path: str | os.PathLike[str] | None) -> None:
    """Confine path-taking tools to ``path``. ``None`` removes the restriction."""
    global _root
    _root = Path(path).expanduser().resolve() if path is not None else None


def get_workspace_root() -> Path | None:
    return _root


def resolve_path(path: str) -> str:
    """Return the absolute path for ``path``, or raise if it escapes the workspace.

    Relative paths resolve against the workspace root when one is set, otherwise against
    the process working directory. Symlinks are resolved before the check, so a link that
    points outside the root is rejected.
    """
    expanded = Path(os.path.expanduser(path))
    if _root is None:
        return os.path.abspath(expanded)
    candidate = expanded if expanded.is_absolute() else _root / expanded
    resolved = Path(os.path.realpath(candidate))
    if resolved != _root and _root not in resolved.parents:
        raise WorkspaceError(
            f"path {path!r} is outside the workspace {str(_root)!r}; "
            f"set {ALLOW_OUTSIDE_ENV}=1 to lift the restriction"
        )
    return str(resolved)
