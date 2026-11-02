"""Export a graph thread's tool calls as a trajectory-replay cassette.

A cassette is the ordered list of tool calls an agent made, each with the text it got back.
With one you can replay a run without a live model, and score a new run's trajectory against it
(see https://github.com/RockENZO/trajectory-replay).

The graph engine already logs every call with its id, name, arguments, result, success flag and
the turn it belonged to, so this only selects and converts. Calls the user rejected were never
run and are left out. The trajectory-replay package is an optional dependency
(``pip install 'odyssey[eval]'``).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


def _replay():
    try:
        import trajectory_replay
    except ImportError as e:
        raise RuntimeError("exporting a cassette needs trajectory-replay: pip install 'odyssey[eval]'") from e
    return trajectory_replay


def calls_from_state(state: dict[str, Any], *, turn_id: str | None = None) -> list:
    """The thread's executed tool calls as cassette steps; ``turn_id`` limits them to one turn."""
    records = [e for e in state.get("tool_log", []) if turn_id is None or e.get("turn") == turn_id]
    return _replay().calls_from_records(records)


def save_cassette(state: dict[str, Any], path: Path, *, turn_id: str | None = None) -> int:
    """Write the cassette to ``path`` and return how many calls it holds."""
    calls = calls_from_state(state, turn_id=turn_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    _replay().save(calls, path)
    return len(calls)
