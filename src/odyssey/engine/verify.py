"""Completion gate: "I'm done" is only a claim until a deterministic check passes.

``ledger_gate`` returns an ``Acceptance`` function for the graph engine. When the model
answers after changing files, the gate

1. records a ``task_complete`` claim in a transactional-agent-ledger database, with the
   tool calls of this turn as provenance (their ids match the checkpoint's tool log);
2. runs a verifier on the project, usually the test suite;
3. marks the claim committed or rejected with the verifier's reason.

On failure the graph hands the reason (the tail of the command's output) back to the model for
a bounded number of repair rounds; if it still fails the turn ends ``unverified``.

What this does not do: the check is not sandboxed, a verifier only checks what its command
checks (an agent that can edit the tests can make them pass; pass an ``ExpectedStateVerifier``
as ``extra`` to pin them), and a failed check does not roll the project back. The ledger's own
``run_step`` can snapshot and restore a workspace, but it copies the whole directory, which is not
safe to do to a real repository on every turn.

The ledger is an optional dependency (``pip install 'odyssey[verify]'``).
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Sequence

from odyssey.engine.graph import Acceptance

# Calls that can change the project. ``bash`` counts because it may.
CHANGING_TOOLS = frozenset({"bash", "write", "edit"})


def ledger_gate(
    argv: Sequence[str],
    *,
    ledger_path: Path,
    timeout: float = 300.0,
    extra: Any = None,
) -> Acceptance:
    """Build the acceptance check. ``extra`` is another ledger verifier to require as well."""
    try:
        from agent_ledger import AllOf, ClaimStatus, CommandVerifier, Ledger, Verdict
    except ImportError as e:  # pragma: no cover - exercised by the CLI test with the module hidden
        raise RuntimeError("--verify needs the ledger: pip install 'odyssey[verify]'") from e

    verifier: Any = CommandVerifier(list(argv), timeout=timeout)
    if extra is not None:
        verifier = AllOf(verifier, extra)

    def check(workspace: Path, claim: tuple) -> Any:
        # The ledger's SQLite connection belongs to the thread that opens it, so open it here.
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        ledger = Ledger(ledger_path)
        try:
            ids = ledger.add_claims([claim])
            try:
                verdict = verifier.check(workspace, ledger.get_many(ids))
            except Exception as exc:  # a crashing verifier is a failed check, as in the ledger's run_step
                verdict = Verdict(False, f"verifier error: {exc}")
            ledger.finalize(ids, ClaimStatus.COMMITTED if verdict.ok else ClaimStatus.REJECTED, verdict.reason)
            return verdict
        finally:
            ledger.close()

    async def accept(state: dict[str, Any]) -> tuple[bool, str]:
        turn = state.get("turn_id")
        calls = [e for e in state.get("tool_log", []) if e.get("turn") == turn and not e.get("rejected")]
        if not any(e["name"] in CHANGING_TOOLS for e in calls):
            return True, "no files were changed this turn, so there was nothing to check"
        claim = (
            "task_complete",
            {"answer": (state.get("final") or "")[:2000], "check": list(argv)},
            {
                "turn": turn,
                "tool_calls": [{"id": e["call_id"], "name": e["name"], "ok": e["ok"]} for e in calls],
            },
        )
        verdict = await asyncio.to_thread(check, Path(state.get("workspace") or "."), claim)
        return verdict.ok, verdict.reason

    return accept
