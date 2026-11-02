"""A conversation thread on the graph engine: submit, approve or reject, resume."""
from __future__ import annotations

import contextlib
import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from odyssey.engine.graph import AgentState, new_turn
from odyssey.engine.policy import ApprovalPolicy


@dataclass
class TurnResult:
    status: str  # running | done | verified | unverified | budget_exhausted | awaiting_approval | idle
    text: str = ""
    approval: list[dict[str, Any]] = field(default_factory=list)  # calls waiting for a decision
    resumed_in_flight: list[dict[str, Any]] = field(default_factory=list)  # calls that may have run before a crash


@contextlib.asynccontextmanager
async def open_checkpointer(path: Path) -> AsyncIterator[AsyncSqliteSaver]:
    path.parent.mkdir(parents=True, exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string(str(path)) as saver:
        yield saver


def new_thread_id() -> str:
    return uuid.uuid4().hex[:12]


class GraphSession:
    def __init__(self, graph: Any, thread_id: str, policy: ApprovalPolicy, workspace: str = "", base: str = "", **limits: int) -> None:
        self.graph = graph
        self.thread_id = thread_id
        self.policy = policy
        self.workspace = workspace
        self.base = base
        self.limits = limits
        self._config = {"configurable": {"thread_id": thread_id}}

    def new_thread(self) -> str:
        """Start a fresh conversation; the old thread stays in the checkpoint file."""
        self.thread_id = new_thread_id()
        self._config = {"configurable": {"thread_id": self.thread_id}}
        return self.thread_id

    def _check_workspace(self, snapshot) -> None:
        """Unfinished work belongs to the directory it was started in; do not run it in another.

        The project root alone is not enough: a relative path in a pending call means a different
        file when Odyssey is launched from another subdirectory of the same repository.
        """
        recorded_root = snapshot.values.get("workspace", "")
        recorded_base = snapshot.values.get("base", "")
        moved_root = self.workspace and recorded_root and recorded_root != self.workspace
        moved_base = self.base and recorded_base and recorded_base != self.base
        if moved_root or moved_base:
            raise RuntimeError(
                f"this thread was started in {recorded_base or recorded_root}, but Odyssey is running in "
                f"{self.base or self.workspace}; start Odyssey from the original directory or begin a new thread with /clear"
            )

    async def _snapshot(self):
        return await self.graph.aget_state(self._config)

    @staticmethod
    def _interrupts(snapshot) -> list[dict[str, Any]]:
        calls: list[dict[str, Any]] = []
        for task in snapshot.tasks:
            for item in task.interrupts:
                calls.extend(item.value.get("calls", []))
        return calls

    async def pending_approval(self) -> list[dict[str, Any]]:
        return self._interrupts(await self._snapshot())

    async def state(self) -> dict[str, Any]:
        return dict((await self._snapshot()).values)

    def _result(self, snapshot) -> TurnResult:
        approval = self._interrupts(snapshot)
        if approval:
            return TurnResult("awaiting_approval", approval=approval)
        values = snapshot.values
        return TurnResult(values.get("status", "idle"), text=values.get("final", ""))

    async def submit(self, text: str) -> TurnResult:
        snapshot = await self._snapshot()
        self._check_workspace(snapshot)
        if self._interrupts(snapshot):
            raise RuntimeError("a tool call is waiting for approval: approve or reject it first")
        if snapshot.next:
            raise RuntimeError("the previous turn did not finish: resume it first")
        await self.graph.ainvoke(new_turn(text, workspace=self.workspace, base=self.base, **self.limits), self._config, durability="sync")
        return self._result(await self._snapshot())

    async def decide(self, approve: bool, reason: str = "", always: bool = False) -> TurnResult:
        waiting = await self.pending_approval()
        if not waiting:
            raise RuntimeError("nothing is waiting for approval")
        self._check_workspace(await self._snapshot())
        if approve and always:
            for call in waiting:
                self.policy.allow(call["name"])
        await self.graph.ainvoke(Command(resume={"approve": approve, "reason": reason}), self._config, durability="sync")
        return self._result(await self._snapshot())

    async def resume(self) -> TurnResult:
        """Continue an unfinished turn from its last checkpoint (for example after a crash)."""
        snapshot = await self._snapshot()
        if not snapshot.next:
            return TurnResult("idle", text=snapshot.values.get("final", ""))
        self._check_workspace(snapshot)
        if self._interrupts(snapshot):
            return self._result(snapshot)
        in_flight = list(snapshot.values.get("pending", [])) if "tools" in snapshot.next else []
        await self.graph.ainvoke(None, self._config, durability="sync")
        result = self._result(await self._snapshot())
        result.resumed_in_flight = in_flight[:1]
        return result


async def unfinished(graph: Any, thread_id: str) -> bool:
    snapshot = await graph.aget_state({"configurable": {"thread_id": thread_id}})
    return bool(snapshot.next)


def default_db_path() -> Path:
    from odyssey.config import get_data_dir
    return get_data_dir() / "checkpoints.sqlite"


def last_thread_file() -> Path:
    """JSON map from launch directory to the last thread started there."""
    from odyssey.config import get_data_dir
    return get_data_dir() / "last_threads.json"


def recall_thread(base: str) -> str:
    """The last thread started from ``base`` ('' if there is none)."""
    try:
        return str(json.loads(last_thread_file().read_text()).get(base, ""))
    except (OSError, ValueError, AttributeError):
        return ""


def remember_thread(base: str, thread_id: str) -> None:
    """Record ``thread_id`` as the last thread of ``base``; other directories' threads are kept."""
    path = last_thread_file()
    try:
        threads = json.loads(path.read_text())
        if not isinstance(threads, dict):
            threads = {}
    except (OSError, ValueError):
        threads = {}
    threads[base] = thread_id
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(threads, indent=2))
