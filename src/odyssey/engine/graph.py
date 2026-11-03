"""The agent as a LangGraph state machine.

    START -> agent -> gate -> approve -> tools -> (tools | agent)
    agent -> verify -> (agent | END)      (when the model answered without calling a tool)

* ``agent`` asks the model for its next move.
* ``gate`` decides which pending calls need approval under the current policy and records them in
  the state (``awaiting``). Doing this in its own step means the question that was asked is saved
  with the checkpoint: a resumed thread answers that question, even if the policy has changed since.
* ``approve`` pauses with ``interrupt`` on the saved ``awaiting`` calls.
* ``tools`` runs ONE pending call per step, so the checkpointer records progress after every call.
* ``verify`` runs an optional acceptance check on the model's final answer and sends a failure
  reason back to the model for a bounded number of repair rounds.

Resume semantics (important): after a crash the graph restarts from the last checkpoint. A tool
call that was running when the process died has no result recorded, so it runs again. Read-only
tools are unaffected; for ``bash``/``write``/``edit`` the call may already have had an effect.
Checkpoints save graph state, not files; undoing file changes is the job of a workspace
transaction (the ledger integration, a later stage).
"""
from __future__ import annotations

import operator
import time
import uuid
from typing import Annotated, Any, Awaitable, Callable, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from odyssey.core.tool_calls import parse_tool_calls
from odyssey.core.tool_loop import MAX_TOOL_RESULT_CHARS, SYSTEM_PROMPT, blocked_tool
from odyssey.engine.policy import ApprovalPolicy
from odyssey.engine.todos import format_todos, parse_todos
from odyssey.tools.registry import get_tool

ChatFn = Callable[[list[dict[str, Any]]], Awaitable[dict[str, Any]]]
# Acceptance check on the final state: returns (ok, reason). ``reason`` is shown to the model on failure.
Acceptance = Callable[[dict[str, Any]], Awaitable[tuple[bool, str]]]

MAX_LOGGED_RESULT_CHARS = 20000


class AgentState(TypedDict, total=False):
    messages: Annotated[list[dict[str, Any]], operator.add]
    tool_log: Annotated[list[dict[str, Any]], operator.add]
    awaiting: list[dict[str, Any]]  # calls the person was asked about; fixed when the question is asked
    approved: list[str]  # ids approved for the CURRENT batch of calls; replaced with every new batch
    todos: list[dict[str, str]]
    pending: list[dict[str, Any]]
    model_calls: int
    tool_calls_made: int
    max_model_calls: int
    max_tool_calls: int
    repairs: int
    max_repairs: int
    status: str  # running | done | verified | unverified | budget_exhausted
    final: str
    verification: dict[str, Any]
    workspace: str  # project root the thread was started in; resume refuses a different one
    turn_id: str  # one per user turn; tool-log entries carry it so a check can find this turn's calls
    base: str  # directory Odyssey was launched from: relative tool paths resolve against it


def new_turn(user_input: str, *, max_model_calls: int = 25, max_tool_calls: int = 40, max_repairs: int = 2, workspace: str = "", base: str = "") -> dict[str, Any]:
    """The state update that starts a new user turn on an existing or new thread."""
    return {
        "workspace": workspace,
        "base": base,
        "turn_id": uuid.uuid4().hex[:8],
        "approved": [],
        "messages": [{"role": "user", "content": user_input}],
        "pending": [],
        "model_calls": 0,
        "tool_calls_made": 0,
        "repairs": 0,
        "max_model_calls": max_model_calls,
        "max_tool_calls": max_tool_calls,
        "max_repairs": max_repairs,
        "status": "running",
        "final": "",
        "verification": {},
    }


def _model_messages(state: AgentState) -> list[dict[str, Any]]:
    system = SYSTEM_PROMPT + "- Use investigate to ask up to four independent read-only questions about the code at once; it cannot change anything\n"
    if state.get("todos"):
        system += "\nCurrent plan:\n" + format_todos(state["todos"])
    return [{"role": "system", "content": system}, *state.get("messages", [])]


def build_graph(
    chat: ChatFn,
    *,
    checkpointer: Any,
    policy: ApprovalPolicy | None = None,
    acceptance: Acceptance | None = None,
):
    policy = policy or ApprovalPolicy()
    from odyssey.engine.investigate import register_investigate

    register_investigate(chat)

    async def agent(state: AgentState) -> dict[str, Any]:
        if state.get("model_calls", 0) >= state.get("max_model_calls", 25):
            text = "Stopped: the model-call budget for this turn was used up. Break the request into smaller steps."
            return {"messages": [{"role": "assistant", "content": text}], "status": "budget_exhausted", "final": text, "pending": []}
        response = await chat(_model_messages(state))
        msg = response["message"]
        content = msg.get("content", "") or ""
        calls = parse_tool_calls(msg.get("tool_calls"))
        update: dict[str, Any] = {"model_calls": state.get("model_calls", 0) + 1}
        if not calls:
            update.update(messages=[{"role": "assistant", "content": content}], pending=[], final=content)
        else:
            update.update(
                messages=[{"role": "assistant", "content": "", "tool_calls": [c.to_message() for c in calls]}],
                pending=[{"id": c.id, "name": c.name, "arguments": c.arguments} for c in calls],
                # An approval covers one batch. A later response may reuse an id for a different call.
                approved=[],
            )
        return update

    def after_agent(state: AgentState) -> str:
        if state.get("status") == "budget_exhausted":
            return END
        return "gate" if state.get("pending") else "verify"

    async def gate(state: AgentState) -> dict[str, Any]:
        already = set(state.get("approved", []))
        needs = [c for c in state.get("pending", []) if c["id"] not in already and policy.needs_approval(c)]
        return {"awaiting": needs}

    def after_gate(state: AgentState) -> str:
        return "approve" if state.get("awaiting") else "tools"

    async def approve(state: AgentState) -> dict[str, Any]:
        # LangGraph re-runs this node from the top when the interrupt is resumed, so everything
        # before ``interrupt`` must be free of side effects. ``awaiting`` comes from the checkpoint,
        # not from the policy, so a changed policy cannot make this node skip a saved question.
        needs = state.get("awaiting", [])
        already = set(state.get("approved", []))
        decision = interrupt({"type": "approval", "calls": needs})
        if decision.get("approve"):
            return {"approved": sorted(already | {c["id"] for c in needs}), "awaiting": []}
        reason = decision.get("reason") or "no reason given"
        rejected = {c["id"] for c in needs}
        messages, log = [], []
        for c in needs:
            text = f"Rejected by the user: {reason}. The call was not run."
            messages.append({"role": "tool", "content": text, "tool_name": c["name"], "tool_call_id": c["id"]})
            log.append({"call_id": c["id"], "name": c["name"], "arguments": c["arguments"], "result": text, "ok": False, "rejected": True, "duration_s": 0.0, "turn": state.get("turn_id")})
        return {
            "messages": messages,
            "tool_log": log,
            "awaiting": [],
            # Other calls of the same batch still run; the rejected ones are dropped.
            "pending": [c for c in state["pending"] if c["id"] not in rejected],
        }

    def after_approve(state: AgentState) -> str:
        return "tools" if state.get("pending") else "agent"

    async def tools(state: AgentState) -> dict[str, Any]:
        if state.get("tool_calls_made", 0) >= state.get("max_tool_calls", 40):
            # A batch can be longer than the budget that is left: stop before running any more.
            text = "Stopped: the tool-call budget for this turn was used up. Break the request into smaller steps."
            skipped = [
                {"role": "tool", "content": "Not run: the tool-call budget for this turn was used up.", "tool_name": c["name"], "tool_call_id": c["id"]}
                for c in state["pending"]
            ]
            return {
                "messages": skipped + [{"role": "assistant", "content": text}],
                "pending": [],
                "status": "budget_exhausted",
                "final": text,
            }
        call, rest = state["pending"][0], state["pending"][1:]
        update: dict[str, Any] = {}
        started = time.monotonic()
        if call["name"] == "todo_write":
            todos = parse_todos(call["arguments"].get("items", ""))
            update["todos"] = todos
            result, ok = "Plan updated:\n" + format_todos(todos), True
        elif (tool := get_tool(call["name"])) is None:
            result, ok = f"Error: tool '{call['name']}' not found", False
        else:
            # A tool that runs its own model/tool loop (``task``) cannot pause for approval, so
            # inside it the tools the policy asks about are refused instead of run unasked.
            token = blocked_tool.set(lambda name: policy.needs_approval({"name": name}))
            try:
                outcome = await tool.execute(**call["arguments"])
            finally:
                blocked_tool.reset(token)
            result, ok = outcome.text, outcome.ok
        update.update(
            messages=[{"role": "tool", "content": result[:MAX_TOOL_RESULT_CHARS], "tool_name": call["name"], "tool_call_id": call["id"]}],
            tool_log=[{"call_id": call["id"], "name": call["name"], "arguments": call["arguments"], "result": result[:MAX_LOGGED_RESULT_CHARS], "ok": ok, "rejected": False, "duration_s": round(time.monotonic() - started, 3), "turn": state.get("turn_id")}],
            pending=rest,
            tool_calls_made=state.get("tool_calls_made", 0) + 1,
        )
        if not rest and update["tool_calls_made"] >= state.get("max_tool_calls", 40):
            text = "Stopped: the tool-call budget for this turn was used up. Break the request into smaller steps."
            update.update(status="budget_exhausted", final=text)
            update["messages"] = update["messages"] + [{"role": "assistant", "content": text}]
        return update

    def after_tools(state: AgentState) -> str:
        if state.get("pending"):
            return "tools"
        return END if state.get("status") == "budget_exhausted" else "agent"

    async def verify(state: AgentState) -> dict[str, Any]:
        if acceptance is None:
            return {"status": "done"}
        ok, reason = await acceptance(dict(state))
        if ok:
            return {"status": "verified", "verification": {"ok": True, "reason": reason}}
        if state.get("repairs", 0) >= state.get("max_repairs", 2):
            return {"status": "unverified", "verification": {"ok": False, "reason": reason}}
        text = f"The acceptance check failed:\n{reason}\nFix the problem, then finish again."
        return {
            "messages": [{"role": "user", "content": text}],
            "repairs": state.get("repairs", 0) + 1,
            "verification": {"ok": False, "reason": reason},
            "status": "running",
        }

    def after_verify(state: AgentState) -> str:
        return "agent" if state.get("status") == "running" else END

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent)
    graph.add_node("gate", gate)
    graph.add_node("approve", approve)
    graph.add_node("tools", tools)
    graph.add_node("verify", verify)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", after_agent, {"gate": "gate", "verify": "verify", END: END})
    graph.add_conditional_edges("gate", after_gate, {"approve": "approve", "tools": "tools"})
    graph.add_conditional_edges("approve", after_approve, {"tools": "tools", "agent": "agent"})
    graph.add_conditional_edges("tools", after_tools, {"tools": "tools", "agent": "agent", END: END})
    graph.add_conditional_edges("verify", after_verify, {"agent": "agent", END: END})
    return graph.compile(checkpointer=checkpointer)
