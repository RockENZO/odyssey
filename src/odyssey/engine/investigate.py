"""Read-only investigator sub-agents, each one a LangGraph subgraph, run side by side.

The ``investigate`` tool takes up to four independent questions about the code. Each one goes to
its own sub-agent: a small graph (``model -> tools -> model``) with its own message history, a
fixed read-only tool set (``read``, ``glob``, ``grep``) and its own budget. The sub-agents run
concurrently and the tool returns their answers together.

Why this is safe to run without asking:

* The tool set is fixed in code, not taken from the approval policy: a model that asks an
  investigator for ``write`` gets a refusal, and the registry is narrowed to the read-only tools
  while the investigators run, so the model is not even shown the others.
* They cannot start more sub-agents: ``investigate`` is not in their tool set.
* They only read, so a run that was interrupted by a crash and repeated on resume changes nothing.

What it does not do: the investigators' own calls are not in the thread's tool log (the log has
the ``investigate`` call and its answer, with a one-line trace per question); their model calls do
not count toward the turn's model-call budget (each is capped, and at most four run at once); and
a local Ollama server may answer concurrent requests one at a time, in which case "side by side"
saves nothing but also costs nothing. Answers are model output about the code, not verified facts.
"""
from __future__ import annotations

import asyncio
import operator
from collections import Counter
from typing import Annotated, Any, Awaitable, Callable, TypedDict

from langgraph.graph import END, START, StateGraph

from odyssey.core.tool_calls import parse_tool_calls
from odyssey.tools import registry
from odyssey.tools.registry import Tool, ToolOutcome, get_tool, only_tools

ChatFn = Callable[[list[dict[str, Any]]], Awaitable[dict[str, Any]]]

INVESTIGATOR_TOOLS = ("read", "glob", "grep")
MAX_QUESTIONS = 4
MAX_QUESTION_CHARS = 2000
MAX_MODEL_CALLS = 8
MAX_TOOL_CALLS = 16
MAX_RESULT_CHARS = 6000   # one tool result, as shown to an investigator
MAX_ANSWER_CHARS = 4000   # one answer, as returned to the main agent

PROMPT = """You are a read-only code investigator working for another agent.
You can only use read, glob and grep. You cannot change files or run commands, and you must not say that you did.
Answer the question you are given from what you actually read. Name the files (and line numbers where you can) your answer rests on.
If you could not find something, say so instead of guessing. Be concise: the answer goes back to the other agent."""


class InvestigatorState(TypedDict, total=False):
    messages: Annotated[list[dict[str, Any]], operator.add]
    pending: list[dict[str, Any]]
    model_calls: int
    tool_calls: int
    used: list[str]
    answer: str


def build_investigator(chat: ChatFn):
    """The subgraph for one question. No checkpointer: it is short and only reads."""

    async def model(state: InvestigatorState) -> dict[str, Any]:
        if state.get("model_calls", 0) >= MAX_MODEL_CALLS:
            return {"answer": "Stopped: this investigator used its model-call budget before it could answer.", "pending": []}
        response = await chat([{"role": "system", "content": PROMPT}, *state["messages"]])
        msg = response["message"]
        calls = parse_tool_calls(msg.get("tool_calls"))
        update: dict[str, Any] = {"model_calls": state.get("model_calls", 0) + 1}
        if not calls:
            update.update(messages=[{"role": "assistant", "content": msg.get("content", "") or ""}], pending=[], answer=(msg.get("content", "") or "").strip())
        else:
            update.update(
                messages=[{"role": "assistant", "content": "", "tool_calls": [c.to_message() for c in calls]}],
                pending=[{"id": c.id, "name": c.name, "arguments": c.arguments} for c in calls],
            )
        return update

    async def tools(state: InvestigatorState) -> dict[str, Any]:
        messages, used, made = [], list(state.get("used", [])), state.get("tool_calls", 0)
        for call in state["pending"]:
            tool = get_tool(call["name"]) if call["name"] in INVESTIGATOR_TOOLS else None
            if made >= MAX_TOOL_CALLS:
                text = "Refused: this investigator used its tool-call budget. Answer from what you have."
            elif tool is None:
                text = f"Refused: '{call['name']}' is not available to an investigator. Only read, glob and grep are."
            else:
                made += 1
                used.append(call["name"])
                text = (await tool.execute(**call["arguments"])).text
            messages.append({"role": "tool", "content": text[:MAX_RESULT_CHARS], "tool_name": call["name"], "tool_call_id": call["id"]})
        return {"messages": messages, "pending": [], "tool_calls": made, "used": used}

    graph = StateGraph(InvestigatorState)
    graph.add_node("model", model)
    graph.add_node("tools", tools)
    graph.add_edge(START, "model")
    graph.add_conditional_edges("model", lambda s: "tools" if s.get("pending") else END, {"tools": "tools", END: END})
    graph.add_edge("tools", "model")
    return graph.compile()


def _trace(state: InvestigatorState) -> str:
    counts = Counter(state.get("used", []))
    tools = ", ".join(f"{name} x{n}" for name, n in counts.items()) or "no tools"
    return f"({tools}; {state.get('model_calls', 0)} model calls)"


async def run_investigators(chat: ChatFn, questions: Any) -> ToolOutcome:
    if not isinstance(questions, list) or not questions or not all(isinstance(q, str) and q.strip() for q in questions):
        return ToolOutcome('Error: "questions" must be a non-empty list of question strings.', False)
    if len(questions) > MAX_QUESTIONS:
        return ToolOutcome(f"Error: at most {MAX_QUESTIONS} questions per call; split the work or ask fewer.", False)
    if any(len(q) > MAX_QUESTION_CHARS for q in questions):
        return ToolOutcome(f"Error: a question is longer than {MAX_QUESTION_CHARS} characters.", False)

    investigator = build_investigator(chat)

    async def one(question: str) -> tuple[str, bool]:
        try:
            state = await investigator.ainvoke({"messages": [{"role": "user", "content": question}], "pending": [], "model_calls": 0, "tool_calls": 0, "used": [], "answer": ""})
        except Exception as e:  # one investigator failing must not lose the others' answers
            return f"Error: {type(e).__name__}: {e}", False
        answer = (state.get("answer") or "").strip() or "No answer."
        return f"{answer[:MAX_ANSWER_CHARS]}\n{_trace(state)}", not answer.startswith("Stopped:")

    # Only the read-only tools exist (and are shown to the model) while the investigators run. The
    # main agent is waiting for this call, so nothing else uses the registry meanwhile.
    with only_tools(INVESTIGATOR_TOOLS):
        results = await asyncio.gather(*(one(q) for q in questions))
    parts = [f"Investigation {i} of {len(questions)}: {q.strip()}\n{text}" for i, (q, (text, _)) in enumerate(zip(questions, results), 1)]
    return ToolOutcome("\n\n".join(parts), any(ok for _, ok in results))


def register_investigate(chat: ChatFn) -> None:
    """Offer the ``investigate`` tool, answering with ``chat``. Called by the graph engine."""

    async def investigate(questions: Any = None) -> ToolOutcome:
        return await run_investigators(chat, questions)

    registry.register(Tool(
        name="investigate",
        description=(
            "Ask up to 4 independent, read-only questions about the code (for example 'where is X defined and who calls it?'). "
            "Each question is answered by a separate sub-agent that can only read, glob and grep, and they run side by side. "
            "Use it for broad exploration before changing anything; it cannot change files. Returns each sub-agent's findings."
        ),
        fn=investigate,
        parameters={
            "type": "object",
            "properties": {"questions": {"type": "array", "items": {"type": "string"}, "description": "1 to 4 self-contained questions"}},
            "required": ["questions"],
        },
        read_only=True,
    ))
