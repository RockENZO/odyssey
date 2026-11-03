"""Read-only investigator sub-agents: concurrency, the read-only boundary, budgets, failures."""
import asyncio

import pytest
from langgraph.checkpoint.memory import MemorySaver

from odyssey.core import workspace
from odyssey.engine import investigate
from odyssey.engine.graph import build_graph
from odyssey.engine.investigate import MAX_MODEL_CALLS, PROMPT, run_investigators
from odyssey.engine.policy import ApprovalPolicy
from odyssey.engine.session import GraphSession
from odyssey.tools import registry
from odyssey.tools.registry import init_tools


@pytest.fixture(autouse=True)
def _env(tmp_path):
    init_tools()
    workspace.set_workspace_root(tmp_path)
    yield
    workspace.set_workspace_root(None)


def call(name, **arguments):
    return {"function": {"name": name, "arguments": arguments}}


def run(coro):
    return asyncio.run(coro)


def is_investigator(messages):
    return messages[0]["content"] == PROMPT


def tool_names():
    return sorted(s["function"]["name"] for s in registry.get_tool_schemas())


def test_the_questions_are_answered_side_by_side(tmp_path):
    """Each investigator's first model call waits until all of them have started: only concurrent runs get past."""
    (tmp_path / "a.txt").write_text("alpha")

    async def go():
        barrier = asyncio.Barrier(3)
        seen = {}

        async def chat(messages):
            question = messages[1]["content"]
            if len(messages) == 2:                                   # first call of this investigator
                await asyncio.wait_for(barrier.wait(), 5)
                return {"message": {"tool_calls": [call("read", path="a.txt")]}}
            seen[question] = messages[-1]["content"]
            return {"message": {"content": f"answer to {question}: {messages[-1]['content'].strip().splitlines()[0]}"}}

        outcome = await run_investigators(chat, ["q1", "q2", "q3"])
        return outcome, seen

    outcome, seen = run(go())
    assert outcome.ok
    for n in (1, 2, 3):
        assert f"Investigation {n} of 3: q{n}" in outcome.text and f"answer to q{n}" in outcome.text
    assert outcome.text.count("(read x1; 2 model calls)") == 3
    assert all("alpha" in text for text in seen.values()), "the investigator did not get the real file text"


def test_an_investigator_cannot_write_run_commands_or_start_another_investigator(tmp_path):
    shown = []
    replies = iter([
        {"tool_calls": [call("write", path="x.txt", content="no"), call("bash", command="touch y.txt"), call("investigate", questions=["deeper"])]},
        {"content": "I could not change anything."},
    ])

    async def chat(messages):
        shown.append(tool_names())
        return {"message": next(replies)}

    outcome = run(run_investigators(chat, ["try to write"]))
    assert not (tmp_path / "x.txt").exists() and not (tmp_path / "y.txt").exists()
    assert shown[0] == ["glob", "grep", "read"], "the model must only be shown the read-only tools"
    assert "(no tools; 2 model calls)" in outcome.text
    assert "write" in tool_names() and "bash" in tool_names(), "the registry must be restored after the investigators"


def test_the_refusals_reach_the_model(tmp_path):
    seen = []
    replies = iter([{"tool_calls": [call("write", path="x.txt", content="no")]}, {"content": "done"}])

    async def chat(messages):
        seen.append(messages[-1].get("content", ""))
        return {"message": next(replies)}

    run(run_investigators(chat, ["q"]))
    assert "Refused: 'write' is not available to an investigator" in seen[1]


@pytest.mark.parametrize("questions", [None, [], "q", [""], ["ok", 3], ["a"] * 5, ["x" * 2001]])
def test_bad_questions_are_an_error_not_a_crash(questions):
    async def chat(messages):
        raise AssertionError("no model call expected")

    outcome = run(run_investigators(chat, questions))
    assert not outcome.ok and outcome.text.startswith("Error")


def test_one_failing_investigator_does_not_lose_the_others():
    async def chat(messages):
        if messages[1]["content"] == "bad":
            raise ConnectionError("ollama is down")
        return {"message": {"content": "fine"}}

    outcome = run(run_investigators(chat, ["good", "bad"]))
    assert outcome.ok
    assert "fine" in outcome.text and "Error: ConnectionError: ollama is down" in outcome.text


def test_an_investigator_that_never_answers_is_stopped_by_its_budget(tmp_path):
    (tmp_path / "a.txt").write_text("x")
    calls = []

    async def chat(messages):
        calls.append(1)
        return {"message": {"tool_calls": [call("read", path="a.txt")]}}

    outcome = run(run_investigators(chat, ["loop forever"]))
    assert len(calls) == MAX_MODEL_CALLS
    assert "Stopped: this investigator used its model-call budget" in outcome.text and not outcome.ok


def test_the_tool_budget_refuses_calls_instead_of_running_them(tmp_path, monkeypatch):
    (tmp_path / "a.txt").write_text("x")
    monkeypatch.setattr(investigate, "MAX_TOOL_CALLS", 2)
    replies = iter([{"tool_calls": [call("read", path="a.txt")] * 4}, {"content": "done"}])
    results = []

    async def chat(messages):
        results.append([m["content"] for m in messages if m["role"] == "tool"])
        return {"message": next(replies)}

    outcome = run(run_investigators(chat, ["q"]))
    assert sum("Refused: this investigator used its tool-call budget" in r for r in results[1]) == 2
    assert "(read x2; 2 model calls)" in outcome.text


def test_the_main_agent_uses_investigate_without_an_approval_prompt(tmp_path):
    (tmp_path / "a.txt").write_text("alpha")
    main_replies = iter([
        {"tool_calls": [call("investigate", questions=["what is in a.txt?", "is there a b.txt?"])]},
        {"content": "summarised"},
    ])
    sub_replies = {}

    async def chat(messages):
        if is_investigator(messages):
            question = messages[1]["content"]
            step = sub_replies.setdefault(question, 0)
            sub_replies[question] += 1
            if step == 0:
                tool = call("read", path="a.txt") if "a.txt?" in question and "what" in question else call("glob", pattern="b.txt")
                return {"message": {"tool_calls": [tool]}}
            return {"message": {"content": "found: " + messages[-1]["content"].strip().splitlines()[0][:40]}}
        return {"message": next(main_replies)}

    async def go():
        policy = ApprovalPolicy()
        session = GraphSession(build_graph(chat, checkpointer=MemorySaver(), policy=policy), "t", policy)
        result = await session.submit("look around")
        return result, (await session.state())["tool_log"]

    result, log = run(go())
    assert (result.status, result.text) == ("done", "summarised")
    [entry] = log
    assert entry["name"] == "investigate" and entry["ok"]
    assert "Investigation 1 of 2: what is in a.txt?" in entry["result"] and "alpha" in entry["result"]
    assert "Investigation 2 of 2" in entry["result"]


def test_a_repeat_after_a_crash_changes_nothing(tmp_path):
    """Investigators only read, so running the same call twice leaves the workspace as it was."""
    (tmp_path / "a.txt").write_text("alpha")

    async def chat(messages):
        return {"message": {"content": "ok"}}

    before = sorted(p.name for p in tmp_path.iterdir())
    run(run_investigators(chat, ["q"]))
    run(run_investigators(chat, ["q"]))
    assert sorted(p.name for p in tmp_path.iterdir()) == before and (tmp_path / "a.txt").read_text() == "alpha"


def test_the_name_filter_holds_even_if_the_registry_is_not_narrowed(tmp_path):
    """Two layers: the registry is narrowed while investigators run, and the subgraph refuses other names itself."""
    replies = iter([{"tool_calls": [call("write", path="x.txt", content="no")]}, {"content": "done"}])

    async def chat(messages):
        return {"message": next(replies)}

    async def go():
        graph = investigate.build_investigator(chat)       # called directly: no registry narrowing
        return await graph.ainvoke({"messages": [{"role": "user", "content": "q"}], "pending": [], "model_calls": 0, "tool_calls": 0, "used": [], "answer": ""})

    state = run(go())
    assert not (tmp_path / "x.txt").exists() and state["used"] == []
