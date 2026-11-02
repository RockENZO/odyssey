"""The LangGraph engine, driven by a scripted model (no Ollama)."""
import asyncio

import pytest
from langgraph.checkpoint.memory import MemorySaver

from odyssey.core import workspace
from odyssey.engine.graph import build_graph
from odyssey.engine.policy import ApprovalPolicy
from odyssey.engine.session import GraphSession, open_checkpointer
from odyssey.engine.todos import format_todos, parse_todos
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


def scripted(*replies):
    queue = list(replies)
    seen = []

    async def chat(messages):
        seen.append(messages)
        return {"message": queue.pop(0)}

    chat.seen = seen
    return chat


def make(chat, *, policy=None, acceptance=None, checkpointer=None, thread="t", **limits):
    policy = policy or ApprovalPolicy()
    graph = build_graph(chat, checkpointer=checkpointer or MemorySaver(), policy=policy, acceptance=acceptance)
    return GraphSession(graph, thread, policy, **limits)


def run(coro):
    return asyncio.run(coro)


# --- approval -------------------------------------------------------------------------------

def test_read_only_tools_run_without_approval(tmp_path):
    (tmp_path / "a.txt").write_text("alpha")
    chat = scripted({"tool_calls": [call("read", path="a.txt")]}, {"content": "it says alpha"})
    result = run(make(chat).submit("read a.txt"))
    assert (result.status, result.text) == ("done", "it says alpha")


def test_write_waits_for_approval_and_runs_only_after_it(tmp_path):
    chat = scripted({"tool_calls": [call("write", path="out.txt", content="hi")]}, {"content": "written"})

    async def go():
        s = make(chat)
        first = await s.submit("write out.txt")
        assert first.status == "awaiting_approval"
        assert [c["name"] for c in first.approval] == ["write"]
        assert not (tmp_path / "out.txt").exists(), "ran before approval"
        final = await s.decide(True)
        return final

    final = run(go())
    assert (final.status, final.text) == ("done", "written")
    assert (tmp_path / "out.txt").read_text() == "hi"


def test_rejection_skips_the_call_and_tells_the_model_why(tmp_path):
    chat = scripted({"tool_calls": [call("write", path="out.txt", content="hi")]}, {"content": "ok, not writing"})

    async def go():
        s = make(chat)
        await s.submit("write")
        final = await s.decide(False, reason="not that file")
        return final, await s.state()

    final, state = run(go())
    assert final.text == "ok, not writing"
    assert not (tmp_path / "out.txt").exists()
    last_tool_msg = [m for m in chat.seen[-1] if m["role"] == "tool"][-1]
    assert "Rejected by the user: not that file" in last_tool_msg["content"]
    assert state["tool_log"][0]["rejected"] is True


def test_always_approve_stops_asking_for_that_tool(tmp_path):
    chat = scripted(
        {"tool_calls": [call("write", path="a.txt", content="1")]},
        {"tool_calls": [call("write", path="b.txt", content="2")]},
        {"content": "both"},
    )

    async def go():
        s = make(chat)
        await s.submit("go")
        return await s.decide(True, always=True)

    assert run(go()).status == "done"
    assert (tmp_path / "b.txt").exists()


def test_approve_all_never_asks(tmp_path):
    chat = scripted({"tool_calls": [call("write", path="a.txt", content="1")]}, {"content": "done"})
    result = run(make(chat, policy=ApprovalPolicy(approve_all=True)).submit("go"))
    assert result.status == "done" and (tmp_path / "a.txt").exists()


def test_submit_is_refused_while_a_call_waits_for_approval():
    chat = scripted({"tool_calls": [call("write", path="a.txt", content="1")]})

    async def go():
        s = make(chat)
        await s.submit("go")
        with pytest.raises(RuntimeError, match="approve or reject"):
            await s.submit("something else")

    run(go())


def test_a_rejected_call_does_not_drop_other_calls_in_the_batch(tmp_path):
    (tmp_path / "r.txt").write_text("readable")
    chat = scripted(
        {"tool_calls": [call("read", path="r.txt"), call("write", path="w.txt", content="x")]},
        {"content": "finished"},
    )

    async def go():
        s = make(chat)
        await s.submit("go")
        await s.decide(False, reason="no")
        return await s.state()

    state = run(go())
    assert [(r["name"], r["rejected"]) for r in state["tool_log"]] == [("write", True), ("read", False)]


# --- plan, budgets, acceptance -------------------------------------------------------------

def test_todo_write_is_kept_in_state_and_shown_to_the_model():
    chat = scripted(
        {"tool_calls": [call("todo_write", items='[{"content": "write tests", "status": "in_progress"}, "ship it"]')]},
        {"content": "ok"},
    )

    async def go():
        s = make(chat)
        await s.submit("plan")
        return await s.state()

    state = run(go())
    assert state["todos"] == [
        {"content": "write tests", "status": "in_progress"},
        {"content": "ship it", "status": "pending"},
    ]
    system_prompt = chat.seen[-1][0]["content"]
    assert "Current plan:" in system_prompt and "[~] write tests" in system_prompt


def test_parse_todos_accepts_plain_text_and_marks():
    todos = parse_todos("- [x] done thing\n- [ ] next thing\n2. numbered")
    assert [t["status"] for t in todos] == ["completed", "pending", "pending"]
    assert [t["content"] for t in todos] == ["done thing", "next thing", "numbered"]
    assert format_todos([]) == "(no plan yet)"


def test_model_call_budget_stops_a_looping_model():
    async def chat(messages):
        return {"message": {"tool_calls": [call("read", path="missing.txt")]}}

    result = run(make(chat, max_model_calls=3, max_tool_calls=100).submit("loop"))
    assert result.status == "budget_exhausted"
    assert "model-call budget" in result.text


def test_tool_call_budget_stops_after_the_limit():
    async def chat(messages):
        return {"message": {"tool_calls": [call("read", path="missing.txt")]}}

    async def go():
        s = make(chat, max_model_calls=100, max_tool_calls=2)
        result = await s.submit("loop")
        return result, await s.state()

    result, state = run(go())
    assert result.status == "budget_exhausted" and "tool-call budget" in result.text
    assert len(state["tool_log"]) == 2


def test_failed_acceptance_goes_back_to_the_model_then_passes():
    verdicts = iter([(False, "1 test failed"), (True, "all tests pass")])

    async def acceptance(state):
        return next(verdicts)

    chat = scripted({"content": "I am done"}, {"content": "fixed it"})

    async def go():
        s = make(chat, acceptance=acceptance)
        return await s.submit("fix"), await s.state()

    result, state = run(go())
    assert (result.status, result.text) == ("verified", "fixed it")
    repair_msg = chat.seen[-1][-1]
    assert repair_msg["role"] == "user" and "1 test failed" in repair_msg["content"]
    assert state["repairs"] == 1 and state["verification"]["ok"] is True


def test_acceptance_that_never_passes_ends_unverified():
    async def acceptance(state):
        return False, "still failing"

    chat = scripted({"content": "a"}, {"content": "b"}, {"content": "c"})
    result = run(make(chat, acceptance=acceptance, max_repairs=2).submit("fix"))
    assert result.status == "unverified"


# --- persistence and resume ----------------------------------------------------------------

class Boom(RuntimeError):
    pass


def test_resume_after_a_crash_between_steps_does_not_rerun_finished_tools(tmp_path):
    db = tmp_path.parent / f"{tmp_path.name}-cp.sqlite"
    log = tmp_path / "log.txt"
    command = f"echo ran >> {log}"

    async def phase_one():
        calls = {"n": 0}

        async def chat(messages):
            calls["n"] += 1
            if calls["n"] == 1:
                return {"message": {"tool_calls": [call("bash", command=command)]}}
            raise Boom("process died")

        async with open_checkpointer(db) as saver:
            s = make(chat, policy=ApprovalPolicy(approve_all=True), checkpointer=saver, thread="job")
            with pytest.raises(Boom):
                await s.submit("do it")

    async def phase_two():
        chat = scripted({"content": "all done"})
        async with open_checkpointer(db) as saver:
            s = make(chat, policy=ApprovalPolicy(approve_all=True), checkpointer=saver, thread="job")
            return await s.resume()

    run(phase_one())
    result = run(phase_two())
    assert (result.status, result.text) == ("done", "all done")
    assert log.read_text().splitlines() == ["ran"], "the finished bash call was run again"
    assert result.resumed_in_flight == []


def test_a_tool_interrupted_mid_call_runs_again_on_resume_and_is_reported(tmp_path, monkeypatch):
    db = tmp_path.parent / f"{tmp_path.name}-cp2.sqlite"
    executions = []
    state = {"die": True}

    async def flaky(path: str) -> str:
        executions.append(path)
        if state["die"]:
            raise asyncio.CancelledError  # what Ctrl+C looks like inside a task; Tool.run only catches Exception
        return "contents"

    monkeypatch.setitem(registry._tools, "read", registry.Tool("read", "r", flaky))
    first = {"tool_calls": [call("read", path="x")]}

    async def phase_one():
        async with open_checkpointer(db) as saver:
            s = make(scripted(first), checkpointer=saver, thread="job")
            with pytest.raises(asyncio.CancelledError):
                await s.submit("go")

    async def phase_two():
        state["die"] = False
        async with open_checkpointer(db) as saver:
            s = make(scripted({"content": "ok"}), checkpointer=saver, thread="job")
            return await s.resume()

    run(phase_one())
    result = run(phase_two())
    assert executions == ["x", "x"]
    assert [c["name"] for c in result.resumed_in_flight] == ["read"]
    assert result.status == "done"


def test_resume_with_nothing_unfinished_is_idle():
    chat = scripted({"content": "hi"})

    async def go():
        s = make(chat)
        await s.submit("hello")
        return await s.resume()

    assert run(go()).status == "idle"


def test_threads_are_independent():
    saver = MemorySaver()
    a = make(scripted({"content": "A"}), checkpointer=saver, thread="a")
    b = make(scripted({"content": "B"}), checkpointer=saver, thread="b")

    async def go():
        await a.submit("one")
        await b.submit("two")
        return (await a.state())["final"], (await b.state())["final"]

    assert run(go()) == ("A", "B")


def test_a_batch_longer_than_the_remaining_budget_is_cut_off(tmp_path):
    (tmp_path / "r.txt").write_text("x")
    batch = [call("read", path="r.txt")] * 4
    chat = scripted({"tool_calls": batch})

    async def go():
        s = make(chat, max_tool_calls=2)
        result = await s.submit("many")
        return result, await s.state()

    result, state = run(go())
    assert result.status == "budget_exhausted"
    assert len(state["tool_log"]) == 2, "calls beyond the budget were run"
    tool_msgs = [m for m in state["messages"] if m["role"] == "tool"]
    assert len(tool_msgs) == 4 and sum("Not run" in m["content"] for m in tool_msgs) == 2
    assert state["pending"] == []


def test_a_thread_is_bound_to_the_workspace_it_started_in(tmp_path):
    saver = MemorySaver()
    chat = scripted({"tool_calls": [call("write", path="a.txt", content="1")]}, {"content": "ok"})
    policy = ApprovalPolicy()
    graph = build_graph(chat, checkpointer=saver, policy=policy)
    in_a = GraphSession(graph, "t", policy, workspace="/project/a")
    in_b = GraphSession(graph, "t", policy, workspace="/project/b")

    async def go():
        await in_a.submit("write")
        for attempt in (in_b.decide(True), in_b.resume(), in_b.submit("hi")):
            with pytest.raises(RuntimeError, match="started in /project/a"):
                await attempt
        assert not (tmp_path / "a.txt").exists(), "the pending write ran in the wrong workspace"
        return await in_a.decide(True)

    assert run(go()).status == "done"


def test_a_thread_is_bound_to_its_launch_directory_not_only_the_project_root(tmp_path):
    """A pending relative write must not run against another subdirectory of the same project."""
    saver = MemorySaver()
    chat = scripted({"tool_calls": [call("write", path="out.txt", content="1")]}, {"content": "ok"})
    policy = ApprovalPolicy()
    graph = build_graph(chat, checkpointer=saver, policy=policy)
    in_a = GraphSession(graph, "t", policy, workspace="/repo", base="/repo/a")
    in_b = GraphSession(graph, "t", policy, workspace="/repo", base="/repo/b")

    async def go():
        await in_a.submit("write")
        for attempt in (in_b.decide(True), in_b.resume(), in_b.submit("hi")):
            with pytest.raises(RuntimeError, match="started in /repo/a"):
                await attempt
        return await in_a.decide(True)

    assert run(go()).status == "done"


def test_an_approval_does_not_carry_over_to_a_later_call_that_reuses_the_id(tmp_path):
    first = {**call("write", path="a.txt", content="1"), "id": "call_1"}
    second = {**call("write", path="b.txt", content="2"), "id": "call_1"}
    chat = scripted({"tool_calls": [first]}, {"tool_calls": [second]}, {"content": "done"})

    async def go():
        s = make(chat)
        await s.submit("go")
        after_first = await s.decide(True)
        # the second call has the same id but is a different operation: it must ask again
        assert after_first.status == "awaiting_approval"
        assert after_first.approval[0]["arguments"]["path"] == "b.txt"
        assert not (tmp_path / "b.txt").exists()
        return await s.decide(True)

    assert run(go()).status == "done"


def _sub_agent_writes(monkeypatch):
    """The model inside the `task` sub-agent tries to write a file, then answers."""
    from odyssey.core import tool_loop

    replies = [{"tool_calls": [call("write", path="sub.txt", content="x")]}, {"content": "sub done"}]

    async def sub_chat(messages):
        return {"message": replies.pop(0)}

    monkeypatch.setattr(tool_loop, "chat_completion", sub_chat)


def test_a_sub_agent_cannot_run_a_tool_that_needs_approval(tmp_path, monkeypatch):
    _sub_agent_writes(monkeypatch)
    chat = scripted({"tool_calls": [call("task", description="write sub.txt")]}, {"content": "finished"})
    result = run(make(chat).submit("delegate"))
    assert result.status == "done"
    assert not (tmp_path / "sub.txt").exists(), "the sub-agent wrote a file without approval"


def test_a_sub_agent_may_use_a_tool_the_user_allowed(tmp_path, monkeypatch):
    _sub_agent_writes(monkeypatch)
    chat = scripted({"tool_calls": [call("task", description="write sub.txt")]}, {"content": "finished"})
    run(make(chat, policy=ApprovalPolicy(approve_all=True)).submit("delegate"))
    assert (tmp_path / "sub.txt").read_text() == "x"


def test_a_saved_rejection_is_honoured_after_the_approval_policy_changes(tmp_path):
    """Restarting with --yes must not turn a rejection of the saved question into an approval."""
    saver = MemorySaver()
    first = make(scripted({"tool_calls": [call("write", path="a.txt", content="1")]}), checkpointer=saver)
    restarted = make(scripted({"content": "ok, skipped"}), checkpointer=saver, policy=ApprovalPolicy(approve_all=True))

    async def go():
        assert (await first.submit("write")).status == "awaiting_approval"
        assert (await restarted.pending_approval())[0]["name"] == "write"
        return await restarted.decide(False, reason="no")

    assert run(go()).status == "done"
    assert not (tmp_path / "a.txt").exists(), "the rejected write ran after the policy changed"

