"""REPL glue for the graph engine: slash commands and thread bookkeeping."""
import asyncio

import pytest
from langgraph.checkpoint.memory import MemorySaver

from odyssey.cli import repl
from odyssey.core import workspace
from odyssey.core.state import Conversation
from odyssey.engine.graph import build_graph
from odyssey.engine.policy import ApprovalPolicy
from odyssey.engine.session import GraphSession, TurnResult, recall_thread
from odyssey.tools.registry import init_tools


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    init_tools()
    workspace.set_workspace_root(tmp_path)
    monkeypatch.setenv("ODYSSEY_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(repl.console, "width", 200)
    yield
    workspace.set_workspace_root(None)


def make_session(*replies):
    queue = list(replies)

    async def chat(messages):
        return {"message": queue.pop(0)}

    policy = ApprovalPolicy()
    return GraphSession(build_graph(chat, checkpointer=MemorySaver(), policy=policy), "t", policy)


def test_submit_approve_flow_through_the_repl_helpers(tmp_path, capsys):
    gs = make_session(
        {"tool_calls": [{"function": {"name": "write", "arguments": {"path": "x.txt", "content": "hi"}}}]},
        {"content": "all done"},
    )

    async def go():
        repl._show_turn(await gs.submit("write x"))
        assert not (tmp_path / "x.txt").exists()
        assert await repl._handle_graph_slash("/approve", gs, Conversation()) is True

    asyncio.run(go())
    out = capsys.readouterr().out
    assert "Waiting for approval" in out and "write" in out and "all done" in out
    assert (tmp_path / "x.txt").read_text() == "hi"


def test_reject_and_plan_commands(tmp_path, capsys):
    gs = make_session(
        {"tool_calls": [{"function": {"name": "todo_write", "arguments": {"items": "- step one\n- step two"}}},
                        {"function": {"name": "write", "arguments": {"path": "x.txt", "content": "hi"}}}]},
        {"content": "understood"},
    )

    async def go():
        await gs.submit("go")
        await repl._handle_graph_slash("/reject not now", gs, Conversation())
        await repl._handle_graph_slash("/plan", gs, Conversation())

    asyncio.run(go())
    out = capsys.readouterr().out
    assert "understood" in out and "1. [ ] step one" in out and "2. [ ] step two" in out
    assert not (tmp_path / "x.txt").exists()


def test_graph_commands_report_misuse_instead_of_crashing(capsys):
    gs = make_session()

    async def go():
        assert await repl._handle_graph_slash("/approve", gs, Conversation()) is True
        assert await repl._handle_graph_slash("/history", gs, Conversation()) is False  # not a graph command

    asyncio.run(go())
    assert "nothing is waiting for approval" in capsys.readouterr().out


def test_start_session_remembers_and_reuses_the_thread(tmp_path):
    async def go():
        first = await repl._start_graph_session(MemorySaver(), resume=False, approve_all=False)
        again = await repl._start_graph_session(MemorySaver(), resume=True, approve_all=False)
        fresh = await repl._start_graph_session(MemorySaver(), resume=False, approve_all=True)
        return first.thread_id, again.thread_id, fresh.thread_id, fresh.policy.approve_all

    first, again, fresh, approve_all = asyncio.run(go())
    assert again == first
    assert fresh != first and approve_all is True
    assert recall_thread("") == fresh


def test_clear_history_and_save_follow_the_graph_thread(tmp_path, capsys):
    gs = make_session({"content": "first answer"}, {"content": "second answer"})
    conv = Conversation()

    async def go():
        await gs.submit("hello")
        await repl._sync_conv(conv, gs)
        assert [m["role"] for m in conv.messages] == ["user", "assistant"]
        before = gs.thread_id
        assert await repl._handle_graph_slash("/clear", gs, conv) is True
        assert conv.messages == [] and gs.thread_id != before
        await gs.submit("again")
        # the new thread does not carry the old conversation
        return (await gs.state())["messages"]

    messages = asyncio.run(go())
    assert [m["content"] for m in messages] == ["again", "second answer"]
    assert recall_thread("") == gs.thread_id


def test_the_double_run_warning_is_shown_even_when_an_approval_follows(capsys):
    repl._show_turn(TurnResult(
        "awaiting_approval",
        approval=[{"id": "c2", "name": "bash", "arguments": {"command": "ls"}}],
        resumed_in_flight=[{"id": "c1", "name": "bash", "arguments": {}}],
    ))
    out = capsys.readouterr().out
    assert "may already have run" in out and "Waiting for approval" in out
    assert out.index("may already have run") < out.index("Waiting for approval")


def test_resume_from_another_directory_does_not_overwrite_the_original_thread(tmp_path, capsys):
    async def go():
        saver = MemorySaver()
        a = await repl._start_graph_session(saver, resume=False, approve_all=False, workspace="/project/a")
        # put something in the thread so it records where it started
        await a.graph.aupdate_state(a._config, {"workspace": "/project/a", "base": "/project/a", "messages": [], "pending": []}, as_node="verify")
        other = await repl._start_graph_session(saver, resume=True, approve_all=False, workspace="/project/b")
        back = await repl._start_graph_session(saver, resume=True, approve_all=False, workspace="/project/a")
        return a.thread_id, other.thread_id, back.thread_id

    first, other, back = asyncio.run(go())
    assert other != first
    assert back == first, "starting in project b must not make project a's thread unreachable"
    assert recall_thread("/project/a") == first and recall_thread("/project/b") == other
    assert "No earlier thread" in capsys.readouterr().out


def test_resume_from_another_subdirectory_of_the_same_project_is_a_different_thread(tmp_path):
    async def go():
        saver = MemorySaver()
        a = await repl._start_graph_session(saver, resume=False, approve_all=False, workspace="/repo", base="/repo/a")
        await a.graph.aupdate_state(a._config, {"workspace": "/repo", "base": "/repo/a", "messages": [], "pending": []}, as_node="verify")
        b = await repl._start_graph_session(saver, resume=True, approve_all=False, workspace="/repo", base="/repo/b")
        again = await repl._start_graph_session(saver, resume=True, approve_all=False, workspace="/repo", base="/repo/a")
        return a.thread_id, b.thread_id, again.thread_id

    a, b, again = asyncio.run(go())
    assert b != a and again == a
