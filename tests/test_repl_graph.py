"""REPL glue for the graph engine: slash commands and thread bookkeeping."""
import asyncio

import pytest
from langgraph.checkpoint.memory import MemorySaver

from odyssey.cli import repl
from odyssey.core import workspace
from odyssey.engine.graph import build_graph
from odyssey.engine.policy import ApprovalPolicy
from odyssey.engine.session import GraphSession
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
        assert await repl._handle_graph_slash("/approve", gs) is True

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
        await repl._handle_graph_slash("/reject not now", gs)
        await repl._handle_graph_slash("/plan", gs)

    asyncio.run(go())
    out = capsys.readouterr().out
    assert "understood" in out and "1. [ ] step one" in out and "2. [ ] step two" in out
    assert not (tmp_path / "x.txt").exists()


def test_graph_commands_report_misuse_instead_of_crashing(capsys):
    gs = make_session()

    async def go():
        assert await repl._handle_graph_slash("/approve", gs) is True
        assert await repl._handle_graph_slash("/clear", gs) is False  # not a graph command

    asyncio.run(go())
    assert "nothing is waiting for approval" in capsys.readouterr().out


def test_start_session_remembers_and_reuses_the_thread(tmp_path):
    from odyssey.engine.session import last_thread_file

    async def go():
        first = await repl._start_graph_session(MemorySaver(), resume=False, approve_all=False)
        again = await repl._start_graph_session(MemorySaver(), resume=True, approve_all=False)
        fresh = await repl._start_graph_session(MemorySaver(), resume=False, approve_all=True)
        return first.thread_id, again.thread_id, fresh.thread_id, fresh.policy.approve_all

    first, again, fresh, approve_all = asyncio.run(go())
    assert again == first
    assert fresh != first and approve_all is True
    assert last_thread_file().read_text() == fresh
