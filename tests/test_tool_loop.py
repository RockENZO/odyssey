"""The tool loop runs end to end against a scripted model, with no Ollama."""
import asyncio

import pytest

from odyssey.core import workspace
from odyssey.core.state import Conversation
from odyssey.core.tool_loop import run_tool_loop
from odyssey.tools.registry import init_tools


@pytest.fixture(autouse=True)
def _tools_and_workspace():
    init_tools()
    yield
    workspace.set_workspace_root(None)


def scripted(*replies):
    """A fake chat function that returns the given replies in order."""
    queue = list(replies)
    seen = []

    async def chat(messages):
        seen.append([dict(m) for m in messages])
        return {"message": queue.pop(0)}

    chat.seen = seen
    return chat


def call(name, **arguments):
    return {"function": {"name": name, "arguments": arguments}}


def run(coro):
    return asyncio.run(coro)


def test_write_then_answer(tmp_path):
    workspace.set_workspace_root(tmp_path)
    target = tmp_path / "greet.py"
    chat = scripted(
        {"content": "", "tool_calls": [call("write", path=str(target), content="print('hi')\n")]},
        {"content": "done"},
    )
    conv = Conversation()
    assert run(run_tool_loop("make greet.py", conv, chat=chat)) == "done"
    assert target.read_text() == "print('hi')\n"


def test_tool_results_keep_the_call_id_and_tool_name(tmp_path):
    workspace.set_workspace_root(tmp_path)
    (tmp_path / "a.txt").write_text("alpha\n")
    chat = scripted(
        {"tool_calls": [call("read", path="a.txt"), call("read", path="a.txt")]},
        {"content": "ok"},
    )
    conv = Conversation()
    run(run_tool_loop("read twice", conv, chat=chat))
    tool_msgs = [m for m in conv.messages if m["role"] == "tool"]
    assistant = next(m for m in conv.messages if m.get("tool_calls"))
    ids = [c["id"] for c in assistant["tool_calls"]]
    assert len(set(ids)) == 2, "two calls to the same tool must get different ids"
    assert [m["tool_call_id"] for m in tool_msgs] == ids
    assert all(m["tool_name"] == "read" for m in tool_msgs)
    assert [r.call_id for r in conv.tool_log] == ids
    assert all(r.ok for r in conv.tool_log)


def test_unknown_tool_and_bad_arguments_are_reported_not_raised():
    chat = scripted(
        {"tool_calls": [call("nope"), call("read", wrong_arg=1)]},
        {"content": "ok"},
    )
    conv = Conversation()
    run(run_tool_loop("x", conv, chat=chat))
    results = [m["content"] for m in conv.messages if m["role"] == "tool"]
    assert "not found" in results[0]
    assert results[1].startswith("Error executing read")
    assert [r.ok for r in conv.tool_log] == [False, False]


def test_string_arguments_are_parsed():
    chat = scripted(
        {"tool_calls": [{"function": {"name": "read", "arguments": '{"path": "/nonexistent-odyssey"}'}}]},
        {"content": "ok"},
    )
    conv = Conversation()
    run(run_tool_loop("x", conv, chat=chat))
    assert conv.tool_log[0].arguments == {"path": "/nonexistent-odyssey"}


def test_round_limit_stops_a_looping_model():
    async def chat(messages):
        return {"message": {"tool_calls": [call("read", path="/nonexistent-odyssey")]}}

    conv = Conversation(max_tool_rounds=3)
    result = run(run_tool_loop("x", conv, chat=chat))
    assert "maximum tool call iterations" in result
    assert len(conv.tool_log) == 3


def test_tool_output_is_truncated_in_context_but_not_in_the_log(tmp_path):
    workspace.set_workspace_root(tmp_path)
    (tmp_path / "big.txt").write_text("x" * 50_000)
    chat = scripted({"tool_calls": [call("read", path="big.txt")]}, {"content": "ok"})
    conv = Conversation()
    run(run_tool_loop("x", conv, chat=chat))
    assert len(next(m for m in conv.messages if m["role"] == "tool")["content"]) == 10000
    assert len(conv.tool_log[0].result) > 10000
