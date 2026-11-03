"""MCP client: allowlist, approvals, failure handling, and the completion gate's view of MCP tools."""
import asyncio
import sys
from pathlib import Path

import pytest

pytest.importorskip("mcp")

from langgraph.checkpoint.memory import MemorySaver  # noqa: E402

from odyssey import mcp_client  # noqa: E402
from odyssey.engine.graph import build_graph  # noqa: E402
from odyssey.engine.policy import ApprovalPolicy  # noqa: E402
from odyssey.engine.session import GraphSession  # noqa: E402
from odyssey.engine.verify import changes_project  # noqa: E402
from odyssey.tools import registry  # noqa: E402
from odyssey.tools.registry import init_tools  # noqa: E402

SERVER = str(Path(__file__).with_name("mcp_fake_server.py"))


@pytest.fixture(autouse=True)
def _tools():
    init_tools()
    before = set(registry._tools)
    yield
    for name in set(registry._tools) - before:
        registry.unregister(name)


def cfg(name="fake", allow=("echo", "save_note"), auto=("echo",), args=(), timeout=20.0, command=sys.executable):
    return mcp_client.ServerConfig(name, command, (SERVER, *args), {}, tuple(allow), tuple(auto), timeout)


def call(name, **arguments):
    return {"function": {"name": name, "arguments": arguments}}


def run(coro):
    return asyncio.run(coro)


# --- config ---------------------------------------------------------------------------------

def test_a_valid_config_parses():
    [c] = mcp_client.parse_config({"servers": {"notes": {"command": "x", "args": ["a"], "allow": ["t"], "auto_approve": ["t"], "timeout": 5}}})
    assert (c.name, c.command, c.args, c.allow, c.auto_approve, c.timeout) == ("notes", "x", ("a",), ("t",), ("t",), 5.0)


@pytest.mark.parametrize("data, message", [
    ([], "must be an object"),
    ({"servers": {}, "extra": 1}, "must be an object"),
    ({"servers": {"a_b": {"command": "x"}}}, "name may only contain"),
    ({"servers": {"a": {"command": "x", "oops": 1}}}, "expected an object with keys"),
    ({"servers": {"a": {}}}, '"command" is required'),
    ({"servers": {"a": {"command": "x", "allow": ["*"]}}}, "wildcard"),
    ({"servers": {"a": {"command": "x", "allow": ["t"], "auto_approve": ["u"]}}}, "subset"),
    ({"servers": {"a": {"command": "x", "args": "no"}}}, "list of strings"),
    ({"servers": {"a": {"command": "x", "env": {"K": 1}}}}, "object of strings"),
    ({"servers": {"a": {"command": "x", "timeout": 0}}}, "positive number"),
])
def test_invalid_configs_are_rejected(data, message):
    with pytest.raises(mcp_client.ConfigError, match=message):
        mcp_client.parse_config(data)


def test_load_config_reports_missing_and_broken_files(tmp_path):
    with pytest.raises(mcp_client.ConfigError, match="cannot read"):
        mcp_client.load_config(tmp_path / "nope.json")
    (tmp_path / "bad.json").write_text("{", encoding="utf-8")
    with pytest.raises(mcp_client.ConfigError, match="not valid JSON"):
        mcp_client.load_config(tmp_path / "bad.json")


# --- allowlist and approvals ----------------------------------------------------------------

def test_only_allowlisted_tools_are_registered_and_they_are_removed_on_exit(tmp_path):
    policy = ApprovalPolicy()

    async def go():
        async with mcp_client.connect([cfg()], policy, tmp_path) as rt:
            names = sorted(rt.tools)
            assert registry.get_tool("mcp__fake__secret") is None, "a tool outside the allowlist was exposed"
            assert registry.get_tool("mcp__fake__boom") is None
            schema = registry.get_tool("mcp__fake__echo").to_openai_tool()["function"]
            return names, schema, set(policy.ask_tools)

    names, schema, asked = run(go())
    assert names == ["mcp__fake__echo", "mcp__fake__save_note"]
    assert schema["parameters"]["properties"]["text"]["type"] == "string" and "[MCP server 'fake']" in schema["description"]
    assert "mcp__fake__save_note" in asked and "mcp__fake__echo" not in asked
    assert registry.get_tool("mcp__fake__echo") is None and "mcp__fake__save_note" not in policy.ask_tools


def test_an_mcp_call_asks_for_approval_unless_auto_approved(tmp_path):
    note = tmp_path / "note.txt"
    policy = ApprovalPolicy()
    replies = iter([
        {"tool_calls": [call("mcp__fake__echo", text="hi")]},
        {"tool_calls": [call("mcp__fake__save_note", text="remember")]},
        {"content": "done"},
    ])

    async def chat(messages):
        return {"message": next(replies)}

    async def go():
        async with mcp_client.connect([cfg(args=(str(note),))], policy, tmp_path):
            session = GraphSession(build_graph(chat, checkpointer=MemorySaver(), policy=policy), "t", policy)
            first = await session.submit("go")
            assert first.status == "awaiting_approval", "the non-auto-approved call ran unasked"
            assert [c["name"] for c in first.approval] == ["mcp__fake__save_note"]
            assert not note.exists()
            final = await session.decide(True)
            return final, (await session.state())["tool_log"]

    final, log = run(go())
    assert final.status == "done" and note.read_text() == "remember"
    assert [(e["name"], e["ok"]) for e in log] == [("mcp__fake__echo", True), ("mcp__fake__save_note", True)]
    assert log[0]["result"] == "echo: hi", "the auto-approved read-only call should have run without a prompt"


def test_a_rejected_mcp_call_does_not_run(tmp_path):
    note = tmp_path / "note.txt"
    policy = ApprovalPolicy()
    replies = iter([{"tool_calls": [call("mcp__fake__save_note", text="x")]}, {"content": "ok"}])

    async def chat(messages):
        return {"message": next(replies)}

    async def go():
        async with mcp_client.connect([cfg(args=(str(note),))], policy, tmp_path):
            session = GraphSession(build_graph(chat, checkpointer=MemorySaver(), policy=policy), "t", policy)
            await session.submit("go")
            return await session.decide(False, reason="no")

    run(go())
    assert not note.exists()


def test_approve_all_is_the_only_thing_that_skips_the_prompt_for_a_non_auto_tool(tmp_path):
    policy = ApprovalPolicy(approve_all=True)

    async def go():
        async with mcp_client.connect([cfg()], policy, tmp_path):
            return policy.needs_approval({"name": "mcp__fake__save_note"})

    assert run(go()) is False


# --- failures -------------------------------------------------------------------------------

def test_an_unknown_allowed_tool_and_a_server_that_fails_to_start_are_reported_not_fatal(tmp_path):
    policy = ApprovalPolicy()
    configs = [cfg(allow=("echo", "ghost"), auto=()), cfg(name="broken", command=str(tmp_path / "no-such-command"))]

    async def go():
        async with mcp_client.connect(configs, policy, tmp_path) as rt:
            return rt.tools, rt.warnings, [(s.name, s.connected) for s in rt.servers]

    tools, warnings, servers = run(go())
    assert tools == ["mcp__fake__echo"]
    assert servers == [("fake", True), ("broken", False)]
    assert any("'ghost' is not offered" in w for w in warnings)
    assert any(w.startswith("broken: could not start") for w in warnings)


def test_an_empty_allowlist_starts_nothing(tmp_path):
    async def go():
        async with mcp_client.connect([cfg(allow=(), auto=())], ApprovalPolicy(), tmp_path) as rt:
            return rt.servers[0], rt.warnings

    status, warnings = run(go())
    assert not status.connected and any('"allow" is empty' in w for w in warnings)


def test_a_failing_and_a_slow_tool_come_back_as_failed_results(tmp_path):
    policy = ApprovalPolicy(approve_all=True)

    async def go():
        async with mcp_client.connect([cfg(allow=("boom", "slow"), auto=(), timeout=1.0)], policy, tmp_path):
            boom = await registry.get_tool("mcp__fake__boom").execute()
            slow = await registry.get_tool("mcp__fake__slow").execute()
            return boom, slow

    boom, slow = run(go())
    assert not boom.ok and boom.text.startswith("Error")
    assert not slow.ok and "did not answer within 1s" in slow.text


def test_a_name_that_is_already_taken_is_skipped(tmp_path):
    registry.register(registry.Tool(name="mcp__fake__echo", description="d", fn=lambda **k: None))

    async def go():
        async with mcp_client.connect([cfg(allow=("echo",), auto=())], ApprovalPolicy(), tmp_path) as rt:
            return rt.tools, rt.warnings

    tools, warnings = run(go())
    assert tools == [] and any("already taken" in w for w in warnings)


# --- completion gate --------------------------------------------------------------------------

def test_the_gate_treats_mcp_tools_as_changing_unless_read_only(tmp_path):
    async def go():
        async with mcp_client.connect([cfg()], ApprovalPolicy(), tmp_path):
            return changes_project("mcp__fake__echo"), changes_project("mcp__fake__save_note")

    read_only, writes = run(go())
    assert read_only is False and writes is True
    assert changes_project("mcp__gone__anything") is True, "an unregistered MCP tool must not skip the check"
    assert changes_project("read") is False and changes_project("bash") is True


# --- the REPL's use of the config ---------------------------------------------------------------

def _write_config(directory: Path, allow=("echo",)) -> Path:
    import json

    path = directory / "mcp.json"
    path.write_text(json.dumps({"servers": {"fake": {"command": sys.executable, "args": [SERVER], "allow": list(allow), "auto_approve": []}}}), encoding="utf-8")
    return path


def test_the_repl_loads_the_users_config_and_only_for_the_graph_engine(tmp_path, monkeypatch):
    from odyssey.cli import repl

    monkeypatch.setenv("ODYSSEY_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("ODYSSEY_DATA_DIR", str(tmp_path / "data"))
    (tmp_path / "cfg").mkdir()

    async def tools(explicit="", graph=True):
        policy = ApprovalPolicy()
        async with repl._mcp_session(policy, explicit, graph) as rt:
            return None if rt is None else (rt.tools, set(policy.ask_tools))

    assert run(tools()) is None, "no config, no MCP"
    _write_config(tmp_path / "cfg")
    found, asked = run(tools())
    assert found == ["mcp__fake__echo"] and "mcp__fake__echo" in asked
    assert run(tools(graph=False)) is None, "the loop has no approval step, so it must not get MCP tools"
    other = tmp_path / "other"
    other.mkdir()
    assert run(tools(explicit=str(_write_config(other, allow=("save_note",)))))[0] == ["mcp__fake__save_note"]


def test_a_broken_config_turns_mcp_off_without_ending_the_session(tmp_path, monkeypatch):
    from odyssey.cli import repl

    monkeypatch.setenv("ODYSSEY_CONFIG_DIR", str(tmp_path / "cfg"))
    (tmp_path / "cfg").mkdir()
    (tmp_path / "cfg" / "mcp.json").write_text("{not json", encoding="utf-8")

    async def go():
        async with repl._mcp_session(ApprovalPolicy(), "", True) as rt:
            return rt

    assert run(go()) is None


def test_an_error_inside_the_session_passes_through_and_the_servers_are_stopped(tmp_path, monkeypatch):
    from odyssey.cli import repl

    monkeypatch.setenv("ODYSSEY_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("ODYSSEY_DATA_DIR", str(tmp_path / "data"))
    (tmp_path / "cfg").mkdir()
    _write_config(tmp_path / "cfg")

    async def go():
        async with repl._mcp_session(ApprovalPolicy(), "", True):
            raise RuntimeError("the session failed")

    with pytest.raises(RuntimeError, match="the session failed"):
        run(go())
    assert registry.get_tool("mcp__fake__echo") is None


def test_a_sub_agent_cannot_use_an_mcp_tool_that_needs_approval(tmp_path, monkeypatch):
    from odyssey.core import tool_loop

    note = tmp_path / "note.txt"
    sub_replies = iter([{"tool_calls": [call("mcp__fake__save_note", text="sneaky")]}, {"content": "sub done"}])

    async def sub_chat(messages):
        return {"message": next(sub_replies)}

    monkeypatch.setattr(tool_loop, "chat_completion", sub_chat)
    main_replies = iter([{"tool_calls": [call("task", description="save a note")]}, {"content": "finished"}])

    async def chat(messages):
        return {"message": next(main_replies)}

    policy = ApprovalPolicy()

    async def go():
        async with mcp_client.connect([cfg(args=(str(note),), auto=())], policy, tmp_path):
            session = GraphSession(build_graph(chat, checkpointer=MemorySaver(), policy=policy), "t", policy)
            return await session.submit("delegate")

    assert run(go()).status == "done"
    assert not note.exists(), "the sub-agent used an MCP tool without approval"
