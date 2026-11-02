"""Export a graph thread as a trajectory-replay cassette, and replay it through the engine."""
import asyncio

import pytest
from langgraph.checkpoint.memory import MemorySaver

trajectory_replay = pytest.importorskip("trajectory_replay")

from trajectory_replay import ReplayMismatch, Replayer, load, score  # noqa: E402

from odyssey.cli import repl  # noqa: E402
from odyssey.core import workspace  # noqa: E402
from odyssey.engine.cassette import calls_from_state, save_cassette  # noqa: E402
from odyssey.engine.graph import build_graph  # noqa: E402
from odyssey.engine.policy import ApprovalPolicy  # noqa: E402
from odyssey.engine.session import GraphSession  # noqa: E402
from odyssey.tools.registry import get_tool, init_tools  # noqa: E402


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    init_tools()
    workspace.set_workspace_root(tmp_path)
    monkeypatch.setenv("ODYSSEY_DATA_DIR", str(tmp_path.parent / f"{tmp_path.name}-data"))
    monkeypatch.setattr(repl.console, "width", 200)
    yield
    workspace.set_workspace_root(None)


def call(name, **arguments):
    return {"function": {"name": name, "arguments": arguments}}


def scripted(*replies):
    queue = list(replies)

    async def chat(messages):
        return {"message": queue.pop(0)}

    return chat


def session(chat, *, policy=None):
    policy = policy or ApprovalPolicy(approve_all=True)
    return GraphSession(build_graph(chat, checkpointer=MemorySaver(), policy=policy), "t", policy)


def run(coro):
    return asyncio.run(coro)


SCRIPT = (
    {"tool_calls": [call("write", path="a.txt", content="hello")]},
    {"tool_calls": [call("read", path="a.txt")]},
    {"content": "done"},
)


def test_the_cassette_holds_each_executed_call_with_its_id_and_result(tmp_path):
    s = session(scripted(*SCRIPT))
    run(s.submit("go"))
    state = run(s.state())
    calls = calls_from_state(state)
    assert [c.name for c in calls] == ["write", "read"]
    assert [c.call_id for c in calls] == [e["call_id"] for e in state["tool_log"]]
    assert calls[1].arguments == {"path": "a.txt"} and "hello" in calls[1].result


def test_saved_cassette_round_trips(tmp_path):
    s = session(scripted(*SCRIPT))
    run(s.submit("go"))
    path = tmp_path.parent / "out" / "c.json"
    assert save_cassette(run(s.state()), path) == 2
    assert [c.name for c in load(path)] == ["write", "read"]


def test_a_rejected_call_is_not_in_the_cassette(tmp_path):
    s = session(scripted({"tool_calls": [call("write", path="a.txt", content="x")]}, {"content": "ok, skipped"}), policy=ApprovalPolicy())

    async def go():
        await s.submit("go")
        await s.decide(False, reason="no")
        return await s.state()

    state = run(go())
    assert [e["rejected"] for e in state["tool_log"]] == [True]
    assert calls_from_state(state) == []


def test_one_turn_can_be_exported_on_its_own(tmp_path):
    s = session(scripted(*SCRIPT, {"tool_calls": [call("read", path="a.txt")]}, {"content": "again"}))

    async def go():
        await s.submit("first")
        first = (await s.state())["turn_id"]
        await s.submit("second")
        return first, await s.state()

    first, state = run(go())
    assert [c.name for c in calls_from_state(state)] == ["write", "read", "read"]
    assert [c.name for c in calls_from_state(state, turn_id=state["turn_id"])] == ["read"]
    assert [c.name for c in calls_from_state(state, turn_id=first)] == ["write", "read"]


def test_cassette_command_saves_the_thread(tmp_path, capsys):
    from odyssey.core.state import Conversation

    s = session(scripted(*SCRIPT))
    target = tmp_path.parent / "cmd" / "t.json"

    async def go():
        await s.submit("go")
        return await repl._handle_graph_slash(f"/cassette {target}", s, Conversation())

    assert run(go()) is True
    assert len(load(target)) == 2 and "Saved 2 tool calls" in capsys.readouterr().out


def _replay_through_the_engine(replayer, monkeypatch, chat):
    """Swap each tool's function for the cassette's replay and run the same model script again."""
    for name, fn in replayer.async_tools().items():
        monkeypatch.setattr(get_tool(name), "fn", fn)
    s = session(chat)
    run(s.submit("go"))
    return run(s.state())


def test_a_recorded_run_replays_through_the_engine_without_touching_files(tmp_path, monkeypatch):
    live = session(scripted(*SCRIPT))
    run(live.submit("go"))
    cassette = calls_from_state(run(live.state()))
    (tmp_path / "a.txt").unlink()                      # the replay must not need the file

    replayer = Replayer(cassette)
    state = _replay_through_the_engine(replayer, monkeypatch, scripted(*SCRIPT))
    replayer.assert_exhausted()
    assert not (tmp_path / "a.txt").exists(), "the replayed write must not run"
    assert score(calls_from_state(state), cassette).exact


def test_a_diverging_run_is_caught_by_the_replay(tmp_path, monkeypatch):
    live = session(scripted(*SCRIPT))
    run(live.submit("go"))
    cassette = calls_from_state(run(live.state()))

    replayer = Replayer(cassette)
    changed = (
        {"tool_calls": [call("write", path="a.txt", content="DIFFERENT")]},
        {"tool_calls": [call("read", path="a.txt")]},
        {"content": "done"},
    )
    _replay_through_the_engine(replayer, monkeypatch, scripted(*changed))
    with pytest.raises(ReplayMismatch):
        replayer.assert_exhausted()                      # the mismatching write never consumed a step


def test_export_without_the_package_says_how_to_install_it(monkeypatch):
    import builtins

    real = builtins.__import__

    def fake(name, *a, **k):
        if name == "trajectory_replay":
            raise ImportError(name)
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake)
    with pytest.raises(RuntimeError, match=r"odyssey\[eval\]"):
        calls_from_state({"tool_log": []})


def test_cassette_command_reports_a_bad_path_and_keeps_the_session(tmp_path, capsys):
    from odyssey.core.state import Conversation

    s = session(scripted(*SCRIPT))
    blocker = tmp_path.parent / "a-file"
    blocker.write_text("not a directory")

    async def go():
        await s.submit("go")
        # a regular file is an ancestor of the target, and the target is a directory
        first = await repl._handle_graph_slash(f"/cassette {blocker}/c.json", s, Conversation())
        second = await repl._handle_graph_slash(f"/cassette {tmp_path}", s, Conversation())
        return first, second

    assert run(go()) == (True, True)
    assert capsys.readouterr().out.count("Errno") == 2
