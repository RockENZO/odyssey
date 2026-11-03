"""The ledger completion gate on the graph engine, driven by a scripted model (no Ollama)."""
import asyncio
import sys

import pytest
from langgraph.checkpoint.memory import MemorySaver

pytest.importorskip("agent_ledger")

from agent_ledger import ClaimStatus, Ledger  # noqa: E402

from odyssey.core import workspace  # noqa: E402
from odyssey.engine.graph import build_graph  # noqa: E402
from odyssey.engine.policy import ApprovalPolicy  # noqa: E402
from odyssey.engine.session import GraphSession  # noqa: E402
from odyssey.engine.verify import ledger_gate  # noqa: E402
from odyssey.tools.registry import init_tools  # noqa: E402

# Passes only when out.txt says "ok".
CHECK = [sys.executable, "-c", "import pathlib, sys; sys.exit(0 if pathlib.Path('out.txt').read_text() == 'ok' else 1)"]


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


def session(tmp_path, chat, argv=CHECK):
    policy = ApprovalPolicy(approve_all=True)
    gate = ledger_gate(argv, ledger_path=tmp_path.parent / f"{tmp_path.name}-ledger" / "ledger.sqlite", timeout=30)
    graph = build_graph(chat, checkpointer=MemorySaver(), policy=policy, acceptance=gate)
    return GraphSession(graph, "t", policy, workspace=str(tmp_path), max_repairs=2)


def ledger_of(tmp_path):
    return Ledger(tmp_path.parent / f"{tmp_path.name}-ledger" / "ledger.sqlite")


def run(coro):
    return asyncio.run(coro)


def test_a_passing_check_commits_a_claim_with_the_turns_tool_calls_as_provenance(tmp_path):
    chat = scripted({"tool_calls": [call("write", path="out.txt", content="ok")]}, {"content": "done"})
    s = session(tmp_path, chat)
    result = run(s.submit("write ok"))
    assert result.status == "verified"
    assert result.verification["ok"] is True
    ledger = ledger_of(tmp_path)
    [claim] = ledger.committed("task_complete")
    log = run(s.state())["tool_log"]
    assert [c["id"] for c in claim.provenance["tool_calls"]] == [e["call_id"] for e in log]
    assert claim.provenance["tool_calls"][0]["name"] == "write"
    assert claim.value["check"] == CHECK
    ledger.close()


def test_a_failing_check_goes_back_to_the_model_and_the_repair_is_recorded(tmp_path):
    chat = scripted(
        {"tool_calls": [call("write", path="out.txt", content="wrong")]},
        {"content": "done"},                                              # gate fails here
        {"tool_calls": [call("write", path="out.txt", content="ok")]},
        {"content": "fixed"},                                             # gate passes here
    )
    result = run(session(tmp_path, chat).submit("write ok"))
    assert result.status == "verified"
    # the model was shown why the check failed
    assert any("acceptance check failed" in m.get("content", "") for m in chat.seen[2])
    ledger = ledger_of(tmp_path)
    assert len(ledger.committed("task_complete")) == 1 and ledger.rejected_count() == 1
    ledger.close()


def test_a_check_that_never_passes_ends_unverified_and_every_claim_is_rejected(tmp_path):
    bad = {"tool_calls": [call("write", path="out.txt", content="wrong")]}
    chat = scripted(bad, {"content": "done"}, bad, {"content": "done again"}, bad, {"content": "still done"})
    result = run(session(tmp_path, chat).submit("write ok"))
    assert result.status == "unverified"
    assert result.verification["ok"] is False and "exit" in result.verification["reason"].lower()
    ledger = ledger_of(tmp_path)
    assert not ledger.committed("task_complete") and ledger.rejected_count() == 3
    assert (tmp_path / "out.txt").read_text() == "wrong", "the gate must not roll the project back"
    ledger.close()


def test_a_turn_that_changed_nothing_is_not_checked(tmp_path):
    (tmp_path / "out.txt").write_text("wrong")           # the check would fail if it ran
    chat = scripted({"tool_calls": [call("read", path="out.txt")]}, {"content": "it says wrong"})
    result = run(session(tmp_path, chat).submit("what does out.txt say?"))
    assert result.status == "verified" and "nothing to check" in result.verification["reason"]
    assert not (tmp_path.parent / f"{tmp_path.name}-ledger").exists(), "no check ran, so no claim was recorded"


def test_only_this_turns_calls_count(tmp_path):
    """A write from an earlier turn must not make a later read-only turn look like a change."""
    chat = scripted(
        {"tool_calls": [call("write", path="out.txt", content="ok")]}, {"content": "one"},
        {"tool_calls": [call("read", path="out.txt")]}, {"content": "two"},
    )
    s = session(tmp_path, chat)
    assert run(s.submit("first")).status == "verified"
    second = run(s.submit("second"))
    assert "nothing to check" in second.verification["reason"]
    ledger = ledger_of(tmp_path)
    assert len(ledger.committed("task_complete")) == 1
    ledger.close()


def test_verify_needs_the_graph_engine():
    from typer.testing import CliRunner

    from odyssey.cli import app as app_module

    result = CliRunner().invoke(app_module.app, ["--verify", "pytest -q"])
    assert result.exit_code != 0 and "graph engine" in result.output


def test_a_turn_that_only_delegated_is_still_checked(tmp_path, monkeypatch):
    """A sub-agent's own calls are not in the graph's tool log, so `task` itself counts as a change."""
    from odyssey.core import tool_loop

    replies = [{"tool_calls": [call("write", path="out.txt", content="wrong")]}, {"content": "sub done"}]

    async def sub_chat(messages):
        return {"message": replies.pop(0)}

    monkeypatch.setattr(tool_loop, "chat_completion", sub_chat)
    chat = scripted({"tool_calls": [call("task", description="write out.txt")]}, {"content": "delegated"},
                    {"content": "still done"}, {"content": "really done"})
    result = run(session(tmp_path, chat).submit("delegate"))
    assert (tmp_path / "out.txt").read_text() == "wrong"      # the sub-agent changed the project (approve_all)
    assert result.status == "unverified", "the failing check must not be skipped"
    ledger = ledger_of(tmp_path)
    assert ledger.rejected_count() == 3
    ledger.close()


def test_crash_resume_demo_runs_end_to_end():
    """The demo script really kills a process mid-turn and finishes the turn in a new one."""
    import subprocess
    import sys
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "docs" / "demo" / "crash_resume.py"
    out = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "exit status -9" in out.stdout
    assert "turn status: verified" in out.stdout and "ledger gate: accepted" in out.stdout
    assert out.stdout.count("2 passed") == 1 and "2 failed" in out.stdout
