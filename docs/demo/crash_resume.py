"""Crash and resume demo for the graph engine.

``python docs/demo/crash_resume.py`` (needs ``odyssey[verify]``) does this in a throw-away directory:

1. runs the project's tests (one fails),
2. starts the agent in a first process: it reads the file and edits it, then the process is
   killed with SIGKILL before the turn is finished,
3. starts a second process on the same checkpoint file, which resumes the turn from the last
   checkpoint and ends it through the ledger gate (the tests must pass for the turn to count).

The model is a SCRIPTED stand-in: it replays three fixed replies, so this shows the engine's
checkpointing, resume and verification, not a model's ability. Everything else is real: the
tools, the SQLite checkpoint, the SIGKILL, pytest and the ledger.
"""
from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

BUGGY = '''def total_with_tax(prices, rate):
    """Sum of the prices plus tax."""
    return sum(prices) * rate
'''
TESTS = '''from calc import total_with_tax


def test_tax_is_added():
    assert total_with_tax([10, 20], 0.1) == 33.0


def test_no_tax():
    assert total_with_tax([5], 0.0) == 5.0
'''
CRASH_MARKER = "[crash point] the first two steps are saved; waiting to be killed"


def scripted_model(crash: bool):
    """Three fixed replies, chosen by how many tool results the conversation already holds."""

    async def chat(messages):
        done = sum(1 for m in messages if m.get("role") == "tool")
        if done:
            last = str([m for m in messages if m.get("role") == "tool"][-1].get("content", "")).strip().splitlines()
            print(f"  tool result: {last[0][:90] if last else ''}", flush=True)
        if done == 0:
            print("  model: read calc.py", flush=True)
            return {"message": {"tool_calls": [{"function": {"name": "read", "arguments": {"path": "calc.py"}}}]}}
        if done == 1:
            print("  model: edit calc.py (multiply -> 1 + rate)", flush=True)
            return {"message": {"tool_calls": [{"function": {"name": "edit", "arguments": {
                "path": "calc.py", "old_string": "sum(prices) * rate", "new_string": "sum(prices) * (1 + rate)"}}}]}}
        if crash:
            print(CRASH_MARKER, flush=True)
            await asyncio.sleep(3600)
        print("  model: done", flush=True)
        return {"message": {"content": "Fixed total_with_tax: the tax was multiplied instead of added."}}

    return chat


async def agent(workdir: Path, db: Path, ledger: Path, thread: str, crash: bool, resume: bool) -> None:
    from agent_ledger import ExpectedStateVerifier  # noqa: F401  (checks the optional extra is installed)

    from odyssey.core import workspace
    from odyssey.engine.graph import build_graph
    from odyssey.engine.policy import ApprovalPolicy
    from odyssey.engine.session import GraphSession, open_checkpointer
    from odyssey.engine.verify import ledger_gate
    from odyssey.tools import bash as bash_tool
    from odyssey.tools.registry import init_tools

    init_tools()
    workspace.set_workspace_root(workdir)
    bash_tool._working_dir = str(workdir)
    policy = ApprovalPolicy(approve_all=True)
    gate = ledger_gate([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"], ledger_path=ledger, timeout=120)
    async with open_checkpointer(db) as saver:
        graph = build_graph(scripted_model(crash), checkpointer=saver, policy=policy, acceptance=gate)
        session = GraphSession(graph, thread, policy, workspace=str(workdir), base=str(workdir))
        if resume:
            saved = [c["name"] for c in (await session.state()).get("tool_log", [])]
            print(f"  restored from the checkpoint: {len(saved)} tool calls already done ({', '.join(saved)})", flush=True)
            result = await session.resume()
            if result.resumed_in_flight:
                call = result.resumed_in_flight[0]
                print(f"  note: {call.get('name')} may have run before the crash", flush=True)
        else:
            result = await session.submit("Fix the failing test in calc.py.")
        if resume:
            after = len((await session.state()).get("tool_log", []))
            print(f"  tool calls after the resume: {after} ({'none repeated' if after == len(saved) else 'REPEATED'})", flush=True)
        print(f"turn status: {result.status}", flush=True)
        if result.verification:
            reason = str(result.verification.get('reason', '')).strip().splitlines()[-1].replace(sys.executable, "python")
            print(f"ledger gate: {'accepted' if result.verification.get('ok') else 'rejected'}: {reason}", flush=True)
        print(f"agent: {result.text}", flush=True)


def pytest(workdir: Path) -> None:
    out = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"], cwd=workdir, capture_output=True, text=True)
    lines = [l for l in out.stdout.strip().splitlines() if l.strip()]
    print("  " + (lines[-1] if lines else out.stderr.strip()[-100:]), flush=True)


def spawn(args: list[str]) -> subprocess.Popen:
    return subprocess.Popen([sys.executable, "-u", __file__, *args], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def drive() -> None:
    root = Path(tempfile.mkdtemp(prefix="odyssey-crash-demo-")).resolve()
    work = root / "project"
    work.mkdir()
    (work / "calc.py").write_text(BUGGY, encoding="utf-8")
    (work / "test_calc.py").write_text(TESTS, encoding="utf-8")
    db, ledger = root / "checkpoints.sqlite", root / "ledger.sqlite"
    base = [str(work), str(db), str(ledger), "demo"]

    print("$ pytest            # the project before the agent runs", flush=True)
    pytest(work)
    print("\n$ python crash_resume.py agent start     # process A: graph engine + ledger gate, scripted model", flush=True)
    first = spawn(["agent", *base, "start"])
    assert first.stdout is not None
    for line in first.stdout:
        print(line.rstrip(), flush=True)
        if CRASH_MARKER in line:
            break
    time.sleep(0.5)
    print(f"\n$ kill -9 {first.pid}      # process A is gone mid-turn", flush=True)
    os.kill(first.pid, signal.SIGKILL)
    first.wait()
    print(f"  (exit status {first.returncode}; the file on disk has the edit, the turn is unfinished)", flush=True)
    print("\n$ python crash_resume.py agent resume    # process B: new process, same checkpoint file", flush=True)
    second = spawn(["agent", *base, "resume"])
    assert second.stdout is not None
    for line in second.stdout:
        print(line.rstrip(), flush=True)
    second.wait()
    print("\n$ pytest            # independent check", flush=True)
    pytest(work)
    print("\nScripted model: this shows checkpoint, resume and the verification gate, not model ability.", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "agent":
        wd, db, ledger, thread, mode = sys.argv[2:7]
        asyncio.run(agent(Path(wd), Path(db), Path(ledger), thread, crash=(mode == "start"), resume=(mode == "resume")))
    else:
        drive()
