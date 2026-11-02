"""Kill the process for real (os._exit mid-turn) and resume from the SQLite checkpoint."""
import subprocess
import sys
import textwrap

SCRIPT = textwrap.dedent('''
    import asyncio, os, sys
    from pathlib import Path
    from odyssey.core import workspace
    from odyssey.engine.graph import build_graph
    from odyssey.engine.policy import ApprovalPolicy
    from odyssey.engine.session import GraphSession, open_checkpointer
    from odyssey.tools.registry import init_tools

    mode, root = sys.argv[1], Path(sys.argv[2])
    init_tools()
    workspace.set_workspace_root(root)
    log = root / "log.txt"

    calls = {"n": 0}
    async def chat(messages):
        calls["n"] += 1
        if mode == "first":
            if calls["n"] == 1:
                return {"message": {"tool_calls": [{"function": {"name": "bash", "arguments": {"command": f"echo ran >> {log}"}}}]}}
            os._exit(137)  # killed during the second model call
        return {"message": {"content": "finished after resume"}}

    async def main():
        policy = ApprovalPolicy(approve_all=True)
        async with open_checkpointer(root.parent / (root.name + ".sqlite")) as saver:
            session = GraphSession(build_graph(chat, checkpointer=saver, policy=policy), "job", policy)
            result = await (session.submit("do the thing") if mode == "first" else session.resume())
            print(result.status, "|", result.text)

    asyncio.run(main())
''')


def run(script, mode, root):
    return subprocess.run([sys.executable, str(script), mode, str(root)], capture_output=True, text=True, timeout=60)


def test_a_killed_process_resumes_without_rerunning_finished_work(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    script = tmp_path / "script.py"
    script.write_text(SCRIPT)

    killed = run(script, "first", root)
    assert killed.returncode == 137, killed.stderr
    assert (root / "log.txt").read_text().splitlines() == ["ran"]

    resumed = run(script, "second", root)
    assert resumed.returncode == 0, resumed.stderr
    assert "done | finished after resume" in resumed.stdout
    assert (root / "log.txt").read_text().splitlines() == ["ran"], "the bash call ran again after the kill"
