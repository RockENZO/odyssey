"""Run the frozen suite through the graph engine and write a report.

For each task the runner copies the starting files into a fresh directory, lets the agent work on
it with the graph engine (no approval prompts; the check is the gate), then decides pass or fail
with its own check, run after the agent has finished: the project's tests, with the test files
pinned to their original text. That independent check is the result. The gate's verdict is
recorded next to it, because the two can differ (for example, a turn that changed nothing is not
gated, but the independent check still fails).

Needs ``odyssey[verify,eval]``.

Two kinds of run are distinguished in the report:

* ``model``: a live model through ``chat_completion`` (Ollama). These are the only results.
* ``oracle``: a scripted reference solution. It proves each task is solvable and each check is
  discriminating. It is NOT a model result and the report says so.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import platform
import shutil
import statistics
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

from langgraph.checkpoint.memory import MemorySaver

from odyssey.tools.registry import only_tools  # noqa: F401  (offer the model only some tools)
from odyssey.evals.suite import SUITE_HASH, SUITE_VERSION, Task, check_argv, get_tasks, suite_hash

ChatFn = Callable[[list[dict[str, Any]]], Awaitable[dict[str, Any]]]

# The tools an evaluated model gets. No shell, no web, and none of the tools that write to the
# user's own memory, journal or task list: a model runs unattended and approvals are off. The
# gate runs the tests and hands failures back, so the model does not need a shell to iterate.
# ``--allow-bash`` adds the shell back; it is not sandboxed.
EVAL_TOOLS = ("read", "write", "edit", "glob", "grep", "todo_write")


@dataclass
class TaskResult:
    id: str
    passed: bool                    # the independent check, after the run
    detail: str                     # why it passed or failed
    gate: str                       # the engine's status: verified / unverified / done / budget_exhausted / timeout / error
    repairs: int = 0
    model_calls: int = 0
    tool_calls: int = 0
    seconds: float = 0.0
    tools: list[str] = field(default_factory=list)
    trajectory_f1: float | None = None
    error: str = ""


def oracle_chat(task: Task) -> ChatFn:
    """A scripted model that writes the reference solution and then says it is done."""
    replies: list[dict[str, Any]] = [
        {"tool_calls": [{"function": {"name": "write", "arguments": {"path": p, "content": c}}}]}
        for p, c in task.solution.items()
    ] + [{"content": "done"}]

    async def chat(messages):
        return {"message": replies.pop(0) if replies else {"content": "done"}}

    return chat


def lazy_chat(task: Task) -> ChatFn:
    """A scripted model that changes nothing. Used to show that the checks fail on the starting files."""
    async def chat(messages):
        return {"message": {"content": "I looked at it and I think it is fine."}}

    return chat


def _independent_check(task: Task, workdir: Path, out_dir: Path) -> tuple[bool, str]:
    """The result: pinned tests, run with the agent's config and conftest ignored, and every expected test must pass.

    Unchanged test text and a zero exit code are not enough: an agent could add a conftest.py that
    skips the tests. So the run writes a JUnit report, and the report must show exactly the expected
    number of test cases with no failure, error or skip.
    """
    import xml.etree.ElementTree as ET

    from agent_ledger import AllOf, CommandVerifier, ExpectedStateVerifier

    junit = out_dir / "junit" / f"{task.id}.xml"
    junit.parent.mkdir(parents=True, exist_ok=True)
    junit.unlink(missing_ok=True)
    pins = ExpectedStateVerifier({p: task.files[p] for p in task.tests})
    verdict = AllOf(pins, CommandVerifier(check_argv(_empty_config(out_dir)) + ["--junitxml", str(junit)], timeout=120)).check(workdir, [])
    if not verdict.ok:
        return False, verdict.reason
    try:
        suite = ET.parse(junit).getroot()
        suite = suite if suite.tag == "testsuite" else suite.find("testsuite")
        tests, bad = int(suite.get("tests", 0)), sum(int(suite.get(k, 0)) for k in ("failures", "errors", "skipped"))
    except Exception as exc:
        return False, f"could not read the test report: {exc}"
    if tests != task.expected_tests or bad:
        return False, f"expected {task.expected_tests} tests to run and pass, got {tests} run with {bad} failed, errored or skipped"
    return True, f"{tests} tests passed"


def _shapes(calls: list) -> list:
    """Calls reduced to tool name and target path, so a rewritten file is compared by where, not by its exact text.

    Scoring the full arguments gave 0.00 for every real model: it would have to reproduce the
    reference file byte for byte (and the model that read first or used edit would never match).
    """
    import dataclasses

    keep = ("path", "file_path", "pattern")
    return [dataclasses.replace(c, arguments={k: v for k, v in c.arguments.items() if k in keep}, result=None, error=None) for c in calls]


def _empty_config(out_dir: Path) -> str:
    """An empty pytest config outside the task directory, so a pytest.ini the agent adds is never read."""
    path = out_dir / "empty-pytest.ini"
    path.write_text("", encoding="utf-8")
    return str(path)


async def run_task(
    task: Task,
    chat: ChatFn,
    out_dir: Path,
    *,
    timeout: float = 600.0,
    gold: list | None = None,
    max_repairs: int = 2,
    keep: bool = False,
    tools: tuple[str, ...] = EVAL_TOOLS,
) -> tuple[TaskResult, list]:
    """Run one task. Returns the result and the executed tool calls as cassette steps."""
    out_dir = out_dir.resolve()  # the checks run in the task directory, so a relative path would point into it
    from agent_ledger import ExpectedStateVerifier

    from odyssey.core import workspace
    from odyssey.engine.cassette import calls_from_state
    from odyssey.engine.graph import build_graph
    from odyssey.engine.policy import ApprovalPolicy
    from odyssey.engine.session import GraphSession
    from odyssey.engine.verify import ledger_gate
    from odyssey.tools import bash as bash_tool
    from odyssey.tools.registry import init_tools

    init_tools()
    workdir = Path(tempfile.mkdtemp(prefix=f"odyssey-eval-{task.id}-")).resolve()
    for rel, text in task.files.items():
        target = workdir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")

    previous_dir = getattr(bash_tool, "_working_dir", None)
    workspace.set_workspace_root(workdir)
    bash_tool._working_dir = str(workdir)
    policy = ApprovalPolicy(approve_all=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    gate = ledger_gate(
        check_argv(_empty_config(out_dir)),
        ledger_path=out_dir / "ledger.sqlite",
        timeout=120,
        extra=ExpectedStateVerifier({p: task.files[p] for p in task.tests}),
    )
    graph = build_graph(chat, checkpointer=MemorySaver(), policy=policy, acceptance=gate)
    session = GraphSession(graph, f"eval-{task.id}", policy, workspace=str(workdir), base=str(workdir), max_repairs=max_repairs)

    started = time.monotonic()
    gate_status, error = "error", ""
    try:
        try:
            with only_tools(tools):
                turn = await asyncio.wait_for(session.submit(task.prompt), timeout)
            gate_status = turn.status
        except asyncio.TimeoutError:
            gate_status = "timeout"
        except Exception as exc:  # a model or connection failure is a result, not a crash of the whole run
            error = f"{type(exc).__name__}: {exc}"
        seconds = time.monotonic() - started
        state = await session.state()
        passed, detail = _independent_check(task, workdir, out_dir)
        calls = calls_from_state(state)
    finally:
        workspace.set_workspace_root(None)
        bash_tool._working_dir = previous_dir
        if not keep:
            shutil.rmtree(workdir, ignore_errors=True)

    result = TaskResult(
        id=task.id, passed=passed, detail=detail.strip().splitlines()[-1] if detail.strip() else "", gate=gate_status,
        repairs=int(state.get("repairs", 0)), model_calls=int(state.get("model_calls", 0)),
        tool_calls=len(calls), seconds=round(seconds, 2), tools=[c.name for c in calls], error=error,
    )
    if gold:
        from trajectory_replay import score
        result.trajectory_f1 = round(score(_shapes(calls), _shapes(gold)).f1, 3)
    (out_dir / "cassettes").mkdir(parents=True, exist_ok=True)
    from trajectory_replay import save
    save(calls, out_dir / "cassettes" / f"{task.id}.json")
    return result, calls


async def gold_trajectory(task: Task, scratch: Path) -> list:
    """The reference solution's tool calls, produced by running it through the same engine."""
    scratch.mkdir(parents=True, exist_ok=True)
    _, calls = await run_task(task, oracle_chat(task), scratch)
    return calls


def environment(model: str, kind: str) -> dict[str, Any]:
    import importlib.metadata as md

    def version(name: str) -> str:
        try:
            return md.version(name)
        except md.PackageNotFoundError:
            return "not installed"

    return {
        "kind": kind,
        "model": model,
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "device": f"{platform.system()} {platform.machine()}, Python {platform.python_version()}",
        "suite": {"version": SUITE_VERSION, "hash": suite_hash(), "frozen_hash_matches": suite_hash() == SUITE_HASH},
        "packages": {n: version(n) for n in ("odyssey", "langgraph", "ollama", "trajectory-replay", "transactional-agent-ledger")},
    }


def summarize(results: list[TaskResult]) -> dict[str, Any]:
    n = len(results)
    f1s = [r.trajectory_f1 for r in results if r.trajectory_f1 is not None]
    return {
        "tasks": n,
        "passed": sum(r.passed for r in results),
        "pass_rate": round(sum(r.passed for r in results) / n, 3) if n else 0.0,
        "gate_verified": sum(r.gate == "verified" for r in results),
        "gate_disagrees": sum((r.gate == "verified") != r.passed for r in results),
        "median_tool_calls": statistics.median(r.tool_calls for r in results) if n else 0,
        "total_seconds": round(sum(r.seconds for r in results), 1),
        "mean_trajectory_f1": round(statistics.mean(f1s), 3) if f1s else None,
    }


def render_markdown(env: dict[str, Any], results: list[TaskResult], summary: dict[str, Any]) -> str:
    lines = [f"# Evaluation report: {env['model']}", ""]
    if env["kind"] == "oracle":
        lines += ["> **Scripted run. This is not a model result.** It validates the harness: the reference solution must pass every task and a do-nothing model must fail every task.", ""]
    lines += [
        f"- Date: {env['date']}",
        f"- Device: {env['device']}",
        f"- Tools offered to the model: {', '.join(env.get('tools', []))}" + ("" if "bash" not in env.get("tools", []) else " (the shell is not sandboxed)"),
        f"- Suite: version {env['suite']['version']}, hash `{env['suite']['hash'][:12]}`"
        + ("" if env["suite"]["frozen_hash_matches"] else " **(does not match the frozen hash)**"),
        "- Packages: " + ", ".join(f"{k} {v}" for k, v in env["packages"].items()),
        "",
        "| Task | Passed | Gate | Repairs | Model calls | Tool calls | Seconds | Trajectory F1 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in results:
        f1 = "" if r.trajectory_f1 is None else f"{r.trajectory_f1:.2f}"
        lines.append(f"| {r.id} | {'yes' if r.passed else 'no'} | {r.gate} | {r.repairs} | {r.model_calls} | {r.tool_calls} | {r.seconds} | {f1} |")
    lines += [
        "",
        f"**Passed {summary['passed']} of {summary['tasks']}** (independent check, test files pinned). "
        f"The gate reported verified for {summary['gate_verified']}; it disagreed with the independent check on {summary['gate_disagrees']}.",
        "",
        "Trajectory F1 compares the model's tool calls with the reference solution's. A coding task has many valid trajectories "
        "(reading first, editing instead of writing), so treat it as a diagnostic, not a score.",
        "",
    ]
    return "\n".join(lines)


async def run_suite(
    tasks: list[Task],
    chat_for: Callable[[Task], ChatFn],
    out_dir: Path,
    *,
    model: str,
    kind: str,
    timeout: float = 600.0,
    with_gold: bool = True,
    tools: tuple[str, ...] = EVAL_TOOLS,
) -> dict[str, Any]:
    import importlib.util

    if importlib.util.find_spec("pytest") is None:
        raise RuntimeError("the checks run pytest, which is not installed: pip install 'odyssey[verify,eval]'")
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[TaskResult] = []
    for task in tasks:
        gold = None
        if with_gold:
            try:
                import trajectory_replay  # noqa: F401
                gold = await gold_trajectory(task, out_dir / "gold-scratch")
            except ImportError:
                gold = None
        result, _ = await run_task(task, chat_for(task), out_dir, timeout=timeout, gold=gold, tools=tools)
        results.append(result)
        print(f"{task.id:18} passed={result.passed!s:5} gate={result.gate:16} tools={result.tool_calls:3} {result.seconds:7.1f}s", flush=True)
    shutil.rmtree(out_dir / "gold-scratch", ignore_errors=True)
    env = environment(model, kind)
    env["tools"] = list(tools)
    summary = summarize(results)
    report = {"environment": env, "summary": summary, "results": [asdict(r) for r in results]}
    (out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (out_dir / "report.md").write_text(render_markdown(env, results, summary), encoding="utf-8")
    return report


async def _model_name(model: str | None) -> str:
    """Model name and digest when Ollama can say; otherwise just the name. Never guessed."""
    from odyssey.llm.client import get_client, get_fast_model

    name = model or get_fast_model()
    try:
        info = await get_client().show(name)
        digest = getattr(info, "digest", None) or (info.get("digest") if hasattr(info, "get") else None)
        return f"{name} (digest {str(digest)[:12]})" if digest else name
    except Exception:
        return f"{name} (digest unavailable)"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m odyssey.evals.runner", description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True, help="directory for report.json, report.md, cassettes/ and ledger.sqlite")
    parser.add_argument("--tasks", help="comma-separated task ids (default: all)")
    parser.add_argument("--model", help="Ollama model (default: the configured fast model)")
    parser.add_argument("--timeout", type=float, default=600.0, help="seconds per task")
    parser.add_argument("--allow-bash", action="store_true", help="also give the model the shell (not sandboxed; the default is file tools only)")
    parser.add_argument("--oracle", action="store_true", help="scripted reference solution instead of a model (not a model result)")
    parser.add_argument("--lazy", action="store_true", help="scripted model that changes nothing (the checks must fail)")
    args = parser.parse_args(argv)

    tasks = get_tasks(args.tasks.split(",") if args.tasks else None)
    if args.oracle or args.lazy:
        kind = "oracle"
        name = "scripted reference solution" if args.oracle else "scripted do-nothing model"
        chat_for = oracle_chat if args.oracle else lazy_chat
    else:
        from functools import partial

        from odyssey.llm.client import chat_completion

        kind = "model"
        name = asyncio.run(_model_name(args.model))
        # Temperature 0 and an explicit model: the report must say what was measured.
        model = args.model

        def chat_for(task, _chat=partial(chat_completion, model=model, temperature=0.0)):
            return _chat

    tools = EVAL_TOOLS + ("bash",) if args.allow_bash else EVAL_TOOLS
    report = asyncio.run(run_suite(tasks, chat_for, args.out, model=name, kind=kind, timeout=args.timeout, tools=tools))
    print(json.dumps(report["summary"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
