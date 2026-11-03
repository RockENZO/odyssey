"""The frozen task suite and the evaluation runner, driven by scripted models (no Ollama)."""
import asyncio
import json

import pytest

pytest.importorskip("agent_ledger")
pytest.importorskip("trajectory_replay")

from odyssey.evals import runner  # noqa: E402
from odyssey.evals.suite import SUITE_HASH, TASKS, get_tasks, suite_hash  # noqa: E402


def run_suite(tmp_path, chat_for, tasks=None, **kw):
    out = tmp_path / "out"
    kw.setdefault("model", "scripted")
    kw.setdefault("kind", "oracle")
    kw.setdefault("with_gold", False)           # the gold run doubles the time; only the test that needs it asks
    report = asyncio.run(runner.run_suite(tasks or list(TASKS), chat_for, out, **kw))
    return out, report


def test_the_suite_is_frozen():
    assert suite_hash() == SUITE_HASH, "a task changed: bump SUITE_VERSION and SUITE_HASH together"


def test_every_task_is_well_formed():
    ids = [t.id for t in TASKS]
    assert len(ids) == len(set(ids)) == 10
    for t in TASKS:
        assert t.tests and all(p in t.files for p in t.tests), t.id
        assert t.solution and all(p not in t.tests for p in t.solution), f"{t.id}: the solution must not touch the tests"


def test_unknown_task_ids_are_reported():
    with pytest.raises(KeyError, match="nope"):
        get_tasks(["off_by_one", "nope"])


def test_the_reference_solution_passes_every_task(tmp_path):
    _, report = run_suite(tmp_path, runner.oracle_chat)
    assert report["summary"]["passed"] == 10 and report["summary"]["gate_disagrees"] == 0


def test_trajectory_f1_is_scored_against_the_reference(tmp_path):
    tasks = get_tasks(["rename_function"])
    _, report = run_suite(tmp_path, runner.oracle_chat, tasks, with_gold=True)
    assert report["summary"]["mean_trajectory_f1"] == 1.0
    _, lazy = run_suite(tmp_path / "lazy", runner.lazy_chat, tasks, with_gold=True)
    assert lazy["summary"]["mean_trajectory_f1"] == 0.0


def test_a_model_that_changes_nothing_fails_every_task(tmp_path):
    """The checks are real: they fail on the starting files. The gate does not gate a turn that changed nothing."""
    _, report = run_suite(tmp_path, runner.lazy_chat)
    assert report["summary"]["passed"] == 0
    assert report["summary"]["gate_disagrees"] == 10


def test_editing_the_tests_does_not_pass(tmp_path):
    task = get_tasks(["off_by_one"])[0]

    def cheater(_task):
        replies = [
            {"tool_calls": [{"function": {"name": "write", "arguments": {"path": "tests/test_calc.py", "content": "def test_x():\n    pass\n"}}}]},
            {"content": "done"},
        ]
        queue = replies * 3                      # the same cheat in every repair round

        async def chat(messages):
            return {"message": queue.pop(0) if queue else {"content": "done"}}

        return chat

    _, report = run_suite(tmp_path, cheater, [task])
    [result] = report["results"]
    assert result["passed"] is False and "wrong content" in result["detail"]
    assert result["gate"] == "unverified" and result["repairs"] == 2


def test_a_failing_model_is_a_result_not_a_crash(tmp_path):
    def broken(_task):
        async def chat(messages):
            raise ConnectionError("ollama is not running")

        return chat

    _, report = run_suite(tmp_path, broken, get_tasks(["fizzbuzz", "slugify"]))
    assert [r["gate"] for r in report["results"]] == ["error", "error"]
    assert all("ConnectionError" in r["error"] and r["passed"] is False for r in report["results"])


def test_a_slow_model_times_out(tmp_path):
    def slow(_task):
        async def chat(messages):
            await asyncio.sleep(5)

        return chat

    _, report = run_suite(tmp_path, slow, get_tasks(["fizzbuzz"]), timeout=0.2)
    assert report["results"][0]["gate"] == "timeout" and report["results"][0]["passed"] is False


def test_the_report_is_written_and_says_when_it_is_not_a_model_result(tmp_path):
    out, report = run_suite(tmp_path, runner.oracle_chat, get_tasks(["off_by_one", "rename_function"]))
    data = json.loads((out / "report.json").read_text())
    assert data["environment"]["kind"] == "oracle" and data["environment"]["suite"]["frozen_hash_matches"] is True
    text = (out / "report.md").read_text()
    assert "not a model result" in text and "off_by_one" in text and "| Task | Passed |" in text
    assert (out / "cassettes" / "rename_function.json").exists()
    assert (out / "ledger.sqlite").exists()


def test_a_live_model_report_has_no_oracle_banner():
    env = runner.environment("some-model", "model")
    text = runner.render_markdown(env, [], runner.summarize([]))
    assert "not a model result" not in text


def test_the_model_gets_file_tools_only_and_the_registry_is_restored(tmp_path):
    from odyssey.tools.registry import get_all_tools, init_tools

    init_tools()
    before = sorted(t.name for t in get_all_tools())
    offered = []

    def spy(_task):
        from odyssey.tools.registry import get_all_tools as live

        async def chat(messages):
            offered.append(sorted(t.name for t in live()))
            return {"message": {"content": "no changes"}}

        return chat

    run_suite(tmp_path, spy, get_tasks(["fizzbuzz"]))
    assert set(offered[0]) == set(runner.EVAL_TOOLS)
    assert "bash" not in offered[0] and "save_memory" not in offered[0] and "web_search" not in offered[0]
    assert sorted(t.name for t in get_all_tools()) == before


def test_a_model_cannot_run_the_shell_unless_allowed(tmp_path):
    marker = tmp_path / "pwned"

    def shell_user(_task):
        queue = [
            {"tool_calls": [{"function": {"name": "bash", "arguments": {"command": f"touch {marker}"}}}]},
            {"content": "done"},
        ]

        async def chat(messages):
            return {"message": queue.pop(0) if queue else {"content": "done"}}

        return chat

    _, report = run_suite(tmp_path / "a", shell_user, get_tasks(["fizzbuzz"]))
    assert not marker.exists(), "the shell must not run by default"
    _, allowed = run_suite(tmp_path / "b", shell_user, get_tasks(["fizzbuzz"]), tools=runner.EVAL_TOOLS + ("bash",))
    assert marker.exists(), "--allow-bash should give the model the shell"
    assert "bash" in json.loads((tmp_path / "b" / "out" / "report.json").read_text())["environment"]["tools"]


def _writes(*files):
    """A scripted model that writes the given files, then says it is done (the same every round)."""
    def make(_task):
        calls = [{"tool_calls": [{"function": {"name": "write", "arguments": {"path": p, "content": c}}}]} for p, c in files]
        queue = (calls + [{"content": "done"}]) * 3

        async def chat(messages):
            return {"message": queue.pop(0) if queue else {"content": "done"}}

        return chat

    return make


def test_a_conftest_that_skips_the_tests_does_not_pass(tmp_path):
    """The pinned test text is unchanged and pytest would exit 0 with everything skipped."""
    skip_all = "import pytest\n\n\ndef pytest_collection_modifyitems(items):\n    for item in items:\n        item.add_marker(pytest.mark.skip(reason='x'))\n"
    _, report = run_suite(tmp_path, _writes(("conftest.py", skip_all)), get_tasks(["off_by_one"]))
    [result] = report["results"]
    assert result["passed"] is False
    assert result["gate"] == "unverified", "the gate runs the same hardened command"


def test_a_pytest_config_that_deselects_the_tests_does_not_pass(tmp_path):
    cfg = "[pytest]\naddopts = -k nothing_matches_this\n"
    _, report = run_suite(tmp_path, _writes(("pytest.ini", cfg)), get_tasks(["off_by_one"]))
    assert report["results"][0]["passed"] is False


def test_the_expected_test_count_is_checked(tmp_path):
    """Every task's reference solution runs exactly the number of tests the suite says it should."""
    _, report = run_suite(tmp_path, runner.oracle_chat)
    assert report["summary"]["passed"] == 10          # a wrong expected_tests would fail the oracle


def test_delegation_is_not_offered_to_an_evaluated_model(tmp_path):
    from odyssey.tools.registry import get_tool, init_tools

    init_tools()
    assert get_tool("task") is not None               # it exists in the registry...
    seen = []

    def spy(_task):
        async def chat(messages):
            seen.append(get_tool("task"))
            return {"message": {"content": "ok"}}

        return chat

    run_suite(tmp_path, spy, get_tasks(["fizzbuzz"]))
    assert seen == [None]                             # ...but not while a model is evaluated


def test_missing_pytest_is_reported_before_the_suite_starts(tmp_path, monkeypatch):
    import importlib.util

    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a, **k: None if name == "pytest" else real(name, *a, **k))
    with pytest.raises(RuntimeError, match=r"odyssey\[verify,eval\]"):
        run_suite(tmp_path, runner.oracle_chat, get_tasks(["fizzbuzz"]))
