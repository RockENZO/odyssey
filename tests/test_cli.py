"""The installed config-path command is usable outside the source checkout."""
import os
from pathlib import Path
import subprocess
import sys


def test_installed_config_path_from_unrelated_directory(tmp_path):
    target = tmp_path / 'new' / 'config'
    env = dict(os.environ, ODYSSEY_CONFIG_DIR=str(target))
    env.pop('PYTHONPATH', None)
    executable = Path(sys.executable).parent / ('odyssey.exe' if os.name == 'nt' else 'odyssey')
    result = subprocess.run([str(executable), 'config-path'], cwd=tmp_path,
                            env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(target)
    assert target.is_dir()
    # Importing CLI configuration must not import inference or vector dependencies.
    probe = subprocess.run([sys.executable, '-c',
        "import sys; import odyssey.cli.app; assert 'ollama' not in sys.modules; "
        "assert 'chromadb' not in sys.modules"], cwd=tmp_path, env=env,
        capture_output=True, text=True, timeout=10)
    assert probe.returncode == 0, probe.stderr


def test_project_root_recognises_git_worktrees_and_submodules(tmp_path):
    """A linked worktree has a `.git` file, not a directory."""
    from odyssey.cli.repl import _find_project_root
    repo = tmp_path / "wt"
    (repo / "src").mkdir(parents=True)
    (repo / ".git").write_text("gitdir: /elsewhere/.git/worktrees/wt\n")
    assert _find_project_root(str(repo / "src")) == str(repo)
    plain = tmp_path / "plain"
    (plain / ".git").mkdir(parents=True)
    assert _find_project_root(str(plain)) == str(plain)


def test_root_options_reach_the_repl_subcommand(monkeypatch):
    from typer.testing import CliRunner

    from odyssey.cli import app as app_module

    seen = []
    monkeypatch.setattr(app_module, "_launch_repl", lambda engine="loop", resume=False, yes=False, verify="", mcp_config="": seen.append((engine, resume, yes)))
    runner = CliRunner()
    for args in (["--engine", "graph", "--resume", "repl"], ["repl", "--engine", "graph", "--resume"],
                 ["--engine", "graph", "repl", "--resume", "--yes"], ["--engine", "graph", "--resume"]):
        assert runner.invoke(app_module.app, args).exit_code == 0
    assert seen == [("graph", True, False), ("graph", True, False), ("graph", True, True), ("graph", True, False)]
    seen.clear()
    assert runner.invoke(app_module.app, ["repl"]).exit_code == 0
    assert seen == [("loop", False, False)]


def test_mcp_config_option_reaches_the_repl_and_is_validated(monkeypatch, tmp_path):
    from typer.testing import CliRunner

    from odyssey.cli import app as app_module

    good = tmp_path / "mcp.json"
    good.write_text('{"servers": {}}', encoding="utf-8")
    bad = tmp_path / "bad.json"
    bad.write_text('{"servers": {"a": {}}}', encoding="utf-8")
    seen = []

    async def fake_run_repl(cwd, engine, resume, yes, verify, mcp_config):
        seen.append((engine, mcp_config))

    monkeypatch.setattr("odyssey.cli.repl.run_repl", fake_run_repl)
    runner = CliRunner()
    for args in (["--engine", "graph", "--mcp-config", str(good), "repl"], ["repl", "--engine", "graph", "--mcp-config", str(good)]):
        assert runner.invoke(app_module.app, args).exit_code == 0
    assert seen == [("graph", str(good)), ("graph", str(good))]
    seen.clear()
    for args, message in ((["--engine", "graph", "--mcp-config", str(bad)], "command"),
                          (["--engine", "graph", "--mcp-config", str(tmp_path / "missing.json")], "cannot read"),
                          (["--mcp-config", str(good)], "graph engine")):
        result = runner.invoke(app_module.app, args)
        assert result.exit_code != 0 and message in result.output, (args, result.output)
    assert seen == []
