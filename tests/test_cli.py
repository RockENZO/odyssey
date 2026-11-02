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
