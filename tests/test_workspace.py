"""File and search tools stay inside the workspace root once one is set."""
import asyncio
import os

import pytest

from odyssey.core import workspace
from odyssey.tools import file_ops, search_tools


@pytest.fixture(autouse=True)
def _reset():
    yield
    workspace.set_workspace_root(None)


def run(coro):
    return asyncio.run(coro)


def test_no_root_means_no_restriction(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("hello")
    assert "hello" in run(file_ops.read_file(str(f)))


def test_paths_inside_the_root_work_and_relative_paths_resolve_against_it(tmp_path):
    workspace.set_workspace_root(tmp_path)
    assert "Written" in run(file_ops.write_file("sub/new.txt", "data"))
    assert (tmp_path / "sub" / "new.txt").read_text() == "data"
    assert "data" in run(file_ops.read_file("sub/new.txt"))


@pytest.mark.parametrize("target", ["../escape.txt", "/etc/passwd", "~/x"])
def test_paths_outside_the_root_are_refused(tmp_path, target):
    root = tmp_path / "root"
    root.mkdir()
    workspace.set_workspace_root(root)
    for result in (
        run(file_ops.read_file(target)),
        run(file_ops.write_file(target, "x")),
        run(file_ops.edit_file(target, "a", "b")),
    ):
        assert result.startswith("Error: path"), result
        assert "outside the workspace" in result
    assert not (tmp_path / "escape.txt").exists()


def test_symlink_pointing_outside_is_refused(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "secret.txt"
    outside.write_text("secret")
    (root / "link.txt").symlink_to(outside)
    workspace.set_workspace_root(root)
    assert "outside the workspace" in run(file_ops.read_file("link.txt"))


def test_a_sibling_directory_with_the_same_prefix_is_not_inside(tmp_path):
    root = tmp_path / "proj"
    sibling = tmp_path / "proj-other"
    root.mkdir()
    sibling.mkdir()
    (sibling / "f.txt").write_text("x")
    workspace.set_workspace_root(root)
    assert "outside the workspace" in run(file_ops.read_file(str(sibling / "f.txt")))


def test_glob_and_grep_do_not_leave_the_root(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "in.py").write_text("needle\n")
    (tmp_path / "out.py").write_text("needle\n")
    workspace.set_workspace_root(root)
    assert "in.py" in run(search_tools.glob("*.py"))
    assert "out.py" not in run(search_tools.glob("../*.py"))
    assert "outside the workspace" in run(search_tools.glob("*.py", path=str(tmp_path)))
    assert "outside the workspace" in run(search_tools.grep("needle", path=str(tmp_path)))
