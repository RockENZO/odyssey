import pytest
import sqlite3
from pathlib import Path
import tempfile

from odyssey.storage.db import add_task, list_tasks, complete_task, delete_task, get_task


@pytest.fixture(autouse=True)
def _patch_db(monkeypatch, tmp_path):
    db_path = tmp_path / "test.db"

    def _get_db():
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                description TEXT DEFAULT '',
                priority TEXT DEFAULT 'medium',
                status TEXT DEFAULT 'pending',
                due_date TEXT,
                tags TEXT DEFAULT '',
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS journal_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                sentiment TEXT DEFAULT '',
                summary TEXT DEFAULT '',
                entry_date TEXT DEFAULT (date('now')),
                created_at TEXT DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                embedding_id TEXT,
                tags TEXT DEFAULT '',
                source TEXT DEFAULT 'manual',
                created_at TEXT DEFAULT (datetime('now'))
            );
        """)
        conn.commit()
        return conn

    monkeypatch.setattr("odyssey.storage.db._conn", None)
    monkeypatch.setattr("odyssey.storage.db.get_db", _get_db)


class TestTasks:
    def test_add_task(self):
        task_id = add_task("test task", priority="high", due_date="2026-06-15")
        assert task_id > 0

    def test_list_tasks(self):
        add_task("task 1")
        add_task("task 2", priority="high")
        tasks = list_tasks()
        assert len(tasks) == 2
        titles = {t["title"] for t in tasks}
        assert "task 1" in titles
        assert "task 2" in titles

    def test_list_tasks_filtered(self):
        add_task("pending task")
        add_task("done task")
        complete_task(2)
        pending = list_tasks(status="pending")
        assert len(pending) == 1
        assert pending[0]["title"] == "pending task"

    def test_complete_task(self):
        task_id = add_task("to complete")
        assert complete_task(task_id) is True
        task = get_task(task_id)
        assert task["status"] == "done"

    def test_complete_nonexistent_task(self):
        assert complete_task(9999) is False

    def test_delete_task(self):
        task_id = add_task("to delete")
        assert delete_task(task_id) is True
        assert get_task(task_id) is None

    def test_get_task(self):
        task_id = add_task("get me")
        task = get_task(task_id)
        assert task is not None
        assert task["title"] == "get me"

    def test_get_nonexistent_task(self):
        assert get_task(9999) is None
